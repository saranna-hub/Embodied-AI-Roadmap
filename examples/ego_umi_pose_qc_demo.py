"""教学用合成 EGO/UMI 标定位姿闭环：不是 Kalibr 或真实设备测量结果。依赖 numpy、opencv-python。"""
import argparse
import numpy as np
import cv2


def make_T(rotvec=(0, 0, 0), trans=(0, 0, 0)):
    R, _ = cv2.Rodrigues(np.asarray(rotvec, dtype=np.float64))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = trans
    return T


def inv_T(T):
    result = np.eye(4)
    result[:3, :3] = T[:3, :3].T
    result[:3, 3] = -(T[:3, :3].T @ T[:3, 3])
    return result


def square_corners(size):
    """OpenCV SOLVEPNP_IPPE_SQUARE 规定的 4 角顺序。"""
    h = size / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


def fisheye_project(object_pts, T_cam_obj, K, D):
    rvec, _ = cv2.Rodrigues(T_cam_obj[:3, :3])
    tvec = np.ascontiguousarray(T_cam_obj[:3, 3].reshape(3, 1))
    px, _ = cv2.fisheye.projectPoints(
        np.ascontiguousarray(object_pts.reshape(-1, 1, 3)),
        np.ascontiguousarray(rvec), tvec, K, D
    )
    return px.reshape(-1, 2)


def fit_fisheye_tag(object_pts, px, K, D):
    """原始鱼眼角点 -> 去畸变归一化点 -> IPPE 双解 -> 原鱼眼重投影选解。
    单帧最小残差仍受平面双解影响；真机应辅以多帧/双目/先验约束。
    """
    normalized = cv2.fisheye.undistortPoints(
        px.reshape(-1, 1, 2), K, D
    ).reshape(-1, 2)
    result = cv2.solvePnPGeneric(
        object_pts, normalized, np.eye(3), None,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    success, rvecs, tvecs = result[:3]
    if not success or len(rvecs) == 0:
        return None, float("inf"), 0
    candidates = []
    for rvec, tvec in zip(rvecs, tvecs):
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = np.asarray(tvec).reshape(3)
        pts_cam = (R @ object_pts.T).T + T[:3, 3]
        if np.any(pts_cam[:, 2] <= 0):
            continue
        predicted = fisheye_project(object_pts, T, K, D)
        rms = float(np.sqrt(np.mean(np.sum((predicted - px) ** 2, axis=1))))
        candidates.append((rms, T))
    if not candidates:
        return None, float("inf"), len(rvecs)
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1], candidates[0][0], len(candidates)


def rotation_error_deg(T1, T2):
    R = T1[:3, :3].T @ T2[:3, :3]
    cosine = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def run(n=80, sigma_px=0.1, seed=7):
    rng = np.random.default_rng(seed)
    K = np.array([[460., 0, 640], [0, 460., 360], [0, 0, 1.]])
    D = np.array([-0.025, 0.005, -0.0008, 0.0001])
    W_T_W = np.eye(4)
    U_T_E = make_T(trans=(0.01, -0.02, 0.12))
    corners_w = square_corners(0.1)
    corners_u = square_corners(0.06)
    errs_mm, errs_deg, reprj, accepted = [], [], [], []
    for i in range(n):
        t = i / 30.0
        W_T_C = make_T((0.07, -0.09, 0.02), (0.02*np.sin(t), -0.01, -0.8))
        W_T_U = make_T(
            (0.14, -0.12, 0.09 + 0.12*t),
            (0.16+0.08*t, 0.07, 0.02)
        )
        if i in (18, 19, 55):  # 标签遮挡：必须保留 invalid，而非伪造真值
            accepted.append(False)
            continue
        C_T_W = inv_T(W_T_C) @ W_T_W
        C_T_U = inv_T(W_T_C) @ W_T_U
        px_w = fisheye_project(corners_w, C_T_W, K, D) + rng.normal(0, sigma_px, (4, 2))
        px_u = fisheye_project(corners_u, C_T_U, K, D) + rng.normal(0, sigma_px, (4, 2))
        C_T_W_hat, ew, _ = fit_fisheye_tag(corners_w, px_w, K, D)
        C_T_U_hat, eu, _ = fit_fisheye_tag(corners_u, px_u, K, D)
        valid = C_T_W_hat is not None and C_T_U_hat is not None and max(ew, eu) < 1.5
        accepted.append(valid)
        if not valid:
            continue
        W_T_E_hat = inv_T(C_T_W_hat) @ C_T_U_hat @ U_T_E
        W_T_E_true = W_T_U @ U_T_E
        errs_mm.append(1000*np.linalg.norm(W_T_E_hat[:3, 3] - W_T_E_true[:3, 3]))
        errs_deg.append(rotation_error_deg(W_T_E_hat, W_T_E_true))
        reprj.append(max(ew, eu))
    if not errs_mm:
        raise RuntimeError("没有有效标签：请检查视场、噪声与尺寸")
    print("DEMO: 纯合成鱼眼角点，不是任何真实设备验收数据")
    print(f"frames={n} valid={sum(accepted)} valid_ratio={np.mean(accepted):.3f}")
    print(f"p95 reprojection={np.percentile(reprj, 95):.3f}px")
    print(f"p95 world tool translation={np.percentile(errs_mm, 95):.2f}mm")
    print(f"p95 world tool rotation={np.percentile(errs_deg, 95):.2f}deg")
    print("time-offset example: v=0.40m/s, dt=15ms -> 6mm position mismatch")
    print("time-offset example: omega=120deg/s, dt=15ms -> 1.8deg angle mismatch")
    print("重要：重投影误差不能替代独立真值和未见场景重复测量")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=80)
    p.add_argument("--noise-px", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    run(args.frames, args.noise_px, args.seed)
