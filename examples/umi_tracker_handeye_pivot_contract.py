#!/usr/bin/env python3
"""CPU synthetic sanity checks for OpenCV hand-eye and UMI Tracker TCP pivot.

No robot, OpenVR, real PnP pixels, or device accuracy measurements are made.
"""
import numpy as np
import cv2

RNG = np.random.default_rng(20261105)


def transform(r, t):
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = t
    return m


def random_rotation():
    axis = RNG.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = RNG.uniform(-1.4, 1.4)
    return cv2.Rodrigues(axis * angle)[0]


def test_handeye():
    # X = tracker-to-rigid-camera, Y = tracking-world-to-stationary-board.
    x = transform(cv2.Rodrigues(np.array([0.17, -0.11, 0.23]))[0],
                  np.array([0.05, -0.04, 0.09]))
    y = transform(cv2.Rodrigues(np.array([-0.2, 0.18, 0.08]))[0],
                  np.array([0.4, 0.2, 0.7]))
    a_list, b_list = [], []
    for _ in range(30):
        a = transform(random_rotation(),
                      RNG.uniform(-0.2, 0.2, 3) + [0.2, 0.1, 0.35])
        b = np.linalg.inv(x) @ np.linalg.inv(a) @ y
        a_list.append(a)
        b_list.append(b)
    r, t = cv2.calibrateHandEye(
        [a[:3, :3] for a in a_list], [a[:3, 3] for a in a_list],
        [b[:3, :3] for b in b_list], [b[:3, 3] for b in b_list],
        method=cv2.CALIB_HAND_EYE_PARK,
    )
    estimated = transform(r, t.reshape(3))
    trans_error_m = np.linalg.norm(estimated[:3, 3] - x[:3, 3])
    rot_error_rad = np.linalg.norm(
        cv2.Rodrigues(estimated[:3, :3] @ x[:3, :3].T)[0]
    )
    closure_m = max(np.linalg.norm((a @ estimated @ b)[:3, 3] - y[:3, 3])
                    for a, b in zip(a_list, b_list))
    assert trans_error_m < 1e-8
    assert rot_error_rad < 1e-7
    assert closure_m < 1e-8
    # Deliberately invert X; this is a negative contract, not merely a loss value.
    wrong_closure = max(
        np.linalg.norm((a @ np.linalg.inv(estimated) @ b)[:3, 3] - y[:3, 3])
        for a, b in zip(a_list, b_list)
    )
    assert wrong_closure > 0.05
    print(f"PASS handeye: translation={trans_error_m:.2e}m "
          f"rotation={rot_error_rad:.2e}rad closure={closure_m:.2e}m")
    print(f"PASS inversion-negative-control: error={wrong_closure:.3f}m")


def test_pivot():
    # A tip p_tool is fixed at one physical point, while the tracker rotates.
    p_tool = np.array([0.02, 0.04, -0.08])
    p_world = np.array([0.3, 0.2, 0.5])
    blocks, observations = [], []
    for _ in range(25):
        r = random_rotation()
        t = p_world - r @ p_tool + RNG.normal(0, 1e-4, 3)
        blocks.append(np.hstack([r, -np.eye(3)]))
        observations.append(-t)
    matrix = np.vstack(blocks)
    rhs = np.concatenate(observations)
    solution, _, rank, singular_values = np.linalg.lstsq(
        matrix, rhs, rcond=None
    )
    tip_error_m = np.linalg.norm(solution[:3] - p_tool)
    assert rank == 6
    assert singular_values[-1] > 0.1
    assert tip_error_m < 5e-4
    print(f"PASS pivot: rank={rank} tip_error={tip_error_m*1000:.3f}mm")


def test_degenerate():
    # Identical orientations do not make the pivot location observable.
    r = np.eye(3)
    blocks = np.vstack([np.hstack([r, -np.eye(3)]) for _ in range(12)])
    rank = np.linalg.matrix_rank(blocks)
    assert rank < 6
    print(f"PASS degeneracy: constant-rotation rank={rank}<6")


def test_sync_budget():
    displacement_m = 0.30 * 0.015
    assert abs(displacement_m - 0.0045) < 1e-12
    print(f"PASS timestamp-budget: 15ms at 0.30m/s = {displacement_m*1000:.1f}mm")


if __name__ == "__main__":
    test_handeye()
    test_pivot()
    test_degenerate()
    test_sync_budget()
    print("PASS ALL: UMI Tracker hand-eye / pivot CPU contracts")
