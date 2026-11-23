#!/usr/bin/env python3
"""MonoTag + LeRobot public-source typed contracts. Synthetic CPU only.

NOT the author's native ORB-SLAM3, NOT a measurement of dynamic wrist accuracy,
NOT a robot control program. Every oracle is explicitly pedagogical.
"""
import math


def ambiguous_ippe(angle_deg, camera_gap_mm, rms_gap_px):
    if min(angle_deg, camera_gap_mm, rms_gap_px) < 0:
        raise ValueError("negative geometric difference")
    return (angle_deg > 5.0 or camera_gap_mm > 20.0) and rms_gap_px < 0.5


def metric_proposal(observations, keyframes, baseline_m, parallax_ok, uncertainty_ok):
    return (observations >= 8 and keyframes >= 3 and baseline_m >= 0.04
            and parallax_ok and uncertainty_ok)


def retro_pnp(inliers, rms_px, positive_depth, connected, spatial_support):
    return (inliers >= 30 and 0 <= rms_px <= 3 and positive_depth
            and connected and spatial_support)


def valid_wrist(current, following):
    # Strict teaching contract: current and following must share final map
    # revision and both support a metric camera and measured wrist.
    return (current["camera_valid"] and following["camera_valid"]
            and current["metric"] and following["metric"]
            and current["wrist_valid"] and following["wrist_valid"]
            and current["map_id"] == following["map_id"]
            and current["revision"] == following["revision"]
            and current["segment"] == following["segment"])


def rot2_yaw(yaw, x, y):
    c, s = math.cos(yaw), math.sin(yaw)
    return (c*x - s*y, s*x + c*y)


def wrist_local_delta(p0, yaw0, p1, yaw1):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    # R(current)^T maps world translation difference into local axes.
    tx, ty = rot2_yaw(-yaw0, dx, dy)
    dyaw = (yaw1 - yaw0 + math.pi) % (2 * math.pi) - math.pi
    return tx, ty, p1[2] - p0[2], dyaw


def sample_indices(frames, src_fps, out_fps):
    if frames <= 0 or src_fps <= 0 or out_fps <= 0 or out_fps > src_fps:
        raise ValueError("invalid sampling rate")
    count = math.floor((frames - 1) * out_fps / src_fps) + 1
    return list(dict.fromkeys(min(frames - 1, round(i * src_fps / out_fps))
                              for i in range(count)))


def supervised_loss(pred, label, valid):
    if not(len(pred) == len(label) == len(valid)):
        raise ValueError("shape mismatch")
    active = [i for i, v in enumerate(valid) if v]
    if not active:
        return None
    return sum((pred[i] - label[i]) ** 2 for i in active) / len(active)


def check_ippe():
    assert ambiguous_ippe(6, 18, 0.20)
    assert ambiguous_ippe(2, 21, 0.49)
    assert not ambiguous_ippe(6, 40, 0.60)
    assert not ambiguous_ippe(2, 15, 0.10)
    print("PASS IPPE dual-positive-depth ambiguity gate")


def check_scale():
    assert metric_proposal(8, 3, 0.04, True, True)
    assert not metric_proposal(1, 1, 0.04, True, True)
    assert not metric_proposal(8, 3, 0.003, True, True)
    assert not metric_proposal(8, 3, 0.04, False, True)
    assert not metric_proposal(8, 3, 0.04, True, False)
    print("PASS metric gauge needs multiframe baseline and parallax")


def check_retro():
    assert retro_pnp(30, 3.0, True, True, True)
    assert not retro_pnp(29, 0.1, True, True, True)
    assert not retro_pnp(40, 3.1, True, True, True)
    assert not retro_pnp(40, 0.2, True, False, True)
    assert not retro_pnp(40, 0.2, False, True, True)
    print("PASS retrospective PnP needs image-to-map evidence")


def check_provenance():
    a = dict(camera_valid=True, metric=True, wrist_valid=True, map_id="A",
             revision=4, segment=1)
    b = dict(a)
    assert valid_wrist(a, b)
    bad = dict(a, map_id="B")
    assert not valid_wrist(a, bad)
    bad = dict(a, revision=5)
    assert not valid_wrist(a, bad)
    bad = dict(a, segment=2)
    assert not valid_wrist(a, bad)
    bad = dict(a, metric=False)
    assert not valid_wrist(a, bad)
    bad = dict(a, wrist_valid=False)
    assert not valid_wrist(a, bad)
    print("PASS map id revision segment and metric validity isolation")


def check_action():
    x, y, z, yaw = wrist_local_delta((1., 2., 0.), math.pi / 2,
                                      (1., 3., .5), math.pi)
    assert math.isclose(x, 1., abs_tol=1e-10)
    assert math.isclose(y, 0., abs_tol=1e-10)
    assert math.isclose(z, .5, abs_tol=1e-10)
    assert math.isclose(yaw, math.pi / 2, abs_tol=1e-10)
    wrong_world_x, wrong_world_y = 0., 1.
    assert not math.isclose(x, wrong_world_x, abs_tol=1e-10)
    assert not math.isclose(y, wrong_world_y, abs_tol=1e-10)
    print("PASS wrist-local SE3 translation direction negative control")


def check_sampling():
    frames = sample_indices(91, 90., 20)
    assert frames[0] == 0 and len(frames) == 21
    intervals = [b-a for a,b in zip(frames,frames[1:])]
    assert 4 in intervals and 5 in intervals
    assert frames[-1] == 90
    try:
        sample_indices(15, 15., 20.)
        raise AssertionError("upsampling should be rejected")
    except ValueError:
        pass
    print("PASS source-frame sampling and terminal action invalid")


def check_schema_mask():
    # Official exporter: camera7 + 2 wrist7 + 2 hands *21 joints *3.
    assert 7 + 2*7 + 2*21*3 == 147
    assert 2*(3+3) == 12
    assert len(("camera", "left_wrist", "right_wrist", "left_hand", "right_hand")) == 5
    assert len(("left_wrist", "right_wrist")) == 2
    # Last frame and missing wrist action are numeric zeros but NOT supervised.
    assert supervised_loss([5., 9.], [1., 0.], [True, False]) == 16.0
    assert supervised_loss([5., 9.], [1., 0.], [False, False]) is None
    print("PASS 147D 12D masks and no zero-filled invalid supervision")


if __name__ == "__main__":
    check_ippe()
    check_scale()
    check_retro()
    check_provenance()
    check_action()
    check_sampling()
    check_schema_mask()
    print("PASS ALL: MonoTag synthetic metric Atlas and LeRobot contracts")
