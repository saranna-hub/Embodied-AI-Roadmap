#!/usr/bin/env python3
"""Industrial EGO/UMI data *contract* demo. stdlib only; NOT a production MCAP converter.

Synthetic tests:
(1) clock drift/offset vs naive timestamp equality;
(2) causal image selection vs future-nearest bug;
(3) SE(3) anchored-relative vs sequential-delta (planar rotation subgroup);
(4) action horizon mask at episode end;
(5) source-clip split leakage;
(6) reversible normalization and frozen dimension.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def affine_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("need >=2 synchronized trigger pairs")
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    var = sum((x - cx) ** 2 for x in xs)
    if var < 1e-16:
        raise ValueError("clock is not observable")
    a = sum((x - cx) * (y - cy) for x, y in zip(xs, ys)) / var
    return a, cy - a * cx


def causal_select(frames: list[float], deadline: float, max_age: float) -> float | None:
    # Sensor exposure timestamp on one common time axis. DO NOT choose future observations.
    valid = [t for t in frames if t <= deadline and deadline - t <= max_age]
    return max(valid) if valid else None


@dataclass(frozen=True)
class Pose:
    # A valid 3D rigid transform subgroup: translation xyz + yaw around z.
    x: float
    y: float
    z: float
    yaw: float


def compose(a: Pose, b: Pose) -> Pose:
    c, s = math.cos(a.yaw), math.sin(a.yaw)
    return Pose(a.x + c * b.x - s * b.y,
                a.y + s * b.x + c * b.y,
                a.z + b.z, a.yaw + b.yaw)


def inverse(a: Pose) -> Pose:
    c, s = math.cos(a.yaw), math.sin(a.yaw)
    return Pose(-c * a.x - s * a.y, s * a.x - c * a.y, -a.z, -a.yaw)


def relative(a: Pose, b: Pose) -> Pose:
    # T_a^{-1} T_b, in the coordinate frame of a; not componentwise world subtraction.
    return compose(inverse(a), b)


def close_pose(a: Pose, b: Pose, eps: float = 1e-9) -> bool:
    return all(abs(x - y) < eps for x, y in zip(
        (a.x, a.y, a.z, math.sin(a.yaw), math.cos(a.yaw)),
        (b.x, b.y, b.z, math.sin(b.yaw), math.cos(b.yaw)),
    ))


def masked_action_horizon(
    episodes: dict[str, list[int]], episode: str, index: int, horizon: int
) -> tuple[list[int], list[bool]]:
    seq = episodes[episode]
    if not 0 <= index < len(seq):
        raise IndexError("invalid anchor")
    actions, mask = [], []
    for k in range(horizon):
        j = index + k
        if j < len(seq):
            actions.append(seq[j])
            mask.append(True)
        else:
            actions.append(seq[-1])  # padding is a placeholder, never train on it
            mask.append(False)
    return actions, mask


def leakage(train: list[dict], val: list[dict]) -> set[str]:
    tr = {x["source_clip_id"] for x in train}
    va = {x["source_clip_id"] for x in val}
    return tr.intersection(va)


def normalize(x: float, q01: float, q99: float) -> float:
    if q99 - q01 <= 1e-10:
        raise ValueError("constant dimension requires explicit mask/policy")
    return 2.0 * (x - q01) / (q99 - q01) - 1.0


def denormalize(x: float, q01: float, q99: float) -> float:
    if q99 - q01 <= 1e-10:
        raise ValueError("constant dimension")
    return q01 + (x + 1.0) * (q99 - q01) / 2.0


def main() -> None:
    # Clock triggers are physical paired events, not MCAP log_time & publish_time.
    device = [0.0, 1.1, 2.4, 3.9, 5.0, 7.2, 9.0]
    ref = [1.0002 * t + 0.031 for t in device]
    a, b = affine_fit(device, ref)
    corrected = [a * x + b for x in device]
    residual = max(abs(v - y) for v, y in zip(corrected, ref))
    naive_error = max(abs(x - y) for x, y in zip(device, ref))
    assert abs(a - 1.0002) < 1e-11 and abs(b - .031) < 1e-11
    assert residual < 1e-11 and naive_error > .03
    print(f"PASS affine-clock: drift={a:.7f} offset={b:.4f}s max_residual={residual:.2e}s")

    # t=.087: mathematically nearest frame is .099 (FUTURE) but policy cannot see it.
    frame_ts = [.000, .033, .066, .099, .132]
    deadline = .087
    nearest = min(frame_ts, key=lambda t: abs(t - deadline))
    selected = causal_select(frame_ts, deadline, max_age=.030)
    assert nearest == .099 and selected == .066
    assert causal_select(frame_ts, deadline, max_age=.010) is None
    print(f"PASS causal-observation: nearest={nearest:.3f}s REJECT; observed={selected:.3f}s")

    # SE(3) demo on the planar subgroup (yaw + xyz), not a full 3D quaternion claim.
    poses = [Pose(1.0, .1, .2, .0),
             Pose(1.1, .2, .2, math.pi / 6),
             Pose(1.4, .3, .2, math.pi / 3)]
    anchored = [relative(poses[0], p) for p in poses]
    sequential = [Pose(0, 0, 0, 0)] + [
        relative(poses[k - 1], poses[k]) for k in range(1, len(poses))
    ]
    assert not close_pose(anchored[-1], sequential[-1])
    assert close_pose(compose(poses[0], anchored[-1]), poses[-1])
    reconstructed = poses[0]
    for delta in sequential[1:]:
        reconstructed = compose(reconstructed, delta)
    assert close_pose(reconstructed, poses[-1])
    print(f"PASS SE3-action: anchored_dx={anchored[-1].x:.5f} sequential_dx={sequential[-1].x:.5f}")

    # One file can contain multiple episodes; label selection must stop at episode boundary.
    episodes = {"A": [10, 11, 12], "B": [20, 21]}
    actions, mask = masked_action_horizon(episodes, "A", 2, 4)
    assert actions == [12, 12, 12, 12] and mask == [True, False, False, False]
    assert sum(mask) == 1 and 20 not in actions
    print(f"PASS episode-mask: valid={sum(mask)}/4; never joined episode B")

    # Re-encoding one raw clip into two derived datasets must not create train/val overlap.
    tr = [{"source_clip_id": "clip_001", "episode": 0}]
    va_bad = [{"source_clip_id": "clip_001", "episode": 15}]
    va_good = [{"source_clip_id": "clip_002", "episode": 15}]
    assert leakage(tr, va_bad) == {"clip_001"}
    assert not leakage(tr, va_good)
    print("PASS source-clip-leakage: same raw video in train/val rejected")

    for x in [-1.0, -0.5, 0.8, 1.4]:
        z = normalize(x, -.8, 1.0)
        assert abs(denormalize(z, -.8, 1.0) - x) < 1e-10
    try:
        normalize(.1, .2, .2)
    except ValueError:
        pass
    else:
        raise AssertionError("constant-dimension normalization must be rejected")
    print("PASS normalization-roundtrip: q01/q99 inverse; frozen dimension protected")
    print("PASS ALL: MCAP -> LeRobot v3 -> openpi synthetic data contracts")


if __name__ == "__main__":
    main()
