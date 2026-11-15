#!/usr/bin/env python3
"""HiFi-UMI-2K synthetic data ABI contracts. CPU stdlib only; NOT official dataset replay.

Primary sources: https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K
https://huggingface.co/docs/lerobot/lerobot-dataset-v3
No real HiFi data files are downloaded in this test.
"""
from __future__ import annotations
import math
import statistics


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1],
            a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0])


def norm(x):
    return math.sqrt(dot(x, x))


def unit(x):
    n = norm(x)
    if n <= 1e-9 or not math.isfinite(n):
        raise ValueError("invalid rot6d row; mask rather than normalize")
    return tuple(v/n for v in x)


def rot6d_rows_to_R(v):
    """Official HiFi-UMI-2K is first two ROWS, not two columns."""
    assert len(v) == 6
    r0 = unit(v[:3])
    projected = tuple(y-dot(v[3:], r0)*x for x, y in zip(r0, v[3:]))
    r1 = unit(projected)
    r2 = cross(r0, r1)
    return (r0, r1, r2)


def matmul(A, B):
    BT = list(zip(*B))
    return tuple(tuple(dot(row, col) for col in BT) for row in A)


def transpose(A):
    return tuple(tuple(x) for x in zip(*A))


def good_rot(R):
    RtR = matmul(transpose(R), R)
    identity = tuple(tuple(float(i == j) for j in range(3)) for i in range(3))
    return all(abs(RtR[i][j]-identity[i][j]) < 1e-9 for i in range(3) for j in range(3))


def valid_horizon(frames, t, H, required_dims=(0, 1, 2, 10, 11, 12)):
    """Strict teaching contract: never cross episode or include invalid dimensions."""
    ep = frames[t]["episode"]
    anchor = frames[t]
    if not anchor["valid_frame"] or not all(anchor["state_valid"][j] for j in required_dims):
        return [[False]*20 for _ in range(H)]
    out = []
    for k in range(1, H+1):
        j = t+k
        if j >= len(frames) or frames[j]["episode"] != ep or not frames[j]["valid_frame"]:
            out.append([False]*20)
        else:
            out.append(list(frames[j]["action_valid"]))
    return out


def test_rotation():
    # A +90-degree Z rotation: first two rows [0,-1,0] and [1,0,0].
    raw = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0)
    R = rot6d_rows_to_R(raw)
    expected = ((0., -1., 0.), (1., 0., 0.), (0., 0., 1.))
    assert R == expected
    assert good_rot(R)
    # WRONG column convention turns this non-symmetric rotation into -90 degrees.
    as_columns = transpose(R)
    assert as_columns != R and as_columns[0][1] == 1.
    try:
        rot6d_rows_to_R((1, 0, 0, 2, 0, 0))
        raise AssertionError("degenerate second row must raise")
    except ValueError:
        pass
    print("PASS rot6d row-vs-column negative-control: +90 is not -90")


def test_video_and_mask():
    # Episode B starts at offset 8s in its shared MP4; 3s inside B => 11s file PTS.
    video_start = 8.0
    episode_time = 3.0
    seek = video_start+episode_time
    assert seek == 11.0 and seek != episode_time
    frames = []
    for e in (0, 1):
        for k in range(6):
            frames.append({
                "episode": e, "frame_index": k,
                "valid_frame": (k != 2),
                "state_valid": [True]*20,
                "action_valid": [True]*20,
            })
    frames[3]["action_valid"][8] = False
    # Keep invalid physical row 2. A post-filter array index is NOT a video index.
    filtered = [f for f in frames if f["valid_frame"]]
    assert filtered[2]["frame_index"] == 3 != 2
    assert len(frames) == 12 and len(filtered) == 10
    masks = valid_horizon(frames, t=1, H=4)
    assert masks[0] == [False]*20
    assert masks[1][8] is False
    assert all(masks[2]) and all(masks[3])
    # Anchor 4: k=1 still episode 0; k=2 crosses into episode 1.
    boundary = valid_horizon(frames, t=4, H=3)
    assert all(boundary[0]) and not any(boundary[1]) and not any(boundary[2])
    # Channel validity is independent of frame-level validity.
    frames[4]["state_valid"][10] = False
    assert not any(any(row) for row in valid_horizon(frames, 4, 2))
    # Unqualified episode ids may collide across outer parts.
    assert ("chunk-0000", "part-0000", 0) != ("chunk-0001", "part-0000", 0)
    print("PASS video-offset: file seek = episode from_timestamp + frame time")
    print("PASS keep-invalid-frames: physical index preserved")
    print("PASS per-dimension masks and horizon no-episode-crossing")


def test_absolute_delta():
    # Official stored action is a target ABSOLUTE next-state; delta must be derived.
    p_now = (0.20, -0.10, 0.35)
    p_target = (0.22, -0.06, 0.36)
    delta = tuple(b-a for a, b in zip(p_now, p_target))
    assert all(abs(x-y) < 1e-12 for x, y in zip(delta, (0.02, 0.04, 0.01)))
    reconstructed = tuple(a+d for a, d in zip(p_now, delta))
    assert all(abs(a-b) < 1e-12 for a, b in zip(reconstructed, p_target))
    # Gripper remains absolute in the referenced openpi example, not part of xyz delta.
    gripper_rad = 0.43
    assert gripper_rad == 0.43
    print("PASS absolute-next-state to xyz-delta roundtrip")


def test_normalization_and_masked_loss():
    train_valid = [(1.0, True, True), (3.0, True, True),
                   (999.0, False, True), (333.0, True, False)]
    accepted = [v for v, frame_valid, action_valid in train_valid
                if frame_valid and action_valid]
    mu = statistics.mean(accepted)
    sigma = statistics.pstdev(accepted)
    assert mu == 2 and sigma == 1
    assert statistics.mean(v for v, _, _ in train_valid) != mu
    prediction = [2., 999., -1.]
    target = [1., float("nan"), 1.]
    mask = [True, False, True]
    # sanitize BEFORE mathematical operations. 0*nan is still nan.
    numerator = sum((prediction[i]-target[i])**2 for i in range(3) if mask[i])
    denom = max(1, sum(mask))
    assert numerator == 5 and numerator/denom == 2.5
    empty_mask = [False]*3
    safe_empty_loss = sum((prediction[i]-target[i])**2 for i in range(3)
                          if empty_mask[i])/max(1, sum(empty_mask))
    assert safe_empty_loss == 0
    print("PASS filtered training-only normalization and finite masked loss")


if __name__ == "__main__":
    test_rotation()
    test_video_and_mask()
    test_absolute_delta()
    test_normalization_and_masked_loss()
    print("PASS ALL: HiFi-UMI-2K synthetic data contracts (not real data replication)")
