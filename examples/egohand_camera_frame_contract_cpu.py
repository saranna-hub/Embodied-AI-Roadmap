#!/usr/bin/env python3
"""CPU-only synthetic contracts for the EgoHand learning page.

No EgoHand, DINO, SOMA or detector weights are loaded. These are geometry and
frame-join invariants, NOT a reproduction of the paper's 3D accuracy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Pinhole:
    fx: float
    fy: float
    px: float
    py: float

    def __post_init__(self):
        assert self.fx > 0 and self.fy > 0


def project(point: tuple[float, float, float], k: Pinhole):
    x, y, z = point
    if z <= 0:
        raise ValueError("invalid camera depth")
    return (k.fx * x / z + k.px, k.fy * y / z + k.py)


def add3(x, y):
    return tuple(a + b for a, b in zip(x, y))


def sub3(x, y):
    return tuple(a - b for a, b in zip(x, y))


def solve_tx_ty(joints, uv, k: Pinhole, depth: float, weights, ridge=0.0):
    """Fixed-depth weighted pinhole reprojection WLS, a teaching contract."""
    assert len(joints) == len(uv) == len(weights) and joints
    if depth <= 0 or ridge < 0:
        raise ValueError("invalid depth/ridge")
    nx = ny = 0.0
    dx = dy = ridge
    for (jx, jy, jz), (u, v), w in zip(joints, uv, weights):
        if w < 0 or not math.isfinite(w) or jz + depth <= 0:
            raise ValueError("invalid confidence / joint depth")
        ax = k.fx / (jz + depth)
        ay = k.fy / (jz + depth)
        nx += w * ax * (u - k.px - ax * jx)
        ny += w * ay * (v - k.py - ay * jy)
        dx += w * ax * ax
        dy += w * ay * ay
    if dx <= 1e-12 or dy <= 1e-12:
        raise ValueError("correspondence solve degenerate")
    return nx / dx, ny / dy


def cameras_to_world(p, rotation_rows, translation):
    return tuple(
        sum(rotation_rows[i][j] * p[j] for j in range(3)) + translation[i]
        for i in range(3)
    )


def worlds_to_camera(pw, rotation_rows, translation):
    centered = sub3(pw, translation)
    return tuple(
        sum(rotation_rows[j][i] * centered[j] for j in range(3))
        for i in range(3)
    )


def strict_frame_join(left_ids, right_ids, pose_ids):
    """Only aligned sample IDs are eligible for joint two-hand world refinement."""
    arrays = [left_ids, right_ids, pose_ids]
    for seq in arrays:
        if len(seq) != len(set(seq)) or any(x < 0 for x in seq):
            raise ValueError("duplicate or negative frame ID")
        if seq != sorted(seq):
            raise ValueError("nonmonotonic frame IDs")
    matched = sorted(set(left_ids) & set(right_ids) & set(pose_ids))
    return matched


def segments_from_frame_ids(ids, fps, max_gap_seconds):
    if fps <= 0 or max_gap_seconds <= 0:
        raise ValueError("bad timing units")
    chunks = [[ids[0]]] if ids else []
    for previous, current in zip(ids, ids[1:]):
        if current <= previous:
            raise ValueError("nonmonotonic source frames")
        if (current - previous) / fps > max_gap_seconds:
            chunks.append([])
        chunks[-1].append(current)
    return chunks


def main():
    k = Pinhole(930.0, 905.0, 640.0, 360.0)
    wrist_relative = [(0, 0, 0), (0.03, -0.05, 0.03), (-0.07, 0.08, -0.02),
                      (0.04, 0.09, 0.015), (0.08, -0.03, -0.04)]
    true_t = (0.125, -0.087, 1.15)
    uv = [project(add3(j, true_t), k) for j in wrist_relative]
    tx, ty = solve_tx_ty(wrist_relative, uv, k, true_t[2], [1.0, 1.0, 0.5, 0.7, 1.0])
    assert abs(tx - true_t[0]) < 1e-12 and abs(ty - true_t[1]) < 1e-12
    assert max(abs(x-y) for x, y in zip(project(add3(wrist_relative[0], (tx, ty, true_t[2])), k), uv[0])) < 1e-9
    wrong_tx, wrong_ty = solve_tx_ty(wrist_relative, uv, k, true_t[2] * 1.25, [1] * len(uv))
    assert abs(wrong_tx - true_t[0]) > 0.01 or abs(wrong_ty - true_t[1]) > 0.01
    print("PASS 2D-to-3D fixed-depth weighted lateral solve and wrong-depth negative control")

    focal_true, focal_bad, box, residual = 900., 1200., 300., math.log(0.8)
    z_good = focal_true / box * math.exp(residual)
    z_bad = focal_bad / box * math.exp(residual)
    assert math.isclose(z_bad / z_good, 4 / 3, rel_tol=1e-12)
    # Scale the complete geometry and camera K together: pixel overlay is unchanged.
    p = (0.1, 0.03, z_good)
    u_good = project(p, Pinhole(focal_true, focal_true, 640, 360))[0]
    p_fake = (p[0], p[1], z_bad)
    u_fake = project(p_fake, Pinhole(focal_bad, focal_bad, 640, 360))[0]
    assert math.isclose(u_good, u_fake, abs_tol=1e-12)
    assert abs(z_good-z_bad)>0.5
    print("PASS focal-depth ratio and self-consistent-overlay counterexample")

    rot = ((0., -1., 0.), (1., 0., 0.), (0., 0., 1.))
    world_shift = (2.0, -3.0, 0.8)
    pc = add3(wrist_relative[1], true_t)
    pw = cameras_to_world(pc, rot, world_shift)
    back = worlds_to_camera(pw, rot, world_shift)
    assert max(abs(x-y) for x, y in zip(pc, back)) < 1e-12
    wrong_inverse = worlds_to_camera(pc, rot, world_shift)
    assert math.dist(wrong_inverse, pw) > 0.5
    print("PASS T_WC world coordinates / inverse and direction-negative control")

    lids = [0, 2, 3, 4]
    rids = [0, 1, 3, 4]
    cids = [0, 2, 3, 4]
    assert len(lids) == len(rids) == len(cids)
    assert strict_frame_join(lids, rids, cids) == [0, 3, 4]
    assert lids[1] != rids[1]
    for bad in ([0, 0, 3], [3, 2, 4], [-1, 1, 3]):
        try:
            strict_frame_join(bad, rids, cids)
        except ValueError:
            continue
        raise AssertionError("malformed frame ids not rejected")
    print("PASS same-length unequal-frame arrays rejected by strict source-ID join")

    segments = segments_from_frame_ids([0, 1, 2, 9, 10, 20], fps=30, max_gap_seconds=0.15)
    assert segments == [[0, 1, 2], [9, 10], [20]]
    try:
        segments_from_frame_ids([0, 2, 1], fps=30, max_gap_seconds=0.15)
    except ValueError:
        pass
    else:
        raise AssertionError("nonmonotonic frame times accepted")
    print("PASS temporal-gap segmentation and nonmonotonic timestamp rejection")
    print("PASS ALL: EgoHand camera/frame/pose CPU synthetic contracts")


if __name__ == "__main__":
    main()
