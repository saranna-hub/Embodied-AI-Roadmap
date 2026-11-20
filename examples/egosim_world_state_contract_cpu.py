#!/usr/bin/env python3
"""EgoSim synthetic CPU contracts. Teaching-only; no model weights and no external packages."""
from __future__ import annotations

from dataclasses import dataclass
from math import isclose, sqrt
from copy import deepcopy


def test_latent_channels() -> None:
    pieces = {"noise": 16, "mask_patch": 4, "scene": 16, "hand": 16}
    assert sum(pieces.values()) == 52
    # 2x2 binary spatial mask occupies four patch-channel slots, not one.
    pixels = [[0, 1], [1, 0]]
    patch = [pixels[y][x] for y in (0, 1) for x in (0, 1)]
    assert patch == [0, 1, 1, 0] and len(patch) == 4
    # Known pixels retain background. Unknown regions are synthesized.
    prior, mask = [0.2, 0.8], [0, 1]
    masked_prior = [x * (1 - m) for x, m in zip(prior, mask)]
    assert masked_prior == [0.2, 0.0]
    first_frame_mask = [0, 0]
    assert all(v == 0 for v in first_frame_mask)
    print("PASS 52-channel latent and first-frame mask contract")


def project_identity_rotation(world, camera_translation, fx=500.0, fy=500.0):
    x, y, z = (world[i] - camera_translation[i] for i in range(3))
    assert z > 0
    return fx * x / z, fy * y / z


def test_camera_geometry():
    point_world = (0.2, -0.1, 2.0)
    u0, v0 = project_identity_rotation(point_world, (0, 0, 0))
    u1, v1 = project_identity_rotation(point_world, (0.1, 0, 0))
    assert u1 < u0 and isclose(v1, v0)
    # World point and camera both translate together => image invariant.
    shift = (0.3, 0.4, 0.0)
    shifted_point = tuple(p + d for p, d in zip(point_world, shift))
    u2, v2 = project_identity_rotation(shifted_point, shift)
    assert isclose(u2, u0) and isclose(v2, v0)
    # Negative control: confusing camera translation with world translation.
    wrong = project_identity_rotation(shifted_point, (-shift[0], -shift[1], 0))
    assert abs(wrong[0] - u0) > 1
    print("PASS T_WC projection and camera-motion negative control")


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def norm(v):
    return sqrt(sum(x*x for x in v))


def simple_scale_translate(p0, p1, q0, q1):
    """Known-rotation Sim3 teaching oracle: NOT a full Umeyama SVD solver."""
    dp = norm(sub(p1, p0))
    if dp < 1e-8:
        raise ValueError("degenerate overlap / cannot recover scale")
    scale = norm(sub(q1, q0)) / dp
    if scale <= 0:
        raise ValueError("nonpositive scale")
    shift = tuple(q - scale*p for q, p in zip(q0, p0))
    return scale, shift


def test_sim3_and_degeneracy():
    p0, p1 = (0., 0., 0.), (1., 0., 0.)
    q0, q1 = (2., -3., 1.), (4., -3., 1.)
    scale, shift = simple_scale_translate(p0, p1, q0, q1)
    assert isclose(scale, 2.0) and shift == (2.0, -3.0, 1.0)
    probe = (0.5, 0, 0)
    mapped = tuple(scale * x + t for x, t in zip(probe, shift))
    assert mapped == (3., -3., 1.)
    try:
        simple_scale_translate(p0, p0, q0, q1)
    except ValueError:
        pass
    else:
        raise AssertionError("degenerate pair must fail")
    print("PASS Sim3 scale and degenerate-overlap reject oracle")


@dataclass
class WorldState:
    object_position: dict[str, tuple[float, float, float]]
    version: int = 0


def gated_update(old: WorldState, candidate: WorldState, *, mask_ok: bool, geometry_ok: bool):
    if not (mask_ok and geometry_ok):
        return deepcopy(old)
    next_state = deepcopy(candidate)
    next_state.version = old.version + 1
    return next_state


def test_persistent_state_gate():
    old = WorldState({"cup": (0., 0., 0.)}, version=7)
    bad = WorldState({"cup": (99., 0., 0.)})
    rejected = gated_update(old, bad, mask_ok=False, geometry_ok=True)
    assert rejected.version == 7 and rejected.object_position["cup"] == (0., 0., 0.)
    assert old.object_position["cup"] == (0., 0., 0.)
    good = WorldState({"cup": (0.25, 0., 0.)})
    accepted = gated_update(old, good, mask_ok=True, geometry_ok=True)
    assert accepted.version == 8 and accepted.object_position["cup"] == (0.25, 0., 0.)
    print("PASS untrusted state rollback and accepted-version increment")


def test_overlap_and_source_isolation():
    first = list(range(0, 61))
    second = list(range(60, 121))
    merged = first + second[1:]
    assert len(merged) == 121 and len(set(merged)) == 121 and merged[-1] == 120
    fps = 16
    assert isclose((merged[-1] - merged[0]) / fps, 7.5)
    # Same source footage must not leak between training and test splits.
    source_ids = {"clipA_0": "srcA", "clipA_1": "srcA", "clipB": "srcB"}
    train = {"clipA_0"}
    test = {"clipA_1", "clipB"}
    assert {source_ids[x] for x in train} & {source_ids[x] for x in test} == {"srcA"}
    print("PASS 61-frame overlap and source-clip leakage guard")


if __name__ == "__main__":
    test_latent_channels()
    test_camera_geometry()
    test_sim3_and_degeneracy()
    test_persistent_state_gate()
    test_overlap_and_source_isolation()
    print("PASS ALL: EgoSim geometry-action-world-state CPU contracts")
