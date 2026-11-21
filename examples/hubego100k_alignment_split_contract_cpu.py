#!/usr/bin/env python3
"""HubEgo-100K synthetic, CPU-only teaching contracts. No HF dataset or GPU needed.

These tests verify interfaces and negative controls, not source-data correctness or
VLA transfer. The official dataset is access-conditioned and is not downloaded here.
"""
from __future__ import annotations

from math import isclose
from dataclasses import dataclass


G0 = 9.80665


def si_acceleration(values, source_unit):
    """Use native MCAP channel metadata. Never infer units from magnitude."""
    if source_unit == "g":
        return tuple(x * G0 for x in values)
    if source_unit == "m/s^2":
        return tuple(values)
    raise ValueError("missing/unknown accelerometer unit")


def test_imu_unit_contract():
    ios = si_acceleration((0.0, 0.0, -1.0), "g")
    android = si_acceleration((0.0, 0.0, -G0), "m/s^2")
    assert all(isclose(a, b, abs_tol=1e-11) for a, b in zip(ios, android))
    assert isclose(ios[2], -G0)
    frame_state = (0., 0., -G0, 0.1, -0.1, 0.)
    assert si_acceleration(frame_state[:3], "m/s^2") == frame_state[:3]
    try:
        si_acceleration(frame_state[:3], "")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown units must reject")
    print("PASS native IMU unit normalization and no double gravity scaling")


def interpolate_offline(times, xs, t, max_gap=0.050):
    assert len(times) == len(xs) >= 2
    if any(b <= a for a, b in zip(times[:-1], times[1:])):
        raise ValueError("non-monotonic native sensor clock")
    for i in range(len(times) - 1):
        l, r = times[i], times[i + 1]
        if l <= t <= r:
            if r - l > max_gap:
                raise ValueError("invalid wide sampling gap")
            a = (t - l) / (r - l)
            return (1-a) * xs[i] + a * xs[i + 1], r
    raise ValueError("no interpolation neighbors / extrapolation forbidden")


def online_hold(times, xs, t, max_staleness=0.020):
    past = [i for i, ts in enumerate(times) if ts <= t]
    if not past:
        raise ValueError("no online-available sample")
    k = past[-1]
    if t - times[k] > max_staleness:
        raise ValueError("stale state")
    return xs[k], times[k]


def test_imu_time_and_causality():
    times, samples = [0.0, .01, .02], [0., 10., 20.]
    target = .015
    offline, latest_used = interpolate_offline(times, samples, target)
    online, latest_allowed = online_hold(times, samples, target)
    assert isclose(offline, 15.) and online == 10.
    assert latest_used > target and latest_allowed <= target
    assert offline != online, "using offline value for online policy leaks future"
    for bad_time in ([0, .01, .01], [0, .03, .025]):
        try:
            interpolate_offline(bad_time, [0, 1, 2], .015)
        except ValueError:
            pass
        else:
            raise AssertionError("repeated/out-of-order IMU clock must reject")
    try:
        interpolate_offline([0, .1], [0, 1], .05, max_gap=.05)
    except ValueError:
        pass
    else:
        raise AssertionError("wide interpolation gap must be rejected")
    print("PASS offline interpolation versus online causal-hold negative control")


def video_qc(timestamps, frames, nominal_fps=30.0):
    if len(timestamps) != len(frames) or len(timestamps) < 2:
        raise ValueError("frame and timestamp length mismatch")
    diffs = [b - a for a, b in zip(timestamps[:-1], timestamps[1:])]
    if any(delta <= 0 for delta in diffs):
        raise ValueError("bad video capture timestamp ordering")
    gaps = sum(delta > 1.5 / nominal_fps for delta in diffs)
    repeats = sum(a == b for a, b in zip(frames[:-1], frames[1:]))
    nominal_duration = (len(frames) - 1) / nominal_fps
    drift = (timestamps[-1] - timestamps[0]) - nominal_duration
    return gaps, repeats, drift


def test_frame_integrity():
    # A dropped or delayed capture interval plus a duplicated re-encoded frame.
    ts = [0., 1/30, 2/30, 4/30]
    gaps, repeats, drift = video_qc(ts, [b"a", b"b", b"b", b"d"])
    assert gaps == 1 and repeats == 1
    assert isclose(drift, 1/30)
    assert video_qc([0, 1/30, 2/30], [1, 2, 3]) == (0, 0, 0)
    try:
        video_qc([0, 1/30, 1/30], [1, 2, 3])
    except ValueError:
        pass
    else:
        raise AssertionError("non-monotonic timestamps must reject")
    print("PASS real frame-time gap, re-encoded repeat and nominal FPS drift")


@dataclass(frozen=True)
class Episode:
    eid: int
    contributor: str
    recording: str
    source: str


def assert_no_source_leakage(episodes, train, val):
    by_id = {e.eid: e for e in episodes}
    if not set(train).isdisjoint(val):
        raise AssertionError("same episode repeated")
    for field in ("contributor", "recording", "source"):
        left = {getattr(by_id[e], field) for e in train}
        right = {getattr(by_id[e], field) for e in val}
        if left & right:
            raise AssertionError(f"split leakage via {field}")


def test_group_split():
    episodes = [
        Episode(1, "alice", "rec-A", "hubego-v1"),
        Episode(2, "alice", "rec-A", "hubego-v1"),
        Episode(3, "bob", "rec-B", "hubego-v1"),
        Episode(4, "carol", "rec-C", "hubego-v1"),
    ]
    # Source dataset name is provenance, not a split key; real splits may share a dataset.
    def no_leak(train, val):
        by_id = {e.eid: e for e in episodes}
        for field in ("contributor", "recording"):
            assert {getattr(by_id[e], field) for e in train}.isdisjoint(
                {getattr(by_id[e], field) for e in val})
    try:
        no_leak({1}, {2})
    except AssertionError:
        pass
    else:
        raise AssertionError("episode-level random split leaked original recording")
    no_leak({1, 2}, {3, 4})
    # Need correct leave-one-contributor-out semantics.
    assert {e.contributor for e in episodes if e.eid in {1,2}} == {"alice"}
    print("PASS contributor and original-recording disjoint split / leakage reject")


def masked_action_loss(pred, target, mask, source):
    if source == "human_ego" and any(mask):
        raise AssertionError("human video must not carry synthetic robot GT")
    if not (len(pred) == len(target) == len(mask)):
        raise ValueError("action interface mismatch")
    valid = [((p-y)**2) for p, y, m in zip(pred, target, mask) if m]
    return sum(valid) / len(valid) if valid else None


def test_actionfree_cotrain():
    human = masked_action_loss([1., 2.], [0., 0.], [False, False], "human_ego")
    assert human is None, "loss contribution must be absent, not a zero-target loss"
    robot = masked_action_loss([1., 2.], [1.5, 3.], [True, True], "robot")
    assert isclose(robot, (.5**2+1.**2)/2)
    try:
        masked_action_loss([1., 2.], [0., 0.], [True, True], "human_ego")
    except AssertionError:
        pass
    else:
        raise AssertionError("human pseudo-zero action must reject")
    print("PASS action-free EGO supervision mask and robot-only action loss")


if __name__ == "__main__":
    test_imu_unit_contract()
    test_imu_time_and_causality()
    test_frame_integrity()
    test_group_split()
    test_actionfree_cotrain()
    print("PASS ALL: HubEgo-100K synthetic RGB/IMU/split/co-training contracts")
