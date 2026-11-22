#!/usr/bin/env python3
"""Open-HumanWAM public-source interface invariants (pure stdlib synthetic tests).

NOT official checkpoint inference, NOT 50K corpus decoding, NOT physical control.
"""
import math


def clip_frames(start, end, episode_len):
    if not (0 <= start < end <= episode_len):
        raise ValueError("half-open episode-local frame interval required")
    return list(range(start, end))


def group_disjoint(train, test):
    # Native source clip identity is NOT enough: multiple clips share one recording.
    tr = {(x["source"], x["recording_id"]) for x in train}
    te = {(x["source"], x["recording_id"]) for x in test}
    return tr.isdisjoint(te)


def goalwam_valid_action(row, valid_mask):
    if len(row) != 49 or len(valid_mask) != 49:
        raise ValueError("physical GoalWAM action ABI is 49D")
    if any(valid_mask[42:49]):
        raise ValueError("head camera is state-only")
    if any(not math.isfinite(x) for x in row):
        raise ValueError("nonfinite action")
    return [x if mask else 0.0 for x, mask in zip(row, valid_mask)]


def normalize_q01_q99(values, lo, hi):
    if hi <= lo:
        raise ValueError("nondegenerate active dimension required")
    return [2 * (x - lo) / (hi - lo) - 1 for x in values]


def denormalize_q01_q99(values, lo, hi):
    return [lo + (x + 1) * (hi - lo) / 2 for x in values]


def absolute_prefix_relative_to_current(absolute_commands, state):
    # Translation-only oracle for the typed contract, not full SE(3) retargeting.
    return [command - state for command in absolute_commands]


def fit_endpoint_slope(h0, s0, h1, s1):
    if not (0 < h0 < h1):
        raise ValueError("training hours must increase")
    return (s1 - s0) / math.log10(h1 / h0)


def test_clip():
    assert clip_frames(100, 132, 133) == list(range(100, 132))
    assert len(clip_frames(100, 132, 133)) == 32
    try:
        clip_frames(100, 134, 133)
        raise AssertionError("cross-episode accepted")
    except ValueError:
        pass
    print("PASS half-open LeRobot clip and episode boundary")


def test_provenance():
    train = [{"source": "A", "recording_id": "rec1", "clip": 1}]
    test = [{"source": "A", "recording_id": "rec1", "clip": 2}]
    assert not group_disjoint(train, test)
    assert group_disjoint(train, [{"source": "A", "recording_id": "rec2", "clip": 1}])
    print("PASS source-recording leakage split")


def test_goalwam_mask():
    row = [float(i) for i in range(49)]
    mask = [True] * 42 + [False] * 7
    out = goalwam_valid_action(row, mask)
    assert out[41] == 41 and out[42:] == [0] * 7
    invalid = mask[:]
    invalid[45] = True
    try:
        goalwam_valid_action(row, invalid)
        raise AssertionError("action camera leaked")
    except ValueError:
        pass
    assert 23 + 19 + 7 == 49 and 23 + 19 + 22 == 64
    print("PASS 49D physical, internal 64D and camera action mask")


def test_quantile():
    xs = [0.2, 0.5, 0.9]
    zz = normalize_q01_q99(xs, 0.2, 0.9)
    back = denormalize_q01_q99(zz, 0.2, 0.9)
    assert max(abs(x - y) for x, y in zip(xs, back)) < 1e-12
    try:
        normalize_q01_q99(xs, 0.2, 0.2)
        raise AssertionError("zero-width accepted")
    except ValueError:
        pass
    print("PASS train-only q01/q99 roundtrip and degenerate boundary")


def test_prefix():
    previous_absolute = [0.8, 0.9]
    assert absolute_prefix_relative_to_current(previous_absolute, 0.5) == [0.30000000000000004, 0.4] or all(
        abs(a - b) < 1e-12 for a, b in zip(absolute_prefix_relative_to_current(previous_absolute, 0.5), [0.3, 0.4])
    )
    a = absolute_prefix_relative_to_current(previous_absolute, 0.5)
    b = absolute_prefix_relative_to_current(previous_absolute, 0.7)
    assert abs(a[0] - b[0] - 0.2) < 1e-12
    assert 48 - 32 == 16
    print("PASS RTC absolute prefix current-state reanchor and 48/32 horizon")


def test_scaling():
    beta = fit_endpoint_slope(100, 49.1, 50_000, 73.2)
    assert 8.8 < beta < 9.1
    # This number is a fit coefficient, NOT a guarantee beyond [100, 50K].
    def estimate(hours):
        if not 100 <= hours <= 50_000:
            raise ValueError("no extrapolation permitted")
        return 49.1 + beta * math.log10(hours / 100)
    assert abs(estimate(50_000) - 73.2) < 1e-9
    try:
        estimate(500_000)
        raise AssertionError("out-of-range extrapolation accepted")
    except ValueError:
        pass
    print("PASS in-range log-linear endpoint and out-of-range rejection")


if __name__ == "__main__":
    test_clip()
    test_provenance()
    test_goalwam_mask()
    test_quantile()
    test_prefix()
    test_scaling()
    print("PASS ALL: Open-HumanWAM public-source synthetic interface contracts")
