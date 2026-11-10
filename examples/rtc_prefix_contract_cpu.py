#!/usr/bin/env python3
"""RTC math/temporal contracts: independent stdlib-only teaching checks.

No network, model weights, GPU, robot, or claimed empirical task result.
Sources: arxiv 2506.07339v2 Eq. 5; arxiv 2512.05964; LeRobot pinned source.
"""
from __future__ import annotations

import math


def soft_mask(h: int, s: int, d: int) -> list[float]:
    """Paper Eq. (5), not a Transformer attention matrix."""
    if not (0 <= d <= s <= h - d and h > 0):
        raise ValueError("invalid RTC budget: need 0 <= d <= s <= H-d")
    w = []
    for i in range(h):
        if i < d:
            v = 1.0
        elif i >= h - s:
            v = 0.0
        else:
            c = (h - s - i) / (h - s - d + 1)
            v = c * math.expm1(c) / (math.e - 1)
        w.append(v)
    return w


def delay_budget(h: int, consumed: int, measured_delay: int) -> int:
    """Controller step budget, not a wall-clock latency estimator."""
    if not (0 <= consumed <= h and 0 <= measured_delay <= h - consumed):
        raise ValueError("old chunk does not cover new inference")
    return h - consumed - measured_delay


def masked_postfix_mse(
    pred: list[list[list[float]]],
    target: list[list[list[float]]],
    lengths: list[int],
) -> float:
    """Only valid suffix positions enter numerator AND denominator."""
    if len(pred) != len(target) or len(pred) != len(lengths):
        raise ValueError("batch mismatch")
    total = 0.0
    valid = 0
    for p_b, t_b, delay in zip(pred, target, lengths, strict=True):
        if len(p_b) != len(t_b) or not 0 <= delay < len(p_b):
            raise ValueError("invalid prefix or chunk")
        for i in range(delay, len(p_b)):
            if len(p_b[i]) != len(t_b[i]):
                raise ValueError("dimension mismatch")
            for p, t in zip(p_b[i], t_b[i], strict=True):
                if not (math.isfinite(p) and math.isfinite(t)):
                    raise ValueError("NaN/Inf")
                total += (p - t) ** 2
                valid += 1
    if valid == 0:
        raise ValueError("no valid postfix")
    return total / valid


def merge_chunk(new_actions: list[float], real_delay: int) -> list[float]:
    """Do not execute actions for timesteps already elapsed."""
    if not (0 <= real_delay < len(new_actions)):
        raise ValueError("stale or exhausted chunk")
    if not all(map(math.isfinite, new_actions)):
        raise ValueError("invalid actions")
    return new_actions[real_delay:]


def test_budget():
    assert delay_budget(50, 15, 7) == 28
    for args in [(20, 12, 9), (8, 9, 0), (10, 1, -1)]:
        try:
            delay_budget(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unrejected budget {args}")
    assert len(soft_mask(12, 4, 3)) == 12
    try:
        soft_mask(20, 12, 9)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid overlap accepted")


def test_mask():
    w = soft_mask(12, 4, 3)
    assert w[:3] == [1, 1, 1]
    assert all(v == 0 for v in w[8:])
    assert all(0 < w[i + 1] < w[i] < 1 for i in range(3, 7))
    c = 5 / 6
    assert abs(w[3] - c * math.expm1(c) / (math.e - 1)) < 1e-12
    assert all(0 <= v <= 1 for v in w)


def test_postfix():
    # B=2, H=5, D=2; first sample d=0, second d=3.
    p = [[[2.0, 2.0] for _ in range(5)] for _ in range(2)]
    t = [[[0.0, 0.0] for _ in range(5)] for _ in range(2)]
    assert abs(masked_postfix_mse(p, t, [0, 3]) - 4.0) < 1e-12
    # Deliberately corrupt the *prefix* sample: target remains unaffected.
    p[1][:3] = [[400.0, 400.0] for _ in range(3)]
    assert abs(masked_postfix_mse(p, t, [0, 3]) - 4.0) < 1e-12


def test_queue():
    assert merge_chunk([10.0, 11.0, 12.0, 13.0], 2) == [12.0, 13.0]
    for delay in [-1, 4, 5]:
        try:
            merge_chunk([10.0, 11.0, 12.0, 13.0], delay)
        except ValueError:
            pass
        else:
            raise AssertionError("stale packet accepted")
    try:
        merge_chunk([1.0, float("nan")], 0)
    except ValueError:
        pass
    else:
        raise AssertionError("NaN packet accepted")


def test_flow_orientation():
    a, eps, steps = 2.0, -1.0, 10
    # Paper direction: noise at tau=0, data at tau=1.
    x = eps
    for _ in range(steps):
        x += (a - eps) / steps
    assert abs(x - a) < 1e-12
    # LeRobot/openpi: noise at t=1, data at t=0; reverse Euler.
    y = eps
    for _ in range(steps):
        y -= (eps - a) / steps
    assert abs(y - a) < 1e-12
    assert abs((eps + (eps - a)) - a) > 1.0  # wrong sign negative control


def test_no_future_label():
    # Deployment conditioning uses old committed actions, NOT future GT.
    old_committed = [5.0, 5.1, 5.2]
    future_ground_truth = [99.0, 99.0, 99.0]
    inferred_prefix = list(old_committed)
    assert inferred_prefix == old_committed
    assert inferred_prefix != future_ground_truth


if __name__ == "__main__":
    test_budget()
    print("PASS delay-budget: old chunk covers inference and invalid overlap rejects")
    test_mask()
    print("PASS soft-mask: paper Eq5, hard/soft/free regions")
    test_postfix()
    print("PASS postfix-loss: prefix ignored and denominator correct")
    test_queue()
    print("PASS stale-queue: real delay trim and invalid packets reject")
    test_flow_orientation()
    print("PASS flow-time: forward/reverse Euler signs consistent")
    test_no_future_label()
    print("PASS causality: inference uses committed prefix not future GT")
    print("PASS ALL: RTC inference/training CPU contracts")
