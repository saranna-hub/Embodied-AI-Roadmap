"""PACT-VLA FlowRes CPU contract test. Synthetic oracle only, NOT official reproduction.
Requires Python 3 standard library. No GPU, model, or robot is used.
Paper: https://arxiv.org/html/2610.28019 (Eq. 5-6, Table 6).
"""
from __future__ import annotations

import math
import random

N, H, D = 4, 50, 32
REAL_D, EXEC_H = 7, 5
TAUS = (0.2, 0.4, 0.6, 0.8)


def action(value):
    return [[value + 0.01 * j if k < REAL_D else 0.0
             for k in range(D)] for j in range(H)]


def flowres(candidate, target, noise, include_padding=False):
    """Use the same verification noise for all N candidates and all M flow times."""
    error = 0.0
    for tau in TAUS:
        for j in range(EXEC_H):
            for k in range(D if include_padding else REAL_D):
                xt = (1.0 - tau) * candidate[j][k] + tau * noise[j][k]
                # Idealized conditional vector field for a fixed 'target'.
                # This oracle is NOT an actual VLA.
                v_hat = (xt - target[j][k]) / tau
                implied = noise[j][k] - candidate[j][k]
                error += (v_hat - implied) ** 2
    return error


def main():
    rng = random.Random(261028019)
    noise = [[rng.gauss(0.0, 1.0) for _ in range(D)] for _ in range(H)]
    old, fresh = action(0.0), action(0.5)
    candidates = [old, fresh, action(-0.4), action(0.95)]
    assert len(candidates) == N
    old_scores = [flowres(a, old, noise) for a in candidates]
    new_scores = [flowres(a, fresh, noise) for a in candidates]
    assert min(range(N), key=lambda n: old_scores[n]) == 0
    assert min(range(N), key=lambda n: new_scores[n]) == 1
    assert math.isclose(new_scores[1], 0.0, abs_tol=1e-24)
    print("PASS precommit switches candidate: stale=0 precommit=1")

    padded = action(0.5)
    for j in range(H):
        for k in range(REAL_D, D):
            padded[j][k] = 1_000_000.0
    assert math.isclose(flowres(padded, fresh, noise), 0.0, abs_tol=1e-24)
    assert flowres(padded, fresh, noise, include_padding=True) > 1e12
    print("PASS masked 7 of 32 dims: padding ignored")

    new_tail = action(0.5)
    for j in range(EXEC_H, H):
        new_tail[j][0] += 2000
    assert math.isclose(flowres(new_tail, fresh, noise), 0.0, abs_tol=1e-24)
    print("PASS executed 5 of 50 positions scored")

    period_ms = 1000.0 / 30.0
    assert math.ceil(35.4 / period_ms) == 2
    assert math.ceil(60.2 / period_ms) == 2
    assert math.ceil(64.8 / period_ms) == 2
    print("PASS latency bucket: 35.4ms, 60.2ms, 64.8ms all d=2 at 30Hz")
    print("PASS ALL: synthetic algebra check; NOT π0.5 inference or robot rollout")


if __name__ == "__main__":
    main()
