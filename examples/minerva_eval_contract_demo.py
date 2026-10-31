#!/usr/bin/env python3
"""MINERVA / LIBERO synthetic evaluation contract (CPU, stdlib only).

This verifies indexing/statistical bookkeeping, not MINERVA weights or rollout scores.
"""
from __future__ import annotations

import math


def wilson(successes: int, count: int, z: float = 1.96) -> tuple[float, float]:
    if not 0 <= successes <= count or count < 1:
        raise ValueError("invalid Bernoulli sample")
    p = successes / count
    d = 1.0 + z * z / count
    center = (p + z * z / (2.0 * count)) / d
    radius = z * math.sqrt(p * (1.0 - p) / count + z * z / (4 * count * count)) / d
    return center - radius, center + radius


def ensemble_action(chunks: list[tuple[int, list[tuple[float, ...]]]],
                    tick: int, decay: float = 0.4) -> tuple[float, ...]:
    valid = []
    for start_tick, actions in chunks:
        offset = tick - start_tick
        if 0 <= offset < len(actions):
            valid.append((math.exp(-decay * offset), actions[offset]))
    if not valid:
        raise ValueError(f"no predicted action at tick={tick}")
    dim = len(valid[0][1])
    if any(len(a) != dim for _, a in valid):
        raise AssertionError("inconsistent action dimensions")
    denom = sum(weight for weight, _ in valid)
    return tuple(sum(weight * action[d] for weight, action in valid) / denom
                 for d in range(dim))


def episode_action_indices(start: int, episode_length: int,
                           horizon: int) -> tuple[list[int], list[bool]]:
    if not 0 <= start < episode_length or horizon < 1:
        raise ValueError("invalid episode/horizon")
    indices = [min(start + offset, episode_length - 1) for offset in range(horizon)]
    valid = [start + offset < episode_length for offset in range(horizon)]
    return indices, valid


def main() -> None:
    horizon, flow_steps, execution_horizon = 16, 10, 1
    assert horizon != flow_steps != execution_horizon
    assert horizon > execution_horizon
    print("PASS H/K/E distinct: predicted=16, flow_iterations=10, executed=1")

    chunks = []
    for start in range(7):
        # Synthetic robot 7D deltas: all chunks agree that physical time is start+offset,
        # but later predictions have a small drift to exercise temporal averaging.
        predicted = [(float(start + offset + 0.03 * start),) + (0.0,) * 6
                     for offset in range(horizon)]
        chunks.append((start, predicted))
    now = 6
    merged = ensemble_action(chunks, now)
    assert len(merged) == 7
    assert 6.0 <= merged[0] <= 6.20
    assert merged[1:] == (0.0,) * 6
    assert len([a for start, acts in chunks if 0 <= now - start < len(acts)]) == 7
    print("PASS temporal ensemble: overlap indices and 7D action dimensions")

    indices, mask = episode_action_indices(start=10, episode_length=12, horizon=16)
    assert indices[:2] == [10, 11] and all(i == 11 for i in indices[2:])
    assert mask == [True, True] + [False] * 14
    print("PASS episode boundary: no cross-episode action supervision")

    ci_small = wilson(19, 20)
    ci_large = wilson(1900, 2000)
    assert ci_small[0] < 0.95 < ci_small[1]
    assert ci_large[0] < 0.95 < ci_large[1]
    assert (ci_large[1] - ci_large[0]) < (ci_small[1] - ci_small[0])
    print(f"PASS Wilson interval: n=20 {ci_small[0]:.3f}-{ci_small[1]:.3f}"
          f", n=2000 {ci_large[0]:.3f}-{ci_large[1]:.3f}")

    task_ids = list(range(10))
    identity = dict(zip(task_ids, task_ids))
    permuted = {i: (i + 1) % len(task_ids) for i in task_ids}
    assert sum(identity[i] == i for i in task_ids) == 10
    assert sum(permuted[i] == i for i in task_ids) == 0
    # This only audits task-ID bookkeeping, NOT real policy success under permutation.
    print("PASS task-ID permutation: baseline and rotated map are distinct")
    print("PASS ALL: CPU synthetic contract only; no model or LIBERO rollout was executed")


if __name__ == "__main__":
    main()
