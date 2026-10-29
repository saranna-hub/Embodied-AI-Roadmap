#!/usr/bin/env python3
"""Educational, dependency-free Pi0.5/LIBERO data-contract checks.

NOT a reimplementation of openpi or a policy/benchmark run.
Pinned source: Physical-Intelligence/openpi@215abfb217dbac7d5f1273282331b9b1866c0479
"""
from __future__ import annotations

import math
import random


def qnorm(value: float, q01: float, q99: float) -> float:
    if q99 <= q01:
        raise ValueError("non-positive quantile span")
    return (value - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def qdenorm(value: float, q01: float, q99: float) -> float:
    if q99 <= q01:
        raise ValueError("non-positive quantile span")
    return (value + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def pad_to_dim(actions: list[list[float]], out_dim: int) -> list[list[float]]:
    if any(len(step) > out_dim for step in actions):
        raise ValueError("action dimension exceeds container dimension")
    return [step + [0.0] * (out_dim - len(step)) for step in actions]


def flow_pair(actions: list[list[float]], time: float, seed: int = 17):
    if not 0 <= time <= 1:
        raise ValueError("flow time must be in [0,1]")
    rng = random.Random(seed)
    noise = [[rng.gauss(0.0, 1.0) for _ in row] for row in actions]
    intermediate = [
        [time * eps + (1.0 - time) * action for eps, action in zip(ni, ai)]
        for ni, ai in zip(noise, actions)
    ]
    target_velocity = [
        [eps - action for eps, action in zip(ni, ai)]
        for ni, ai in zip(noise, actions)
    ]
    return noise, intermediate, target_velocity


def mse(velocity, target, dim: int) -> float:
    squared = [
        (velocity[t][j] - target[t][j]) ** 2
        for t in range(len(target))
        for j in range(dim)
    ]
    return sum(squared) / len(squared)


def run() -> None:
    state = [0.70, -0.10, 0.25, 0.0, 0.3, -0.3, 0.75]
    # LIBERO-like action is ALREADY delta; 7th dimension is a gripper command.
    horizon = 10
    raw_delta = [
        [0.02 + 0.001 * t, 0.01, -0.005, 0.0, 0.0, 0.0, 1.0]
        for t in range(horizon)
    ]
    assert len(raw_delta) == 10 and all(len(step) == 7 for step in raw_delta)
    wrong_delta = [
        [a - state[i] if i < 6 else a for i, a in enumerate(step)]
        for step in raw_delta
    ]
    assert math.isclose(raw_delta[0][0], 0.02)
    assert math.isclose(wrong_delta[0][0], -0.68)
    print("PASS delta-contract: correct_first=0.02 double_delta_first=-0.68")

    q01, q99 = -0.20, 0.20
    for physical in (-0.2, 0.0, 0.1, 0.2):
        assert abs(qdenorm(qnorm(physical, q01, q99), q01, q99) - physical) < 1e-12
    print("PASS q01/q99 normalization round-trip")

    norm_actions = [[qnorm(a, -1.0, 1.0) for a in step] for step in raw_delta]
    padded = pad_to_dim(norm_actions, 32)
    assert len(padded) == horizon and all(len(x) == 32 for x in padded)
    assert all(all(x == 0.0 for x in row[7:]) for row in padded)
    print("PASS action padding [10,7] -> [10,32]")

    noise, x_t, target = flow_pair(padded, time=0.7)
    for t in range(horizon):
        for j in range(32):
            assert math.isclose(
                x_t[t][j], 0.7 * noise[t][j] + 0.3 * padded[t][j], abs_tol=1e-12
            )
            assert math.isclose(
                target[t][j], noise[t][j] - padded[t][j], abs_tol=1e-12
            )
    assert mse(target, target, dim=32) == 0.0
    print("PASS FM time convention and target: x_t=t*noise+(1-t)*action")

    # Artificial residual 1 only on 7 valid dims, zero on 25 padding dims.
    prediction = [
        [target[t][j] + (1.0 if j < 7 else 0.0) for j in range(32)]
        for t in range(horizon)
    ]
    full_loss, valid_loss = mse(prediction, target, 32), mse(prediction, target, 7)
    assert math.isclose(full_loss, 7 / 32, rel_tol=1e-12)
    assert math.isclose(valid_loss, 1.0, rel_tol=1e-12)
    print(f"PASS loss denominator: full32={full_loss:.6f} valid7={valid_loss:.6f}")

    fps, execute = 20, 4
    offsets = [t / fps for t in range(horizon)]
    assert math.isclose(offsets[-1], 0.45)
    assert math.isclose(execute / fps, 0.2)
    print(
        f"PASS action times: H=10 fps={fps} "
        f"last_target={offsets[-1]:.2f}s execute={execute / fps:.2f}s"
    )
    print("PASS ALL: mathematical data-contract checks (no openpi training or inference executed)")


if __name__ == "__main__":
    run()
