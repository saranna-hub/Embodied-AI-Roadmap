#!/usr/bin/env python3
"""EgoLAP mathematical/data-contract synthetic tests (educational, NOT author code).
Paper: https://arxiv.org/abs/2610.08726 (2026-10-06).
Only Python stdlib. No VLA checkpoint, video model, GRPO training or robot control.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence


def rotate_z(theta: float, xyz: Sequence[float]) -> tuple[float, float, float]:
    """R_z(theta) p, world coords; right-handed Z-up teaching example."""
    x, y, z = xyz
    return (math.cos(theta) * x - math.sin(theta) * y,
            math.sin(theta) * x + math.cos(theta) * y,
            z)


def egoframe_translation(
    world_start: Sequence[float],
    world_end: Sequence[float],
    frame_origin: Sequence[float],
    frame_yaw: float,
) -> tuple[float, float, float]:
    """pbar_end - pbar_start, both expressed in SAME start-of-window frame."""
    st = tuple(a - b for a, b in zip(world_start, frame_origin))
    en = tuple(a - b for a, b in zip(world_end, frame_origin))
    p_st = rotate_z(-frame_yaw, st)
    p_en = rotate_z(-frame_yaw, en)
    return tuple(e - s for s, e in zip(p_st, p_en))


def nearest_quantum(x: float, step: float) -> int:
    """Nearest-integer step, half-away-from-zero (document a tie policy)."""
    return math.copysign(math.floor(abs(x) / step + 0.5), x).__int__()


def language_action(
    arm: str,
    delta_m: Sequence[float],
    delta_deg_xyz: Sequence[float],
    closed: bool,
) -> str:
    """Subset of paper grammar; axis order x,z,y then roll,pitch,yaw.
    Does NOT implement full 3D extrinsic XYZ Euler extraction or authors' parser.
    """
    if arm not in ("Left", "Right"):
        raise ValueError("arm must be Left or Right")
    if len(delta_m) != 3 or len(delta_deg_xyz) != 3:
        raise ValueError("must supply 3 translation and 3 angular components")
    terms: list[str] = []
    for value, positive, negative in (
        (delta_m[0], "move forward", "move back"),
        (delta_m[2], "move up", "move down"),
        (delta_m[1], "move left", "move right"),
    ):
        num = nearest_quantum(value * 100.0, 1.0)
        if num:
            terms.append(f"{positive if num > 0 else negative} {abs(num)} cm")
    for value, positive, negative in (
        (delta_deg_xyz[0], "tilt left", "tilt right"),
        (delta_deg_xyz[1], "tilt back", "tilt forward"),
        (delta_deg_xyz[2], "rotate counterclockwise", "rotate clockwise"),
    ):
        num = nearest_quantum(value, 5.0) * 5
        if num:
            terms.append(f"{positive if num > 0 else negative} {abs(num)} degrees")
    terms.append("close gripper" if closed else "open gripper")
    return f"{arm} arm: " + ", ".join(terms)


def state_to_bins(values: Sequence[float]) -> list[int]:
    """Teaching uniform map for state normalized to [-1,1].
    Verify actual per-category q01/q99 transform before using physical data.
    """
    return [int(math.floor((min(1.0, max(-1.0, float(v))) + 1.0)
                           * 255.0 / 2.0 + 0.5)) for v in values]


def attention_mask(p: int, t: int, a: int) -> list[list[bool]]:
    """Each row is Query, each column Key; P/T/A semantics from paper Fig.3."""
    n = p + t + a
    m = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i < p:                      # prefix: ONLY prefix (bidirectional)
                m[i][j] = j < p
            elif i < p + t:               # text: prefix + earlier/self text
                m[i][j] = j < p or (p <= j <= i)
            else:                         # actions: prefix + ALL action tokens
                m[i][j] = j < p or j >= p + t
    return m


def transitive_reachable(graph: list[list[bool]]) -> list[list[bool]]:
    """If allowed attention were a graph, what can leak across MANY layers?"""
    reach = [row[:] for row in graph]
    n = len(graph)
    for k in range(n):
        for i in range(n):
            if reach[i][k]:
                for j in range(n):
                    reach[i][j] |= reach[k][j]
    return reach


def flow_training_sample(
    action: Sequence[float], eps: Sequence[float], tau: float
) -> tuple[list[float], list[float]]:
    if len(action) != len(eps) or not 0 <= tau <= 1:
        raise ValueError("invalid shapes/time")
    return ([(1.0 - tau) * a + tau * e for a, e in zip(action, eps)],
            [e - a for a, e in zip(action, eps)])


def inverse_euler_with_oracle(eps: Sequence[float], true_velocity: Sequence[float],
                              num_steps: int = 10) -> list[float]:
    """Exact for constant oracle field, a data-contract test only."""
    current = list(eps)
    dt = 1.0 / num_steps
    for _ in range(num_steps):
        current = [x - dt * v for x, v in zip(current, true_velocity)]
    return current


def sampled_reasoning_mode(rng: random.Random, has_label: bool) -> str:
    """B=Bern(0.5) controls context; C=Bern(0.5) controls reason CE."""
    if not has_label:
        return "direct"
    b = rng.random() < 0.5
    c = rng.random() < 0.5
    return ("direct" if not b else ("context_and_ce" if c else "context_only"))


@dataclass(frozen=True)
class ParsedMotion:
    """Tiny stand-in for the paper's full parseable motion grammar."""
    translation_cm: tuple[float, float, float]
    angle_deg: float
    closed: bool


def demo_reward(pred: ParsedMotion, target: ParsedMotion) -> float:
    """Paper Eq.9 specialization: translations, single-axis rotation, gripper."""
    trans = sum((p - t) ** 2 for p, t
                in zip(pred.translation_cm, target.translation_cm)) / 4.0
    wrapped = (pred.angle_deg - target.angle_deg + 180.0) % 360.0 - 180.0
    angle = (math.radians(wrapped) / (math.pi / 18.0)) ** 2
    grip = float(pred.closed != target.closed)
    return -(trans + angle + grip)


def group_advantages(rewards: Sequence[float]) -> list[float]:
    """Paper Eq.10: population standard deviation; zero for degenerate groups."""
    if not rewards or max(rewards) - min(rewards) <= 1e-6:
        return [0.0] * len(rewards)
    mu = sum(rewards) / len(rewards)
    sigma = math.sqrt(sum((x - mu) ** 2 for x in rewards) / len(rewards))
    return [max(-4.0, min(4.0, (x - mu) / max(sigma, 1e-4)))
            for x in rewards]


def check_all() -> None:
    # 1) Z-up example with a head frame rotated 90 degrees from world.
    dp = egoframe_translation((1., 1., 0.0), (1., 1.05, 0.0),
                              (1., 1., 0.0), math.pi / 2)
    assert abs(dp[0] - 0.05) < 1e-12 and abs(dp[1]) < 1e-12
    labels = language_action("Left", dp, (0., 0., 12.5), True)
    assert "move forward 5 cm" in labels
    assert "rotate counterclockwise 15 degrees" in labels
    assert labels.endswith("close gripper")
    for x in [0.0024, -0.0024, 0.0334, -0.0334, 0.1101]:
        cm_est = nearest_quantum(100.0 * x, 1.0) / 100.0
        assert abs(x - cm_est) <= 0.00500001
    assert state_to_bins([-2., -1., 0., 1., 2.]) == [0, 0, 128, 255, 255]
    print("PASS canonical coordinates and language action quantization:", labels)

    # 2) The action tokens can never see TRAINING-ONLY text, even indirectly.
    p, t, a = 2, 3, 4
    m = attention_mask(p, t, a)
    reach = transitive_reachable(m)
    for i in range(p):  # prefix cannot access training targets or action noise
        assert all(not m[i][j] for j in range(p, p + t + a))
    for i in range(p, p + t):  # text causal
        assert all(not m[i][j] for j in range(i + 1, p + t + a))
    for i in range(p + t, p + t + a):
        assert all(not m[i][j] and not reach[i][j]
                   for j in range(p, p + t))
        assert all(m[i][j] for j in range(p + t, p + t + a))
    print("PASS three-block causal mask prevents cross-layer target leakage:", len(m), "tokens")

    # 3) Flow is a 30x14 physical-action tensor, NOT 10 or 32 steps.
    rng = random.Random(4)
    K, D = 30, 14
    action = [rng.uniform(-0.5, 0.5) for _ in range(K * D)]
    eps = [rng.gauss(0., 1.) for _ in range(K * D)]
    tau = 0.001 + 0.999 * rng.betavariate(1.5, 1.)
    noisy, velocity = flow_training_sample(action, eps, tau)
    assert abs(noisy[7] - ((1 - tau) * action[7] + tau * eps[7])) < 1e-12
    estimated = inverse_euler_with_oracle(eps, velocity, 10)
    assert max(abs(a - b) for a, b in zip(action, estimated)) < 1e-12
    fm_loss_when_oracle = sum((v - (e - a)) ** 2
                              for v, e, a in zip(velocity, eps, action)) / (K * D)
    assert fm_loss_when_oracle == 0.0
    print("PASS flow orientation and 10-step inverse Euler oracle:", K, "x", D, "loss", fm_loss_when_oracle)

    # 4) For a reasoning-annotated sample: 50% direct, 25% reason context,
    # 25% reason context with reason CE (statistical test not exact distribution).
    draws = 10000
    counts = {"direct": 0, "context_only": 0, "context_and_ce": 0}
    for _ in range(draws):
        counts[sampled_reasoning_mode(rng, True)] += 1
    for k, expected in [("direct", 0.5), ("context_only", 0.25),
                        ("context_and_ce", 0.25)]:
        assert abs(counts[k] / draws - expected) < 0.025
    assert all(sampled_reasoning_mode(rng, False) == "direct" for _ in range(100))
    print("PASS stochastic reason context/loss masks:", counts)

    # 5) Eq.9, Eq.10; a perfect language action has reward 0 (not positive).
    tgt = ParsedMotion((4., -2., 0.), 15., True)
    perfect = demo_reward(tgt, tgt)
    dx = demo_reward(ParsedMotion((6., -2., 0.), 15., True), tgt)
    dtheta = demo_reward(ParsedMotion((4., -2., 0.), 25., True), tgt)
    dgrip = demo_reward(ParsedMotion((4., -2., 0.), 15., False), tgt)
    assert perfect == 0.0
    assert all(abs(x + 1.0) < 1e-12 for x in [dx, dtheta, dgrip])
    rewards = [0., -1., -2., -9.]
    adv = group_advantages(rewards)
    assert abs(sum(adv)) < 1e-10 and adv[0] > adv[-1]
    assert group_advantages([-1., -1., -1.]) == [0., 0., 0.]
    print("PASS deterministic demonstration-matching reward and group advantage:", adv)
    print("PASS ALL: EGO-LAP educational contracts; NOT model or robot replication")


if __name__ == "__main__":
    check_all()
