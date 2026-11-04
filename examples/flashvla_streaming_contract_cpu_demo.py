#!/usr/bin/env python3
"""FlashVLA 数学/结构契约验证，零第三方依赖。
仅检验 source-aligned 的缓冲长度、配置隔离、流时间方向与异步动作窗口；
不是官方模型、GPU 性能或 LIBERO 成功率复现。
"""
from __future__ import annotations

from math import isclose

N = 4
C = 10
D = 7


def packed_action_contract() -> tuple[int, int, int]:
    horizon = N * C
    masks = []
    for k in range(1, N + 1):
        masks.extend([True] * (k * C) + [False] * (horizon - k * C))
    assert len(masks) == N * N * C == 160
    assert sum(masks) == 100
    assert len(masks) - sum(masks) == 60
    return len(masks), sum(masks), len(masks) - sum(masks)


def block_causal_allow(q: int, k: int) -> bool:
    """q/k: 0=Prefix; 1..N=chunk。Prefix 不得读后缀。"""
    if q == 0:
        return k == 0
    return k == 0 or (1 <= k <= q)


def shared_observation_allow(q_config: int, k_config: int,
                             query_is_suffix: bool = True,
                             key_is_suffix: bool = True) -> bool:
    if not query_is_suffix and key_is_suffix:
        return False
    if query_is_suffix and key_is_suffix and q_config != k_config:
        return False
    return True


def mask_contract() -> None:
    for query in range(N + 1):
        for key in range(N + 1):
            assert block_causal_allow(query, key) == (key == 0 or (query >= key > 0))
    assert block_causal_allow(4, 1)
    assert not block_causal_allow(1, 4)
    assert not block_causal_allow(0, 4)
    for q in range(N):
        for k in range(N):
            assert shared_observation_allow(q, k) == (q == k)


def oracle_streaming_contract() -> None:
    """对每个 chunk，严格用 N 次正确方向 Euler 积分去噪。

    理想化速度场恒为 eps-target，测试符号约定与摊销次数，
    不模拟真实 VLA 的多任务拟合误差。
    """
    targets = [0.25, -0.45, 0.7, -0.2, 1.1, 0.05, -0.7, 0.4]
    epsilons = [1.2, -0.9, 0.3, 1.8, -1.5, 2.1, -0.1, 0.9]
    active = []  # dict {id, x, updates}
    outputs = []
    # one new chunk enters per forward; each active chunk performs one inverse Euler.
    for t in range(len(targets) + N - 1):
        if t < len(targets):
            active.append({"id": t, "x": epsilons[t], "updates": 0})
        for obj in active:
            j = obj["id"]
            velocity = epsilons[j] - targets[j]
            obj["x"] -= velocity / N
            obj["updates"] += 1
        if active[0]["updates"] == N:
            done = active.pop(0)
            assert isclose(done["x"], targets[done["id"]], abs_tol=1e-12)
            assert done["updates"] == N
            outputs.append(done["id"])
        if t < N - 1:
            assert len(outputs) == 0, "cold-start must not emit policy chunks"
    assert outputs == list(range(len(targets)))


def async_window_contract(K: int, overlap: int, size: int) -> None:
    if not (0 <= overlap <= K and overlap + K <= size):
        raise ValueError("invalid async overlap or stale-action window")
    full = list(range(size))
    selected = full[overlap:overlap + K]
    assert len(selected) == K
    assert selected[0] == overlap
    assert selected[-1] == overlap + K - 1


def test_bad_time_direction() -> None:
    target, eps = 0.25, 1.2
    wrong = eps + (eps - target) / N
    right = eps - (eps - target) / N
    assert abs(right - target) < abs(wrong - target)


if __name__ == "__main__":
    length, valid, padded = packed_action_contract()
    print(f"PASS packed training tensor: {length} tokens, valid={valid}, padded={padded}")
    mask_contract()
    print("PASS chunk-causal and cross-configuration attention masks")
    oracle_streaming_contract()
    print("PASS inverse Euler oracle: each chunk 4 steps, no early cold-start output")
    async_window_contract(K=5, overlap=2, size=C)
    try:
        async_window_contract(K=9, overlap=2, size=C)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid overlap must fail")
    print("PASS async windows: d+K<=C and invalid overlap rejected")
    test_bad_time_direction()
    print("PASS flow time convention: data=0, noise=1, update subtracts velocity")
    print("PASS ALL: FlashVLA CPU contracts")
