#!/usr/bin/env python3
"""CPU-only educational contracts of StableVLA's channel attention: no model/weights."""
import math


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def channel_gram(tokens, head_dim):
    """x is [N,C], q_proj=identity only for this synthetic shape test."""
    n, c = len(tokens), len(tokens[0])
    assert c % head_dim == 0
    heads = c // head_dim
    grams = []
    for h in range(heads):
        grams.append([[sum(tokens[t][h * head_dim + i] *
                           tokens[t][h * head_dim + j] for t in range(n)) / math.sqrt(head_dim)
                      for j in range(head_dim)] for i in range(head_dim)])
    return grams


def gated_value(tokens, head_dim):
    """Simplified Q=K=V=X, temperature=1; not the official learned module."""
    gram = channel_gram(tokens, head_dim)
    n, c = len(tokens), len(tokens[0])
    result = [[0.0] * c for _ in range(n)]
    for h, logits in enumerate(gram):
        for t in range(n):
            for i in range(head_dim):
                result[t][h * head_dim + i] = sum(
                    sigmoid(logits[i][j]) * tokens[t][h * head_dim + j]
                    for j in range(head_dim))
    return result


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_main():
    tokens = [[1., 0.2, 0.0, 0.8], [0.3, 0.6, 0.7, 0.1], [0.8, 0.4, 0.2, 0.4]]
    gram = channel_gram(tokens, 2)
    assert len(gram) == 2 and all(len(m) == 2 and len(m[0]) == 2 for m in gram)
    out = gated_value(tokens, 2)
    assert len(out) == 3 and len(out[0]) == 4
    print('PASS B/H/D/N channel Gram shape: [1,2,2,2], output [1,3,4]')

    shuffled = [tokens[2], tokens[0], tokens[1]]
    gram_shuffled = channel_gram(shuffled, 2)
    assert all(close(gram[h][i][j], gram_shuffled[h][i][j])
               for h in range(2) for i in range(2) for j in range(2))
    shuffled_out = gated_value(shuffled, 2)
    for source, output in [(2, 0), (0, 1), (1, 2)]:
        assert all(close(out[source][j], shuffled_out[output][j]) for j in range(4))
    print('PASS patch permutation: channel Gram invariant; output equivariant')

    # sigmoid is NOT a normalized distribution; zero correlation -> 0.5, not zero.
    sig0 = sigmoid(0.)
    assert close(sig0, 0.5)
    weights = [sigmoid(0.)] * 3
    assert close(sum(weights), 1.5)
    print('PASS sigmoid-vs-softmax: independent gates, no row-sum-one constraint')

    fixed_coefficient = math.tanh(0.3)
    assert 0.29 < fixed_coefficient < 0.30 and not close(fixed_coefficient, .3)
    print('PASS fixed buffer fusion gate: tanh(0.3) ~= %.6f' % fixed_coefficient)

    baseline_s5, ours_s5 = 58.5, 82.0
    assert close(ours_s5 - baseline_s5, 23.5)
    assert close(ours_s5 / baseline_s5 - 1, 0.4017094017094016)
    print('PASS metrics: LIBERO-Spatial S5 +23.5pp, not +23.5% relative')
    print('PASS ALL: StableVLA channel-gating CPU mathematical contracts')


if __name__ == '__main__':
    test_main()
