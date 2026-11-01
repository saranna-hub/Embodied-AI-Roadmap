#!/usr/bin/env python3
"""WING-inspired CPU-only synthetic contracts. Not the author's training code.

Verifies (1) an affine camera warp vs. interaction residual,
(2) orthonormal temporal DCT and low-frequency truncation,
(3) an inference-time spectral guidance function without future frames.

This deliberately does NOT train WING-LAM, a WAM, or a robot policy.
"""
import numpy as np


def compensate_with_background(background, moved_background, hand, moved_hand):
    """Fit an affine warp on background only; return compensated motion."""
    matrix = np.column_stack([background, np.ones(len(background))])
    affine, *_ = np.linalg.lstsq(matrix, moved_background, rcond=None)
    linear, shift = affine[:2, :].T, affine[2, :]

    def undo(points):
        return np.linalg.solve(linear, (points - shift).T).T

    return undo(moved_background) - background, undo(moved_hand) - hand


def dct_ii_orthonormal(horizon):
    """DCT-II basis across the *time* axis, not across latent channels."""
    k = np.arange(horizon, dtype=np.float64)[:, None]
    t = np.arange(horizon, dtype=np.float64)[None, :]
    transform = np.sqrt(2.0 / horizon) * np.cos(
        np.pi / horizon * (t + 0.5) * k
    )
    transform[0, :] = np.sqrt(1.0 / horizon)
    return transform


def predict_guidance_from_present(image_feature, language_feature, robot_state, k=4, dz=32):
    """Illustrative present-only interface; no future frames accepted."""
    present = (np.mean(image_feature) + np.mean(language_feature)
               + np.mean(robot_state))
    return np.full((k, dz), np.tanh(present), dtype=np.float64)


def main():
    rng = np.random.default_rng(20261101)
    background = rng.uniform(-2.0, 2.0, size=(48, 2))
    hand = rng.uniform(-0.5, 0.5, size=(8, 2))
    linear = np.array([[1.015, -0.030], [0.020, 0.995]])
    shift = np.array([0.35, -0.27])
    interaction = np.array([0.12, -0.07])
    moved_background = background @ linear.T + shift
    moved_hand = hand @ linear.T + shift + interaction

    residual_background, residual_hand = compensate_with_background(
        background, moved_background, hand, moved_hand
    )
    bg_error = float(np.max(np.abs(residual_background)))
    hand_signal = float(np.mean(np.linalg.norm(residual_hand, axis=-1)))
    assert bg_error < 1e-10, bg_error
    assert hand_signal > 0.10, hand_signal
    print(f"PASS camera motion removed: max_background_residual={bg_error:.3e}")
    print(f"PASS interaction residual retained: mean_hand_motion={hand_signal:.6f}")

    horizon, latent_dim, retained = 8, 32, 4
    temporal_basis = dct_ii_orthonormal(horizon)
    assert np.allclose(temporal_basis @ temporal_basis.T, np.eye(horizon))
    t = np.arange(horizon)
    common_slow_motion = 0.5 + 0.4 * np.sin(np.pi * t / (horizon - 1))
    different_fast_motion = (-1.0) ** t
    channel_scales = np.linspace(0.5, 1.5, latent_dim)
    human = (common_slow_motion[:, None] + 0.25 * different_fast_motion[:, None]) * channel_scales
    robot = (common_slow_motion[:, None] - 0.25 * different_fast_motion[:, None]) * channel_scales

    human_coeff = temporal_basis @ human
    robot_coeff = temporal_basis @ robot
    assert human_coeff.shape == (horizon, latent_dim)
    assert human_coeff[:retained].shape == (retained, latent_dim)
    assert np.allclose(temporal_basis.T @ human_coeff, human, atol=1e-12)
    low_delta = float(np.linalg.norm(human_coeff[:retained] - robot_coeff[:retained]))
    full_delta = float(np.linalg.norm(human_coeff - robot_coeff))
    assert low_delta < full_delta, (low_delta, full_delta)
    assert low_delta > 0.0
    print(
        "PASS DCT and spectral bandwidth:"
        f" coeff_shape={human_coeff.shape} retained={retained}"
        f" low_gap={low_delta:.4f} full_gap={full_delta:.4f}"
    )

    # Spectral guidance does not rewrite the robot's own physical action contract.
    robot_action_chunk = np.zeros((50, 7))
    assert robot_action_chunk.shape == (50, 7)
    present_image_feature = rng.normal(size=(16,))
    present_language_feature = rng.normal(size=(8,))
    present_robot_state = rng.normal(size=(7,))
    predicted = predict_guidance_from_present(
        present_image_feature, present_language_feature, present_robot_state,
        retained, latent_dim
    )
    assert predicted.shape == (retained, latent_dim)

    # Changing hypothetical future data must not affect the present-only interface.
    future_a = human_coeff[:retained].copy()
    future_b = robot_coeff[:retained].copy()
    assert not np.allclose(future_a, future_b)
    predicted_again = predict_guidance_from_present(
        present_image_feature, present_language_feature, present_robot_state,
        retained, latent_dim
    )
    assert np.array_equal(predicted, predicted_again)
    print("PASS no-future inference contract: current_only_shape=(4, 32)")
    print("PASS ALL: WING synthetic camera/DCT contracts only; no model or robot tested")


if __name__ == "__main__":
    main()
