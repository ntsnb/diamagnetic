from __future__ import annotations

import torch

from train.stage3_signals import Stage3SignalState, build_stage3_drift_vector


def test_level_scalar_matches_original_global_mean_repeat():
    drift = torch.tensor(
        [
            [1.0, -2.0, 3.0],
            [2.0, -1.0, 4.0],
        ]
    )

    result = build_stage3_drift_vector(
        drift,
        mode="level_scalar",
        bottleneck_dim=5,
        state=Stage3SignalState(),
    )

    expected = (drift.mean() / (drift.abs().mean() + 1e-6)).repeat(5)
    assert result.drift_vec.shape == (5,)
    assert torch.allclose(result.drift_vec, expected)
    assert result.drift_vec.unique().numel() == 1


def test_level_vector_preserves_embedding_dimensions():
    drift = torch.tensor(
        [
            [1.0, -2.0, 3.0],
            [2.0, -1.0, 5.0],
        ]
    )

    result = build_stage3_drift_vector(
        drift,
        mode="level_vector",
        bottleneck_dim=5,
        state=Stage3SignalState(),
    )

    expected = drift.mean(dim=0) / (drift.abs().mean() + 1e-6)
    assert result.drift_vec.shape == (3,)
    assert torch.allclose(result.drift_vec, expected)
    assert not torch.allclose(result.drift_vec, result.drift_vec.mean().repeat(3))


def test_delta_vector_uses_previous_sample_and_previous_batch_tail():
    first = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [2.0, 4.0, 8.0],
        ]
    )
    second = torch.tensor(
        [
            [5.0, 7.0, 11.0],
            [6.0, 8.0, 13.0],
        ]
    )
    state = Stage3SignalState()

    first_result = build_stage3_drift_vector(
        first,
        mode="delta_vector",
        bottleneck_dim=4,
        state=state,
    )
    second_result = build_stage3_drift_vector(
        second,
        mode="delta_vector",
        bottleneck_dim=4,
        state=state,
    )

    expected_first_signal = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 3.0, 7.0],
        ]
    )
    expected_second_signal = torch.tensor(
        [
            [3.0, 3.0, 3.0],
            [1.0, 1.0, 2.0],
        ]
    )
    expected_second_vec = expected_second_signal.mean(dim=0) / (
        expected_second_signal.abs().mean() + 1e-6
    )

    assert torch.allclose(first_result.signal, expected_first_signal)
    assert torch.allclose(second_result.signal, expected_second_signal)
    assert torch.allclose(second_result.drift_vec, expected_second_vec)
    assert second_result.drift_vec.shape == (3,)


def test_delta_scalar_is_repeated_temporal_difference_ratio():
    drift = torch.tensor(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [5.0, 7.0],
        ]
    )
    state = Stage3SignalState()

    result = build_stage3_drift_vector(
        drift,
        mode="delta_scalar",
        bottleneck_dim=6,
        state=state,
    )

    expected_signal = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [3.0, 3.0],
        ]
    )
    expected_ratio = expected_signal.mean() / (expected_signal.abs().mean() + 1e-6)
    assert torch.allclose(result.signal, expected_signal)
    assert result.drift_vec.shape == (6,)
    assert torch.allclose(result.drift_vec, expected_ratio.repeat(6))
    assert result.drift_vec.unique().numel() == 1


def test_drift_polarity_explicitly_flips_level_vector_signal():
    drift = torch.tensor([[1.0, -2.0, 3.0]])

    result = build_stage3_drift_vector(
        drift,
        mode="level_vector",
        bottleneck_dim=3,
        state=Stage3SignalState(),
        normalize_vector=False,
        polarity=-1.0,
    )

    assert torch.allclose(result.raw, -drift)
    assert torch.allclose(result.signal, -drift)
    assert torch.allclose(result.drift_vec, -drift.squeeze(0))
