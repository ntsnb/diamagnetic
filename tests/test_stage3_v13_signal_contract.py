from __future__ import annotations

import pytest
import torch

from train.stage3_v13_signal_contract import Stage3SignalBuilderV13, compute_v13_sequence_delta


def test_v13_fluctuation_delta_uses_level_offset_transition() -> None:
    level = torch.tensor(
        [
            [1.0, 2.0, 4.0],
            [3.0, 5.0, 9.0],
            [4.0, 8.0, 15.0],
        ]
    )
    builder = Stage3SignalBuilderV13(drift_dim=3, mode="fluctuation_delta", normalize=False)
    builder.fit(level, split="train")

    result = builder.transform(level, counterfactual_mode="real")

    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 3.0, 5.0],
            [1.0, 3.0, 6.0],
        ]
    )
    assert torch.allclose(result.level_offset, level)
    assert torch.allclose(result.fluctuation, expected)
    assert torch.allclose(result.signal, expected)
    assert result.mode == "fluctuation_delta"
    assert result.protocol_version == builder.protocol_version


def test_v13_fluctuation_delta_uses_previous_tail() -> None:
    train = torch.tensor([[10.0, 20.0], [13.0, 25.0]])
    validation = torch.tensor([[18.0, 31.0], [20.0, 40.0]])
    builder = Stage3SignalBuilderV13(drift_dim=2, mode="fluctuation_delta", normalize=False)
    builder.fit(train, split="train")

    result = builder.transform(validation, previous_level_tail=train[-1:], counterfactual_mode="real")

    assert torch.allclose(result.fluctuation[0], torch.tensor([5.0, 6.0]))
    assert torch.allclose(result.fluctuation[1], torch.tensor([2.0, 9.0]))


def test_v13_variant_temporal_delta_requires_actual_variant_sequence() -> None:
    phi_v = torch.tensor([[2.0, 4.0], [5.0, 8.0], [6.0, 9.0]])
    previous_phi_v = torch.tensor([[1.0, 3.0]])

    delta = compute_v13_sequence_delta(phi_v, previous_tail=previous_phi_v)

    assert torch.allclose(delta[0], torch.tensor([1.0, 1.0]))
    assert torch.allclose(delta[1], torch.tensor([3.0, 4.0]))
    assert torch.allclose(delta[2], torch.tensor([1.0, 1.0]))


def test_v13_train_only_scaling_and_zero_preservation() -> None:
    train = torch.cat([torch.ones(16, 3), 3.0 * torch.ones(16, 3)], dim=0)
    builder = Stage3SignalBuilderV13(
        drift_dim=3,
        mode="fluctuation_delta",
        normalize=True,
        robust_scale_quantile=0.75,
    )

    with pytest.raises(ValueError, match="train split"):
        builder.fit(train, split="validation")

    builder.fit(train, split="train")
    zero = builder.transform(torch.zeros(4, 3), counterfactual_mode="zero")

    assert torch.allclose(zero.signal, torch.zeros_like(zero.signal), atol=1e-8)
    assert builder.diagnostic_summary()["signal_contract_version"] == builder.protocol_version


def test_v13_counterfactual_modes_and_stop_diagnostics() -> None:
    level = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    builder = Stage3SignalBuilderV13(drift_dim=3, mode="fluctuation_delta", normalize=False)
    builder.fit(level, split="train")

    real = builder.transform(level, counterfactual_mode="real")
    flipped = builder.transform(level, counterfactual_mode="sign_flipped")
    zero = builder.transform(level, counterfactual_mode="zero")
    amplified = builder.transform(level, counterfactual_mode="amplified_2x")
    random_matched = builder.transform(level, counterfactual_mode="random_matched_norm")

    assert torch.allclose(flipped.signal, -real.signal)
    assert torch.allclose(zero.signal, torch.zeros_like(real.signal))
    assert torch.allclose(amplified.signal, 2.0 * real.signal)
    assert random_matched.signal.shape == real.signal.shape

    stop = builder.stop_diagnostic(real.signal, zero_fraction_threshold=0.99)
    assert stop["stop_condition_triggered"] is False
    assert stop["stop_condition_name"] == ""

