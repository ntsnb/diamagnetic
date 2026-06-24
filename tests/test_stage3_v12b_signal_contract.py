from __future__ import annotations

import pytest
import torch

from train.stage3_v12b_signal_contract import Stage3TeacherSignalBuilderV12b


def test_v12b_signal_contract_is_train_only_and_zero_preserving():
    drift = torch.randn(64, 4)
    builder = Stage3TeacherSignalBuilderV12b(drift_dim=4, mode="level_vector")

    with pytest.raises(ValueError, match="train split"):
        builder.fit(drift, split="validation")

    builder.fit(drift, split="train")
    zero = builder.transform(torch.zeros(8, 4)).signal

    assert torch.allclose(zero, torch.zeros_like(zero), atol=1e-8)
    assert builder.diagnostic_summary()["signal_contract_version"] == builder.protocol_version


def test_v12b_signal_contract_uses_train_scale_not_batch_abs_mean():
    train = torch.cat([torch.ones(16, 3), 2.0 * torch.ones(16, 3)], dim=0)
    small = 0.01 * torch.ones(4, 3)
    builder = Stage3TeacherSignalBuilderV12b(drift_dim=3, mode="level_vector", robust_scale_quantile=0.5)
    builder.fit(train, split="train")
    transformed = builder.transform(small).signal

    assert transformed.abs().max().item() < 0.02


def test_v12b_delta_signal_counterfactual_modes_are_consistent():
    drift = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    builder = Stage3TeacherSignalBuilderV12b(drift_dim=3, mode="delta_vector", normalize=False)
    builder.fit(drift, split="train")

    real = builder.transform(drift, counterfactual_mode="real").signal
    flipped = builder.transform(drift, counterfactual_mode="sign_flipped").signal
    zero = builder.transform(drift, counterfactual_mode="zero").signal
    amplified = builder.transform(drift, counterfactual_mode="amplified_2x").signal

    assert torch.allclose(flipped, -real)
    assert torch.allclose(zero, torch.zeros_like(real))
    assert torch.allclose(amplified, 2.0 * real)
