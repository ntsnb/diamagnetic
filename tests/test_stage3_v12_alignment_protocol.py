from __future__ import annotations

import pytest
import torch

from models.stage3_alignment_v12 import TrainOnlyOutcomeAlignedTransportAlignerV12


def _toy_alignment_data(sample_count: int = 120) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(1203)
    phi_i = torch.randn(sample_count, 4)
    time = torch.linspace(-1.0, 1.0, sample_count).unsqueeze(1)
    invariant_leakage = 1.7 * phi_i[:, :1]
    useful_delta = torch.cat([time, -0.5 * time, torch.zeros(sample_count, 2)], dim=1)
    raw_delta = torch.cat([invariant_leakage, torch.zeros(sample_count, 3)], dim=1) + useful_delta
    phi_v = phi_i + raw_delta
    y_base = torch.zeros(sample_count, 2)
    residual = torch.cat([0.8 * time, -0.4 * time], dim=1)
    return phi_i, phi_v, y_base, y_base + residual


def test_v12_aligner_preserves_zero_identity_and_uses_shared_anchor():
    phi_i, phi_v, y_base, y = _toy_alignment_data()
    aligner = TrainOnlyOutcomeAlignedTransportAlignerV12(
        anchor_rank=4,
        max_rank=2,
        max_leakage_rank=2,
        min_calibration_r2=0.0,
        min_block_transfer_r2=0.0,
        min_block_sign_agreement=0.0,
        bootstrap_samples=0,
        reliability_gate_enabled=False,
    ).fit(
        phi_i_train=phi_i,
        phi_v_train=phi_v,
        y_base_train=y_base,
        y_train=y,
        split="train",
    )

    zero = aligner.transform_phi_pair(phi_i[:8], phi_i[:8])
    real = aligner.transform_phi_pair(phi_i[:8], phi_v[:8])
    summary = aligner.diagnostic_summary()

    assert summary["alignment_method"] == "outcome_aligned_transport_v12"
    assert summary["anchor_rank"] > 0
    assert summary["alignment_gate_pass"] is True
    assert torch.allclose(zero, torch.zeros_like(zero), atol=1e-7)
    assert torch.linalg.vector_norm(real).item() > 0.0


def test_v12_aligner_rejects_non_train_fit_and_removes_invariant_leakage():
    phi_i, phi_v, y_base, y = _toy_alignment_data()
    aligner = TrainOnlyOutcomeAlignedTransportAlignerV12(
        anchor_rank=4,
        max_rank=2,
        max_leakage_rank=2,
        min_calibration_r2=0.0,
        min_block_transfer_r2=0.0,
        min_block_sign_agreement=0.0,
        bootstrap_samples=0,
        reliability_gate_enabled=False,
    )

    with pytest.raises(ValueError, match="train split"):
        aligner.fit(
            phi_i_train=phi_i,
            phi_v_train=phi_v,
            y_base_train=y_base,
            y_train=y,
            split="validation",
        )

    aligner.fit(
        phi_i_train=phi_i,
        phi_v_train=phi_v,
        y_base_train=y_base,
        y_train=y,
        split="train",
    )
    summary = aligner.diagnostic_summary()

    assert summary["selected_leakage_rank"] > 0
    assert summary["oof_transfer_mse_gain"] > 0.0
    assert summary["active_rank"] > 0


def test_v12_aligner_fails_closed_when_residual_transfer_is_not_stable():
    phi_i, phi_v, y_base, y = _toy_alignment_data()
    shuffled_y = y[torch.randperm(y.shape[0])]
    aligner = TrainOnlyOutcomeAlignedTransportAlignerV12(
        anchor_rank=4,
        max_rank=2,
        max_leakage_rank=2,
        min_calibration_r2=0.2,
        min_block_transfer_r2=0.2,
        min_block_sign_agreement=0.75,
        bootstrap_samples=0,
        reliability_gate_enabled=False,
    ).fit(
        phi_i_train=phi_i,
        phi_v_train=phi_v,
        y_base_train=y_base,
        y_train=shuffled_y,
        split="train",
    )

    transformed = aligner.transform_phi_pair(phi_i[:12], phi_v[:12])
    summary = aligner.diagnostic_summary()

    assert summary["alignment_gate_pass"] is False
    assert summary["active_rank"] == 0
    assert torch.allclose(transformed, torch.zeros_like(transformed), atol=1e-7)


def test_v12_aligner_round_trips_state_dict():
    phi_i, phi_v, y_base, y = _toy_alignment_data()
    aligner = TrainOnlyOutcomeAlignedTransportAlignerV12(
        anchor_rank=4,
        max_rank=2,
        max_leakage_rank=2,
        min_calibration_r2=0.0,
        min_block_transfer_r2=0.0,
        min_block_sign_agreement=0.0,
        bootstrap_samples=0,
        reliability_gate_enabled=True,
    ).fit(
        phi_i_train=phi_i,
        phi_v_train=phi_v,
        y_base_train=y_base,
        y_train=y,
        split="train",
    )

    restored = TrainOnlyOutcomeAlignedTransportAlignerV12().load_state_dict(aligner.state_dict())

    assert torch.allclose(
        restored.transform_phi_pair(phi_i[:10], phi_v[:10]),
        aligner.transform_phi_pair(phi_i[:10], phi_v[:10]),
    )
    assert restored.diagnostic_summary()["alignment_method"] == "outcome_aligned_transport_v12"
