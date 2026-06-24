from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from models.adaptation_v10 import ResidualBasisHyperGeneratorV10
from models.stage3_alignment_v11 import TrainSetCrossFittedResidualTransferPLSAlignerV11
from models.stage3_residual_teacher_v11 import (
    TrainOnlyRobustBlockResidualTeacherV11,
    load_stage3_v11_teacher_checkpoint,
    save_stage3_v11_teacher_checkpoint,
)
from train.stage3_generator_training_v10 import load_stage3_generator_checkpoint, save_stage3_generator_checkpoint
from train.stage3_v11_artifacts import load_stage3_v11_artifact_bundle


def _toy_teacher_data(sample_count: int = 96) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    drift = torch.linspace(-1.0, 1.0, sample_count).unsqueeze(1)
    drift = torch.cat([drift, drift.square(), torch.sin(drift * 2.0)], dim=1)
    y_base = torch.zeros(sample_count, 2)
    correction = torch.stack([0.4 * drift[:, 0] - 0.2 * drift[:, 2], -0.3 * drift[:, 0]], dim=1)
    return drift, y_base, y_base + correction


def test_v11_teacher_block_calibrates_helpful_correction_and_preserves_zero_identity():
    drift, y_base, y = _toy_teacher_data()
    teacher = TrainOnlyRobustBlockResidualTeacherV11(
        drift_dim=3,
        output_shape=(2,),
        ridge=1e-6,
        shrinkage_grid=[0.0, 0.25, 0.5, 0.75, 1.0],
        alpha_mode="block_scalar",
        calibration_block_count=4,
        min_block_mse_gain=0.0,
        min_block_mae_gain=0.0,
        min_block_sign_agreement=0.5,
        reliability_gate_enabled=False,
    ).fit(drift_train=drift, y_base_train=y_base, y_train=y, split="train")

    zero_correction = teacher.predict_correction(torch.zeros_like(drift[:4]))
    helpful_correction = teacher.predict_correction(drift)
    base_mse = torch.mean((y - y_base).pow(2)).item()
    teacher_mse = torch.mean((y - (y_base + helpful_correction)).pow(2)).item()

    assert torch.allclose(zero_correction, torch.zeros_like(zero_correction), atol=1e-7)
    assert teacher.shrinkage_alpha > 0.0
    assert teacher_mse < base_mse
    summary = teacher.diagnostic_summary()
    assert summary["teacher_alpha_source"] == "block_scalar_grid"
    assert summary["teacher_block_pass_count"] == summary["teacher_effective_block_count"]


def test_v11_teacher_harmful_chronological_block_abstains_with_zero_alpha():
    drift, y_base, y = _toy_teacher_data(sample_count=96)
    y = y.clone()
    y[72:] = y_base[72:] - (y[72:] - y_base[72:])
    teacher = TrainOnlyRobustBlockResidualTeacherV11(
        drift_dim=3,
        output_shape=(2,),
        ridge=1e-6,
        inner_fit_fraction=0.5,
        shrinkage_grid=[0.0, 0.25, 0.5, 1.0],
        alpha_mode="block_scalar",
        calibration_block_count=2,
        min_block_mse_gain=0.001,
        min_block_mae_gain=0.001,
        min_block_sign_agreement=0.5,
        reliability_gate_enabled=False,
    ).fit(drift_train=drift, y_base_train=y_base, y_train=y, split="train")

    correction = teacher.predict_correction(drift)

    assert teacher.shrinkage_alpha == pytest.approx(0.0)
    assert torch.allclose(correction, torch.zeros_like(correction), atol=1e-7)
    assert teacher.diagnostic_summary()["teacher_alpha_source"] == "fallback_block_gate"


def test_v11_teacher_rejects_non_train_update_and_round_trips(tmp_path: Path):
    drift, y_base, y = _toy_teacher_data()
    teacher = TrainOnlyRobustBlockResidualTeacherV11(
        drift_dim=3,
        output_shape=(2,),
        ridge=1e-5,
        reliability_gate_enabled=False,
    )
    teacher.fit(drift_train=drift, y_base_train=y_base, y_train=y, split="train")
    before = teacher.state_hash()

    with pytest.raises(ValueError, match="train split"):
        teacher.fit(drift_train=drift, y_base_train=y_base, y_train=y, split="validation")
    with pytest.raises(ValueError, match="train split"):
        teacher.update(drift=drift, y_base=y_base, y=y, split="test")
    assert teacher.state_hash() == before

    path = tmp_path / "teacher_v11.pt"
    save_stage3_v11_teacher_checkpoint(path=path, teacher=teacher, metadata={"run_id": "v11_roundtrip"})
    loaded, metadata = load_stage3_v11_teacher_checkpoint(path, expected_drift_dim=3, expected_output_shape=(2,))

    assert metadata["run_id"] == "v11_roundtrip"
    assert metadata["protocol_version"] == TrainOnlyRobustBlockResidualTeacherV11.protocol_version
    assert loaded.shrinkage_alpha == pytest.approx(teacher.shrinkage_alpha)
    assert torch.allclose(teacher.predict_correction(drift), loaded.predict_correction(drift))


def test_v11_residual_transfer_pls_preserves_zero_and_serializes():
    torch.manual_seed(1111)
    sample_count = 120
    phi_i = torch.randn(sample_count, 4)
    latent = torch.linspace(-1.0, 1.0, sample_count).unsqueeze(1)
    raw_delta = torch.zeros(sample_count, 4)
    raw_delta[:, 0] = 1.5 * phi_i[:, 0]
    raw_delta[:, 1] = latent[:, 0]
    raw_delta[:, 2] = 0.5 * latent[:, 0]
    phi_v = phi_i + raw_delta
    y_base = torch.zeros(sample_count, 2)
    y = y_base + torch.cat([latent, -0.5 * latent], dim=1)

    aligner = TrainSetCrossFittedResidualTransferPLSAlignerV11(
        max_rank=2,
        min_fit_correlation=0.0,
        min_calibration_correlation=0.0,
        min_calibration_r2=0.0,
        bootstrap_samples=0,
        min_bootstrap_sign_stability=0.0,
        max_leakage_rank=2,
        max_invariant_predictability_r2=1.0,
        min_retained_variance_fraction=0.0,
        min_teacher_residual_r2=-1.0,
        min_prefix_samples=8,
        direct_block_count=3,
        min_direct_calibration_correlation=0.0,
        min_block_transfer_r2=0.0,
    ).fit(
        phi_i_train=phi_i,
        phi_v_train=phi_v,
        y_base_train=y_base,
        y_train=y,
    )

    transformed_zero = aligner.transform_phi_pair(phi_i[:5], phi_i[:5])
    transformed_real = aligner.transform_phi_pair(phi_i[:5], phi_v[:5])
    restored = TrainSetCrossFittedResidualTransferPLSAlignerV11().load_state_dict(aligner.state_dict())

    assert aligner.diagnostic_summary()["alignment_method"] == "cross_fitted_residual_transfer_pls_v11"
    assert torch.allclose(transformed_zero, torch.zeros_like(transformed_zero), atol=1e-7)
    assert torch.linalg.vector_norm(transformed_real).item() > 0.0
    assert torch.allclose(restored.transform_phi_pair(phi_i[:5], phi_v[:5]), transformed_real)


def test_v11_generator_and_artifact_loader_round_trip(tmp_path: Path):
    model = nn.Linear(3, 2, bias=True)
    generator = ResidualBasisHyperGeneratorV10(
        named_param_shapes={"weight": tuple(model.weight.shape), "bias": tuple(model.bias.shape)},
        drift_dim=3,
        hidden_dim=8,
        rank=2,
        output_shape=(2,),
        max_scale=0.1,
    )
    generator_path = tmp_path / "generator_v10.pt"
    save_stage3_generator_checkpoint(path=generator_path, generator=generator, metadata={"run_id": "artifact_v11"})

    loaded_generator, metadata = load_stage3_generator_checkpoint(generator_path, model=model)
    assert isinstance(loaded_generator, ResidualBasisHyperGeneratorV10)
    assert metadata["run_id"] == "artifact_v11"

    drift, y_base, y = _toy_teacher_data()
    teacher = TrainOnlyRobustBlockResidualTeacherV11(
        drift_dim=3,
        output_shape=(2,),
        ridge=1e-5,
        reliability_gate_enabled=False,
    ).fit(
        drift_train=drift,
        y_base_train=y_base,
        y_train=y,
        split="train",
    )
    teacher_path = tmp_path / "teacher_v11.pt"
    save_stage3_v11_teacher_checkpoint(path=teacher_path, teacher=teacher, metadata={"run_id": "artifact_v11"})

    phi_i = torch.randn(96, 3)
    phi_v = phi_i + drift
    aligner = TrainSetCrossFittedResidualTransferPLSAlignerV11(
        max_rank=2,
        min_fit_correlation=0.0,
        min_calibration_correlation=0.0,
        min_calibration_r2=0.0,
        bootstrap_samples=0,
        min_bootstrap_sign_stability=0.0,
        max_leakage_rank=1,
        max_invariant_predictability_r2=1.0,
        min_retained_variance_fraction=0.0,
        min_prefix_samples=8,
        direct_block_count=2,
        min_direct_calibration_correlation=0.0,
        min_block_transfer_r2=0.0,
    ).fit(
        phi_i_train=phi_i,
        phi_v_train=phi_v,
        y_base_train=y_base,
        y_train=y,
    )
    aligner_path = tmp_path / "aligner_v11.pt"
    torch.save(aligner.state_dict(), aligner_path)

    bundle = load_stage3_v11_artifact_bundle(
        generator_checkpoint=generator_path,
        model=model,
        residual_canonicalizer_checkpoint=aligner_path,
        residual_teacher_checkpoint=teacher_path,
    )

    assert isinstance(bundle.generator, ResidualBasisHyperGeneratorV10)
    assert isinstance(bundle.residual_teacher, TrainOnlyRobustBlockResidualTeacherV11)
    assert isinstance(bundle.residual_canonicalizer, TrainSetCrossFittedResidualTransferPLSAlignerV11)
