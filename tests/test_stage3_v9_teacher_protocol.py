from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from models.adaptation_v9 import ResidualBasisHyperGeneratorV9
from models.stage3_alignment_v9 import TrainSetTeacherResidualizedReducedRankCCAAligner
from models.stage3_residual_teacher_v9 import (
    TrainOnlyResidualTeacherV9,
    load_stage3_v9_teacher_checkpoint,
    save_stage3_v9_teacher_checkpoint,
)
from train.online_loop_stage3_v2 import run_online_adaptation_v2
from train.stage3_v2_artifacts import load_stage3_v2_artifact_bundle
from train.stage3_generator_training import save_stage3_generator_checkpoint
from train.stage3_v9_teacher_diagnostics import (
    compute_teacher_correction_metrics,
    summarize_teacher_mode_rows,
)


def _toy_teacher_data(sample_count: int = 32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    drift = torch.linspace(-1.0, 1.0, sample_count).unsqueeze(1)
    drift = torch.cat([drift, drift.square(), torch.sin(drift * 2.0)], dim=1)
    y_base = torch.zeros(sample_count, 2)
    correction = torch.stack([0.4 * drift[:, 0] - 0.2 * drift[:, 2], -0.3 * drift[:, 0]], dim=1)
    y = y_base + correction
    return drift, y_base, y


def test_v9_residual_teacher_updates_only_on_train_split():
    drift, y_base, y = _toy_teacher_data()
    sample_ids = torch.arange(1000, 1000 + drift.shape[0])

    teacher = TrainOnlyResidualTeacherV9(drift_dim=3, output_shape=(2,), ridge=1e-5)
    teacher.fit(
        drift_train=drift,
        y_base_train=y_base,
        y_train=y,
        split="train",
        sample_ids=sample_ids,
    )

    assert teacher.fit_sample_count == drift.shape[0]
    assert teacher.fit_split == "train"
    assert teacher.train_sample_ids.tolist() == sample_ids.tolist()
    assert not any(item in teacher.train_sample_ids.tolist() for item in [7777, 8888])


def test_v9_residual_teacher_rejects_validation_or_test_update():
    drift, y_base, y = _toy_teacher_data()
    teacher = TrainOnlyResidualTeacherV9(drift_dim=3, output_shape=(2,), ridge=1e-5)
    teacher.fit(drift_train=drift, y_base_train=y_base, y_train=y, split="train")
    before = teacher.state_hash()

    with pytest.raises(ValueError, match="train split"):
        teacher.fit(drift_train=drift, y_base_train=y_base, y_train=y, split="validation")
    with pytest.raises(ValueError, match="train split"):
        teacher.update(drift=drift, y_base=y_base, y=y, split="test")

    assert teacher.state_hash() == before


def test_v9_teacher_update_reduces_train_residual_on_linear_toy():
    drift, y_base, y = _toy_teacher_data(sample_count=64)
    teacher = TrainOnlyResidualTeacherV9(drift_dim=3, output_shape=(2,), ridge=1e-6)
    before_mse = torch.mean((y - y_base).pow(2)).item()

    teacher.fit(drift_train=drift, y_base_train=y_base, y_train=y, split="train")
    corrected = y_base + teacher.predict_correction(drift)
    after_mse = torch.mean((y - corrected).pow(2)).item()

    assert after_mse < before_mse * 0.05
    assert teacher.diagnostic_summary()["teacher_residual_mse_after"] < before_mse * 0.05


def test_v9_teacher_only_diagnostics_rank_real_above_zero_and_sign_flipped():
    drift, y_base, y = _toy_teacher_data(sample_count=64)
    teacher = TrainOnlyResidualTeacherV9(drift_dim=3, output_shape=(2,), ridge=1e-6)
    teacher.fit(drift_train=drift, y_base_train=y_base, y_train=y, split="train")

    real_metrics = compute_teacher_correction_metrics(
        y_base=y_base,
        y_true=y,
        correction=teacher.predict_correction(drift),
    )
    zero_metrics = compute_teacher_correction_metrics(
        y_base=y_base,
        y_true=y,
        correction=teacher.predict_correction(torch.zeros_like(drift)),
    )
    sign_flipped_metrics = compute_teacher_correction_metrics(
        y_base=y_base,
        y_true=y,
        correction=teacher.predict_correction(-drift),
    )
    summary = summarize_teacher_mode_rows(
        [
            {"mode": "real", **real_metrics},
            {"mode": "zero", **zero_metrics},
            {"mode": "sign_flipped", **sign_flipped_metrics},
        ]
    )

    assert real_metrics["teacher_mse_gain"] > 0.95
    assert zero_metrics["teacher_mse_gain"] == pytest.approx(0.0)
    assert sign_flipped_metrics["teacher_mse_gain"] < 0.0
    assert real_metrics["teacher_sign_agreement_rate"] > 0.95
    assert summary["teacher_real_is_best_mode"] is True
    assert summary["teacher_best_counterfactual_mode"] == "zero"
    assert summary["teacher_real_vs_best_counterfactual_mse_gain_margin"] > 0.95


def test_v9_teacher_checkpoint_round_trip_preserves_correction_and_metadata(tmp_path: Path):
    drift, y_base, y = _toy_teacher_data()
    teacher = TrainOnlyResidualTeacherV9(drift_dim=3, output_shape=(2,), ridge=1e-5)
    teacher.fit(drift_train=drift, y_base_train=y_base, y_train=y, split="train")
    path = tmp_path / "residual_teacher_v9.pt"

    save_stage3_v9_teacher_checkpoint(
        path=path,
        teacher=teacher,
        metadata={"run_id": "v9_teacher_roundtrip"},
    )
    loaded, metadata = load_stage3_v9_teacher_checkpoint(path, expected_drift_dim=3, expected_output_shape=(2,))

    assert metadata["run_id"] == "v9_teacher_roundtrip"
    assert metadata["protocol_version"] == "v9_residual_teacher"
    assert torch.allclose(teacher.predict_correction(drift), loaded.predict_correction(drift))
    assert loaded.fit_sample_count == teacher.fit_sample_count

    with pytest.raises(ValueError, match="drift_dim"):
        load_stage3_v9_teacher_checkpoint(path, expected_drift_dim=4, expected_output_shape=(2,))


def test_v9_teacher_aware_aligner_uses_train_data_and_preserves_physical_zero_drift():
    torch.manual_seed(909)
    sample_count = 36
    phi_i = torch.randn(sample_count, 4)
    latent = torch.cumsum(torch.randn(sample_count, 2), dim=0)
    raw_signal = torch.zeros(sample_count, 4)
    raw_signal[:, 0] = latent[:, 0]
    raw_signal[:, 2] = -0.5 * latent[:, 1]
    phi_v = phi_i + raw_signal
    y_base = torch.zeros(sample_count, 2)
    y = y_base + torch.stack([latent[:, 0], latent[:, 1]], dim=1)

    aligner = TrainSetTeacherResidualizedReducedRankCCAAligner(
        max_rank=2,
        min_fit_correlation=0.01,
        min_calibration_correlation=0.01,
        min_calibration_r2=0.0001,
        bootstrap_samples=8,
        min_bootstrap_sign_stability=0.5,
    ).fit(
        phi_i_train=phi_i,
        phi_v_train=phi_v,
        y_base_train=y_base,
        y_train=y,
    )

    transformed_zero = aligner.transform_phi_pair(phi_i[:3], phi_i[:3])
    assert torch.allclose(transformed_zero, torch.zeros_like(transformed_zero), atol=1e-7)
    summary = aligner.diagnostic_summary()
    assert summary["alignment_method"] == "teacher_residual_rrcca_v9"
    assert summary["fit_sample_count"] == sample_count
    assert summary["teacher_residual_r2"] >= 0.0


def test_v9_zero_drift_identity_with_teacher_and_generator_enabled():
    model = nn.Linear(2, 1, bias=True)
    generator = ResidualBasisHyperGeneratorV9(
        named_param_shapes={"weight": tuple(model.weight.shape), "bias": tuple(model.bias.shape)},
        drift_dim=3,
        hidden_dim=8,
        rank=2,
        output_shape=(1,),
        max_scale=0.2,
    )
    x = torch.randn(5, 2)
    y = model(x).detach()
    drift = torch.zeros(3)

    result = run_online_adaptation_v2(
        stream=[(x[:1], y[:1], torch.zeros(1, 3), torch.zeros(1, 3))],
        model=model,
        generator=generator,
        cfg={
            "online": {
                "stage3": {
                    "enable": True,
                    "apply_transient": True,
                    "allow_untrained_generator": False,
                    "require_aligned_embeddings": True,
                    "causal_online": True,
                    "signal_mode": "level_vector",
                    "normalize_vector": False,
                    "target_names": ["weight", "bias"],
                    "max_delta": 0.05,
                    "max_pred_shift": 0.25,
                    "blend_alpha": 1.0,
                    "risk_gate_enable": False,
                }
            }
        },
        device=torch.device("cpu"),
        return_step_rows=True,
    )

    assert result["online_stage3_use_rate"] == 1.0
    assert result["online_stage3_delta_norm_pre_clamp"] == 0.0
    assert result["online_stage3_pred_shift"] == 0.0
    assert result["online_stage3_mse"] == pytest.approx(result["online_base_mse"])
    assert generator.residual_correction(drift, torch.zeros(1, 1)).abs().max().item() == pytest.approx(0.0)


def test_v9_composite_artifact_round_trip_loads_generator_teacher_canonicalizer(tmp_path: Path):
    model = nn.Linear(3, 2, bias=True)
    generator = ResidualBasisHyperGeneratorV9(
        named_param_shapes={"weight": tuple(model.weight.shape), "bias": tuple(model.bias.shape)},
        drift_dim=3,
        hidden_dim=8,
        rank=2,
        output_shape=(2,),
        max_scale=0.1,
    )
    generator_path = tmp_path / "generator_v9.pt"
    save_stage3_generator_checkpoint(
        path=generator_path,
        generator=generator,
        metadata={"run_id": "artifact_v9"},
    )

    drift, y_base, y = _toy_teacher_data()
    teacher = TrainOnlyResidualTeacherV9(drift_dim=3, output_shape=(2,), ridge=1e-5)
    teacher.fit(drift_train=drift, y_base_train=y_base, y_train=y, split="train")
    teacher_path = tmp_path / "teacher_v9.pt"
    save_stage3_v9_teacher_checkpoint(path=teacher_path, teacher=teacher, metadata={"run_id": "artifact_v9"})

    phi_i = torch.randn(36, 3)
    phi_v = phi_i + drift.repeat(2, 1)[:36]
    y_base_align = torch.zeros(36, 2)
    y_align = y_base_align + torch.randn(36, 2) * 0.1
    aligner = TrainSetTeacherResidualizedReducedRankCCAAligner(
        max_rank=2,
        min_fit_correlation=0.0,
        min_calibration_correlation=0.0,
        min_calibration_r2=0.0,
        bootstrap_samples=0,
        min_bootstrap_sign_stability=0.0,
    ).fit(
        phi_i_train=phi_i,
        phi_v_train=phi_v,
        y_base_train=y_base_align,
        y_train=y_align,
    )
    aligner_path = tmp_path / "aligner_v9.pt"
    torch.save(aligner.state_dict(), aligner_path)

    bundle = load_stage3_v2_artifact_bundle(
        generator_checkpoint=generator_path,
        model=model,
        residual_canonicalizer_checkpoint=aligner_path,
        residual_teacher_checkpoint=teacher_path,
    )

    assert isinstance(bundle.generator, ResidualBasisHyperGeneratorV9)
    assert isinstance(bundle.residual_teacher, TrainOnlyResidualTeacherV9)
    assert isinstance(bundle.residual_canonicalizer, TrainSetTeacherResidualizedReducedRankCCAAligner)
    assert torch.allclose(bundle.residual_teacher.predict_correction(drift), teacher.predict_correction(drift))
