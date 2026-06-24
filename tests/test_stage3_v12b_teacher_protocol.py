from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models.stage3_residual_teacher_v12b import (
    TrainOnlyChronologicalOOFResidualTeacherV12b,
    load_stage3_v12b_teacher_checkpoint,
    save_stage3_v12b_teacher_checkpoint,
)


def _stable_teacher_data(sample_count: int = 160) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = torch.linspace(-1.0, 1.0, sample_count).unsqueeze(1)
    drift = torch.cat([t, t.square(), torch.sin(3.0 * t)], dim=1)
    y_base = torch.zeros(sample_count, 3)
    residual = torch.cat(
        [
            0.55 * drift[:, :1] - 0.15 * drift[:, 2:3],
            -0.35 * drift[:, :1],
            0.2 * drift[:, 2:3],
        ],
        dim=1,
    )
    return drift, y_base, y_base + residual


def test_v12b_teacher_train_only_zero_identity_and_checkpoint_round_trip(tmp_path: Path):
    drift, y_base, y = _stable_teacher_data()
    teacher = TrainOnlyChronologicalOOFResidualTeacherV12b(
        drift_dim=3,
        output_shape=(3,),
        ridge_grid=[1e-6, 1e-4],
        alpha_grid=[0.0, 0.5, 1.0],
        fold_count=4,
        min_prefix_samples=32,
        min_block_mse_gain=-0.05,
        min_block_mae_gain=-0.05,
        min_block_sign_agreement=0.45,
        min_block_pass_fraction=0.75,
        reliability_gate_enabled=False,
    )

    with pytest.raises(ValueError, match="train split"):
        teacher.fit(drift_train=drift, y_base_train=y_base, y_train=y, split="validation")

    teacher.fit(drift_train=drift, y_base_train=y_base, y_train=y, split="train")
    summary = teacher.diagnostic_summary()

    assert summary["teacher_method"] == teacher.protocol_version
    assert summary["teacher_alpha_nonzero_fraction"] > 0.0
    assert summary["teacher_oof_mse_gain"] > 0.0
    assert torch.allclose(teacher.predict_correction(torch.zeros(5, 3)), torch.zeros(5, 3), atol=1e-7)

    path = tmp_path / "residual_teacher_v12b.pt"
    save_stage3_v12b_teacher_checkpoint(path=path, teacher=teacher, metadata={"run_id": "toy"})
    loaded, metadata = load_stage3_v12b_teacher_checkpoint(path, expected_drift_dim=3, expected_output_shape=(3,))

    assert metadata["protocol_version"] == teacher.protocol_version
    assert torch.allclose(teacher.predict_correction(drift), loaded.predict_correction(drift))


def test_v12b_teacher_fails_closed_when_oof_residual_is_unstable():
    drift, y_base, y = _stable_teacher_data()
    shuffled_y = y[torch.randperm(y.shape[0])]
    teacher = TrainOnlyChronologicalOOFResidualTeacherV12b(
        drift_dim=3,
        output_shape=(3,),
        ridge_grid=[1e-4],
        alpha_grid=[0.0, 0.5, 1.0],
        fold_count=4,
        min_prefix_samples=32,
        min_block_mse_gain=0.05,
        min_block_mae_gain=0.02,
        min_block_sign_agreement=0.65,
        min_block_pass_fraction=1.0,
        reliability_gate_enabled=False,
    ).fit(drift_train=drift, y_base_train=y_base, y_train=shuffled_y, split="train")

    summary = teacher.diagnostic_summary()

    assert summary["teacher_alpha_nonzero_fraction"] == 0.0
    assert summary["teacher_alpha_source"].startswith("fallback")
    assert torch.allclose(teacher.predict_correction(drift[:8]), torch.zeros(8, 3), atol=1e-7)
