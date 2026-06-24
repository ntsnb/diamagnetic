from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models.stage3_residual_teacher_v12 import (
    TrainOnlyOutcomeAlignedBlockResidualTeacherV12,
    load_stage3_v12_teacher_checkpoint,
    save_stage3_v12_teacher_checkpoint,
)


def _toy_teacher_data(sample_count: int = 96) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    drift = torch.linspace(-1.0, 1.0, sample_count).unsqueeze(1)
    drift = torch.cat([drift, drift.square(), torch.sin(drift * 2.0)], dim=1)
    y_base = torch.zeros(sample_count, 2)
    correction = torch.stack([0.4 * drift[:, 0] - 0.2 * drift[:, 2], -0.3 * drift[:, 0]], dim=1)
    return drift, y_base, y_base + correction


def test_v12_teacher_uses_v12_protocol_and_preserves_v11_gate_behavior(tmp_path: Path):
    drift, y_base, y = _toy_teacher_data()
    teacher = TrainOnlyOutcomeAlignedBlockResidualTeacherV12(
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

    path = tmp_path / "residual_teacher_v12.pt"
    save_stage3_v12_teacher_checkpoint(path=path, teacher=teacher, metadata={"run_id": "v12_teacher"})
    loaded, metadata = load_stage3_v12_teacher_checkpoint(path, expected_drift_dim=3, expected_output_shape=(2,))

    assert metadata["protocol_version"] == TrainOnlyOutcomeAlignedBlockResidualTeacherV12.protocol_version
    assert loaded.diagnostic_summary()["teacher_method"] == TrainOnlyOutcomeAlignedBlockResidualTeacherV12.protocol_version
    assert loaded.shrinkage_alpha == pytest.approx(teacher.shrinkage_alpha)
    assert torch.allclose(loaded.predict_correction(torch.zeros(4, 3)), torch.zeros(4, 2), atol=1e-7)
    assert torch.allclose(teacher.predict_correction(drift), loaded.predict_correction(drift))
