from __future__ import annotations

import torch

from models.stage3_residual_teacher_v12b import TrainOnlyChronologicalOOFResidualTeacherV12b


def test_v12b_oof_teacher_does_not_use_future_calibration_targets():
    sample_count = 140
    torch.manual_seed(20260601)
    drift = torch.randn(sample_count, 4)
    y_base = torch.zeros(sample_count, 2)
    residual = torch.zeros(sample_count, 2)
    residual[70:] = torch.randn(sample_count - 70, 2)
    y = y_base + residual

    teacher = TrainOnlyChronologicalOOFResidualTeacherV12b(
        drift_dim=4,
        output_shape=(2,),
        ridge_grid=[1e-4],
        alpha_grid=[0.0, 0.5, 1.0],
        fold_count=4,
        min_prefix_samples=40,
        purge_gap=4,
        min_block_mse_gain=0.01,
        min_block_mae_gain=0.01,
        min_block_sign_agreement=0.6,
        min_block_pass_fraction=1.0,
        reliability_gate_enabled=False,
    ).fit(drift_train=drift, y_base_train=y_base, y_train=y, split="train")

    summary = teacher.diagnostic_summary()

    assert summary["oof_fold_count"] > 0
    assert summary["teacher_alpha_nonzero_fraction"] == 0.0
    assert torch.allclose(teacher.predict_correction(drift[70:80]), torch.zeros(10, 2), atol=1e-7)
