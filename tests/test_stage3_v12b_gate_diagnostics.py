from __future__ import annotations

import torch

from models.stage3_residual_teacher_v12b import TrainOnlyChronologicalOOFResidualTeacherV12b


def test_v12b_teacher_records_alpha_block_and_ridge_diagnostics():
    n = 128
    t = torch.linspace(-1.0, 1.0, n).unsqueeze(1)
    drift = torch.cat([t, torch.sin(2.0 * t)], dim=1)
    y_base = torch.zeros(n, 2)
    y = y_base + torch.cat([0.4 * drift[:, :1], -0.2 * drift[:, 1:2]], dim=1)

    teacher = TrainOnlyChronologicalOOFResidualTeacherV12b(
        drift_dim=2,
        output_shape=(2,),
        ridge_grid=[1e-6, 1e-4],
        alpha_grid=[0.0, 0.5, 1.0],
        fold_count=3,
        min_prefix_samples=32,
        min_block_mse_gain=-0.05,
        min_block_mae_gain=-0.05,
        min_block_sign_agreement=0.4,
        min_block_pass_fraction=0.67,
        reliability_gate_enabled=False,
    ).fit(drift_train=drift, y_base_train=y_base, y_train=y, split="train")

    alpha_rows = teacher.alpha_block_diagnostic_rows()
    ridge_rows = teacher.ridge_diagnostic_rows()

    assert alpha_rows
    assert ridge_rows
    assert {"ridge", "output_idx", "alpha", "fold_idx", "mse_gain", "mae_gain", "sign_agreement"} <= set(
        alpha_rows[0]
    )
    assert {"ridge", "oof_mse_gain", "output_pass_count", "selected_alpha_nonzero_fraction"} <= set(ridge_rows[0])
