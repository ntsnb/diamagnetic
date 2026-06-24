import torch

from models.stage3_residual_teacher_v12d import TrainOnlyContextOOFResidualTeacherV12d
from train.stage3_v12d_context_features import CausalResidualHistoryFeatureBuilderV12d


def test_v12d_history_teacher_beats_drift_only_when_residual_is_autoregressive() -> None:
    n = 96
    residual = torch.zeros(n, 1)
    residual[0, 0] = 0.25
    for idx in range(1, n):
        residual[idx, 0] = 0.82 * residual[idx - 1, 0] + 0.04 * torch.sin(torch.tensor(float(idx) / 5.0))
    y_base = torch.zeros_like(residual)
    y_true = y_base + residual
    drift = torch.zeros(n, 2)

    history_builder = CausalResidualHistoryFeatureBuilderV12d(
        output_shape=(1,),
        lags=[1, 2, 3],
        include_availability_mask=True,
        standardize=True,
    ).fit(y_base_train=y_base, y_train=y_true, split="train")
    history = history_builder.transform(y_base=y_base, y_true=y_true).features

    history_teacher = TrainOnlyContextOOFResidualTeacherV12d(
        drift_dim=history.shape[1],
        output_shape=(1,),
        context_feature_mode="history_only",
        history_lags=[1, 2, 3],
        ridge_grid=[1e-4, 1e-3],
        alpha_grid=[0.0, 0.25, 0.5, 0.75, 1.0],
        fold_count=4,
        min_prefix_samples=24,
        purge_gap=0,
        min_block_pass_fraction=0.0,
        min_block_mse_gain=-1.0,
        min_block_mae_gain=-1.0,
        min_block_sign_agreement=0.0,
        min_output_gain=-1.0,
        reliability_gate_enabled=False,
    ).fit(drift_train=history, y_base_train=y_base, y_train=y_true, split="train")

    drift_teacher = TrainOnlyContextOOFResidualTeacherV12d(
        drift_dim=drift.shape[1],
        output_shape=(1,),
        context_feature_mode="drift_only",
        ridge_grid=[1e-4, 1e-3],
        alpha_grid=[0.0, 0.25, 0.5, 0.75, 1.0],
        fold_count=4,
        min_prefix_samples=24,
        purge_gap=0,
        min_block_pass_fraction=0.0,
        min_block_mse_gain=-1.0,
        min_block_mae_gain=-1.0,
        min_block_sign_agreement=0.0,
        min_output_gain=-1.0,
        reliability_gate_enabled=False,
    ).fit(drift_train=drift, y_base_train=y_base, y_train=y_true, split="train")

    assert history_teacher.oof_mse_gain > drift_teacher.oof_mse_gain + 0.10
    assert history_teacher.diagnostic_summary()["teacher_context_feature_mode"] == "history_only"
