import torch

from models.stage3_residual_teacher_v12c import TrainOnlyChronologicalOOFResidualTeacherV12c
from models.stage3_residual_teacher_v12c import load_stage3_v12c_teacher_checkpoint
from models.stage3_residual_teacher_v12c import save_stage3_v12c_teacher_checkpoint
from train.stage3_v12c_signal_contract import Stage3TeacherSignalBuilderV12c


def _piecewise_residual_data(n: int = 72) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    trend = torch.linspace(-1.0, 1.0, n).unsqueeze(1)
    drift = torch.cat([trend, trend.square()], dim=1)
    y_base = torch.zeros(n, 1)
    residual = 0.4 * trend
    residual[-12:] = -0.4 * trend[-12:]
    y = y_base + residual
    return drift, y_base, y


def test_v12c_relaxed_policy_keeps_alpha_when_fraction_passes_but_strict_abstains() -> None:
    drift, y_base, y = _piecewise_residual_data()
    teacher = TrainOnlyChronologicalOOFResidualTeacherV12c(
        drift_dim=2,
        output_shape=(1,),
        ridge_grid=[1e-4],
        alpha_grid=[0.0, 0.25, 0.5, 1.0],
        fold_count=3,
        min_prefix_samples=18,
        purge_gap=0,
        min_block_pass_fraction=0.6,
        min_block_mse_gain=0.0,
        min_block_mae_gain=-1.0,
        min_block_sign_agreement=0.0,
        min_output_gain=-1.0,
        reliability_gate_enabled=False,
    ).fit(drift_train=drift, y_base_train=y_base, y_train=y, split="train")

    assert teacher.alpha_policy_names() == ("ungated", "relaxed", "strict")
    assert teacher.selected_alpha_source == "v12c_multi_policy_oof"
    assert teacher.alpha_by_policy["relaxed"].abs().max().item() > 0.0
    assert teacher.alpha_by_policy["strict"].abs().max().item() == 0.0

    relaxed = teacher.predict_correction(drift, policy="relaxed")
    strict = teacher.predict_correction(drift, policy="strict")
    assert relaxed.abs().max().item() > 0.0
    assert strict.abs().max().item() == 0.0


def test_v12c_diagnostics_report_all_policy_families() -> None:
    drift, y_base, y = _piecewise_residual_data()
    teacher = TrainOnlyChronologicalOOFResidualTeacherV12c(
        drift_dim=2,
        output_shape=(1,),
        ridge_grid=[1e-4],
        alpha_grid=[0.0, 0.5],
        fold_count=3,
        min_prefix_samples=18,
        min_block_pass_fraction=0.6,
        min_block_mse_gain=0.0,
        min_block_mae_gain=-1.0,
        min_block_sign_agreement=0.0,
        min_output_gain=-1.0,
        reliability_gate_enabled=False,
    ).fit(drift_train=drift, y_base_train=y_base, y_train=y, split="train")

    summary = teacher.diagnostic_summary()
    assert summary["teacher_policy_relaxed_alpha_nonzero_fraction"] == 1.0
    assert summary["teacher_policy_strict_alpha_nonzero_fraction"] == 0.0
    assert "teacher_policy_ungated_oof_mse_gain" in summary
    assert "teacher_policy_relaxed_oof_mse_gain" in summary
    assert "teacher_policy_strict_oof_mse_gain" in summary


def test_v12c_checkpoint_round_trip_preserves_policy_predictions(tmp_path) -> None:
    drift, y_base, y = _piecewise_residual_data()
    teacher = TrainOnlyChronologicalOOFResidualTeacherV12c(
        drift_dim=2,
        output_shape=(1,),
        ridge_grid=[1e-4],
        alpha_grid=[0.0, 0.25, 0.5, 1.0],
        fold_count=3,
        min_prefix_samples=18,
        min_block_pass_fraction=0.6,
        min_block_mse_gain=0.0,
        min_block_mae_gain=-1.0,
        min_block_sign_agreement=0.0,
        min_output_gain=-1.0,
        reliability_gate_enabled=False,
    ).fit(drift_train=drift, y_base_train=y_base, y_train=y, split="train")
    before_relaxed = teacher.predict_correction(drift[:5], policy="relaxed")
    before_strict = teacher.predict_correction(drift[:5], policy="strict")

    path = tmp_path / "teacher_v12c.pt"
    save_stage3_v12c_teacher_checkpoint(path=path, teacher=teacher)
    loaded, metadata = load_stage3_v12c_teacher_checkpoint(path)

    assert metadata["protocol_version"] == TrainOnlyChronologicalOOFResidualTeacherV12c.protocol_version
    assert loaded.selected_policy == teacher.selected_policy
    assert loaded.claim_policy == teacher.claim_policy
    assert torch.allclose(loaded.predict_correction(drift[:5], policy="relaxed"), before_relaxed)
    assert torch.allclose(loaded.predict_correction(drift[:5], policy="strict"), before_strict)


def test_v12c_checkpoint_metadata_preserves_signal_contract(tmp_path) -> None:
    drift, y_base, y = _piecewise_residual_data()
    signal_builder = Stage3TeacherSignalBuilderV12c(drift_dim=2, mode="delta_vector").fit(drift, split="train")
    signal = signal_builder.transform(drift).signal
    teacher = TrainOnlyChronologicalOOFResidualTeacherV12c(
        drift_dim=2,
        output_shape=(1,),
        ridge_grid=[1e-4],
        alpha_grid=[0.0, 0.5],
        fold_count=3,
        min_prefix_samples=18,
        min_block_pass_fraction=0.6,
        min_block_mse_gain=0.0,
        min_block_mae_gain=-1.0,
        min_block_sign_agreement=0.0,
        min_output_gain=-1.0,
        reliability_gate_enabled=False,
    ).fit(drift_train=signal, y_base_train=y_base, y_train=y, split="train")
    teacher.stage3_signal_builder_v12c = signal_builder

    path = tmp_path / "teacher_v12c_with_signal.pt"
    save_stage3_v12c_teacher_checkpoint(path=path, teacher=teacher)
    _, metadata = load_stage3_v12c_teacher_checkpoint(path)

    state = metadata["stage3_signal_builder_v12c_state"]
    restored = Stage3TeacherSignalBuilderV12c(drift_dim=2).load_state_dict(state)
    assert torch.allclose(restored.transform(drift).signal, signal)
