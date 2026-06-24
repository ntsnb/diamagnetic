from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models.stage3_residual_teacher_v13 import TrainOnlyFluctuationContextResidualTeacherV13
from models.stage3_residual_teacher_v14 import (
    TrainOnlyStabilityFluctuationResidualTeacherV14,
    build_stage3_v14_context_features,
    load_stage3_v14_teacher_checkpoint,
    save_stage3_v14_teacher_checkpoint,
)
from train.run_stage3_v14_stability_teacher_probe import _validate_v14_gate_modes


def _fit_v14_teacher(
    teacher: TrainOnlyStabilityFluctuationResidualTeacherV14,
) -> TrainOnlyStabilityFluctuationResidualTeacherV14:
    features = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.1, 0.0],
            [0.4, 0.1, 0.1],
            [0.6, 0.2, 0.1],
            [0.8, 0.2, 0.2],
            [1.0, 0.3, 0.2],
            [1.2, 0.3, 0.3],
            [1.4, 0.4, 0.3],
            [1.6, 0.4, 0.4],
            [1.8, 0.5, 0.4],
        ],
        dtype=torch.float32,
    )
    y_base = torch.zeros(features.shape[0], 2)
    y = torch.stack([0.15 * features[:, 0], -0.25 * features[:, 1]], dim=1)
    return teacher.fit(drift_train=features, y_base_train=y_base, y_train=y, split="train")


def test_v14_context_features_keep_v13_signal_but_version_metadata_is_v14() -> None:
    fluctuation = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    history = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    level = 10.0 + fluctuation

    context = build_stage3_v14_context_features(
        fluctuation_signal=fluctuation,
        history_features=history,
        level_offset=level,
        mode="fluctuation_plus_history",
    )

    assert context.protocol_version == "v14_stability_fluctuation_context_features"
    assert context.mode == "fluctuation_plus_history"
    assert context.feature_dim == 5
    assert torch.allclose(context.features[:, :3], fluctuation)
    assert torch.allclose(context.features[:, 3:], history)


def test_v14_default_readout_is_linear_and_does_not_expand_like_v12b() -> None:
    teacher = TrainOnlyStabilityFluctuationResidualTeacherV14(
        drift_dim=3,
        output_shape=(2,),
        context_feature_mode="fluctuation_only",
        fluctuation_signal_dim=3,
        history_feature_dim=0,
        level_offset_dim=3,
        ridge_grid=[1e-3],
        alpha_grid=[0.0, 0.1],
        fold_count=2,
        min_prefix_samples=3,
        purge_gap=0,
    )

    assert teacher.readout_mode == "linear"
    assert teacher.feature_dim == 3
    mapped = teacher.fit_readout_scale(torch.tensor([[1.0, -2.0, 3.0], [2.0, 0.0, -4.0]], dtype=torch.float32))._feature_map(
        torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.float32)
    )

    assert mapped.shape == (1, 3)
    assert "abs" not in ",".join(teacher.readout_feature_names())
    assert "square" not in ",".join(teacher.readout_feature_names())
    assert "norm" not in ",".join(teacher.readout_feature_names())


def test_v14_train_only_scale_preserves_zero_context() -> None:
    teacher = TrainOnlyStabilityFluctuationResidualTeacherV14(
        drift_dim=3,
        output_shape=(2,),
        context_feature_mode="fluctuation_only",
        fluctuation_signal_dim=3,
        history_feature_dim=0,
        level_offset_dim=3,
        ridge_grid=[1e-3],
        alpha_grid=[0.0],
        fold_count=2,
        min_prefix_samples=3,
        purge_gap=0,
    )

    with pytest.raises(ValueError, match="train split"):
        teacher.fit(drift_train=torch.ones(8, 3), y_base_train=torch.zeros(8, 2), y_train=torch.zeros(8, 2), split="validation")

    _fit_v14_teacher(teacher)
    zero_features = teacher._feature_map(torch.zeros(2, 3))
    correction = teacher.predict_correction(torch.zeros(2, 3))

    assert torch.allclose(zero_features, torch.zeros_like(zero_features), atol=1e-8)
    assert torch.allclose(correction, torch.zeros_like(correction), atol=1e-8)
    assert teacher.readout_scale_fit_split == "train"


def test_v14_defaults_to_strict_policy_and_records_protocol_metadata() -> None:
    teacher = TrainOnlyStabilityFluctuationResidualTeacherV14(
        drift_dim=3,
        output_shape=(2,),
        context_feature_mode="fluctuation_only",
        fluctuation_signal_dim=3,
        history_feature_dim=0,
        level_offset_dim=3,
        ridge_grid=[1e-3],
        alpha_grid=[0.0, 0.1],
        fold_count=2,
        min_prefix_samples=3,
        purge_gap=0,
    )
    _fit_v14_teacher(teacher)

    summary = teacher.diagnostic_summary()

    assert teacher.selected_policy == "strict"
    assert teacher.claim_policy == "strict"
    assert summary["teacher_method"] == teacher.protocol_version
    assert summary["teacher_readout_mode"] == "linear"
    assert summary["teacher_selected_policy"] == "strict"
    assert summary["teacher_context_feature_mode"] == "fluctuation_only"


def test_v14_selected_policy_controls_default_prediction_path() -> None:
    teacher = TrainOnlyStabilityFluctuationResidualTeacherV14(
        drift_dim=2,
        output_shape=(1,),
        context_feature_mode="fluctuation_only",
        fluctuation_signal_dim=2,
        history_feature_dim=0,
        level_offset_dim=2,
        ridge_grid=[1e-3],
        alpha_grid=[0.0, 1.0],
        fold_count=2,
        min_prefix_samples=3,
        purge_gap=0,
    )
    teacher.readout_train_scale = torch.ones(2)
    teacher.readout_scale_fit_split = "train"
    teacher.correction_weight = torch.ones(2, 1)
    teacher.alpha_by_policy["strict"] = torch.zeros(1)
    teacher.alpha_by_policy["relaxed"] = torch.ones(1)
    teacher.selected_policy = "strict"

    default_correction = teacher.predict_correction(torch.ones(1, 2))
    relaxed_correction = teacher.predict_correction(torch.ones(1, 2), policy="relaxed")

    assert torch.allclose(default_correction, torch.zeros_like(default_correction))
    assert torch.allclose(relaxed_correction, torch.tensor([[2.0]]))


def test_v14_checkpoint_refuses_overwrite_and_rejects_v13_checkpoint(tmp_path: Path) -> None:
    teacher = TrainOnlyStabilityFluctuationResidualTeacherV14(
        drift_dim=3,
        output_shape=(2,),
        context_feature_mode="fluctuation_only",
        fluctuation_signal_dim=3,
        history_feature_dim=0,
        level_offset_dim=3,
        ridge_grid=[1e-3],
        alpha_grid=[0.0],
        fold_count=2,
        min_prefix_samples=3,
        purge_gap=0,
    )
    _fit_v14_teacher(teacher)
    path = tmp_path / "teacher_v14.pt"

    save_stage3_v14_teacher_checkpoint(path=path, teacher=teacher, metadata={"run_id": "unit"})

    with pytest.raises(FileExistsError, match="Refuse to overwrite"):
        save_stage3_v14_teacher_checkpoint(path=path, teacher=teacher, metadata={"run_id": "unit"})

    loaded, metadata = load_stage3_v14_teacher_checkpoint(path)
    assert loaded.protocol_version == teacher.protocol_version
    assert loaded.readout_mode == "linear"
    assert metadata["protocol_version"] == teacher.protocol_version

    v13_teacher = TrainOnlyFluctuationContextResidualTeacherV13(
        drift_dim=3,
        output_shape=(2,),
        context_feature_mode="fluctuation_only",
        fluctuation_signal_dim=3,
        history_feature_dim=0,
        level_offset_dim=3,
        ridge_grid=[1e-3],
        alpha_grid=[0.0],
        fold_count=2,
        min_prefix_samples=3,
        purge_gap=0,
    )
    v13_teacher.fit(drift_train=torch.ones(8, 3), y_base_train=torch.zeros(8, 2), y_train=torch.zeros(8, 2), split="train")
    v13_path = tmp_path / "teacher_v13.pt"
    torch.save({"protocol_version": v13_teacher.protocol_version, "state": v13_teacher.state_dict()}, v13_path)

    with pytest.raises(ValueError, match="not a V14"):
        load_stage3_v14_teacher_checkpoint(v13_path)


def test_v14_config_locks_linear_strict_v12e_and_full_counterfactual_gate() -> None:
    config_text = Path("configs/stage3_v14_stability_teacher_probe.yaml").read_text(encoding="utf-8")

    assert "protocol_version: v14_stability_teacher_probe" in config_text
    assert "version: v14_stability_teacher_probe" in config_text
    assert "v14_feature_modes:" in config_text
    assert "readout_mode: linear" in config_text
    assert "selected_policy: strict" in config_text
    assert "claim_policy: strict" in config_text
    assert "protocol: v12e_per_horizon_label_availability" in config_text
    assert "conditional_shuffled" in config_text
    assert "conditional_random_matched_norm" in config_text
    assert "v13_feature_modes:" not in config_text
    assert "version: v13" not in config_text


def test_v14_runner_requires_full_counterfactual_gate_modes() -> None:
    with pytest.raises(ValueError, match="requires full counterfactual mode set"):
        _validate_v14_gate_modes(["real", "zero", "sign_flipped"])
