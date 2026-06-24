from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models.stage3_residual_teacher_v13 import (
    V13_CONTEXT_FEATURE_MODES,
    TrainOnlyFluctuationContextResidualTeacherV13,
    build_stage3_v13_context_features,
    load_stage3_v13_teacher_checkpoint,
    save_stage3_v13_teacher_checkpoint,
)
from train.run_stage3_v13_fluctuation_teacher_probe import _collect_split_tensors, _validate_v13_gate_modes


def _fit_teacher_for_unit_test(teacher: TrainOnlyFluctuationContextResidualTeacherV13) -> None:
    features = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.1, 0.2],
            [1.0, 0.2, 0.4],
            [1.5, 0.3, 0.6],
            [2.0, 0.4, 0.8],
            [2.5, 0.5, 1.0],
            [3.0, 0.6, 1.2],
            [3.5, 0.7, 1.4],
        ],
        dtype=torch.float32,
    )
    y_base = torch.zeros(features.shape[0], 2)
    y = torch.stack([0.1 * features[:, 0], -0.2 * features[:, 1]], dim=1)
    teacher.fit(drift_train=features, y_base_train=y_base, y_train=y, split="train")


def test_v13_context_feature_modes_are_explicit() -> None:
    fluctuation = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    level = 10.0 + fluctuation
    history = torch.arange(8, dtype=torch.float32).reshape(4, 2)

    fluct_only = build_stage3_v13_context_features(
        fluctuation_signal=fluctuation,
        history_features=history,
        level_offset=level,
        mode="fluctuation_only",
    )
    history_only = build_stage3_v13_context_features(
        fluctuation_signal=fluctuation,
        history_features=history,
        level_offset=level,
        mode="history_only",
    )
    both = build_stage3_v13_context_features(
        fluctuation_signal=fluctuation,
        history_features=history,
        level_offset=level,
        mode="fluctuation_plus_history",
    )
    level_diag = build_stage3_v13_context_features(
        fluctuation_signal=fluctuation,
        history_features=history,
        level_offset=level,
        mode="level_offset_only",
    )

    assert "fluctuation_plus_history" in V13_CONTEXT_FEATURE_MODES
    assert torch.allclose(fluct_only.features, fluctuation)
    assert torch.allclose(history_only.features, history)
    assert both.features.shape == (4, 5)
    assert torch.allclose(level_diag.features, level)
    assert both.fluctuation_dim == 3
    assert both.history_dim == 2
    assert both.level_offset_dim == 3


def test_v13_context_features_are_shape_safe() -> None:
    fluctuation = torch.zeros(4, 3)
    history = torch.zeros(5, 2)
    level = torch.zeros(4, 3)

    with pytest.raises(ValueError, match="same sample count"):
        build_stage3_v13_context_features(
            fluctuation_signal=fluctuation,
            history_features=history,
            level_offset=level,
            mode="fluctuation_plus_history",
        )


def test_v13_teacher_diagnostic_summary_has_v13_metadata() -> None:
    teacher = TrainOnlyFluctuationContextResidualTeacherV13(
        drift_dim=3,
        output_shape=(2,),
        context_feature_mode="fluctuation_only",
        fluctuation_signal_dim=3,
        history_feature_dim=0,
        level_offset_dim=3,
        ridge_grid=[1e-3],
        alpha_grid=[0.0],
        fold_count=2,
        min_prefix_samples=2,
        purge_gap=0,
    )
    _fit_teacher_for_unit_test(teacher)

    summary = teacher.diagnostic_summary()

    assert summary["teacher_method"] == teacher.protocol_version
    assert summary["teacher_context_feature_mode"] == "fluctuation_only"
    assert summary["teacher_fluctuation_signal_dim"] == 3


def test_v13_teacher_checkpoint_refuses_overwrite(tmp_path: Path) -> None:
    teacher = TrainOnlyFluctuationContextResidualTeacherV13(
        drift_dim=3,
        output_shape=(2,),
        context_feature_mode="fluctuation_only",
        fluctuation_signal_dim=3,
        history_feature_dim=0,
        level_offset_dim=3,
        ridge_grid=[1e-3],
        alpha_grid=[0.0],
        fold_count=2,
        min_prefix_samples=2,
        purge_gap=0,
    )
    _fit_teacher_for_unit_test(teacher)
    path = tmp_path / "teacher_v13.pt"

    save_stage3_v13_teacher_checkpoint(path=path, teacher=teacher, metadata={"run_id": "unit"})

    with pytest.raises(FileExistsError, match="Refuse to overwrite"):
        save_stage3_v13_teacher_checkpoint(path=path, teacher=teacher, metadata={"run_id": "unit"})

    loaded, metadata = load_stage3_v13_teacher_checkpoint(path)
    assert loaded.protocol_version == teacher.protocol_version
    assert loaded.context_feature_mode == "fluctuation_only"
    assert metadata["protocol_version"] == teacher.protocol_version


def test_v13_runner_rejects_transformed_main_signal_path() -> None:
    cfg = {"online": {"stage3": {"drift_source": "delta_phi_v"}}}

    with pytest.raises(ValueError, match="raw delta_phi_v"):
        _collect_split_tensors(
            cfg=cfg,
            dataset=[],
            batch_size=1,
            device=torch.device("cpu"),
            aligned_model=None,
            model=None,
            max_batches=None,
            delta_standardizer=object(),
            residual_canonicalizer=None,
        )


def test_v13_config_locks_raw_signal_and_full_counterfactual_gate() -> None:
    config_text = Path("configs/stage3_v13_fluctuation_teacher_probe.yaml").read_text(encoding="utf-8")

    assert "protocol_version: v13_fluctuation_teacher_probe" in config_text
    assert "signal_mode: fluctuation_delta" in config_text
    assert "canonicalize_delta: false" in config_text
    assert "diagnostic_include_offline_counterfactuals: true" in config_text
    assert "conditional_shuffled" in config_text
    assert "conditional_random_matched_norm" in config_text


def test_v13_runner_requires_full_counterfactual_gate_modes() -> None:
    with pytest.raises(ValueError, match="requires full counterfactual mode set"):
        _validate_v13_gate_modes(["real", "zero", "sign_flipped"])


def test_v13_probe_cli_rejects_level_offset_main_signal() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "train/run_stage3_v13_fluctuation_teacher_probe.py",
            "--signal-mode",
            "level_offset",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "invalid choice" in result.stderr
