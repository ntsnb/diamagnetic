from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.run_stage3_v15_residual_target_predictability_probe import (
    DEFAULT_V15_FEATURE_MODES,
    PRIMARY_V15_TARGETS,
    build_v15_target_tensor,
    v15_directional_stop_reasons,
    v15_target_to_teacher_tensors,
    validate_v15_config_contract,
)


def test_v15_primary_targets_are_predeclared() -> None:
    assert PRIMARY_V15_TARGETS == ("raw_residual_vector", "residual_change_vector")


def test_v15_feature_modes_include_primary_and_auxiliary_controls() -> None:
    assert DEFAULT_V15_FEATURE_MODES == (
        "history_only",
        "fluctuation_only",
        "fluctuation_plus_history",
        "level_offset_only",
        "level_offset_plus_history",
    )


def test_v15_raw_residual_target_is_y_minus_base() -> None:
    y_base = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y_true = torch.tensor([[1.5, 1.0], [2.0, 6.0]])

    target = build_v15_target_tensor("raw_residual_vector", y_base=y_base, y_true=y_true)

    assert torch.allclose(target, torch.tensor([[0.5, -1.0], [-1.0, 2.0]]))


def test_v15_residual_change_uses_previous_tail() -> None:
    y_base = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y_true = torch.tensor([[1.5, 1.0], [2.0, 6.0]])
    previous_tail = torch.tensor([[0.25, -0.25]])

    target = build_v15_target_tensor(
        "residual_change_vector",
        y_base=y_base,
        y_true=y_true,
        previous_residual_tail=previous_tail,
    )

    assert torch.allclose(target, torch.tensor([[0.25, -0.75], [-1.5, 3.0]]))


def test_v15_residual_change_without_tail_zeros_first_transition() -> None:
    y_base = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y_true = torch.tensor([[1.5, 1.0], [2.0, 6.0]])

    target = build_v15_target_tensor("residual_change_vector", y_base=y_base, y_true=y_true)

    assert torch.allclose(target, torch.tensor([[0.0, 0.0], [-1.5, 3.0]]))


def test_v15_residual_sign_is_secondary_target() -> None:
    y_base = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y_true = torch.tensor([[1.5, 1.0], [3.0, 6.0]])

    target = build_v15_target_tensor("residual_sign", y_base=y_base, y_true=y_true)

    assert torch.allclose(target, torch.tensor([[1.0, -1.0], [0.0, 1.0]]))


def test_v15_target_to_teacher_tensors_make_target_the_residual() -> None:
    target = torch.tensor([[0.5, -1.0], [-1.0, 2.0]])

    y_base_target, y_true_target = v15_target_to_teacher_tensors(target)

    assert torch.allclose(y_base_target, torch.zeros_like(target))
    assert torch.allclose(y_true_target, target)


def test_v15_stop_reasons_block_generator_when_primary_targets_fail() -> None:
    rows = [
        {"target_name": "raw_residual_vector", "target_role": "primary", "primary_gate_pass": False},
        {"target_name": "residual_change_vector", "target_role": "primary", "primary_gate_pass": False},
    ]

    reasons = v15_directional_stop_reasons(rows)

    assert "no_primary_target_passed" in reasons
    assert "do_not_train_generator" in reasons


def test_v15_stop_reasons_empty_when_primary_target_passes() -> None:
    rows = [
        {"target_name": "raw_residual_vector", "target_role": "primary", "primary_gate_pass": True},
        {"target_name": "residual_change_vector", "target_role": "primary", "primary_gate_pass": False},
    ]

    assert v15_directional_stop_reasons(rows) == []


def test_v15_config_locks_non_overwriting_diagnostic_protocol() -> None:
    text = Path("configs/stage3_v15_residual_target_predictability_probe.yaml").read_text(encoding="utf-8")

    assert "protocol_version: v15_residual_target_predictability_probe" in text
    assert "version: v15_residual_target_predictability_probe" in text
    assert "signal_mode: fluctuation_delta" in text
    assert "drift_source: delta_phi_v" in text
    assert "logs/s3v15target" in text
    assert "ckpt/s3v15target" in text
    assert "results/s3v15target" in text
    assert "raw_residual_vector" in text
    assert "residual_change_vector" in text
    assert "fluctuation_plus_history" in text
    assert "history_only" in text
    assert "run_generator" not in text


def test_v15_config_validator_rejects_non_v15_protocol() -> None:
    cfg = {
        "stage3_v2": {"protocol_version": "v14_stability_teacher_probe"},
        "online": {"stage3": {"version": "v15_residual_target_predictability_probe"}},
    }

    with pytest.raises(ValueError, match="protocol_version"):
        validate_v15_config_contract(cfg)
