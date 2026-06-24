from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _load_matrix_module():
    try:
        return importlib.import_module("train.plan_stage3_v29_teacher_gate_tuning_matrix")
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing V29 teacher-gate tuning matrix module: {exc}")


def _base_config() -> dict[str, object]:
    return {
        "model": {
            "num_experts": 4,
            "invariant_release_hidden_dim": 128,
            "router_invariant_hidden_dim": 64,
        },
        "train": {"epochs": 5, "lr": 0.00038},
        "invariant": {
            "lambda_release_first_increment": 0.4,
            "lambda_history_after_release": 0.08,
            "lambda_release_temporal": 0.04,
            "lambda_router_stability": 0.12,
            "lambda_anchor_preservation": 0.12,
            "lambda_change_readout": 0.3,
            "lambda_delayed_change": 0.28,
            "lambda_release_state_advantage": 0.12,
            "lambda_completion_after_release": 0.08,
            "lambda_release_null": 0.05,
            "lambda_counterfactual_shuffle": 0.08,
            "counterfactual_shuffle_margin": 0.001,
            "lambda_history_budget_cap": 0.035,
            "lambda_support_alignment": 0.14,
            "lambda_invariant_correction_budget": 0.02,
        },
        "stage3_v2": {
            "protocol_version": "v29_expert_local_invariant_release_probe",
            "run_id": "manual_stage3_v29_expert_local_invariant_release_probe_tune_stability_release_deconstrained",
        },
    }


def _blocked_preflight() -> dict[str, object]:
    return {
        "protocol_version": "stage3_v29_generator_unlock_preflight_v1",
        "preflight_pass": False,
        "decision": "do_not_train_generator",
        "candidate_reports": [
            {
                "run_id": "v29stable_weather_s42_b60g60e1_epochdiag01",
                "blockers": [
                    "target_gate_not_passed",
                    "no_primary_gate_passed",
                    "teacher_abort_written",
                    "generator_not_unlocked",
                    "stop_reason:do_not_train_generator",
                    "stop_reason:no_primary_target_passed",
                    "positive_partial_windows_below_required",
                    "min_partial_mse_below_floor",
                    "min_partial_mae_below_floor",
                    "counterfactual_margin_negative",
                ],
                "metrics": {
                    "positive_partial_window_count": 2,
                    "required_positive_partial_window_count": 4,
                    "min_partial_mse_gain_vs_history": -0.3467,
                    "min_partial_mae_gain_vs_history": -0.4360,
                    "counterfactual_margin": -0.0117,
                    "window0_fh_mse_gain": 0.4725,
                },
            }
        ],
    }


def test_teacher_gate_tuning_matrix_is_driven_by_preflight_blockers() -> None:
    module = _load_matrix_module()

    candidates = module.build_teacher_gate_tuning_matrix(
        base_config=_base_config(),
        preflight_report=_blocked_preflight(),
        run_id_prefix="v29tgunit",
    )

    assert len(candidates) >= 4
    axes = {candidate["axis"] for candidate in candidates}
    assert {
        "history_guarded_release_floor",
        "counterfactual_margin_micro_repair",
        "late_window_partial_floor_guard",
    }.issubset(axes)
    assert all(candidate["protocol_version"] == module.PROTOCOL_VERSION for candidate in candidates)
    assert all(candidate["source_preflight_decision"] == "do_not_train_generator" for candidate in candidates)
    assert any("counterfactual_margin_negative" in row["addresses_blockers"] for row in candidates)
    assert any("min_partial_mae_below_floor" in row["addresses_blockers"] for row in candidates)


def test_teacher_gate_tuning_matrix_keeps_architecture_frozen_and_avoids_single_axis_repeats() -> None:
    module = _load_matrix_module()

    candidates = module.build_teacher_gate_tuning_matrix(
        base_config=_base_config(),
        preflight_report=_blocked_preflight(),
        run_id_prefix="v29tgunit",
    )

    for candidate in candidates:
        assert candidate["architecture_policy"] == "frozen_v29_no_model_topology_change"
        assert candidate["known_failed_axis_guard"] == "coupled_overrides_only_no_single_axis_repeat"
        assert candidate["model_overrides"] == {}
        assert len(candidate["invariant_overrides"]) >= 2
        assert "suggested_probe_command" in candidate
        assert "--max-generator-batches 60" in candidate["suggested_probe_command"]
        assert "--epochs 1" in candidate["suggested_probe_command"]


def test_teacher_gate_tuning_matrix_writes_configs_and_refuses_overwrite(tmp_path: Path) -> None:
    module = _load_matrix_module()
    candidates = module.build_teacher_gate_tuning_matrix(
        base_config=_base_config(),
        preflight_report=_blocked_preflight(),
        run_id_prefix="v29tgunit",
    )

    paths = module.write_teacher_gate_tuning_matrix_artifacts(
        candidates=candidates,
        base_config=_base_config(),
        out_dir=tmp_path / "artifacts",
        config_dir=tmp_path / "configs",
        prefix="unit_teacher_gate_matrix",
    )

    assert Path(paths["csv"]).exists()
    assert Path(paths["markdown"]).exists()
    config_paths = [Path(path) for path in paths["configs"]]
    assert len(config_paths) == len(candidates)
    first_config = yaml.safe_load(config_paths[0].read_text(encoding="utf-8"))
    assert first_config["stage3_v2"]["protocol_version"] == "v29_expert_local_invariant_release_probe"
    assert first_config["stage3_v2"]["run_id"].startswith("v29tgunit_")
    assert first_config["model"] == _base_config()["model"]

    with pytest.raises(FileExistsError):
        module.write_teacher_gate_tuning_matrix_artifacts(
            candidates=candidates,
            base_config=_base_config(),
            out_dir=tmp_path / "artifacts",
            config_dir=tmp_path / "configs",
            prefix="unit_teacher_gate_matrix",
        )


def test_teacher_gate_tuning_matrix_cli_writes_real_artifact_shape(tmp_path: Path) -> None:
    base_config_path = tmp_path / "base.yaml"
    preflight_path = tmp_path / "preflight.json"
    base_config_path.write_text(yaml.safe_dump(_base_config(), sort_keys=False), encoding="utf-8")
    preflight_path.write_text(json.dumps(_blocked_preflight()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "train/plan_stage3_v29_teacher_gate_tuning_matrix.py",
            "--base-config",
            str(base_config_path),
            "--preflight-json",
            str(preflight_path),
            "--out-dir",
            str(tmp_path / "artifacts"),
            "--config-dir",
            str(tmp_path / "configs"),
            "--prefix",
            "unit_teacher_gate_matrix",
            "--run-id-prefix",
            "v29tgunit",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "artifacts" / "unit_teacher_gate_matrix.csv").exists()
    assert (tmp_path / "artifacts" / "unit_teacher_gate_matrix.md").exists()
    assert list((tmp_path / "configs").glob("*.yaml"))
