from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys

import pytest
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train.plan_stage3_v29_history_gate_recovery_matrix import (
    build_history_gate_recovery_matrix,
    write_history_gate_recovery_matrix_artifacts,
)


OLD_AXES = {
    "history_guarded_release_floor",
    "counterfactual_margin_micro_repair",
    "late_window_partial_floor_guard",
    "delayed_change_tempered_history",
    "release_null_coupled_guard",
    "window0_partial_floor_rebalance",
    "counterfactual_ordering_without_floor_pressure",
    "dual_axis_floor_and_cf_coupled",
    "mode_abstention_teacher_selection_probe",
}


def _sample_config() -> dict[str, object]:
    return {
        "data": {"path": "dataset/weather.csv"},
        "model_v29": {"n_experts": 4, "repr_dim": 64},
        "train": {"lr": 0.00038, "epochs": 5},
        "stage3_v2": {"run_id": "v29stable_weather_s42_b60g60e1_epochdiag01"},
        "invariant": {
            "lambda_release_first_increment": 0.4,
            "lambda_history_after_release": 0.08,
            "lambda_history_budget_cap": 0.035,
            "lambda_counterfactual_shuffle": 0.08,
            "counterfactual_shuffle_margin": 0.001,
            "lambda_router_stability": 0.12,
            "lambda_anchor_preservation": 0.12,
            "lambda_support_alignment": 0.14,
            "lambda_support_conditioned_advantage": 0.08,
            "lambda_suppression_alignment": 0.1,
            "lambda_suppression_gate_sparsity": 0.025,
            "lambda_residual_release_alignment": 0.03,
            "lambda_residual_release_advantage": 0.08,
            "lambda_release_priority_advantage": 0.18,
            "lambda_release_state_advantage": 0.12,
            "lambda_local_env_consistency": 0.04,
        },
    }


def _sample_decomposition() -> dict[str, object]:
    return {
        "protocol_version": "stage3_v29_gate_failure_decomposition_v1",
        "matrix_decision": "stop_local_scalar_matrix",
        "gate_pass_count": 0,
        "generator_unlocked_count": 0,
        "dominant_failure_axes": [
            "no_primary_teacher_gate_pass",
            "positive_partial_window_shortfall",
            "partial_floor_violation",
            "counterfactual_ordering_failure",
            "window_level_conflict",
            "generator_unlock_blocked",
        ],
        "primary_blocker_counts": {
            "counterfactual_margin_negative": 32,
            "min_partial_mse_below_floor": 32,
            "min_partial_mae_below_floor": 32,
            "positive_partial_windows_below_required": 17,
        },
        "window_summaries": [
            {
                "validation_window_id": "0",
                "row_count": 32,
                "partial_mse_floor_failures": 32,
                "partial_mae_floor_failures": 31,
                "counterfactual_negative_rows": 30,
                "real_best_rows": 2,
            },
            {
                "validation_window_id": "2",
                "row_count": 32,
                "partial_mse_floor_failures": 0,
                "partial_mae_floor_failures": 0,
                "counterfactual_negative_rows": 30,
                "real_best_rows": 2,
            },
        ],
    }


def _sample_selector_rows() -> list[dict[str, str]]:
    return [
        {
            "run_id": "v29stable_weather_s42_b60g60e1_epochdiag01",
            "strict_rank": "1",
            "target_name": "raw_residual_vector",
            "feature_mode": "fluctuation_plus_history",
            "target_gate_pass": "False",
            "generator_unlocked": "False",
            "positive_partial_window_count": "2",
            "required_positive_partial_window_count": "4",
            "window0_fh_mse_gain": "0.47259663917691164",
            "min_partial_mse_gain_vs_history": "-0.34678779319025144",
            "counterfactual_margin": "-0.011714003971613818",
        },
        {
            "run_id": "v29first042_weather_s42_b60_01",
            "strict_rank": "15",
            "target_name": "raw_residual_vector",
            "feature_mode": "fluctuation_plus_history",
            "target_gate_pass": "False",
            "generator_unlocked": "False",
            "positive_partial_window_count": "4",
            "required_positive_partial_window_count": "4",
            "window0_fh_mse_gain": "-0.42083107598996183",
            "min_partial_mse_gain_vs_history": "-0.42083107598996183",
            "counterfactual_margin": "-0.42083107598996183",
        },
    ]


def _sample_window_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for window_id, partial, margin, history in [
        ("0", "-0.3242256598289538", "0.001773456637841675", "0.7968222990058654"),
        ("1", "-0.08091132096264686", "-0.0004500174746387975", "0.9190952547891369"),
        ("2", "0.7920882869253715", "-0.011714003971613818", "0.0"),
        ("3", "0.27826005673487364", "-0.002761645844430727", "0.5892884277396425"),
        ("4", "-0.34678779319025144", "0.0007148644800215598", "0.990893094028307"),
    ]:
        rows.append(
            {
                "run_id": "v29stable_weather_s42_b60g60e1_epochdiag01",
                "strict_rank": "1",
                "target_name": "raw_residual_vector",
                "feature_mode": "fluctuation_plus_history",
                "validation_window_id": window_id,
                "partial_mse_gain_vs_history": partial,
                "counterfactual_margin": margin,
                "history_mse_gain": history,
                "fh_mse_gain": "0.5",
                "real_is_best_mode": "False",
            }
        )
    return rows


def test_history_gate_recovery_matrix_requires_exhausted_repair_and_changes_mechanism() -> None:
    rows = build_history_gate_recovery_matrix(
        base_config=_sample_config(),
        decomposition_report=_sample_decomposition(),
        selector_rows=_sample_selector_rows(),
        window_rows=_sample_window_rows(),
        run_id_prefix="v29hgr_conf015",
    )

    assert len(rows) == 4
    axes = {row["axis"] for row in rows}
    assert axes.isdisjoint(OLD_AXES)
    assert {
        "window0_safe_release_ramp",
        "late_history_dominance_bridge",
        "counterfactual_tie_breaker_floor_safe",
        "history_cap_release_balance",
    } == axes
    assert all(row["architecture_policy"] == "frozen_v29_no_model_topology_change" for row in rows)
    assert all(row["generator_policy"] == "preflight_required_do_not_unlock_from_matrix" for row in rows)
    assert all(row["repair_matrix_status_required"] == "exhausted_meaningful_rows" for row in rows)
    assert any("history_dominance_partial_floor_conflict" in row["addresses_failure_axes"] for row in rows)
    assert any("counterfactual_tie_near_gate" in row["addresses_failure_axes"] for row in rows)
    assert all(len(row["run_id"]) <= 48 for row in rows)
    assert all(len(row["config_name"]) <= 64 for row in rows)
    assert not any("--overwrite" in row["suggested_probe_command"] for row in rows)


def test_history_gate_recovery_matrix_refuses_non_final_or_gate_positive_reports() -> None:
    not_stopped = dict(_sample_decomposition())
    not_stopped["matrix_decision"] = "continue_preflight_controlled_tuning"
    assert (
        build_history_gate_recovery_matrix(
            base_config=_sample_config(),
            decomposition_report=not_stopped,
            selector_rows=_sample_selector_rows(),
            window_rows=_sample_window_rows(),
        )
        == []
    )

    gate_positive = dict(_sample_decomposition())
    gate_positive["gate_pass_count"] = 1
    assert (
        build_history_gate_recovery_matrix(
            base_config=_sample_config(),
            decomposition_report=gate_positive,
            selector_rows=_sample_selector_rows(),
            window_rows=_sample_window_rows(),
        )
        == []
    )


def test_write_history_gate_recovery_matrix_outputs_configs_and_no_overwrite(tmp_path: Path) -> None:
    rows = build_history_gate_recovery_matrix(
        base_config=_sample_config(),
        decomposition_report=_sample_decomposition(),
        selector_rows=_sample_selector_rows(),
        window_rows=_sample_window_rows(),
        run_id_prefix="v29hgr_conf015",
    )

    outputs = write_history_gate_recovery_matrix_artifacts(
        rows=rows,
        base_config=_sample_config(),
        config_dir=tmp_path / "configs",
        out_dir=tmp_path / "artifacts",
        prefix="sample_history_gate_recovery",
        overwrite=False,
    )

    matrix_csv = Path(outputs["csv"])
    matrix_md = Path(outputs["markdown"])
    assert matrix_csv.exists()
    assert matrix_md.exists()
    assert len(outputs["configs"]) == 4

    csv_rows = list(csv.DictReader(matrix_csv.open(encoding="utf-8")))
    assert csv_rows[0]["protocol_version"] == "stage3_v29_history_gate_recovery_matrix_v1"
    assert "history_dominance_partial_floor_conflict" in csv_rows[0]["addresses_failure_axes"]

    config_payload = yaml.safe_load(Path(outputs["configs"][0]).read_text(encoding="utf-8"))
    assert config_payload["stage3_v2"]["run_id"] == csv_rows[0]["run_id"]
    assert config_payload["model_v29"] == _sample_config()["model_v29"]
    assert config_payload["history_gate_recovery_matrix"]["generator_policy"] == "preflight_required_do_not_unlock_from_matrix"
    assert config_payload["history_gate_recovery_matrix"]["repair_matrix_status_required"] == "exhausted_meaningful_rows"

    markdown = matrix_md.read_text(encoding="utf-8")
    assert "history-dominance" in markdown
    assert "does not train the generator" in markdown
    assert "does not change V29 model topology" in markdown

    with pytest.raises(FileExistsError):
        write_history_gate_recovery_matrix_artifacts(
            rows=rows,
            base_config=_sample_config(),
            config_dir=tmp_path / "configs",
            out_dir=tmp_path / "artifacts",
            prefix="sample_history_gate_recovery",
            overwrite=False,
        )


def test_history_gate_recovery_matrix_cli_writes_artifacts(tmp_path: Path) -> None:
    base_config = tmp_path / "base.yaml"
    decomposition = tmp_path / "decomposition.json"
    selector = tmp_path / "selector.csv"
    windows = tmp_path / "windows.csv"
    base_config.write_text(yaml.safe_dump(_sample_config(), sort_keys=False), encoding="utf-8")
    decomposition.write_text(json.dumps(_sample_decomposition()), encoding="utf-8")

    with selector.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_sample_selector_rows()[0]))
        writer.writeheader()
        writer.writerows(_sample_selector_rows())
    with windows.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_sample_window_rows()[0]))
        writer.writeheader()
        writer.writerows(_sample_window_rows())

    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "train/plan_stage3_v29_history_gate_recovery_matrix.py",
            "--base-config",
            str(base_config),
            "--decomposition-json",
            str(decomposition),
            "--selector-csv",
            str(selector),
            "--windows-csv",
            str(windows),
            "--config-dir",
            str(tmp_path / "configs"),
            "--out-dir",
            str(tmp_path / "artifacts"),
            "--prefix",
            "cli_history_gate_recovery",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "artifacts" / "cli_history_gate_recovery.csv").exists()
    assert (tmp_path / "artifacts" / "cli_history_gate_recovery.md").exists()
