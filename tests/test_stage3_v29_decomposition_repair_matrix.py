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

from train.plan_stage3_v29_decomposition_repair_matrix import (
    build_decomposition_repair_matrix,
    write_decomposition_repair_matrix_artifacts,
)


OLD_SCALAR_AXES = {
    "history_guarded_release_floor",
    "counterfactual_margin_micro_repair",
    "late_window_partial_floor_guard",
    "delayed_change_tempered_history",
    "release_null_coupled_guard",
}


def _sample_config() -> dict[str, object]:
    return {
        "dataset": {"path": "dataset/weather.csv"},
        "train": {"seed": 42},
        "stage3_v2": {"run_id": "v29stable_weather_s42_b60g60e1_epochdiag01"},
        "invariant": {
            "lambda_release_first_increment": 0.4,
            "lambda_history_budget_cap": 0.035,
            "lambda_history_after_release": 0.08,
            "lambda_invariant_correction_budget": 0.02,
            "lambda_counterfactual_shuffle": 0.08,
            "counterfactual_shuffle_margin": 0.001,
            "lambda_release_temporal": 0.04,
            "lambda_router_stability": 0.12,
            "lambda_anchor_preservation": 0.12,
            "lambda_release_state_advantage": 0.12,
            "lambda_completion_after_release": 0.08,
            "lambda_support_alignment": 0.14,
            "lambda_suppression_alignment": 0.11,
            "lambda_suppression_gate_sparsity": 0.03,
            "lambda_residual_release_alignment": 0.16,
            "lambda_residual_release_advantage": 0.12,
        },
    }


def _sample_decomposition() -> dict[str, object]:
    return {
        "protocol_version": "stage3_v29_gate_failure_decomposition_v1",
        "matrix_decision": "stop_local_scalar_matrix",
        "dominant_failure_axes": [
            "no_primary_teacher_gate_pass",
            "positive_partial_window_shortfall",
            "partial_floor_violation",
            "counterfactual_ordering_failure",
            "window_level_conflict",
            "generator_unlock_blocked",
        ],
        "window_summaries": [
            {
                "validation_window_id": "0",
                "row_count": 27,
                "partial_mse_floor_failures": 27,
                "partial_mae_floor_failures": 27,
                "counterfactual_negative_rows": 25,
                "real_best_rows": 2,
            },
            {
                "validation_window_id": "2",
                "row_count": 27,
                "partial_mse_floor_failures": 0,
                "partial_mae_floor_failures": 0,
                "counterfactual_negative_rows": 25,
                "real_best_rows": 2,
            },
        ],
    }


def test_decomposition_repair_matrix_uses_failure_axes_and_avoids_old_axes() -> None:
    rows = build_decomposition_repair_matrix(
        base_config=_sample_config(),
        decomposition_report=_sample_decomposition(),
        run_id_prefix="v29repair_conf015",
    )

    assert len(rows) == 4
    assert {row["axis"] for row in rows}.isdisjoint(OLD_SCALAR_AXES)
    assert all(row["architecture_policy"] == "frozen_v29_no_model_topology_change" for row in rows)
    assert all(row["generator_policy"] == "preflight_required_do_not_unlock_from_matrix" for row in rows)
    assert all(row["known_failed_axis_guard"] == "not_one_of_prior_five_scalar_axes" for row in rows)
    assert any("partial_floor_violation" in row["addresses_failure_axes"] for row in rows)
    assert any("counterfactual_ordering_failure" in row["addresses_failure_axes"] for row in rows)
    assert any("window_level_conflict" in row["addresses_failure_axes"] for row in rows)
    assert all(len(row["run_id"]) <= 48 for row in rows)
    assert all(len(row["config_name"]) <= 64 for row in rows)
    assert all("suggested_probe_command" in row for row in rows)
    assert not any("--overwrite" in row["suggested_probe_command"] for row in rows)


def test_decomposition_repair_matrix_refuses_when_decomposition_does_not_stop_matrix() -> None:
    decomposition = dict(_sample_decomposition())
    decomposition["matrix_decision"] = "continue_preflight_controlled_tuning"

    rows = build_decomposition_repair_matrix(
        base_config=_sample_config(),
        decomposition_report=decomposition,
        run_id_prefix="v29repair_conf015",
    )

    assert rows == []


def test_write_decomposition_repair_matrix_outputs_configs_and_no_overwrite(tmp_path: Path) -> None:
    rows = build_decomposition_repair_matrix(
        base_config=_sample_config(),
        decomposition_report=_sample_decomposition(),
        run_id_prefix="v29repair_conf015",
    )
    config_dir = tmp_path / "configs"
    out_dir = tmp_path / "artifacts"

    outputs = write_decomposition_repair_matrix_artifacts(
        rows=rows,
        base_config=_sample_config(),
        config_dir=config_dir,
        out_dir=out_dir,
        prefix="sample_repair_matrix",
        overwrite=False,
    )

    matrix_csv = Path(outputs["csv"])
    matrix_md = Path(outputs["markdown"])
    assert matrix_csv.exists()
    assert matrix_md.exists()
    assert len(outputs["configs"]) == 4
    assert all(Path(path).exists() for path in outputs["configs"])

    csv_rows = list(csv.DictReader(matrix_csv.open(encoding="utf-8")))
    assert len(csv_rows) == 4
    assert csv_rows[0]["source_decomposition_protocol"] == "stage3_v29_gate_failure_decomposition_v1"
    assert csv_rows[0]["source_matrix_decision"] == "stop_local_scalar_matrix"
    assert "partial_floor_violation" in csv_rows[0]["addresses_failure_axes"]

    config_payload = yaml.safe_load(Path(outputs["configs"][0]).read_text(encoding="utf-8"))
    assert config_payload["stage3_v2"]["run_id"] == csv_rows[0]["run_id"]
    assert config_payload["decomposition_repair_matrix"]["generator_policy"] == "preflight_required_do_not_unlock_from_matrix"
    assert config_payload["decomposition_repair_matrix"]["known_failed_axis_guard"] == "not_one_of_prior_five_scalar_axes"

    markdown = matrix_md.read_text(encoding="utf-8")
    assert "does not train the generator" in markdown
    assert "partial-floor stability" in markdown
    assert "counterfactual ordering" in markdown

    with pytest.raises(FileExistsError):
        write_decomposition_repair_matrix_artifacts(
            rows=rows,
            base_config=_sample_config(),
            config_dir=config_dir,
            out_dir=out_dir,
            prefix="sample_repair_matrix",
            overwrite=False,
        )
