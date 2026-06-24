import csv
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train.analyze_stage3_v29_gate_failure_decomposition import (
    analyze_gate_failure_decomposition,
    write_gate_failure_decomposition,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _sample_selector_rows() -> list[dict[str, object]]:
    return [
        {
            "run_dir": r"logs\s3v29expert_tuning_diag\anchor\weather\seed_42",
            "run_id": "anchor_gate_near",
            "strict_rank": 1,
            "selection_label": "gate_near_diagnostic",
            "target_gate_pass": "False",
            "any_primary_gate_pass": "False",
            "teacher_abort_written": "True",
            "generator_unlocked": "False",
            "positive_partial_window_count": 2,
            "required_positive_partial_window_count": 4,
            "min_partial_mse_gain_vs_history": -0.34,
            "min_partial_mae_gain_vs_history": -0.43,
            "counterfactual_margin": -0.01,
            "window0_fh_mse_gain": 0.47,
            "validation_partial_mse_gain_vs_history": 0.07,
            "validation_partial_mae_gain_vs_history": -0.02,
        },
        {
            "run_dir": r"logs\s3v29expert_teacher_gate_matrix\candidate05\weather\seed_42",
            "run_id": "candidate05",
            "strict_rank": 8,
            "selection_label": "gate_negative",
            "target_gate_pass": "False",
            "any_primary_gate_pass": "False",
            "teacher_abort_written": "True",
            "generator_unlocked": "False",
            "positive_partial_window_count": 2,
            "required_positive_partial_window_count": 4,
            "min_partial_mse_gain_vs_history": -0.46,
            "min_partial_mae_gain_vs_history": -0.48,
            "counterfactual_margin": -0.04,
            "window0_fh_mse_gain": 0.33,
            "validation_partial_mse_gain_vs_history": 0.02,
            "validation_partial_mae_gain_vs_history": -0.06,
        },
    ]


def _sample_window_rows() -> list[dict[str, object]]:
    return [
        {
            "run_id": "anchor_gate_near",
            "strict_rank": 1,
            "validation_window_id": 0,
            "fh_mse_gain": 0.47,
            "history_mse_gain": 0.79,
            "partial_mse_gain_vs_history": -0.32,
            "partial_mae_gain_vs_history": -0.35,
            "counterfactual_margin": 0.001,
            "real_is_best_mode": "True",
            "failure_reasons": "partial_mse_below_gate_floor;partial_mae_below_gate_floor",
        },
        {
            "run_id": "anchor_gate_near",
            "strict_rank": 1,
            "validation_window_id": 1,
            "fh_mse_gain": 0.83,
            "history_mse_gain": 0.91,
            "partial_mse_gain_vs_history": -0.08,
            "partial_mae_gain_vs_history": -0.20,
            "counterfactual_margin": -0.0004,
            "real_is_best_mode": "False",
            "failure_reasons": "partial_mse_below_gate_floor;counterfactual_negative",
        },
        {
            "run_id": "candidate05",
            "strict_rank": 8,
            "validation_window_id": 0,
            "fh_mse_gain": 0.33,
            "history_mse_gain": 0.70,
            "partial_mse_gain_vs_history": -0.37,
            "partial_mae_gain_vs_history": -0.41,
            "counterfactual_margin": -0.02,
            "real_is_best_mode": "False",
            "failure_reasons": "partial_mse_below_gate_floor;counterfactual_negative",
        },
        {
            "run_id": "candidate05",
            "strict_rank": 8,
            "validation_window_id": 4,
            "fh_mse_gain": 0.12,
            "history_mse_gain": 0.50,
            "partial_mse_gain_vs_history": -0.38,
            "partial_mae_gain_vs_history": -0.45,
            "counterfactual_margin": -0.04,
            "real_is_best_mode": "False",
            "failure_reasons": "partial_mse_below_gate_floor;partial_mae_below_gate_floor;counterfactual_negative",
        },
    ]


def _sample_preflight() -> dict[str, object]:
    blockers = [
        "target_gate_not_passed",
        "no_primary_gate_passed",
        "teacher_abort_written",
        "generator_not_unlocked",
        "positive_partial_windows_below_required",
        "min_partial_mse_below_floor",
        "min_partial_mae_below_floor",
        "counterfactual_margin_negative",
    ]
    return {
        "preflight_pass": False,
        "decision": "do_not_train_generator",
        "candidate_reports": [
            {"run_id": "anchor_gate_near", "blockers": blockers, "metrics": {}},
            {"run_id": "candidate05", "blockers": blockers, "metrics": {}},
        ],
    }


def test_decomposition_identifies_repeated_gate_failures_and_stop_decision() -> None:
    report = analyze_gate_failure_decomposition(
        selector_rows=_sample_selector_rows(),
        window_rows=_sample_window_rows(),
        preflight_report=_sample_preflight(),
    )

    assert report["candidate_count"] == 2
    assert report["gate_pass_count"] == 0
    assert report["generator_unlocked_count"] == 0
    assert report["matrix_decision"] == "stop_local_scalar_matrix"
    assert report["primary_blocker_counts"]["counterfactual_margin_negative"] == 2
    assert report["primary_blocker_counts"]["positive_partial_windows_below_required"] == 2
    assert "partial_floor_violation" in report["dominant_failure_axes"]
    assert "counterfactual_ordering_failure" in report["dominant_failure_axes"]
    assert "window_level_conflict" in report["dominant_failure_axes"]
    assert report["candidate_diagnoses"][0]["conflict_tags"] == [
        "window0_nonharm_but_partial_floor_fails",
        "aggregate_partial_positive_but_min_window_fails",
        "counterfactual_margin_negative",
        "generator_locked",
    ]


def test_decomposition_writes_json_csv_and_markdown_without_overwrite(tmp_path: Path) -> None:
    selector_csv = tmp_path / "selector.csv"
    windows_csv = tmp_path / "windows.csv"
    preflight_json = tmp_path / "preflight.json"
    out_dir = tmp_path / "out"

    _write_csv(selector_csv, _sample_selector_rows())
    _write_csv(windows_csv, _sample_window_rows())
    preflight_json.write_text(json.dumps(_sample_preflight()), encoding="utf-8")

    report = write_gate_failure_decomposition(
        selector_csv=selector_csv,
        windows_csv=windows_csv,
        preflight_json=preflight_json,
        out_dir=out_dir,
        prefix="sample",
        overwrite=False,
    )

    assert report["matrix_decision"] == "stop_local_scalar_matrix"
    assert (out_dir / "sample_gate_failure_decomposition.json").exists()
    assert (out_dir / "sample_gate_failure_decomposition.csv").exists()
    assert (out_dir / "sample_gate_failure_decomposition.md").exists()
    markdown = (out_dir / "sample_gate_failure_decomposition.md").read_text(encoding="utf-8")
    assert "Do not train the generator" in markdown
    assert "stop this local scalar matrix" in markdown

    with pytest.raises(FileExistsError):
        write_gate_failure_decomposition(
            selector_csv=selector_csv,
            windows_csv=windows_csv,
            preflight_json=preflight_json,
            out_dir=out_dir,
            prefix="sample",
            overwrite=False,
        )
