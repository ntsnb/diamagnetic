from __future__ import annotations

import csv
import os
from pathlib import Path
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train.analyze_stage3_v29_tuning_selection import (
    analyze_selection,
    build_feature_mode_policy_diagnostics,
    build_mode_gate_label_rows,
    build_online_gate_feature_rows,
    build_window_breakdown,
    select_runs,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_v29_run(
    root: Path,
    *,
    run_id: str,
    positive_windows: int,
    min_partial_mse: float,
    min_partial_mae: float,
    counterfactual_margin: float,
    validation_fh_mse: float,
    validation_history_mse: float,
    validation_partial_mse: float,
    validation_partial_mae: float,
    window0_fh_mse: float,
    window0_history_mse: float,
    window0_partial_mse: float,
    window0_counterfactual_margin: float,
    abort: bool = True,
) -> Path:
    run_dir = root / run_id / "weather" / "seed_42"
    run_dir.mkdir(parents=True)
    _write_csv(
        run_dir / "teacher_only_v29_target_gate.csv",
        [
            {
                "target_name": "raw_residual_vector",
                "target_role": "primary",
                "primary_gate_pass": "False",
                "positive_partial_window_count": positive_windows,
                "required_positive_partial_window_count": 4,
                "window_count": 5,
                "min_partial_mse_gain_vs_history": min_partial_mse,
                "min_partial_mae_gain_vs_history": min_partial_mae,
                "counterfactual_margin": counterfactual_margin,
                "counterfactual_pass": "False",
            }
        ],
    )
    _write_csv(
        run_dir / "teacher_only_validation_diagnostic_summary_v29.csv",
        [
            {
                "run_id": run_id,
                "dataset": "weather",
                "seed": "42",
                "split": "validation",
                "target_name": "raw_residual_vector",
                "target_role": "primary",
                "feature_mode": "fluctuation_plus_history",
                "teacher_real_mse_gain": validation_fh_mse,
                "teacher_real_mae_gain": "0.5",
                "teacher_real_sign_agreement_rate": "0.7",
                "teacher_best_counterfactual_mode": "shuffled",
                "teacher_best_counterfactual_mse_gain": validation_fh_mse - counterfactual_margin,
                "teacher_real_vs_best_counterfactual_mse_gain_margin": counterfactual_margin,
                "teacher_real_is_best_mode": "False",
            },
            {
                "run_id": run_id,
                "dataset": "weather",
                "seed": "42",
                "split": "validation",
                "target_name": "raw_residual_vector",
                "target_role": "primary",
                "feature_mode": "history_only",
                "teacher_real_mse_gain": validation_history_mse,
                "teacher_real_mae_gain": "0.4",
                "teacher_real_sign_agreement_rate": "0.6",
                "teacher_best_counterfactual_mode": "zero",
                "teacher_best_counterfactual_mse_gain": validation_history_mse,
                "teacher_real_vs_best_counterfactual_mse_gain_margin": "0.0",
                "teacher_real_is_best_mode": "False",
            },
        ],
    )
    _write_csv(
        run_dir / "teacher_only_validation_partial_gain_vs_history_v29.csv",
        [
            {
                "run_id": run_id,
                "dataset": "weather",
                "seed": "42",
                "split": "validation",
                "target_name": "raw_residual_vector",
                "target_role": "primary",
                "history_only_real_mse_gain": validation_history_mse,
                "fluctuation_plus_history_real_mse_gain": validation_fh_mse,
                "partial_mse_gain_vs_history": validation_partial_mse,
                "history_only_real_mae_gain": "0.4",
                "fluctuation_plus_history_real_mae_gain": "0.5",
                "partial_mae_gain_vs_history": validation_partial_mae,
                "history_only_sign_agreement": "0.6",
                "fluctuation_plus_history_sign_agreement": "0.7",
                "partial_sign_agreement_vs_history": "0.1",
            }
        ],
    )
    _write_csv(
        run_dir / "teacher_only_rolling_validation_window_summary_v29.csv",
        [
            {
                "run_id": run_id,
                "dataset": "weather",
                "seed": "42",
                "split": "validation_window_0",
                "validation_window_id": "0",
                "validation_window_start_sample": "0",
                "validation_window_end_sample": "12",
                "target_name": "raw_residual_vector",
                "target_role": "primary",
                "feature_mode": "fluctuation_plus_history",
                "teacher_real_mse_gain": window0_fh_mse,
                "teacher_real_mae_gain": "0.3",
                "teacher_real_sign_agreement_rate": "0.6",
                "teacher_best_counterfactual_mode": "shuffled",
                "teacher_best_counterfactual_mse_gain": window0_fh_mse - window0_counterfactual_margin,
                "teacher_real_vs_best_counterfactual_mse_gain_margin": window0_counterfactual_margin,
                "teacher_real_is_best_mode": "True",
            },
            {
                "run_id": run_id,
                "dataset": "weather",
                "seed": "42",
                "split": "validation_window_0",
                "validation_window_id": "0",
                "validation_window_start_sample": "0",
                "validation_window_end_sample": "12",
                "target_name": "raw_residual_vector",
                "target_role": "primary",
                "feature_mode": "history_only",
                "teacher_real_mse_gain": window0_history_mse,
                "teacher_real_mae_gain": "0.4",
                "teacher_real_sign_agreement_rate": "0.7",
                "teacher_best_counterfactual_mode": "zero",
                "teacher_best_counterfactual_mse_gain": window0_history_mse,
                "teacher_real_vs_best_counterfactual_mse_gain_margin": "0.0",
                "teacher_real_is_best_mode": "False",
            },
        ],
    )
    _write_csv(
        run_dir / "teacher_only_rolling_partial_gain_vs_history_v29.csv",
        [
            {
                "run_id": run_id,
                "dataset": "weather",
                "seed": "42",
                "split": "validation_window_0",
                "target_name": "raw_residual_vector",
                "target_role": "primary",
                "history_only_real_mse_gain": window0_history_mse,
                "fluctuation_plus_history_real_mse_gain": window0_fh_mse,
                "partial_mse_gain_vs_history": window0_partial_mse,
                "history_only_real_mae_gain": "0.4",
                "fluctuation_plus_history_real_mae_gain": "0.3",
                "partial_mae_gain_vs_history": "0.1",
                "validation_window_id": "0",
                "validation_window_start_sample": "0",
                "validation_window_end_sample": "12",
            }
        ],
    )
    if abort:
        (run_dir / "teacher_only_abort.json").write_text(
            '{"stop_reasons":["do_not_train_generator","no_primary_target_passed"]}',
            encoding="utf-8",
        )
    return run_dir


def _make_incomplete_v29_run(root: Path, *, run_id: str) -> Path:
    run_dir = root / run_id / "weather" / "seed_42"
    run_dir.mkdir(parents=True)
    _write_csv(
        run_dir / "teacher_only_v29_target_gate.csv",
        [
            {
                "target_name": "raw_residual_vector",
                "target_role": "primary",
                "primary_gate_pass": "False",
                "positive_partial_window_count": 0,
                "required_positive_partial_window_count": 4,
                "window_count": 5,
                "min_partial_mse_gain_vs_history": "nan",
                "min_partial_mae_gain_vs_history": "nan",
                "counterfactual_margin": 0.0,
                "counterfactual_pass": "False",
            }
        ],
    )
    _write_csv(
        run_dir / "teacher_only_validation_diagnostic_summary_v29.csv",
        [
            {
                "run_id": run_id,
                "dataset": "weather",
                "seed": "42",
                "split": "validation",
                "target_name": "raw_residual_vector",
                "target_role": "primary",
                "feature_mode": "fluctuation_plus_history",
                "teacher_real_mse_gain": "0.0",
                "teacher_real_mae_gain": "0.0",
                "teacher_real_sign_agreement_rate": "0.0",
                "teacher_best_counterfactual_mode": "zero",
                "teacher_best_counterfactual_mse_gain": "0.0",
                "teacher_real_vs_best_counterfactual_mse_gain_margin": "0.0",
                "teacher_real_is_best_mode": "False",
            }
        ],
    )
    _write_csv(run_dir / "teacher_only_validation_partial_gain_vs_history_v29.csv", [])
    _write_csv(
        run_dir / "teacher_only_rolling_validation_window_summary_v29.csv",
        [
            {
                "run_id": run_id,
                "dataset": "weather",
                "seed": "42",
                "split": "validation_window_0",
                "validation_window_id": "0",
                "validation_window_start_sample": "0",
                "validation_window_end_sample": "12",
                "target_name": "raw_residual_vector",
                "target_role": "primary",
                "feature_mode": "fluctuation_plus_history",
                "teacher_real_mse_gain": "0.0",
                "teacher_real_mae_gain": "0.0",
                "teacher_real_sign_agreement_rate": "0.0",
                "teacher_best_counterfactual_mode": "zero",
                "teacher_best_counterfactual_mse_gain": "0.0",
                "teacher_real_vs_best_counterfactual_mse_gain_margin": "0.0",
                "teacher_real_is_best_mode": "False",
            }
        ],
    )
    _write_csv(run_dir / "teacher_only_rolling_partial_gain_vs_history_v29.csv", [])
    return run_dir


def test_v29_selector_prefers_safer_epoch_one_like_candidate(tmp_path: Path) -> None:
    e1 = _make_v29_run(
        tmp_path,
        run_id="epoch1_like",
        positive_windows=2,
        min_partial_mse=-0.346,
        min_partial_mae=-0.436,
        counterfactual_margin=-0.011,
        validation_fh_mse=0.736,
        validation_history_mse=0.663,
        validation_partial_mse=0.073,
        validation_partial_mae=-0.023,
        window0_fh_mse=0.472,
        window0_history_mse=0.796,
        window0_partial_mse=-0.324,
        window0_counterfactual_margin=0.001,
    )
    e2 = _make_v29_run(
        tmp_path,
        run_id="epoch2_like",
        positive_windows=2,
        min_partial_mse=-0.591,
        min_partial_mae=-0.622,
        counterfactual_margin=-0.314,
        validation_fh_mse=0.632,
        validation_history_mse=0.644,
        validation_partial_mse=-0.011,
        validation_partial_mae=-0.083,
        window0_fh_mse=0.292,
        window0_history_mse=0.5,
        window0_partial_mse=-0.2,
        window0_counterfactual_margin=0.001,
    )
    e5 = _make_v29_run(
        tmp_path,
        run_id="epoch5_like",
        positive_windows=4,
        min_partial_mse=-0.146,
        min_partial_mae=-0.058,
        counterfactual_margin=-0.724,
        validation_fh_mse=0.480,
        validation_history_mse=0.132,
        validation_partial_mse=0.347,
        validation_partial_mae=0.261,
        window0_fh_mse=-0.724,
        window0_history_mse=0.0,
        window0_partial_mse=-0.724,
        window0_counterfactual_margin=-0.724,
    )

    ranked, skipped = select_runs([e2, e5, e1])

    assert skipped == []
    assert [row["run_id"] for row in ranked] == ["epoch1_like", "epoch2_like", "epoch5_like"]
    assert ranked[0]["selection_label"] == "gate_near_diagnostic"


def test_v29_selector_skips_incomplete_run_dirs(tmp_path: Path) -> None:
    incomplete = tmp_path / "incomplete" / "weather" / "seed_42"
    incomplete.mkdir(parents=True)

    ranked, skipped = select_runs([incomplete])

    assert ranked == []
    assert skipped[0]["skip_reason"] == "missing_required_v29_teacher_files"
    assert "teacher_only_v29_target_gate.csv" in skipped[0]["missing_files"]


def test_v29_selector_demotes_incomplete_nan_diagnostic_runs(tmp_path: Path) -> None:
    complete = _make_v29_run(
        tmp_path,
        run_id="complete_gate_near",
        positive_windows=2,
        min_partial_mse=-0.34,
        min_partial_mae=-0.43,
        counterfactual_margin=-0.011,
        validation_fh_mse=0.736,
        validation_history_mse=0.663,
        validation_partial_mse=0.073,
        validation_partial_mae=-0.023,
        window0_fh_mse=0.472,
        window0_history_mse=0.796,
        window0_partial_mse=-0.324,
        window0_counterfactual_margin=0.001,
    )
    incomplete = _make_incomplete_v29_run(tmp_path, run_id="incomplete_nan_diag")

    ranked, skipped = select_runs([incomplete, complete])

    assert skipped == []
    assert ranked[0]["run_id"] == "complete_gate_near"
    assert ranked[1]["run_id"] == "incomplete_nan_diag"
    assert ranked[1]["selection_label"] == "incomplete_diagnostic"


def test_v29_selector_complete_only_writes_incomplete_rows_to_skipped(tmp_path: Path) -> None:
    complete = _make_v29_run(
        tmp_path,
        run_id="complete_gate_near",
        positive_windows=2,
        min_partial_mse=-0.34,
        min_partial_mae=-0.43,
        counterfactual_margin=-0.011,
        validation_fh_mse=0.736,
        validation_history_mse=0.663,
        validation_partial_mse=0.073,
        validation_partial_mae=-0.023,
        window0_fh_mse=0.472,
        window0_history_mse=0.796,
        window0_partial_mse=-0.324,
        window0_counterfactual_margin=0.001,
    )
    incomplete = _make_incomplete_v29_run(tmp_path, run_id="incomplete_nan_diag")
    out_dir = tmp_path / "artifacts"

    candidates, skipped = analyze_selection(
        run_dirs=[incomplete, complete],
        roots=[],
        out_dir=out_dir,
        prefix="complete_only",
        require_complete_selector_metrics=True,
    )

    assert [row["run_id"] for row in candidates] == ["complete_gate_near"]
    assert skipped[-1]["skip_reason"] == "selector_metric_incomplete"
    assert skipped[-1]["run_id"] == "incomplete_nan_diag"
    online_rows = list(csv.DictReader((out_dir / "complete_only_online_gate_features.csv").open(encoding="utf-8")))
    assert {row["run_id"] for row in online_rows} == {"complete_gate_near"}


def test_v29_window_breakdown_reports_failure_flags(tmp_path: Path) -> None:
    run_dir = _make_v29_run(
        tmp_path,
        run_id="late_failure_like",
        positive_windows=2,
        min_partial_mse=-0.34,
        min_partial_mae=-0.43,
        counterfactual_margin=-0.02,
        validation_fh_mse=0.74,
        validation_history_mse=0.66,
        validation_partial_mse=0.08,
        validation_partial_mae=-0.01,
        window0_fh_mse=0.46,
        window0_history_mse=0.79,
        window0_partial_mse=-0.33,
        window0_counterfactual_margin=-0.01,
    )
    ranked, skipped = select_runs([run_dir])

    window_rows = build_window_breakdown(ranked)

    assert skipped == []
    assert window_rows[0]["run_id"] == "late_failure_like"
    assert "partial_mse_negative" in window_rows[0]["failure_reasons"]
    assert "partial_mse_below_gate_floor" in window_rows[0]["failure_reasons"]
    assert "counterfactual_negative" in window_rows[0]["failure_reasons"]


def test_v29_feature_mode_policy_diagnostic_reports_oracle_history_protection(tmp_path: Path) -> None:
    run_dir = _make_v29_run(
        tmp_path,
        run_id="history_dominated_like",
        positive_windows=1,
        min_partial_mse=-0.33,
        min_partial_mae=-0.37,
        counterfactual_margin=-0.01,
        validation_fh_mse=0.46,
        validation_history_mse=0.79,
        validation_partial_mse=-0.33,
        validation_partial_mae=-0.37,
        window0_fh_mse=0.46,
        window0_history_mse=0.79,
        window0_partial_mse=-0.33,
        window0_counterfactual_margin=-0.01,
    )
    ranked, skipped = select_runs([run_dir])

    policy_rows = build_feature_mode_policy_diagnostics(ranked)

    assert skipped == []
    by_policy = {row["policy_name"]: row for row in policy_rows}
    protected = by_policy["history_protected_oracle"]
    assert protected["selected_feature_modes_by_window"] == "0:history_only"
    assert protected["positive_partial_window_count"] == 0
    assert protected["min_partial_mse_gain_vs_history"] == 0.0
    assert protected["gate_like_pass"] is False


def test_v29_mode_gate_labels_mark_oracle_as_not_online(tmp_path: Path) -> None:
    run_dir = _make_v29_run(
        tmp_path,
        run_id="mode_gate_label_like",
        positive_windows=1,
        min_partial_mse=-0.33,
        min_partial_mae=-0.37,
        counterfactual_margin=-0.01,
        validation_fh_mse=0.46,
        validation_history_mse=0.79,
        validation_partial_mse=-0.33,
        validation_partial_mae=-0.37,
        window0_fh_mse=0.46,
        window0_history_mse=0.79,
        window0_partial_mse=-0.33,
        window0_counterfactual_margin=-0.01,
    )
    ranked, skipped = select_runs([run_dir])

    label_rows = build_mode_gate_label_rows(ranked)

    assert skipped == []
    assert label_rows[0]["best_available_feature_label"] == "history_only"
    assert label_rows[0]["needs_non_fh_mode"] is True
    assert label_rows[0]["online_label_available"] is False
    assert label_rows[0]["label_source"] == "validation_window_outcome_oracle"


def test_v29_online_gate_features_exclude_outcome_leakage(tmp_path: Path) -> None:
    run_dir = _make_v29_run(
        tmp_path,
        run_id="online_gate_feature_like",
        positive_windows=1,
        min_partial_mse=-0.33,
        min_partial_mae=-0.37,
        counterfactual_margin=-0.01,
        validation_fh_mse=0.46,
        validation_history_mse=0.79,
        validation_partial_mse=-0.33,
        validation_partial_mae=-0.37,
        window0_fh_mse=0.46,
        window0_history_mse=0.79,
        window0_partial_mse=-0.33,
        window0_counterfactual_margin=-0.01,
    )
    _write_csv(
        run_dir / "stage3_v29_signal_contract_diagnostics.csv",
        [
            {
                "signal_contract_version": "v13_fluctuation_first_signal_contract",
                "signal_mode": "fluctuation_delta",
                "signal_normalize": "True",
                "signal_fit_split": "train",
                "signal_fit_sample_count": "60",
                "signal_scale_abs_mean": "0.1",
                "signal_scale_min": "0.01",
                "signal_scale_max": "1.0",
            }
        ],
    )
    _write_csv(
        run_dir / "stage3_v29_history_contract_diagnostics.csv",
        [
            {
                "history_feature_protocol": "v12d_causal_residual_history_features",
                "history_lags": "1,2,3,6,12,24",
                "history_feedback_delay": "0",
                "history_include_availability_mask": "True",
                "history_standardize": "True",
                "history_fit_split": "train",
                "history_fit_sample_count": "60",
                "history_value_dim": "72",
                "history_feature_dim": "144",
                "history_scale_abs_mean": "1.2",
                "history_scale_min": "1.0",
                "history_scale_max": "1.4",
            }
        ],
    )
    (run_dir / "resolved_config.json").write_text(
        '{"data":{"lookback":24,"horizon":12},"invariant":{"lambda_release_first_increment":0.4}}',
        encoding="utf-8",
    )
    ranked, skipped = select_runs([run_dir])

    feature_rows = build_online_gate_feature_rows(ranked)

    assert skipped == []
    assert feature_rows[0]["feature_window_sample_count"] == 12
    assert feature_rows[0]["feature_signal_fit_split"] == "train"
    assert feature_rows[0]["feature_history_fit_split"] == "train"
    assert feature_rows[0]["feature_dynamic_signal_status"] == "missing_current_serving_window_statistics"
    assert feature_rows[0]["oracle_label_best_available_feature"] == "history_only"
    banned_tokens = ("gain", "counterfactual", "label", "partial", "teacher_real", "outcome", "best")
    leaking_feature_columns = [
        key
        for key in feature_rows[0]
        if key.startswith("feature_") and any(token in key for token in banned_tokens)
    ]
    assert leaking_feature_columns == []


def test_v29_online_gate_features_merge_dynamic_forward_rows(tmp_path: Path) -> None:
    run_dir = _make_v29_run(
        tmp_path,
        run_id="online_dynamic_feature_like",
        positive_windows=1,
        min_partial_mse=-0.33,
        min_partial_mae=-0.37,
        counterfactual_margin=-0.01,
        validation_fh_mse=0.46,
        validation_history_mse=0.79,
        validation_partial_mse=-0.33,
        validation_partial_mae=-0.37,
        window0_fh_mse=0.46,
        window0_history_mse=0.79,
        window0_partial_mse=-0.33,
        window0_counterfactual_margin=-0.01,
    )
    _write_csv(
        run_dir / "stage3_v29_online_gate_dynamic_features.csv",
        [
            {
                "target_name": "raw_residual_vector",
                "validation_window_id": "0",
                "feature_dynamic_status": "available_from_model_forward",
                "feature_dynamic_sample_count": "12",
                "feature_dynamic_alpha_entropy_mean": "0.25",
                "feature_dynamic_servable_change_norm_mean": "1.5",
            }
        ],
    )
    ranked, skipped = select_runs([run_dir])

    feature_rows = build_online_gate_feature_rows(ranked)

    assert skipped == []
    assert feature_rows[0]["feature_dynamic_signal_status"] == "available_from_model_forward"
    assert feature_rows[0]["feature_dynamic_sample_count"] == 12
    assert feature_rows[0]["feature_dynamic_alpha_entropy_mean"] == 0.25
    assert feature_rows[0]["feature_dynamic_servable_change_norm_mean"] == 1.5


def test_v29_selector_direct_script_invocation_writes_outputs(tmp_path: Path) -> None:
    run_dir = _make_v29_run(
        tmp_path,
        run_id="epoch1_like",
        positive_windows=2,
        min_partial_mse=-0.346,
        min_partial_mae=-0.436,
        counterfactual_margin=-0.011,
        validation_fh_mse=0.736,
        validation_history_mse=0.663,
        validation_partial_mse=0.073,
        validation_partial_mae=-0.023,
        window0_fh_mse=0.472,
        window0_history_mse=0.796,
        window0_partial_mse=-0.324,
        window0_counterfactual_margin=0.001,
    )
    out_dir = tmp_path / "selection"

    result = subprocess.run(
        [
            sys.executable,
            "train/analyze_stage3_v29_tuning_selection.py",
            "--run-dir",
            str(run_dir),
            "--out-dir",
            str(out_dir),
            "--prefix",
            "sample",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert (out_dir / "sample.csv").exists()
    assert (out_dir / "sample.md").exists()
    assert (out_dir / "sample_windows.csv").exists()
    assert (out_dir / "sample_windows.md").exists()
    assert (out_dir / "sample_feature_mode_policy.csv").exists()
    assert (out_dir / "sample_feature_mode_policy.md").exists()
    assert (out_dir / "sample_mode_gate_labels.csv").exists()
    assert (out_dir / "sample_mode_gate_labels.md").exists()
    assert (out_dir / "sample_online_gate_features.csv").exists()
    assert (out_dir / "sample_online_gate_features.md").exists()
    assert (out_dir / "sample_skipped.csv").exists()
