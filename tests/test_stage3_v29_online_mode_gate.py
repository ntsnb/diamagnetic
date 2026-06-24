from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train.stage3_v29_online_mode_gate import (
    OnlineModeGatePolicy,
    apply_online_mode_gate_policy,
    build_online_mode_gate_artifacts,
    evaluate_online_mode_gate_serving_outcomes,
    load_served_online_mode_gate_policy,
    load_online_mode_gate_policy,
    promote_online_mode_gate_serving_package,
    select_online_feature_columns,
    split_rows_for_heldout_windows,
)
from train.online_loop_stage3_v29 import select_stage3_v29_serving_mode


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


def _rows() -> list[dict[str, object]]:
    base = {
        "strict_rank": "1",
        "run_id": "gate_train_like",
        "target_name": "raw_residual_vector",
        "feature_dataset": "weather",
        "feature_seed": "42",
        "feature_candidate_mode": "fluctuation_plus_history",
        "feature_available_modes": "fluctuation_plus_history;history_only",
        "feature_available_mode_count": "2",
        "feature_has_history_mode": "True",
        "feature_signal_fit_split": "train",
        "feature_history_fit_split": "train",
        "feature_dynamic_signal_status": "available_from_model_forward",
        "feature_counterfactual_margin_should_not_enter": "999",
        "feature_best_gain_should_not_enter": "999",
        "oracle_label_source": "validation_window_outcome_oracle",
    }
    rows: list[dict[str, object]] = []
    for window, norm, label in [
        (0, 0.10, "history_only"),
        (1, 1.30, "fluctuation_plus_history"),
        (2, 0.20, "history_only"),
        (3, 1.40, "fluctuation_plus_history"),
        (4, 0.15, "history_only"),
    ]:
        row = dict(base)
        row.update(
            {
                "validation_window_id": str(window),
                "feature_window_position_index": str(window),
                "feature_window_start_sample": str(window * 12),
                "feature_window_end_sample": str((window + 1) * 12),
                "feature_window_sample_count": "12",
                "feature_dynamic_servable_change_norm_mean": str(norm),
                "feature_dynamic_alpha_entropy_mean": str(1.0 - norm / 2.0),
                "oracle_label_cf_safe_feature": label,
                "oracle_label_best_available_feature": label,
            }
        )
        rows.append(row)
    return rows


def _rolling_outcome_rows() -> list[dict[str, object]]:
    gains = {
        0: {"fluctuation_plus_history": -0.30, "history_only": 0.40},
        1: {"fluctuation_plus_history": 0.80, "history_only": 0.20},
        2: {"fluctuation_plus_history": -0.20, "history_only": 0.30},
        3: {"fluctuation_plus_history": 0.90, "history_only": 0.10},
        4: {"fluctuation_plus_history": -0.40, "history_only": 0.50},
    }
    rows: list[dict[str, object]] = []
    for window, by_mode in gains.items():
        for mode, mse_gain in by_mode.items():
            rows.append(
                {
                    "run_id": "gate_train_like",
                    "dataset": "weather",
                    "seed": "42",
                    "split": f"validation_window_{window}",
                    "validation_window_id": str(window),
                    "validation_window_start_sample": str(window * 12),
                    "validation_window_end_sample": str((window + 1) * 12),
                    "target_name": "raw_residual_vector",
                    "target_role": "primary",
                    "feature_mode": mode,
                    "teacher_real_mse_gain": str(mse_gain),
                    "teacher_real_mae_gain": str(mse_gain / 2.0),
                    "teacher_real_vs_best_counterfactual_mse_gain_margin": str(max(mse_gain, 0.0)),
                }
            )
    return rows


def test_split_rows_for_heldout_windows_uses_future_windows_only() -> None:
    train_rows, eval_rows, split = split_rows_for_heldout_windows(_rows(), heldout_tail_windows=2)

    assert split["heldout_start_window"] == 3
    assert [row["validation_window_id"] for row in train_rows] == ["0", "1", "2"]
    assert [row["validation_window_id"] for row in eval_rows] == ["3", "4"]


def test_online_mode_gate_uses_only_nonleaky_feature_columns_and_scores_heldout() -> None:
    artifacts = build_online_mode_gate_artifacts(
        _rows(),
        label_column="oracle_label_cf_safe_feature",
        heldout_tail_windows=2,
        min_confidence=0.0,
        fallback_mode="history_only",
    )

    assert artifacts.summary["status"] == "evaluated"
    assert artifacts.summary["train_rows"] == 3
    assert artifacts.summary["eval_rows"] == 2
    assert artifacts.summary["heldout_accuracy"] == 1.0
    assert "feature_dynamic_servable_change_norm_mean" in artifacts.model["feature_columns"]
    assert "feature_counterfactual_margin_should_not_enter" not in artifacts.model["feature_columns"]
    assert "feature_best_gain_should_not_enter" not in artifacts.model["feature_columns"]
    assert all(not key.startswith("oracle_label_") for key in artifacts.model["feature_columns"])
    assert [row["split"] for row in artifacts.prediction_rows] == ["train", "train", "train", "heldout", "heldout"]


def test_online_mode_gate_cli_writes_model_predictions_and_report(tmp_path: Path) -> None:
    features_csv = tmp_path / "online_gate_features.csv"
    out_dir = tmp_path / "gate"
    _write_csv(features_csv, _rows())

    result = subprocess.run(
        [
            sys.executable,
            "train/stage3_v29_online_mode_gate.py",
            "--features-csv",
            str(features_csv),
            "--out-dir",
            str(out_dir),
            "--prefix",
            "sample_gate",
            "--heldout-tail-windows",
            "2",
            "--min-confidence",
            "0.0",
            "--overwrite",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads((out_dir / "sample_gate_summary.json").read_text(encoding="utf-8"))
    model = json.loads((out_dir / "sample_gate_model.json").read_text(encoding="utf-8"))
    predictions = list(csv.DictReader((out_dir / "sample_gate_predictions.csv").open(encoding="utf-8")))
    assert summary["status"] == "evaluated"
    assert summary["heldout_accuracy"] == 1.0
    assert model["label_column"] == "oracle_label_cf_safe_feature"
    assert len(predictions) == 5
    assert (out_dir / "sample_gate.md").exists()


def test_online_mode_gate_serving_policy_fails_closed_without_oracle_labels() -> None:
    artifacts = build_online_mode_gate_artifacts(
        _rows(),
        label_column="oracle_label_cf_safe_feature",
        heldout_tail_windows=2,
        min_confidence=0.35,
        fallback_mode="history_only",
    )
    serving_row = dict(_rows()[3])
    serving_row.pop("oracle_label_cf_safe_feature")
    serving_row.pop("oracle_label_best_available_feature")
    serving_row.pop("oracle_label_source")
    serving_row["feature_dynamic_servable_change_norm_mean"] = "0.15"

    decision = apply_online_mode_gate_policy(
        serving_row,
        model=artifacts.model,
        min_confidence=0.35,
        fallback_mode="history_only",
    )

    assert decision["selected_mode"] == "history_only"
    assert decision["used_fallback"] is True
    assert decision["oracle_label_columns_seen"] == []
    assert decision["policy_status"] == "applied"


def test_online_mode_gate_policy_loads_for_v29_serving_helper(tmp_path: Path) -> None:
    artifacts = build_online_mode_gate_artifacts(
        _rows(),
        label_column="oracle_label_cf_safe_feature",
        heldout_tail_windows=2,
        min_confidence=0.35,
        fallback_mode="history_only",
    )
    out_dir = tmp_path / "gate"
    from train.stage3_v29_online_mode_gate import write_online_mode_gate_artifacts

    write_online_mode_gate_artifacts(out_dir=out_dir, prefix="sample_gate", artifacts=artifacts, overwrite=True)
    policy = load_online_mode_gate_policy(out_dir=out_dir, prefix="sample_gate")

    decision = select_stage3_v29_serving_mode(_rows()[4], online_mode_gate_policy=policy)

    assert decision["selected_mode"] == "history_only"
    assert decision["fallback_mode"] == "history_only"
    assert decision["source_prefix"] == "sample_gate"


def test_online_mode_gate_serving_outcome_replay_scores_selected_modes() -> None:
    artifacts = build_online_mode_gate_artifacts(
        _rows(),
        label_column="oracle_label_cf_safe_feature",
        heldout_tail_windows=2,
        min_confidence=0.0,
        fallback_mode="history_only",
    )
    policy = OnlineModeGatePolicy(
        summary=artifacts.summary,
        model=artifacts.model,
        source_prefix="synthetic_gate",
        source_dir="memory",
    )

    outcome_artifacts = evaluate_online_mode_gate_serving_outcomes(
        _rows(),
        rolling_outcome_rows=_rolling_outcome_rows(),
        policy=policy,
        candidate_mode="fluctuation_plus_history",
        history_mode="history_only",
    )

    assert outcome_artifacts.summary["status"] == "evaluated"
    assert outcome_artifacts.summary["evaluated_rows"] == 5
    assert outcome_artifacts.summary["selected_harmful_window_count"] == 0
    assert outcome_artifacts.summary["candidate_harmful_window_count"] == 3
    assert outcome_artifacts.summary["selected_mean_mse_gain"] > outcome_artifacts.summary["candidate_mean_mse_gain"]
    assert outcome_artifacts.summary["selected_mean_mse_gain"] > outcome_artifacts.summary["history_mean_mse_gain"]
    assert all(row["oracle_label_columns_seen"] == "" for row in outcome_artifacts.rows)


def test_online_mode_gate_outcome_cli_writes_serving_replay_artifacts(tmp_path: Path) -> None:
    features_csv = tmp_path / "online_gate_features.csv"
    run_dir = tmp_path / "run" / "weather" / "seed_42"
    gate_dir = tmp_path / "gate"
    out_dir = tmp_path / "outcome"
    _write_csv(features_csv, _rows())
    _write_csv(run_dir / "teacher_only_rolling_validation_window_summary_v29.csv", _rolling_outcome_rows())
    artifacts = build_online_mode_gate_artifacts(
        _rows(),
        label_column="oracle_label_cf_safe_feature",
        heldout_tail_windows=2,
        min_confidence=0.0,
        fallback_mode="history_only",
    )
    from train.stage3_v29_online_mode_gate import write_online_mode_gate_artifacts

    write_online_mode_gate_artifacts(out_dir=gate_dir, prefix="sample_gate", artifacts=artifacts, overwrite=True)

    result = subprocess.run(
        [
            sys.executable,
            "train/stage3_v29_online_mode_gate.py",
            "--evaluate-outcomes",
            "--features-csv",
            str(features_csv),
            "--run-dir",
            str(run_dir),
            "--policy-dir",
            str(gate_dir),
            "--policy-prefix",
            "sample_gate",
            "--out-dir",
            str(out_dir),
            "--prefix",
            "sample_outcome",
            "--overwrite",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads((out_dir / "sample_outcome_summary.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((out_dir / "sample_outcome_rows.csv").open(encoding="utf-8")))
    assert summary["status"] == "evaluated"
    assert summary["selected_harmful_window_count"] == 0
    assert len(rows) == 5
    assert (out_dir / "sample_outcome.md").exists()


def test_promote_online_mode_gate_serving_package_writes_manifest_and_loads_policy(tmp_path: Path) -> None:
    features_csv = tmp_path / "online_gate_features.csv"
    run_dir = tmp_path / "run" / "weather" / "seed_42"
    gate_dir = tmp_path / "gate"
    served_dir = tmp_path / "served"
    _write_csv(features_csv, _rows())
    _write_csv(run_dir / "teacher_only_rolling_validation_window_summary_v29.csv", _rolling_outcome_rows())
    artifacts = build_online_mode_gate_artifacts(
        _rows(),
        label_column="oracle_label_cf_safe_feature",
        heldout_tail_windows=2,
        min_confidence=0.0,
        fallback_mode="history_only",
    )
    from train.stage3_v29_online_mode_gate import write_online_mode_gate_artifacts, write_online_mode_gate_outcome_artifacts

    write_online_mode_gate_artifacts(out_dir=gate_dir, prefix="sample_gate", artifacts=artifacts, overwrite=True)
    policy = load_online_mode_gate_policy(out_dir=gate_dir, prefix="sample_gate")
    outcome_artifacts = evaluate_online_mode_gate_serving_outcomes(
        _rows(),
        rolling_outcome_rows=_rolling_outcome_rows(),
        policy=policy,
        candidate_mode="fluctuation_plus_history",
        history_mode="history_only",
    )
    write_online_mode_gate_outcome_artifacts(
        out_dir=gate_dir,
        prefix="sample_gate_outcome",
        artifacts=outcome_artifacts,
        overwrite=True,
    )

    manifest = promote_online_mode_gate_serving_package(
        policy_dir=gate_dir,
        policy_prefix="sample_gate",
        outcome_prefix="sample_gate_outcome",
        out_dir=served_dir,
        package_id="sample_served_gate",
        overwrite=True,
    )
    loaded_policy = load_served_online_mode_gate_policy(served_dir / "sample_served_gate_manifest.json")

    assert manifest["admission_pass"] is True
    assert manifest["servable_scope"] == "mode_selection_only_no_generator_unlock"
    assert manifest["artifacts"]["policy_model"]["sha256"] != ""
    assert loaded_policy is not None
    decision = select_stage3_v29_serving_mode(_rows()[4], online_mode_gate_policy=loaded_policy)
    assert decision["source_prefix"] == "sample_gate"


def test_served_online_mode_gate_loader_fails_closed_on_hash_mismatch(tmp_path: Path) -> None:
    gate_dir = tmp_path / "gate"
    served_dir = tmp_path / "served"
    artifacts = build_online_mode_gate_artifacts(
        _rows(),
        label_column="oracle_label_cf_safe_feature",
        heldout_tail_windows=2,
        min_confidence=0.0,
        fallback_mode="history_only",
    )
    from train.stage3_v29_online_mode_gate import write_online_mode_gate_artifacts, write_online_mode_gate_outcome_artifacts

    write_online_mode_gate_artifacts(out_dir=gate_dir, prefix="sample_gate", artifacts=artifacts, overwrite=True)
    policy = load_online_mode_gate_policy(out_dir=gate_dir, prefix="sample_gate")
    outcome_artifacts = evaluate_online_mode_gate_serving_outcomes(
        _rows(),
        rolling_outcome_rows=_rolling_outcome_rows(),
        policy=policy,
        candidate_mode="fluctuation_plus_history",
        history_mode="history_only",
    )
    write_online_mode_gate_outcome_artifacts(
        out_dir=gate_dir,
        prefix="sample_gate_outcome",
        artifacts=outcome_artifacts,
        overwrite=True,
    )
    promote_online_mode_gate_serving_package(
        policy_dir=gate_dir,
        policy_prefix="sample_gate",
        outcome_prefix="sample_gate_outcome",
        out_dir=served_dir,
        package_id="sample_served_gate",
        overwrite=True,
    )
    (gate_dir / "sample_gate_model.json").write_text('{"tampered": true}', encoding="utf-8")

    assert load_served_online_mode_gate_policy(served_dir / "sample_served_gate_manifest.json") is None
