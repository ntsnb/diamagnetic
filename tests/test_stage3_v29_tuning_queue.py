from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train.plan_stage3_v29_tuning_queue import (
    build_tuning_queue,
    write_tuning_queue_artifacts,
)


def _write_config(path: Path, *, run_id: str, overrides: dict[str, float] | None = None) -> None:
    overrides = overrides or {}
    invariant_lines = "\n".join(f"  {key}: {value}" for key, value in overrides.items())
    path.write_text(
        "\n".join(
            [
                "seed: 42",
                "data:",
                "  path: dataset/weather.csv",
                "stage3_v2:",
                "  protocol_version: v29_expert_local_invariant_release_probe",
                "  log_root: logs/s3v29expert_tuning",
                "  checkpoint_root: ckpt/s3v29expert_tuning",
                f"  run_id: {run_id}",
                "  validation_windows: 5",
                "online:",
                "  stage3:",
                "    version: v29_expert_local_invariant_release_probe",
                "    drift_source: pooled_servable_change",
                "    signal_mode: fluctuation_delta",
                "invariant:",
                invariant_lines,
                "",
            ]
        ),
        encoding="utf-8",
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


def _write_completed_run(root: Path, *, run_id: str, gate_pass: bool = False, dataset: str = "weather") -> Path:
    run_dir = root / run_id / dataset / "seed_42"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest_v29.json").write_text(
        json.dumps(
            {
                "protocol_version": "v29_expert_local_invariant_release_probe",
                "run_id": run_id,
                "dataset": dataset,
                "seed": 42,
                "stage3_stop_reasons": ["do_not_train_generator", "no_primary_target_passed"],
                "stage3_generator_candidate_count": 0,
                "stage3_generator_enabled": True,
            }
        ),
        encoding="utf-8",
    )
    _write_csv(
        run_dir / "teacher_only_v29_target_gate.csv",
        [
            {
                "target_name": "raw_residual_vector",
                "target_role": "primary",
                "primary_gate_pass": str(gate_pass),
                "positive_partial_window_count": 2,
                "required_positive_partial_window_count": 4,
                "window_count": 5,
                "min_partial_mse_gain_vs_history": -0.34,
                "counterfactual_margin": -0.01,
                "counterfactual_pass": "False",
            }
        ],
    )
    return run_dir


def _write_completed_run_with_resolved_config(
    root: Path,
    *,
    run_id: str,
    resolved_invariant: dict[str, float],
    dataset: str = "weather",
) -> Path:
    run_dir = _write_completed_run(root, run_id=run_id, dataset=dataset)
    resolved_path = run_dir / "resolved_config.json"
    resolved_path.write_text(
        json.dumps({"train": {"lr": 0.00038}, "invariant": resolved_invariant}),
        encoding="utf-8",
    )
    manifest = json.loads((run_dir / "run_manifest_v29.json").read_text(encoding="utf-8"))
    manifest["resolved_config_json"] = {"resolved_config_path": str(resolved_path)}
    (run_dir / "run_manifest_v29.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


def test_tuning_queue_marks_selector_refresh_before_new_training(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    _write_config(
        config_dir / "stage3_v29_expert_local_invariant_release_probe_tune_release_first_043_only.yaml",
        run_id="release_first_043",
        overrides={"lambda_release_first_increment": 0.43},
    )
    _write_config(
        config_dir / "stage3_v29_expert_local_invariant_release_probe_tune_release_null_0060_only.yaml",
        run_id="release_null_0060",
        overrides={"lambda_release_null": 0.06},
    )
    _write_config(
        config_dir / "stage3_v29_expert_local_invariant_release_probe_tune_lr_00036_only.yaml",
        run_id="lr_00036",
    )

    log_root = tmp_path / "logs"
    _write_completed_run(log_root, run_id="release_first_043")
    _write_completed_run(log_root, run_id="release_null_0060")
    selector_csv = tmp_path / "selector.csv"
    _write_csv(selector_csv, [{"run_id": "release_first_043", "strict_rank": 1}])

    rows = build_tuning_queue(
        config_paths=sorted(config_dir.glob("*.yaml")),
        log_roots=[log_root],
        selector_csv=selector_csv,
    )

    by_run = {row["run_id"]: row for row in rows}
    assert by_run["release_first_043"]["status"] == "covered_by_selector"
    assert by_run["release_null_0060"]["status"] == "completed_needs_selector_refresh"
    assert by_run["lr_00036"]["status"] == "queued_not_run"
    assert by_run["release_null_0060"]["next_action"] == "refresh_selector_before_training"
    assert by_run["release_null_0060"]["priority_band"] < by_run["lr_00036"]["priority_band"]
    assert by_run["release_first_043"]["candidate_axis"] == "release_first"
    assert by_run["release_null_0060"]["candidate_axis"] == "release_null"
    assert by_run["lr_00036"]["candidate_axis"] == "learning_rate"


def test_tuning_queue_uses_filename_axis_and_resolved_config_signature(tmp_path: Path) -> None:
    config_path = tmp_path / "stage3_v29_expert_local_invariant_release_probe_tune_cf_shuffle_007_only.yaml"
    _write_config(
        config_path,
        run_id="manual_cf007",
        overrides={
            "lambda_release_first_increment": 0.4,
            "lambda_release_null": 0.05,
            "lambda_counterfactual_shuffle": 0.07,
        },
    )
    log_root = tmp_path / "logs"
    _write_completed_run_with_resolved_config(
        log_root,
        run_id="v29cf007_weather_s42_b60_01",
        resolved_invariant={
            "lambda_release_first_increment": 0.4,
            "lambda_release_null": 0.05,
            "lambda_counterfactual_shuffle": 0.07,
        },
    )

    rows = build_tuning_queue(config_paths=[config_path], log_roots=[log_root])

    assert rows[0]["candidate_axis"] == "counterfactual_shuffle"
    assert rows[0]["status"] == "completed_needs_selector_refresh"
    assert rows[0]["matched_completed_run_ids"] == "v29cf007_weather_s42_b60_01"


def test_tuning_queue_does_not_match_dynamic_smoke_by_shared_signature(tmp_path: Path) -> None:
    config_path = tmp_path / "stage3_v29_expert_local_invariant_release_probe_tune_stability_release_deconstrained.yaml"
    _write_config(
        config_path,
        run_id="manual_stability",
        overrides={
            "lambda_release_first_increment": 0.4,
            "lambda_release_null": 0.05,
        },
    )
    logs_root = tmp_path / "logs"
    dynamic_run = _write_completed_run_with_resolved_config(
        logs_root / "s3v29expert_dynamic_smoke",
        run_id="v29dynamic_manifest_smoke",
        resolved_invariant={
            "lambda_release_first_increment": 0.4,
            "lambda_release_null": 0.05,
        },
    )
    tuning_run = _write_completed_run_with_resolved_config(
        logs_root / "s3v29expert_tuning_screen",
        run_id="v29stable_weather_s42_b60_01",
        resolved_invariant={
            "lambda_release_first_increment": 0.4,
            "lambda_release_null": 0.05,
        },
    )

    rows = build_tuning_queue(config_paths=[config_path], log_roots=[logs_root])

    assert rows[0]["matched_completed_run_ids"] == "v29stable_weather_s42_b60_01"
    assert rows[0]["completed_run_dir"] == str(tuning_run)
    assert str(dynamic_run) not in rows[0]["matched_completed_run_dirs"]


def test_tuning_queue_does_not_match_different_dataset_by_shared_signature(tmp_path: Path) -> None:
    config_path = tmp_path / "stage3_v29_expert_local_invariant_release_probe_tune_stability_release_deconstrained.yaml"
    _write_config(
        config_path,
        run_id="manual_stability",
        overrides={
            "lambda_release_first_increment": 0.4,
            "lambda_release_null": 0.05,
        },
    )
    logs_root = tmp_path / "logs"
    panama_run = _write_completed_run_with_resolved_config(
        logs_root / "s3v29expert_tuning_screen",
        run_id="v29stable_panama_s42_b60_01",
        dataset="panama",
        resolved_invariant={
            "lambda_release_first_increment": 0.4,
            "lambda_release_null": 0.05,
        },
    )
    weather_run = _write_completed_run_with_resolved_config(
        logs_root / "s3v29expert_tuning_screen",
        run_id="v29stable_weather_s42_b60_01",
        dataset="weather",
        resolved_invariant={
            "lambda_release_first_increment": 0.4,
            "lambda_release_null": 0.05,
        },
    )

    rows = build_tuning_queue(config_paths=[config_path], log_roots=[logs_root])

    assert rows[0]["matched_completed_run_ids"] == "v29stable_weather_s42_b60_01"
    assert rows[0]["completed_run_dir"] == str(weather_run)
    assert str(panama_run) not in rows[0]["matched_completed_run_dirs"]


def test_tuning_queue_writes_non_overwriting_csv_and_report(tmp_path: Path) -> None:
    config_path = tmp_path / "stage3_v29_expert_local_invariant_release_probe_tune_release_first_043_only.yaml"
    _write_config(
        config_path,
        run_id="release_first_043",
        overrides={"lambda_release_first_increment": 0.43},
    )
    rows = build_tuning_queue(config_paths=[config_path], log_roots=[tmp_path / "missing_logs"])
    out_dir = tmp_path / "artifacts"

    written = write_tuning_queue_artifacts(rows, out_dir=out_dir, prefix="queue")

    assert written["csv"].exists()
    assert written["md"].exists()
    csv_text = written["csv"].read_text(encoding="utf-8")
    md_text = written["md"].read_text(encoding="utf-8")
    assert "release_first_043" in csv_text
    assert "queued_not_run" in md_text
    assert "Do not treat queued rows as performance evidence" in md_text

    try:
        write_tuning_queue_artifacts(rows, out_dir=out_dir, prefix="queue")
    except FileExistsError:
        pass
    else:
        raise AssertionError("queue artifacts must refuse accidental overwrite")
