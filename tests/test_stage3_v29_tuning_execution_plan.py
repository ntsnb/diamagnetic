from __future__ import annotations

import csv
import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _load_module():
    try:
        return importlib.import_module("train.plan_stage3_v29_tuning_execution")
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing V29 tuning execution planner module: {exc}")


def _matrix_rows(config_dir: Path) -> list[dict[str, object]]:
    config_dir.mkdir(parents=True, exist_ok=True)
    first = config_dir / "stage3_v29_candidate_01.yaml"
    second = config_dir / "stage3_v29_candidate_02.yaml"
    first.write_text("stage3_v2:\n  run_id: candidate_01\n", encoding="utf-8")
    second.write_text("stage3_v2:\n  run_id: candidate_02\n", encoding="utf-8")
    return [
        {
            "candidate_id": "01_history_guarded_release_floor",
            "priority": 10,
            "axis": "history_guarded_release_floor",
            "run_id": "candidate_01",
            "config_name": first.name,
        },
        {
            "candidate_id": "02_counterfactual_margin_micro_repair",
            "priority": 20,
            "axis": "counterfactual_margin_micro_repair",
            "run_id": "candidate_02",
            "config_name": second.name,
        },
    ]


def _write_matrix_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def test_execution_plan_builds_non_overwriting_bounded_commands(tmp_path: Path) -> None:
    module = _load_module()
    config_dir = tmp_path / "configs"
    rows = _matrix_rows(config_dir)

    plan = module.build_tuning_execution_plan(
        matrix_rows=rows,
        config_dir=config_dir,
        out_root=tmp_path / "logs" / "s3v29teacher_gate_exec",
        checkpoint_root=tmp_path / "ckpt" / "s3v29teacher_gate_exec",
        dataset="weather",
        seed=42,
    )

    assert len(plan) == 2
    first = plan[0]
    assert first["execution_status"] == "ready_to_run"
    assert first["overwrite_policy"] == "no_overwrite"
    assert "--overwrite" not in first["command"]
    assert "--out-root" in first["command"]
    assert "--checkpoint-root" in first["command"]
    assert "--max-generator-batches 60" in first["command"]
    assert Path(str(first["run_output_dir"])).parts[-3:] == ("candidate_01", "weather", "seed_42")


def test_execution_plan_omits_weather_dataset_override_but_keeps_output_label(tmp_path: Path) -> None:
    module = _load_module()
    config_dir = tmp_path / "configs"
    rows = _matrix_rows(config_dir)

    plan = module.build_tuning_execution_plan(
        matrix_rows=rows,
        config_dir=config_dir,
        out_root=tmp_path / "logs",
        checkpoint_root=tmp_path / "ckpt",
        dataset="weather",
        seed=42,
    )

    assert " --dataset weather " not in plan[0]["command"]
    assert Path(str(plan[0]["run_output_dir"])).parts[-3:] == ("candidate_01", "weather", "seed_42")


def test_execution_plan_keeps_supported_public_dataset_override(tmp_path: Path) -> None:
    module = _load_module()
    config_dir = tmp_path / "configs"
    rows = _matrix_rows(config_dir)

    plan = module.build_tuning_execution_plan(
        matrix_rows=rows,
        config_dir=config_dir,
        out_root=tmp_path / "logs",
        checkpoint_root=tmp_path / "ckpt",
        dataset="panama",
        seed=42,
    )

    assert " --dataset panama " in plan[0]["command"]
    assert Path(str(plan[0]["run_output_dir"])).parts[-3:] == ("candidate_01", "panama", "seed_42")


def test_execution_plan_blocks_existing_output_paths(tmp_path: Path) -> None:
    module = _load_module()
    config_dir = tmp_path / "configs"
    rows = _matrix_rows(config_dir)
    existing = tmp_path / "logs" / "s3v29teacher_gate_exec" / "candidate_01" / "weather" / "seed_42"
    existing.mkdir(parents=True)

    plan = module.build_tuning_execution_plan(
        matrix_rows=rows,
        config_dir=config_dir,
        out_root=tmp_path / "logs" / "s3v29teacher_gate_exec",
        checkpoint_root=tmp_path / "ckpt" / "s3v29teacher_gate_exec",
        dataset="weather",
        seed=42,
    )

    by_run = {row["run_id"]: row for row in plan}
    assert by_run["candidate_01"]["execution_status"] == "blocked_existing_output"
    assert "run_output_dir_exists" in by_run["candidate_01"]["block_reasons"]
    assert by_run["candidate_02"]["execution_status"] == "ready_to_run"


def test_execution_plan_blocks_windows_checkpoint_paths_that_can_break_torch_save(tmp_path: Path) -> None:
    module = _load_module()
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config = config_dir / "stage3_v29_long.yaml"
    config.write_text("stage3_v2:\n  run_id: long_path\n", encoding="utf-8")
    long_run_id = "v29repair_conf015_02_" + ("counterfactual_ordering_without_floor_pressure_" * 4)
    rows = [
        {
            "candidate_id": "02_counterfactual_ordering_without_floor_pressure",
            "priority": 20,
            "axis": "counterfactual_ordering_without_floor_pressure",
            "run_id": long_run_id,
            "config_name": config.name,
        }
    ]

    plan = module.build_tuning_execution_plan(
        matrix_rows=rows,
        config_dir=config_dir,
        out_root=tmp_path / "logs" / "s3v29expert_decomposition_repair_matrix",
        checkpoint_root=tmp_path / "ckpt" / "s3v29expert_decomposition_repair_matrix",
        dataset="weather",
        seed=42,
    )

    row = plan[0]
    assert row["execution_status"] == "blocked_runtime_path_length"
    assert "checkpoint_file_path_too_long" in row["block_reasons"]
    assert int(row["max_checkpoint_file_path_length"]) > int(row["max_windows_path_length"])


def test_execution_plan_writes_artifacts_and_refuses_overwrite(tmp_path: Path) -> None:
    module = _load_module()
    config_dir = tmp_path / "configs"
    rows = _matrix_rows(config_dir)
    plan = module.build_tuning_execution_plan(
        matrix_rows=rows,
        config_dir=config_dir,
        out_root=tmp_path / "logs",
        checkpoint_root=tmp_path / "ckpt",
        dataset="weather",
        seed=42,
    )

    paths = module.write_tuning_execution_plan_artifacts(
        rows=plan,
        out_dir=tmp_path / "artifacts",
        prefix="unit_execution_plan",
    )

    assert Path(paths["csv"]).exists()
    assert Path(paths["markdown"]).exists()
    with pytest.raises(FileExistsError):
        module.write_tuning_execution_plan_artifacts(
            rows=plan,
            out_dir=tmp_path / "artifacts",
            prefix="unit_execution_plan",
        )


def test_execution_plan_cli_writes_plan_from_matrix_csv(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    rows = _matrix_rows(config_dir)
    matrix_csv = tmp_path / "matrix.csv"
    _write_matrix_csv(matrix_csv, rows)

    result = subprocess.run(
        [
            sys.executable,
            "train/plan_stage3_v29_tuning_execution.py",
            "--matrix-csv",
            str(matrix_csv),
            "--config-dir",
            str(config_dir),
            "--out-root",
            str(tmp_path / "logs"),
            "--checkpoint-root",
            str(tmp_path / "ckpt"),
            "--out-dir",
            str(tmp_path / "artifacts"),
            "--prefix",
            "unit_execution_plan",
            "--dataset",
            "weather",
            "--seed",
            "42",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "artifacts" / "unit_execution_plan.csv").exists()
    assert (tmp_path / "artifacts" / "unit_execution_plan.md").exists()
