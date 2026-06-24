from __future__ import annotations

import csv
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _load_preflight_module():
    try:
        return importlib.import_module("train.stage3_v29_generator_unlock_preflight")
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing Stage III V29 generator-unlock preflight module: {exc}")


def _served_manifest(*, admission_pass: bool = True) -> dict[str, object]:
    return {
        "protocol_version": "v29_served_online_mode_gate_manifest_v1",
        "package_id": "unit_mode_gate",
        "servable_scope": "mode_selection_only_no_generator_unlock",
        "admission_pass": admission_pass,
        "admission_reasons": [] if admission_pass else ["missing_outcome_rows"],
    }


def _passing_selector_row() -> dict[str, object]:
    return {
        "strict_rank": 1,
        "run_id": "generator_ready",
        "selector_metric_complete": True,
        "selection_label": "generator_unlocked_candidate",
        "target_gate_pass": True,
        "any_primary_gate_pass": True,
        "teacher_abort_written": False,
        "generator_unlocked": True,
        "stop_reasons": "",
        "positive_partial_window_count": 5,
        "required_positive_partial_window_count": 4,
        "min_partial_mse_gain_vs_history": -0.01,
        "min_partial_mae_gain_vs_history": -0.02,
        "counterfactual_margin": 0.02,
        "window0_fh_mse_gain": 0.05,
    }


def _blocked_selector_row() -> dict[str, object]:
    return {
        "strict_rank": 1,
        "run_id": "current_gate_negative",
        "selector_metric_complete": True,
        "selection_label": "gate_near_diagnostic",
        "target_gate_pass": False,
        "any_primary_gate_pass": False,
        "teacher_abort_written": True,
        "generator_unlocked": False,
        "stop_reasons": "do_not_train_generator;no_primary_target_passed",
        "positive_partial_window_count": 2,
        "required_positive_partial_window_count": 4,
        "min_partial_mse_gain_vs_history": -0.346,
        "min_partial_mae_gain_vs_history": -0.436,
        "counterfactual_margin": -0.011,
        "window0_fh_mse_gain": 0.472,
    }


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


def test_generator_unlock_preflight_passes_when_teacher_gate_and_mode_gate_are_ready() -> None:
    module = _load_preflight_module()

    report = module.build_generator_unlock_preflight(
        [_passing_selector_row()],
        _served_manifest(admission_pass=True),
        top_k=1,
    )

    assert report["preflight_pass"] is True
    assert report["decision"] == "allow_generator_unlock_experiment"
    assert report["candidate_count"] == 1
    assert report["mode_gate_prerequisite_pass"] is True
    assert report["candidate_reports"][0]["blockers"] == []


def test_generator_unlock_preflight_reports_current_gate_negative_blockers() -> None:
    module = _load_preflight_module()

    report = module.build_generator_unlock_preflight(
        [_blocked_selector_row()],
        _served_manifest(admission_pass=True),
        top_k=1,
    )

    blockers = set(report["candidate_reports"][0]["blockers"])
    assert report["preflight_pass"] is False
    assert report["decision"] == "do_not_train_generator"
    assert {
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
    }.issubset(blockers)


def test_generator_unlock_preflight_artifact_write_is_no_overwrite_by_default(tmp_path: Path) -> None:
    module = _load_preflight_module()
    report = module.build_generator_unlock_preflight(
        [_passing_selector_row()],
        _served_manifest(admission_pass=True),
        top_k=1,
    )

    paths = module.write_generator_unlock_preflight_artifacts(
        out_dir=tmp_path,
        prefix="unit_preflight",
        report=report,
    )

    assert Path(paths["json"]).exists()
    assert Path(paths["markdown"]).exists()
    with pytest.raises(FileExistsError):
        module.write_generator_unlock_preflight_artifacts(
            out_dir=tmp_path,
            prefix="unit_preflight",
            report=report,
        )


def test_generator_unlock_preflight_cli_writes_blocking_report(tmp_path: Path) -> None:
    selector_path = tmp_path / "selector.csv"
    manifest_path = tmp_path / "manifest.json"
    out_dir = tmp_path / "preflight"
    _write_csv(selector_path, [_blocked_selector_row()])
    manifest_path.write_text(json.dumps(_served_manifest(admission_pass=True)), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "train/stage3_v29_generator_unlock_preflight.py",
            "--selector-csv",
            str(selector_path),
            "--served-manifest",
            str(manifest_path),
            "--out-dir",
            str(out_dir),
            "--prefix",
            "unit_blocked",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((out_dir / "unit_blocked.json").read_text(encoding="utf-8"))
    assert payload["preflight_pass"] is False
    assert "teacher_abort_written" in payload["candidate_reports"][0]["blockers"]
    assert (out_dir / "unit_blocked.md").exists()
