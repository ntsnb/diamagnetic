# Stage III V29 Joint Gate Recovery Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a frozen-topology V29 Joint Gate Recovery planner that emits a short, non-overwriting matrix after HGR exhaustion, then validate it through tests and one-at-a-time bounded execution planning.

**Architecture:** Add a dedicated planner beside the existing V29 matrix planners, following the history-gate recovery planner shape but using new joint axes that are not old scalar, repair, or HGR axes. Generated configs must deep-copy the base config, change only `invariant` weights plus matrix metadata, preserve `model_v29` and `online.stage3.drift_source=pooled_servable_change`, and keep generator unlock gated by the existing preflight. Artifacts remain under new Joint Gate Recovery paths so previous V29/V28 evidence is not overwritten.

**Tech Stack:** Python 3, PyYAML, pytest, CSV/JSON artifacts, existing Stage III V29 runner and execution planner.

---

## Project Boundaries

- Do not create V30.
- Do not modify `models/factor_mechanism_v29.py`.
- Do not modify `train/run_stage3_v29_expert_local_invariant_release_probe.py`.
- Do not change `model_v29`, target module names, target names, or `online.stage3.drift_source`.
- Do not add `--overwrite` to generated candidate commands.
- Do not start full training from this plan.
- Do not unlock generator training from matrix generation.
- Do not commit changes unless the user explicitly requests a commit. This overrides the generic Superpowers commit cadence because `AGENTS.md` forbids unrequested commits.
- If a path already exists and `--overwrite` is not passed, raise `FileExistsError` rather than replacing it.

## Files

- Create: `tests/test_stage3_v29_joint_gate_recovery_matrix.py`
- Create: `train/plan_stage3_v29_joint_gate_recovery_matrix.py`
- Create directory when generating real configs: `configs/stage3_v29_joint_gate_recovery_matrix/`
- Create artifact when generating real matrix: `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_joint_gate_recovery_matrix.csv`
- Create artifact when generating real matrix: `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_joint_gate_recovery_matrix.md`
- Create artifact when generating execution plan: `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_joint_gate_recovery_execution_plan_after_01.csv`
- Create artifact when generating execution plan: `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_joint_gate_recovery_execution_plan_after_01.md`
- Modify after verified artifact generation: `docs/audit/STAGE3_V29_EXPERT_LOCAL_INVARIANT_RELEASE_IMPLEMENTATION_RECORD.md`
- Modify after verified artifact generation: `docs/audit/EXPERIMENT_EVIDENCE_LEDGER.md`
- Modify after verified artifact generation: `docs/audit/STAGE3_V29_TUNING_PRIORITY_CORRECTION_REPORT.md`

## Required Constants

Use these exact protocol and path names in tests and implementation:

```python
PROTOCOL_VERSION = "stage3_v29_joint_gate_recovery_matrix_v1"
SOURCE_DECOMPOSITION_PROTOCOL = "stage3_v29_gate_failure_decomposition_v1"
DEFAULT_CONFIG_DIR = Path("configs/stage3_v29_joint_gate_recovery_matrix")
DEFAULT_BUDGET_ARGS = (
    "--epochs 1",
    "--max-train-batches 60",
    "--max-generator-batches 60",
    "--max-eval-batches 60",
    "--validation-windows 5",
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
    "window0_safe_release_ramp",
    "late_history_dominance_bridge",
    "counterfactual_tie_breaker_floor_safe",
    "history_cap_release_balance",
}
NEW_AXES = {
    "joint_floor_cf_balance",
    "window0_preserving_late_floor",
    "real_beats_shuffle_margin",
    "history_residual_rebalance",
}
```

---

### Task 1: Write Failing Planner Tests

**Files:**
- Create: `tests/test_stage3_v29_joint_gate_recovery_matrix.py`
- Read: `tests/test_stage3_v29_history_gate_recovery_matrix.py`
- Read: `docs/superpowers/specs/2026-06-13-stage3-v29-joint-gate-recovery-design.md`

- [ ] **Step 1: Create the test module imports and fixtures**

Use this test file start:

```python
from __future__ import annotations

import csv
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

from train.plan_stage3_v29_joint_gate_recovery_matrix import (
    build_joint_gate_recovery_matrix,
    write_joint_gate_recovery_matrix_artifacts,
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
    "window0_safe_release_ramp",
    "late_history_dominance_bridge",
    "counterfactual_tie_breaker_floor_safe",
    "history_cap_release_balance",
}

NEW_AXES = {
    "joint_floor_cf_balance",
    "window0_preserving_late_floor",
    "real_beats_shuffle_margin",
    "history_residual_rebalance",
}


def _sample_config() -> dict[str, object]:
    return {
        "data": {"path": "dataset/weather.csv"},
        "model_v29": {"n_experts": 4, "repr_dim": 64, "target_modules": ["encoder"]},
        "target_modules": ["encoder"],
        "target_names": ["raw_residual_vector"],
        "online": {"stage3": {"drift_source": "pooled_servable_change"}},
        "train": {"lr": 0.00038, "epochs": 5},
        "stage3_v2": {"run_id": "v29_complete_conf015_hgr04"},
        "invariant": {
            "lambda_release_first_increment": 0.405,
            "lambda_history_after_release": 0.08,
            "lambda_history_budget_cap": 0.035,
            "lambda_counterfactual_shuffle": 0.0812,
            "counterfactual_shuffle_margin": 0.00112,
            "lambda_router_stability": 0.145,
            "lambda_anchor_preservation": 0.14,
            "lambda_support_alignment": 0.14,
            "lambda_support_conditioned_advantage": 0.08,
            "lambda_suppression_alignment": 0.1,
            "lambda_suppression_gate_sparsity": 0.025,
            "lambda_residual_release_alignment": 0.03,
            "lambda_residual_release_advantage": 0.08,
            "lambda_release_priority_advantage": 0.18,
            "lambda_release_state_advantage": 0.12,
            "lambda_local_env_consistency": 0.055,
        },
    }


def _sample_decomposition() -> dict[str, object]:
    return {
        "protocol_version": "stage3_v29_gate_failure_decomposition_v1",
        "matrix_decision": "stop_local_scalar_matrix",
        "candidate_count": 36,
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
            "counterfactual_margin_negative": 36,
            "min_partial_mse_below_floor": 36,
            "min_partial_mae_below_floor": 36,
            "positive_partial_windows_below_required": 21,
        },
        "window_summaries": [
            {
                "validation_window_id": "0",
                "row_count": 36,
                "partial_mse_floor_failures": 36,
                "partial_mae_floor_failures": 36,
                "counterfactual_negative_rows": 35,
                "real_best_rows": 1,
            },
            {
                "validation_window_id": "2",
                "row_count": 36,
                "partial_mse_floor_failures": 0,
                "partial_mae_floor_failures": 0,
                "counterfactual_negative_rows": 34,
                "real_best_rows": 2,
            },
        ],
    }


def _sample_selector_rows() -> list[dict[str, str]]:
    return [
        {
            "run_id": "v29hgr_conf015_03_cftie",
            "strict_rank": "1",
            "target_name": "raw_residual_vector",
            "feature_mode": "fluctuation_plus_history",
            "target_gate_pass": "False",
            "generator_unlocked": "False",
            "positive_partial_window_count": "2",
            "required_positive_partial_window_count": "4",
            "min_partial_mse_gain_vs_history": "-0.41351019804173306",
            "min_partial_mae_gain_vs_history": "-0.466581375695994",
            "counterfactual_margin": "-0.006009365114644338",
        },
        {
            "run_id": "v29hgr_conf015_04_histcap",
            "strict_rank": "4",
            "target_name": "raw_residual_vector",
            "feature_mode": "fluctuation_plus_history",
            "target_gate_pass": "False",
            "generator_unlocked": "False",
            "positive_partial_window_count": "2",
            "required_positive_partial_window_count": "4",
            "min_partial_mse_gain_vs_history": "-0.5100",
            "min_partial_mae_gain_vs_history": "-0.6100",
            "counterfactual_margin": "-0.0120",
        },
    ]


def _sample_window_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for window_id, partial, margin, history, real_best in [
        ("0", "-0.41351019804173306", "0.0011", "0.80", "False"),
        ("1", "-0.115", "-0.0025", "0.91", "False"),
        ("2", "0.792", "-0.006009365114644338", "0.0", "False"),
        ("3", "0.278", "-0.0011", "0.59", "False"),
        ("4", "-0.466", "0.0007", "0.99", "False"),
    ]:
        rows.append(
            {
                "run_id": "v29hgr_conf015_03_cftie",
                "strict_rank": "1",
                "target_name": "raw_residual_vector",
                "feature_mode": "fluctuation_plus_history",
                "validation_window_id": window_id,
                "partial_mse_gain_vs_history": partial,
                "partial_mae_gain_vs_history": partial,
                "counterfactual_margin": margin,
                "history_mse_gain": history,
                "fh_mse_gain": "0.5",
                "real_is_best_mode": real_best,
            }
        )
    return rows
```

- [ ] **Step 2: Add tests for matrix construction boundaries**

Append these tests:

```python
def test_joint_gate_recovery_matrix_requires_exhausted_hgr_and_uses_new_axes() -> None:
    rows = build_joint_gate_recovery_matrix(
        base_config=_sample_config(),
        decomposition_report=_sample_decomposition(),
        selector_rows=_sample_selector_rows(),
        window_rows=_sample_window_rows(),
        run_id_prefix="v29jgr_conf015",
    )

    assert len(rows) == 4
    axes = {row["axis"] for row in rows}
    assert axes == NEW_AXES
    assert axes.isdisjoint(OLD_AXES)
    assert all(row["protocol_version"] == "stage3_v29_joint_gate_recovery_matrix_v1" for row in rows)
    assert all(row["architecture_policy"] == "frozen_v29_no_model_topology_change" for row in rows)
    assert all(row["generator_policy"] == "preflight_required_do_not_unlock_from_matrix" for row in rows)
    assert all(row["source_matrix_status_required"] == "hgr_exhausted_meaningful_rows" for row in rows)
    assert all(row["known_failed_axis_guard"] == "not_old_scalar_repair_or_hgr_axis" for row in rows)
    assert any("joint_floor_counterfactual_conflict" in row["addresses_failure_axes"] for row in rows)
    assert any("real_vs_shuffle_ordering_failure" in row["addresses_failure_axes"] for row in rows)
    assert all(row["model_overrides"] == {} for row in rows)
    assert all(row["train_overrides"] == {} for row in rows)
    assert all(isinstance(row["invariant_overrides"], dict) for row in rows)
    assert all(len(row["run_id"]) <= 48 for row in rows)
    assert all(len(row["config_name"]) <= 64 for row in rows)
    assert all("--overwrite" not in row["suggested_probe_command"] for row in rows)
    assert all("--epochs 1" in row["suggested_probe_command"] for row in rows)
    assert all("--max-train-batches 60" in row["suggested_probe_command"] for row in rows)
    assert all("--max-generator-batches 60" in row["suggested_probe_command"] for row in rows)
    assert all("--max-eval-batches 60" in row["suggested_probe_command"] for row in rows)
    assert all("--validation-windows 5" in row["suggested_probe_command"] for row in rows)
```

- [ ] **Step 3: Add tests for source-report refusal**

Append these tests:

```python
def test_joint_gate_recovery_matrix_refuses_non_final_gate_positive_or_unlocked_reports() -> None:
    not_stopped = dict(_sample_decomposition())
    not_stopped["matrix_decision"] = "continue_preflight_controlled_tuning"
    assert build_joint_gate_recovery_matrix(
        base_config=_sample_config(),
        decomposition_report=not_stopped,
        selector_rows=_sample_selector_rows(),
        window_rows=_sample_window_rows(),
    ) == []

    gate_positive = dict(_sample_decomposition())
    gate_positive["gate_pass_count"] = 1
    assert build_joint_gate_recovery_matrix(
        base_config=_sample_config(),
        decomposition_report=gate_positive,
        selector_rows=_sample_selector_rows(),
        window_rows=_sample_window_rows(),
    ) == []

    generator_unlocked = dict(_sample_decomposition())
    generator_unlocked["generator_unlocked_count"] = 1
    assert build_joint_gate_recovery_matrix(
        base_config=_sample_config(),
        decomposition_report=generator_unlocked,
        selector_rows=_sample_selector_rows(),
        window_rows=_sample_window_rows(),
    ) == []

    missing_blocker = dict(_sample_decomposition())
    missing_blocker["primary_blocker_counts"] = {
        "counterfactual_margin_negative": 36,
        "min_partial_mse_below_floor": 36,
        "min_partial_mae_below_floor": 36,
        "positive_partial_windows_below_required": 0,
    }
    assert build_joint_gate_recovery_matrix(
        base_config=_sample_config(),
        decomposition_report=missing_blocker,
        selector_rows=_sample_selector_rows(),
        window_rows=_sample_window_rows(),
    ) == []
```

- [ ] **Step 4: Add tests for artifact writing and topology preservation**

Append these tests:

```python
def test_write_joint_gate_recovery_matrix_outputs_configs_without_topology_changes(tmp_path: Path) -> None:
    rows = build_joint_gate_recovery_matrix(
        base_config=_sample_config(),
        decomposition_report=_sample_decomposition(),
        selector_rows=_sample_selector_rows(),
        window_rows=_sample_window_rows(),
        run_id_prefix="v29jgr_conf015",
    )

    outputs = write_joint_gate_recovery_matrix_artifacts(
        rows=rows,
        base_config=_sample_config(),
        config_dir=tmp_path / "configs",
        out_dir=tmp_path / "artifacts",
        prefix="sample_joint_gate_recovery",
        overwrite=False,
    )

    matrix_csv = Path(outputs["csv"])
    matrix_md = Path(outputs["markdown"])
    assert matrix_csv.exists()
    assert matrix_md.exists()
    assert len(outputs["configs"]) == 4

    csv_rows = list(csv.DictReader(matrix_csv.open(encoding="utf-8")))
    assert csv_rows[0]["protocol_version"] == "stage3_v29_joint_gate_recovery_matrix_v1"
    assert "joint_floor_counterfactual_conflict" in "".join(row["addresses_failure_axes"] for row in csv_rows)

    config_payload = yaml.safe_load(Path(outputs["configs"][0]).read_text(encoding="utf-8"))
    assert config_payload["stage3_v2"]["run_id"] == csv_rows[0]["run_id"]
    assert config_payload["stage3_v2"]["joint_gate_recovery_matrix_protocol"] == "stage3_v29_joint_gate_recovery_matrix_v1"
    assert config_payload["model_v29"] == _sample_config()["model_v29"]
    assert config_payload["target_modules"] == _sample_config()["target_modules"]
    assert config_payload["target_names"] == _sample_config()["target_names"]
    assert config_payload["online"]["stage3"]["drift_source"] == "pooled_servable_change"
    assert config_payload["joint_gate_recovery_matrix"]["generator_policy"] == "preflight_required_do_not_unlock_from_matrix"
    assert config_payload["joint_gate_recovery_matrix"]["source_matrix_status_required"] == "hgr_exhausted_meaningful_rows"
    assert set(config_payload["joint_gate_recovery_matrix"]["invariant_overrides"]) <= set(config_payload["invariant"])

    markdown = matrix_md.read_text(encoding="utf-8")
    assert "Joint Gate Recovery" in markdown
    assert "does not train the generator" in markdown
    assert "does not change V29 model topology" in markdown
    assert "not manuscript performance evidence" in markdown

    with pytest.raises(FileExistsError):
        write_joint_gate_recovery_matrix_artifacts(
            rows=rows,
            base_config=_sample_config(),
            config_dir=tmp_path / "configs",
            out_dir=tmp_path / "artifacts",
            prefix="sample_joint_gate_recovery",
            overwrite=False,
        )
```

- [ ] **Step 5: Add a CLI smoke test**

Append this test:

```python
def test_joint_gate_recovery_matrix_cli_writes_artifacts(tmp_path: Path) -> None:
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

    result = subprocess.run(
        [
            sys.executable,
            "train/plan_stage3_v29_joint_gate_recovery_matrix.py",
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
            "cli_joint_gate_recovery",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "artifacts" / "cli_joint_gate_recovery.csv").exists()
    assert (tmp_path / "artifacts" / "cli_joint_gate_recovery.md").exists()
```

- [ ] **Step 6: Run the new test and verify it fails before implementation**

Run:

```powershell
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_stage3_v29_joint_gate_recovery_matrix.py -q
```

Expected result before implementation:

```text
ModuleNotFoundError: No module named 'train.plan_stage3_v29_joint_gate_recovery_matrix'
```

- [ ] **Step 7: Check worktree without committing**

Run:

```powershell
git status --short
```

Expected result:

```text
git reports tests/test_stage3_v29_joint_gate_recovery_matrix.py as an untracked file.
```

The repository may also show existing untracked files and the known `pytest_tmp_stage3_review/` permission warning. Do not clean them.

---

### Task 2: Implement Joint Gate Recovery Planner

**Files:**
- Create: `train/plan_stage3_v29_joint_gate_recovery_matrix.py`
- Test: `tests/test_stage3_v29_joint_gate_recovery_matrix.py`

- [ ] **Step 1: Create the planner module skeleton and shared readers**

Create `train/plan_stage3_v29_joint_gate_recovery_matrix.py` with this start:

```python
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

import yaml


PROTOCOL_VERSION = "stage3_v29_joint_gate_recovery_matrix_v1"
SOURCE_DECOMPOSITION_PROTOCOL = "stage3_v29_gate_failure_decomposition_v1"
DEFAULT_CONFIG_DIR = Path("configs/stage3_v29_joint_gate_recovery_matrix")
DEFAULT_BUDGET_ARGS = (
    "--epochs 1",
    "--max-train-batches 60",
    "--max-generator-batches 60",
    "--max-eval-batches 60",
    "--validation-windows 5",
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
    "window0_safe_release_ramp",
    "late_history_dominance_bridge",
    "counterfactual_tie_breaker_floor_safe",
    "history_cap_release_balance",
}


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else {}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _rounded(value: float) -> float:
    return float(f"{value:.12g}")


def _num(mapping: dict[str, Any], key: str, default: float) -> float:
    try:
        value = mapping.get(key, default)
        if value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value == "" or value is None:
            return default
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _base_invariant(base_config: dict[str, Any]) -> dict[str, Any]:
    invariant = base_config.get("invariant", {})
    return invariant if isinstance(invariant, dict) else {}
```

- [ ] **Step 2: Add source-artifact interpretation helpers**

Append:

```python
def _window_summary(report: dict[str, Any], window_id: str) -> dict[str, Any]:
    rows = report.get("window_summaries", [])
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if isinstance(row, dict) and str(row.get("validation_window_id")) == str(window_id):
            return row
    return {}


def _top_gate_near_run_id(selector_rows: list[dict[str, str]]) -> str:
    raw_rows = [
        row
        for row in selector_rows
        if row.get("target_name") == "raw_residual_vector"
        and row.get("feature_mode") == "fluctuation_plus_history"
    ]
    if not raw_rows:
        return ""
    raw_rows.sort(key=lambda row: _as_int(row.get("strict_rank"), 9999))
    return str(raw_rows[0].get("run_id", ""))


def _top_window_rows(window_rows: list[dict[str, str]], run_id: str) -> list[dict[str, str]]:
    rows = [
        row
        for row in window_rows
        if row.get("run_id") == run_id
        and row.get("target_name") == "raw_residual_vector"
        and row.get("feature_mode") == "fluctuation_plus_history"
    ]
    return sorted(rows, key=lambda row: _as_int(row.get("validation_window_id"), 9999))


def _history_dominant_windows(window_rows: list[dict[str, str]], run_id: str) -> list[str]:
    dominant: list[str] = []
    for row in _top_window_rows(window_rows, run_id):
        partial = _num(row, "partial_mse_gain_vs_history", 0.0)
        history_gain = _num(row, "history_mse_gain", 0.0)
        if partial < 0.0 and history_gain > 0.0:
            dominant.append(str(row.get("validation_window_id", "")))
    return dominant


def _counterfactual_negative_windows(window_rows: list[dict[str, str]], run_id: str) -> list[str]:
    negative: list[str] = []
    for row in _top_window_rows(window_rows, run_id):
        margin = _num(row, "counterfactual_margin", 0.0)
        if margin < 0.0:
            negative.append(str(row.get("validation_window_id", "")))
    return negative


def _real_not_best_windows(window_rows: list[dict[str, str]], run_id: str) -> list[str]:
    not_best: list[str] = []
    for row in _top_window_rows(window_rows, run_id):
        if str(row.get("real_is_best_mode", "")).lower() not in {"true", "1", "yes"}:
            not_best.append(str(row.get("validation_window_id", "")))
    return not_best


def _preconditions_hold(decomposition_report: dict[str, Any]) -> bool:
    if decomposition_report.get("protocol_version") != SOURCE_DECOMPOSITION_PROTOCOL:
        return False
    if decomposition_report.get("matrix_decision") != "stop_local_scalar_matrix":
        return False
    if _as_int(decomposition_report.get("gate_pass_count"), 0) != 0:
        return False
    if _as_int(decomposition_report.get("generator_unlocked_count"), 0) != 0:
        return False
    blocker_counts = decomposition_report.get("primary_blocker_counts", {})
    if not isinstance(blocker_counts, dict):
        return False
    required_blockers = (
        "counterfactual_margin_negative",
        "min_partial_mse_below_floor",
        "min_partial_mae_below_floor",
        "positive_partial_windows_below_required",
    )
    return all(_as_int(blocker_counts.get(key), 0) > 0 for key in required_blockers)
```

- [ ] **Step 3: Add candidate templates**

Append:

```python
def _candidate_template(
    *,
    axis: str,
    run_slug: str,
    priority: int,
    addresses_failure_axes: list[str],
    invariant_overrides: dict[str, float],
    mechanism_shift: str,
    rationale: str,
    window_focus: str,
    diagnostic_metadata: dict[str, Any],
) -> dict[str, Any]:
    if axis in OLD_AXES:
        raise ValueError(f"Joint-gate recovery axis repeats an exhausted axis: {axis}")
    return {
        "axis": axis,
        "run_slug": run_slug,
        "priority": priority,
        "addresses_failure_axes": addresses_failure_axes,
        "invariant_overrides": invariant_overrides,
        "mechanism_shift": mechanism_shift,
        "rationale": rationale,
        "window_focus": window_focus,
        "diagnostic_metadata": diagnostic_metadata,
    }


def _candidate_templates(
    *,
    base_config: dict[str, Any],
    decomposition_report: dict[str, Any],
    selector_rows: list[dict[str, str]],
    window_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    invariant = _base_invariant(base_config)
    top_run_id = _top_gate_near_run_id(selector_rows)
    history_dominant = _history_dominant_windows(window_rows, top_run_id)
    cf_negative = _counterfactual_negative_windows(window_rows, top_run_id)
    real_not_best = _real_not_best_windows(window_rows, top_run_id)
    window0 = _window_summary(decomposition_report, "0")
    window2 = _window_summary(decomposition_report, "2")

    shared_meta = {
        "top_gate_near_run_id": top_run_id,
        "history_dominant_windows": history_dominant,
        "counterfactual_negative_windows": cf_negative,
        "real_not_best_windows": real_not_best,
        "window0_partial_mse_floor_failures": _as_int(window0.get("partial_mse_floor_failures"), 0),
        "window0_row_count": _as_int(window0.get("row_count"), 0),
        "window2_counterfactual_negative_rows": _as_int(window2.get("counterfactual_negative_rows"), 0),
    }

    return [
        _candidate_template(
            axis="joint_floor_cf_balance",
            run_slug="jfloorcf",
            priority=10,
            addresses_failure_axes=[
                "joint_floor_counterfactual_conflict",
                "partial_floor_violation",
                "counterfactual_ordering_failure",
                "window_level_conflict",
            ],
            invariant_overrides={
                "lambda_counterfactual_shuffle": _rounded(_num(invariant, "lambda_counterfactual_shuffle", 0.0812) + 0.0010),
                "counterfactual_shuffle_margin": _rounded(_num(invariant, "counterfactual_shuffle_margin", 0.00112) + 0.00010),
                "lambda_residual_release_advantage": _rounded(_num(invariant, "lambda_residual_release_advantage", 0.08) + 0.018),
                "lambda_support_conditioned_advantage": _rounded(_num(invariant, "lambda_support_conditioned_advantage", 0.08) + 0.020),
                "lambda_router_stability": _rounded(_num(invariant, "lambda_router_stability", 0.145) + 0.010),
                "lambda_anchor_preservation": _rounded(_num(invariant, "lambda_anchor_preservation", 0.14) + 0.012),
            },
            mechanism_shift="joint_partial_floor_and_counterfactual_balance",
            rationale=(
                "HGR failed floors and counterfactual ordering together. This candidate increases "
                "counterfactual pressure and residual/support advantage while preserving router and anchor stability."
            ),
            window_focus="all_gate_conditions",
            diagnostic_metadata=shared_meta,
        ),
        _candidate_template(
            axis="window0_preserving_late_floor",
            run_slug="w0late",
            priority=20,
            addresses_failure_axes=[
                "window0_floor_safety",
                "positive_partial_window_shortfall",
                "partial_floor_violation",
                "history_dominance_partial_floor_conflict",
            ],
            invariant_overrides={
                "lambda_support_alignment": _rounded(_num(invariant, "lambda_support_alignment", 0.14) + 0.025),
                "lambda_support_conditioned_advantage": _rounded(_num(invariant, "lambda_support_conditioned_advantage", 0.08) + 0.030),
                "lambda_residual_release_alignment": _rounded(_num(invariant, "lambda_residual_release_alignment", 0.03) + 0.016),
                "lambda_release_state_advantage": _rounded(_num(invariant, "lambda_release_state_advantage", 0.12) + 0.012),
                "lambda_anchor_preservation": _rounded(_num(invariant, "lambda_anchor_preservation", 0.14) + 0.020),
                "lambda_release_first_increment": _rounded(_num(invariant, "lambda_release_first_increment", 0.405) + 0.004),
            },
            mechanism_shift="late_window_floor_push_with_window0_anchor_preservation",
            rationale=(
                "Window 0 floor failures are universal in the source matrix, while later windows still need "
                "residual signal. This tests late floor repair without a large release-amplitude jump."
            ),
            window_focus="window0_and_history_dominant_late_windows",
            diagnostic_metadata=shared_meta,
        ),
        _candidate_template(
            axis="real_beats_shuffle_margin",
            run_slug="realcf",
            priority=30,
            addresses_failure_axes=[
                "real_vs_shuffle_ordering_failure",
                "counterfactual_ordering_failure",
                "counterfactual_negative_margin",
            ],
            invariant_overrides={
                "lambda_counterfactual_shuffle": _rounded(_num(invariant, "lambda_counterfactual_shuffle", 0.0812) + 0.0018),
                "counterfactual_shuffle_margin": _rounded(_num(invariant, "counterfactual_shuffle_margin", 0.00112) + 0.00018),
                "lambda_local_env_consistency": _rounded(_num(invariant, "lambda_local_env_consistency", 0.055) + 0.020),
                "lambda_router_stability": _rounded(_num(invariant, "lambda_router_stability", 0.145) + 0.014),
                "lambda_release_first_increment": _rounded(max(0.0, _num(invariant, "lambda_release_first_increment", 0.405) - 0.004)),
            },
            mechanism_shift="real_environment_signal_ordering_over_shuffle_controls",
            rationale=(
                "The source matrix shows real signal losing to counterfactual controls. This candidate "
                "raises real-versus-shuffle pressure and local environment consistency while slightly reducing release amplitude."
            ),
            window_focus="counterfactual_negative_windows",
            diagnostic_metadata=shared_meta,
        ),
        _candidate_template(
            axis="history_residual_rebalance",
            run_slug="histreb",
            priority=40,
            addresses_failure_axes=[
                "history_residual_imbalance",
                "positive_partial_window_shortfall",
                "partial_floor_violation",
                "counterfactual_ordering_failure",
            ],
            invariant_overrides={
                "lambda_history_after_release": _rounded(max(0.0, _num(invariant, "lambda_history_after_release", 0.08) - 0.012)),
                "lambda_history_budget_cap": _rounded(max(0.0, _num(invariant, "lambda_history_budget_cap", 0.035) - 0.006)),
                "lambda_release_priority_advantage": _rounded(_num(invariant, "lambda_release_priority_advantage", 0.18) + 0.022),
                "lambda_support_alignment": _rounded(_num(invariant, "lambda_support_alignment", 0.14) + 0.018),
                "lambda_suppression_alignment": _rounded(_num(invariant, "lambda_suppression_alignment", 0.1) + 0.008),
                "lambda_suppression_gate_sparsity": _rounded(_num(invariant, "lambda_suppression_gate_sparsity", 0.025) + 0.003),
            },
            mechanism_shift="rebalance_history_dominance_against_residual_release",
            rationale=(
                "HGR still leaves history-only stronger than fluctuation-plus-history in key windows. "
                "This candidate reduces history dominance and raises residual release priority conservatively."
            ),
            window_focus="history_dominant_windows",
            diagnostic_metadata=shared_meta,
        ),
    ]
```

- [ ] **Step 4: Add row builder and suggested command generation**

Append:

```python
def _suggested_command(*, config_name: str, run_id: str, config_dir: Path = DEFAULT_CONFIG_DIR) -> str:
    budget = " ".join(DEFAULT_BUDGET_ARGS)
    return (
        "python train\\run_stage3_v29_expert_local_invariant_release_probe.py "
        f"--config {config_dir / config_name} "
        f"--run-id {run_id} {budget} "
        "--out-root logs\\s3v29jgr --checkpoint-root ckpt\\s3v29jgr"
    )


def build_joint_gate_recovery_matrix(
    *,
    base_config: dict[str, Any],
    decomposition_report: dict[str, Any],
    selector_rows: list[dict[str, str]],
    window_rows: list[dict[str, str]],
    run_id_prefix: str = "v29jgr_conf015",
) -> list[dict[str, Any]]:
    if not _preconditions_hold(decomposition_report):
        return []

    rows: list[dict[str, Any]] = []
    for index, template in enumerate(
        _candidate_templates(
            base_config=base_config,
            decomposition_report=decomposition_report,
            selector_rows=selector_rows,
            window_rows=window_rows,
        ),
        start=1,
    ):
        run_id = f"{run_id_prefix}_{index:02d}_{template['run_slug']}"
        config_name = f"stage3_v29_{run_id}.yaml"
        rows.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "candidate_id": f"{index:02d}_{template['axis']}",
                "priority": template["priority"],
                "axis": template["axis"],
                "run_slug": template["run_slug"],
                "run_id": run_id,
                "config_name": config_name,
                "source_decomposition_protocol": SOURCE_DECOMPOSITION_PROTOCOL,
                "source_matrix_decision": str(decomposition_report.get("matrix_decision", "")),
                "source_candidate_count": _as_int(decomposition_report.get("candidate_count"), 0),
                "addresses_failure_axes": template["addresses_failure_axes"],
                "window_focus": template["window_focus"],
                "architecture_policy": "frozen_v29_no_model_topology_change",
                "generator_policy": "preflight_required_do_not_unlock_from_matrix",
                "source_matrix_status_required": "hgr_exhausted_meaningful_rows",
                "known_failed_axis_guard": "not_old_scalar_repair_or_hgr_axis",
                "mechanism_shift": template["mechanism_shift"],
                "model_overrides": {},
                "train_overrides": {},
                "invariant_overrides": template["invariant_overrides"],
                "diagnostic_metadata": template["diagnostic_metadata"],
                "command_budget": " ".join(DEFAULT_BUDGET_ARGS),
                "suggested_probe_command": _suggested_command(config_name=config_name, run_id=run_id),
                "rationale": template["rationale"],
                "promotion_gate": (
                    "rerun complete selector, gate-failure decomposition, and generator-unlock preflight; "
                    "do not train generator unless preflight_pass becomes true"
                ),
            }
        )
    return sorted(rows, key=lambda row: (int(row["priority"]), str(row["run_id"])))
```

- [ ] **Step 5: Add config merge, CSV, Markdown, writer, and CLI**

Append:

```python
def _merge_config(base_config: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(base_config)
    invariant = cfg.setdefault("invariant", {})
    if not isinstance(invariant, dict):
        invariant = {}
        cfg["invariant"] = invariant
    invariant.update(dict(candidate["invariant_overrides"]))

    stage3 = cfg.setdefault("stage3_v2", {})
    if not isinstance(stage3, dict):
        stage3 = {}
        cfg["stage3_v2"] = stage3
    stage3["run_id"] = str(candidate["run_id"])
    stage3["joint_gate_recovery_matrix_protocol"] = PROTOCOL_VERSION
    stage3["joint_gate_recovery_candidate_id"] = str(candidate["candidate_id"])

    cfg["joint_gate_recovery_matrix"] = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate_id": candidate["candidate_id"],
        "axis": candidate["axis"],
        "source_decomposition_protocol": candidate["source_decomposition_protocol"],
        "source_matrix_decision": candidate["source_matrix_decision"],
        "source_candidate_count": candidate["source_candidate_count"],
        "addresses_failure_axes": candidate["addresses_failure_axes"],
        "window_focus": candidate["window_focus"],
        "architecture_policy": candidate["architecture_policy"],
        "generator_policy": candidate["generator_policy"],
        "source_matrix_status_required": candidate["source_matrix_status_required"],
        "known_failed_axis_guard": candidate["known_failed_axis_guard"],
        "mechanism_shift": candidate["mechanism_shift"],
        "invariant_overrides": candidate["invariant_overrides"],
        "diagnostic_metadata": candidate["diagnostic_metadata"],
        "promotion_gate": candidate["promotion_gate"],
    }
    return cfg


def _csv_scalar(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refuse to overwrite existing V29 joint-gate recovery artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_scalar(value) for key, value in row.items()})


def _markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage III V29 Joint Gate Recovery Matrix",
        "",
        f"Protocol: `{PROTOCOL_VERSION}`.",
        f"Rows: `{len(rows)}`.",
        "",
        "This matrix does not train the generator and does not change V29 model topology. "
        "It is a post-HGR tuning plan for joint partial-floor, positive-window, and counterfactual-ordering recovery.",
        "",
        "| Priority | Candidate | Axis | Mechanism shift | Failure axes |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {priority} | `{candidate}` | `{axis}` | `{mechanism}` | `{axes}` |".format(
                priority=row["priority"],
                candidate=row["candidate_id"],
                axis=row["axis"],
                mechanism=row["mechanism_shift"],
                axes=";".join(row["addresses_failure_axes"]),
            )
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "These candidates are next-run tuning controls only. They are not generator-unlock evidence, "
            "not online-validation evidence, and not manuscript performance evidence. After each run, rerun "
            "the complete selector, gate-failure decomposition, and generator-unlock preflight.",
            "",
        ]
    )
    return "\n".join(lines)


def write_joint_gate_recovery_matrix_artifacts(
    *,
    rows: list[dict[str, Any]],
    base_config: dict[str, Any],
    config_dir: Path,
    out_dir: Path,
    prefix: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    csv_path = out_dir / f"{prefix}.csv"
    md_path = out_dir / f"{prefix}.md"
    if md_path.exists() and not overwrite:
        raise FileExistsError(f"Refuse to overwrite existing V29 joint-gate recovery artifact: {md_path}")
    _write_csv(csv_path, rows, overwrite=overwrite)
    config_dir.mkdir(parents=True, exist_ok=True)
    config_paths: list[str] = []
    for row in rows:
        config_path = config_dir / str(row["config_name"])
        if config_path.exists() and not overwrite:
            raise FileExistsError(f"Refuse to overwrite existing V29 joint-gate recovery config: {config_path}")
        payload = _merge_config(base_config, row)
        config_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")
        config_paths.append(str(config_path))
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_markdown(rows), encoding="utf-8")
    return {"csv": str(csv_path), "markdown": str(md_path), "configs": config_paths}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Stage III V29 joint-gate recovery matrix configs.")
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--decomposition-json", type=Path, required=True)
    parser.add_argument("--selector-csv", type=Path, required=True)
    parser.add_argument("--windows-csv", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="v29_joint_gate_recovery_matrix")
    parser.add_argument("--run-id-prefix", default="v29jgr_conf015")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    base_config = _read_yaml(args.base_config)
    decomposition_report = _read_json(args.decomposition_json)
    selector_rows = _read_csv(args.selector_csv)
    window_rows = _read_csv(args.windows_csv)
    rows = build_joint_gate_recovery_matrix(
        base_config=base_config,
        decomposition_report=decomposition_report,
        selector_rows=selector_rows,
        window_rows=window_rows,
        run_id_prefix=str(args.run_id_prefix),
    )
    outputs = write_joint_gate_recovery_matrix_artifacts(
        rows=rows,
        base_config=base_config,
        config_dir=args.config_dir,
        out_dir=args.out_dir,
        prefix=str(args.prefix),
        overwrite=bool(args.overwrite),
    )
    print(f"rows={len(rows)} csv={outputs['csv']} markdown={outputs['markdown']} configs={len(outputs['configs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Run the focused test**

Run:

```powershell
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_stage3_v29_joint_gate_recovery_matrix.py -q
```

Expected result:

```text
4 passed
```

- [ ] **Step 7: Compile the planner**

Run:

```powershell
C:\Users\admin\anaconda3\python.exe -m py_compile train\plan_stage3_v29_joint_gate_recovery_matrix.py
```

Expected result:

```text
<no output, exit code 0>
```

- [ ] **Step 8: Check worktree without committing**

Run:

```powershell
git status --short
```

Expected result includes:

```text
git reports train/plan_stage3_v29_joint_gate_recovery_matrix.py as an untracked file.
git reports tests/test_stage3_v29_joint_gate_recovery_matrix.py as an untracked file.
```

Do not run `git add` or `git commit`.

---

### Task 3: Generate Real Joint Gate Recovery Matrix Artifacts

**Files:**
- Read: `configs/stage3_v29_expert_local_invariant_release_probe_tune_stability_release_deconstrained.yaml`
- Read: `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr04_gate_failure_decomposition.json`
- Read: `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr04_selection.csv`
- Read: `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr04_selection_windows.csv`
- Create directory: `configs/stage3_v29_joint_gate_recovery_matrix/`
- Create: `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_joint_gate_recovery_matrix.csv`
- Create: `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_joint_gate_recovery_matrix.md`

- [ ] **Step 1: Verify source artifacts exist**

Run:

```powershell
Test-Path configs\stage3_v29_expert_local_invariant_release_probe_tune_stability_release_deconstrained.yaml
Test-Path artifacts\stage3_v29_teacher_gate_tuning_matrix\v29_complete_conf015_hgr04_gate_failure_decomposition.json
Test-Path artifacts\stage3_v29_teacher_gate_tuning_matrix\v29_complete_conf015_hgr04_selection.csv
Test-Path artifacts\stage3_v29_teacher_gate_tuning_matrix\v29_complete_conf015_hgr04_selection_windows.csv
```

Expected result:

```text
True
True
True
True
```

- [ ] **Step 2: Verify output paths are not already occupied**

Run:

```powershell
Test-Path artifacts\stage3_v29_teacher_gate_tuning_matrix\v29_complete_conf015_joint_gate_recovery_matrix.csv
Test-Path artifacts\stage3_v29_teacher_gate_tuning_matrix\v29_complete_conf015_joint_gate_recovery_matrix.md
Test-Path configs\stage3_v29_joint_gate_recovery_matrix
```

Expected result before first generation:

```text
False
False
False
```

If `configs\stage3_v29_joint_gate_recovery_matrix` exists from a prior attempt, inspect file names with:

```powershell
Get-ChildItem configs\stage3_v29_joint_gate_recovery_matrix | Select-Object Name,Length,LastWriteTime
```

Proceed only if no generated `stage3_v29_v29jgr_conf015_*.yaml` file would be replaced.

- [ ] **Step 3: Generate matrix artifacts without overwrite**

Run:

```powershell
C:\Users\admin\anaconda3\python.exe train\plan_stage3_v29_joint_gate_recovery_matrix.py --base-config configs\stage3_v29_expert_local_invariant_release_probe_tune_stability_release_deconstrained.yaml --decomposition-json artifacts\stage3_v29_teacher_gate_tuning_matrix\v29_complete_conf015_hgr04_gate_failure_decomposition.json --selector-csv artifacts\stage3_v29_teacher_gate_tuning_matrix\v29_complete_conf015_hgr04_selection.csv --windows-csv artifacts\stage3_v29_teacher_gate_tuning_matrix\v29_complete_conf015_hgr04_selection_windows.csv --config-dir configs\stage3_v29_joint_gate_recovery_matrix --out-dir artifacts\stage3_v29_teacher_gate_tuning_matrix --prefix v29_complete_conf015_joint_gate_recovery_matrix --run-id-prefix v29jgr_conf015
```

Expected result:

```text
rows=4 csv=artifacts\stage3_v29_teacher_gate_tuning_matrix\v29_complete_conf015_joint_gate_recovery_matrix.csv markdown=artifacts\stage3_v29_teacher_gate_tuning_matrix\v29_complete_conf015_joint_gate_recovery_matrix.md configs=4
```

- [ ] **Step 4: Inspect generated matrix rows**

Run:

```powershell
C:\Users\admin\anaconda3\python.exe - <<'PY'
import csv
from pathlib import Path
path = Path("artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_joint_gate_recovery_matrix.csv")
rows = list(csv.DictReader(path.open(encoding="utf-8")))
print(len(rows))
for row in rows:
    print(row["candidate_id"], row["axis"], row["run_id"], row["config_name"], row["command_budget"])
PY
```

Expected result:

```text
4
01_joint_floor_cf_balance joint_floor_cf_balance v29jgr_conf015_01_jfloorcf stage3_v29_v29jgr_conf015_01_jfloorcf.yaml --epochs 1 --max-train-batches 60 --max-generator-batches 60 --max-eval-batches 60 --validation-windows 5
02_window0_preserving_late_floor window0_preserving_late_floor v29jgr_conf015_02_w0late stage3_v29_v29jgr_conf015_02_w0late.yaml --epochs 1 --max-train-batches 60 --max-generator-batches 60 --max-eval-batches 60 --validation-windows 5
03_real_beats_shuffle_margin real_beats_shuffle_margin v29jgr_conf015_03_realcf stage3_v29_v29jgr_conf015_03_realcf.yaml --epochs 1 --max-train-batches 60 --max-generator-batches 60 --max-eval-batches 60 --validation-windows 5
04_history_residual_rebalance history_residual_rebalance v29jgr_conf015_04_histreb stage3_v29_v29jgr_conf015_04_histreb.yaml --epochs 1 --max-train-batches 60 --max-generator-batches 60 --max-eval-batches 60 --validation-windows 5
```

- [ ] **Step 5: Verify generated configs preserve frozen topology**

Run:

```powershell
C:\Users\admin\anaconda3\python.exe - <<'PY'
import yaml
from pathlib import Path
base = yaml.safe_load(Path("configs/stage3_v29_expert_local_invariant_release_probe_tune_stability_release_deconstrained.yaml").read_text(encoding="utf-8"))
for path in sorted(Path("configs/stage3_v29_joint_gate_recovery_matrix").glob("stage3_v29_v29jgr_conf015_*.yaml")):
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert cfg.get("model_v29") == base.get("model_v29"), path
    assert cfg.get("online", {}).get("stage3", {}).get("drift_source") == "pooled_servable_change", path
    assert cfg.get("joint_gate_recovery_matrix", {}).get("architecture_policy") == "frozen_v29_no_model_topology_change", path
    assert cfg.get("joint_gate_recovery_matrix", {}).get("generator_policy") == "preflight_required_do_not_unlock_from_matrix", path
    print(path.name, cfg["stage3_v2"]["run_id"], cfg["joint_gate_recovery_matrix"]["axis"])
PY
```

Expected result:

```text
stage3_v29_v29jgr_conf015_01_jfloorcf.yaml v29jgr_conf015_01_jfloorcf joint_floor_cf_balance
stage3_v29_v29jgr_conf015_02_w0late.yaml v29jgr_conf015_02_w0late window0_preserving_late_floor
stage3_v29_v29jgr_conf015_03_realcf.yaml v29jgr_conf015_03_realcf real_beats_shuffle_margin
stage3_v29_v29jgr_conf015_04_histreb.yaml v29jgr_conf015_04_histreb history_residual_rebalance
```

- [ ] **Step 6: Check worktree without committing**

Run:

```powershell
git status --short
```

Expected result includes:

```text
git reports artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_joint_gate_recovery_matrix.csv as an untracked file.
git reports artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_joint_gate_recovery_matrix.md as an untracked file.
git reports configs/stage3_v29_joint_gate_recovery_matrix/ as an untracked directory.
```

Do not commit.

---

### Task 4: Generate Non-Overwriting Execution Plan

**Files:**
- Read: `train/plan_stage3_v29_tuning_execution.py`
- Read: `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_joint_gate_recovery_matrix.csv`
- Create: `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_joint_gate_recovery_execution_plan_after_01.csv`
- Create: `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_joint_gate_recovery_execution_plan_after_01.md`

- [ ] **Step 1: Generate an execution plan with the existing planner**

Run:

```powershell
C:\Users\admin\anaconda3\python.exe train\plan_stage3_v29_tuning_execution.py --matrix-csv artifacts\stage3_v29_teacher_gate_tuning_matrix\v29_complete_conf015_joint_gate_recovery_matrix.csv --config-dir configs\stage3_v29_joint_gate_recovery_matrix --out-root logs\s3v29jgr --checkpoint-root ckpt\s3v29jgr --out-dir artifacts\stage3_v29_teacher_gate_tuning_matrix --prefix v29_complete_conf015_joint_gate_recovery_execution_plan_after_01
```

Expected result:

```text
rows=4
```

If the existing execution planner prints a longer summary, verify that it writes the two expected output files and does not mark any existing run path as safe when a collision is present.

- [ ] **Step 2: Inspect execution rows for Windows path safety and no overwrite**

Run:

```powershell
C:\Users\admin\anaconda3\python.exe - <<'PY'
import csv
from pathlib import Path
path = Path("artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_joint_gate_recovery_execution_plan_after_01.csv")
rows = list(csv.DictReader(path.open(encoding="utf-8")))
print(len(rows))
for row in rows:
    text = " ".join(str(value) for value in row.values())
    assert "--overwrite" not in text, row
    print(row.get("run_id", ""), row.get("config_path", ""), row.get("checkpoint_dir", ""))
PY
```

Expected result:

```text
4
v29jgr_conf015_01_jfloorcf configs\stage3_v29_joint_gate_recovery_matrix\stage3_v29_v29jgr_conf015_01_jfloorcf.yaml ckpt\s3v29jgr\v29jgr_conf015_01_jfloorcf
v29jgr_conf015_02_w0late configs\stage3_v29_joint_gate_recovery_matrix\stage3_v29_v29jgr_conf015_02_w0late.yaml ckpt\s3v29jgr\v29jgr_conf015_02_w0late
v29jgr_conf015_03_realcf configs\stage3_v29_joint_gate_recovery_matrix\stage3_v29_v29jgr_conf015_03_realcf.yaml ckpt\s3v29jgr\v29jgr_conf015_03_realcf
v29jgr_conf015_04_histreb configs\stage3_v29_joint_gate_recovery_matrix\stage3_v29_v29jgr_conf015_04_histreb.yaml ckpt\s3v29jgr\v29jgr_conf015_04_histreb
```

- [ ] **Step 3: Select exactly one first bounded candidate**

Use priority order unless the execution planner reports a path collision. The first candidate is:

```text
v29jgr_conf015_01_jfloorcf
```

The command must remain bounded:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v29_expert_local_invariant_release_probe.py --config configs\stage3_v29_joint_gate_recovery_matrix\stage3_v29_v29jgr_conf015_01_jfloorcf.yaml --run-id v29jgr_conf015_01_jfloorcf --seed 42 --device auto --epochs 1 --max-train-batches 60 --max-generator-batches 60 --max-eval-batches 60 --validation-windows 5 --out-root logs\s3v29jgr --checkpoint-root ckpt\s3v29jgr
```

Do not run this command during plan writing. Run it only when executing the plan and only after the execution plan confirms no path collision.

---

### Task 5: Bounded First Candidate Probe

**Files:**
- Read: `configs/stage3_v29_joint_gate_recovery_matrix/stage3_v29_v29jgr_conf015_01_jfloorcf.yaml`
- Create under bounded run: `logs/s3v29jgr/v29jgr_conf015_01_jfloorcf/`
- Create under bounded run: `ckpt/s3v29jgr/v29jgr_conf015_01_jfloorcf/`

- [ ] **Step 1: Run only the first bounded candidate**

Run:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v29_expert_local_invariant_release_probe.py --config configs\stage3_v29_joint_gate_recovery_matrix\stage3_v29_v29jgr_conf015_01_jfloorcf.yaml --run-id v29jgr_conf015_01_jfloorcf --seed 42 --device auto --epochs 1 --max-train-batches 60 --max-generator-batches 60 --max-eval-batches 60 --validation-windows 5 --out-root logs\s3v29jgr --checkpoint-root ckpt\s3v29jgr
```

Expected result:

```text
The command completes with exit code 0 and writes a bounded one-epoch candidate log/checkpoint under the v29jgr paths.
```

- [ ] **Step 2: Refresh selector for the current evidence set**

Run the repository's existing selector command used for HGR, replacing only the matrix/log roots with the Joint Gate Recovery paths. If the selector script requires explicit file paths, use:

```text
matrix source: artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_joint_gate_recovery_matrix.csv
log root: logs/s3v29jgr
candidate run id: v29jgr_conf015_01_jfloorcf
```

Expected result:

```text
The selector artifact contains v29jgr_conf015_01_jfloorcf and reports target_gate_pass plus generator_unlocked as explicit fields.
```

- [ ] **Step 3: Refresh generator-unlock preflight**

Run the existing V29 generator-unlock preflight command against the refreshed selector output. Expected result:

```text
preflight_pass is either true or false, and no generator training command is started by the preflight.
```

- [ ] **Step 4: Refresh gate-failure decomposition**

Run the existing V29 gate-failure decomposition command against the refreshed selector/window artifacts. Expected result:

```text
The decomposition reports candidate_count, gate_pass_count, generator_unlocked_count, primary_blocker_counts, and matrix_decision.
```

- [ ] **Step 5: Apply the stop/continue rule**

Use this decision rule exactly:

```text
If target_gate_pass is false, do not run generator training and do not run the next candidate until selector, preflight, and decomposition artifacts are recorded.
If target_gate_pass is true and preflight_pass is false, do not run generator training; record the preflight blocker.
If target_gate_pass is true and preflight_pass is true, the candidate becomes eligible for a bounded generator run, but it is still not paper-grade evidence until online validation, cross-dataset checks, and seed checks exist.
```

---

### Task 6: Update Audit Documentation After Generated Evidence Exists

**Files:**
- Modify: `docs/audit/STAGE3_V29_EXPERT_LOCAL_INVARIANT_RELEASE_IMPLEMENTATION_RECORD.md`
- Modify: `docs/audit/EXPERIMENT_EVIDENCE_LEDGER.md`
- Modify: `docs/audit/STAGE3_V29_TUNING_PRIORITY_CORRECTION_REPORT.md`
- Modify: `docs/superpowers/plans/2026-06-14-stage3-v29-joint-gate-recovery-matrix.md`

- [ ] **Step 1: Add a Joint Gate Recovery section to the implementation record**

Add a section with these facts after artifact generation:

```markdown
## Joint Gate Recovery Matrix

- Protocol: `stage3_v29_joint_gate_recovery_matrix_v1`.
- Planner: `train/plan_stage3_v29_joint_gate_recovery_matrix.py`.
- Config directory: `configs/stage3_v29_joint_gate_recovery_matrix/`.
- Matrix artifacts:
  - `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_joint_gate_recovery_matrix.csv`
  - `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_joint_gate_recovery_matrix.md`
- Boundary: frozen V29 topology; only `invariant` weights plus matrix metadata change.
- Generator boundary: matrix generation does not unlock generator training; existing preflight remains mandatory.
- Evidence status: tuning-plan artifact only until bounded candidate logs, selector, decomposition, and preflight artifacts exist.
```

- [ ] **Step 2: Add ledger rows for planner tests and matrix artifacts**

Add concise entries to `docs/audit/EXPERIMENT_EVIDENCE_LEDGER.md` containing:

```markdown
| Stage III V29 Joint Gate Recovery planner tests | `tests/test_stage3_v29_joint_gate_recovery_matrix.py`; `train/plan_stage3_v29_joint_gate_recovery_matrix.py` | Verifies exhausted-HGR precondition, new axes, frozen topology, no overwrite, bounded command strings | Implementation smoke evidence only |
| Stage III V29 Joint Gate Recovery matrix | `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_joint_gate_recovery_matrix.csv`; `configs/stage3_v29_joint_gate_recovery_matrix/` | Four predeclared frozen-topology candidates after HGR exhaustion | Not performance evidence; no generator unlock |
```

- [ ] **Step 3: Add tuning-priority correction note**

Add this statement to `docs/audit/STAGE3_V29_TUNING_PRIORITY_CORRECTION_REPORT.md`:

```markdown
The next permitted V29 step after HGR exhaustion is the Joint Gate Recovery matrix, not another local scalar sweep. Its purpose is to test coupled recovery of partial floors, positive-window count, and real-versus-counterfactual ordering while preserving frozen V29 topology and the served drift source `pooled_servable_change`.
```

- [ ] **Step 4: Record executed commands in this plan**

Append a `## Execution Log` section to this file after real execution:

```markdown
## Execution Log

- Planner tests: `<command>` -> `<observed result>`.
- Planner compile: `<command>` -> `<observed result>`.
- Matrix generation: `<command>` -> `<observed result>`.
- Execution-plan generation: `<command>` -> `<observed result>`.
- First bounded candidate: `<command>` -> `<observed result or not run reason>`.
```

Replace each angle-bracket field with the exact command and observed result before saving. Do not leave angle-bracket markers in the committed document.

---

### Task 7: Final Verification

**Files:**
- Read: generated tests, planner, configs, artifacts, docs

- [ ] **Step 1: Run focused planner tests**

Run:

```powershell
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_stage3_v29_joint_gate_recovery_matrix.py -q
```

Expected result:

```text
4 passed
```

- [ ] **Step 2: Run V29 matrix-planner regression tests**

Run:

```powershell
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_stage3_v29_history_gate_recovery_matrix.py tests\test_stage3_v29_decomposition_repair_matrix.py tests\test_stage3_v29_teacher_gate_tuning_matrix.py tests\test_stage3_v29_tuning_execution_plan.py tests\test_stage3_v29_joint_gate_recovery_matrix.py -q
```

Expected result:

```text
All selected tests pass.
```

- [ ] **Step 3: Compile changed Python files**

Run:

```powershell
C:\Users\admin\anaconda3\python.exe -m py_compile train\plan_stage3_v29_joint_gate_recovery_matrix.py
```

Expected result:

```text
<no output, exit code 0>
```

- [ ] **Step 4: Scan for disallowed draft markers and generated overwrite commands**

Run:

```powershell
rg -n "T[B]D|T[O]DO|place[ ]?holder|\?\?" docs\superpowers\plans\2026-06-14-stage3-v29-joint-gate-recovery-matrix.md train\plan_stage3_v29_joint_gate_recovery_matrix.py tests\test_stage3_v29_joint_gate_recovery_matrix.py
rg -n -- "--overwrite" artifacts\stage3_v29_teacher_gate_tuning_matrix\v29_complete_conf015_joint_gate_recovery_matrix.csv configs\stage3_v29_joint_gate_recovery_matrix
```

Expected result:

```text
No matches for draft markers in the plan/source/tests.
No matches for --overwrite in generated matrix commands/configs.
```

The planner source may contain the string `--overwrite` only as an argparse flag and writer guard. Candidate matrix CSV rows and suggested commands must not contain it.

- [ ] **Step 5: Report residual scientific risk**

Final response must state:

```text
This is a tuning matrix and bounded-probe pathway, not accuracy-improvement evidence. Generator training remains locked unless the existing preflight passes. Online validation, cross-dataset checks, and seed checks remain absent until separately run.
```

Do not describe the model correction as effective unless the refreshed selector, decomposition, preflight, and online validation artifacts support that statement.
