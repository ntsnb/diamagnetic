# Stage III V15 Residual-Target Predictability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a non-overwriting V15 diagnostic that tests whether `fluctuation_delta` predicts pre-declared residual targets beyond causal residual history.

**Architecture:** V15 reuses the existing V13 signal contract, V12e causal history builder, and V14 stability teacher readout. It adds a thin target-predictability runner, a locked V15 config, focused tests, and an audit record. It does not train a generator or modify V12-V14 files.

**Tech Stack:** Python, PyTorch, pytest, YAML config, existing Stage III training utilities.

---

## File Structure

- Create `tests/test_stage3_v15_residual_target_predictability.py`
  - Unit tests for target construction, feature-mode configuration, stop rules, and no-overwrite defaults.
- Create `configs/stage3_v15_residual_target_predictability_probe.yaml`
  - Versioned config with V15-only output roots and pre-declared targets.
- Create `train/run_stage3_v15_residual_target_predictability_probe.py`
  - Thin diagnostic runner that reuses V14 data/model/signal/history utilities and trains V14-compatible teachers against V15 residual targets.
- Create `train/run_stage3_v15_residual_target_predictability_matrix.py`
  - Optional matrix runner for multi-dataset or multi-seed execution.
- Create `docs/audit/STAGE3_V15_RESIDUAL_TARGET_PREDICTABILITY_RECORD.md`
  - Design, command history, results, and interpretation ledger.
- Update `docs/audit/EXPERIMENT_EVIDENCE_LEDGER.md`
  - Add the V15 diagnostic run and artifact pointers after execution.
- Update `docs/audit/RISK_REGISTER.md`
  - Add remaining risk around target predictability and target shopping.
- Do not modify V12, V13, or V14 implementation files.

## Task 1: Tests For V15 Target Contract

**Files:**
- Create: `tests/test_stage3_v15_residual_target_predictability.py`
- Create later: `train/run_stage3_v15_residual_target_predictability_probe.py`

- [ ] **Step 1: Write failing tests**

Add tests that import the future V15 helpers:

```python
import torch
import pytest

from train.run_stage3_v15_residual_target_predictability_probe import (
    PRIMARY_V15_TARGETS,
    build_v15_target_tensor,
    v15_target_to_teacher_tensors,
)


def test_v15_primary_targets_are_predeclared() -> None:
    assert PRIMARY_V15_TARGETS == ("raw_residual_vector", "residual_change_vector")


def test_v15_raw_residual_target_is_y_minus_base() -> None:
    y_base = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y_true = torch.tensor([[1.5, 1.0], [2.0, 6.0]])
    target = build_v15_target_tensor("raw_residual_vector", y_base=y_base, y_true=y_true)
    assert torch.allclose(target, torch.tensor([[0.5, -1.0], [-1.0, 2.0]]))


def test_v15_residual_change_uses_previous_tail() -> None:
    y_base = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y_true = torch.tensor([[1.5, 1.0], [2.0, 6.0]])
    previous_tail = torch.tensor([[0.25, -0.25]])
    target = build_v15_target_tensor(
        "residual_change_vector",
        y_base=y_base,
        y_true=y_true,
        previous_residual_tail=previous_tail,
    )
    assert torch.allclose(target, torch.tensor([[0.25, -0.75], [-1.5, 3.0]]))


def test_v15_teacher_tensors_make_target_the_residual() -> None:
    target = torch.tensor([[0.5, -1.0], [-1.0, 2.0]])
    y_base_target, y_true_target = v15_target_to_teacher_tensors(target)
    assert torch.allclose(y_base_target, torch.zeros_like(target))
    assert torch.allclose(y_true_target, target)
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
pytest tests\test_stage3_v15_residual_target_predictability.py -q
```

Expected: fail because the V15 runner module does not exist yet.

## Task 2: V15 Helper Implementation

**Files:**
- Create: `train/run_stage3_v15_residual_target_predictability_probe.py`
- Test: `tests/test_stage3_v15_residual_target_predictability.py`

- [ ] **Step 1: Add minimal helper code**

Add constants and helper functions:

```python
PRIMARY_V15_TARGETS = ("raw_residual_vector", "residual_change_vector")
SECONDARY_V15_TARGETS = ("residual_sign",)
V15_TARGETS = PRIMARY_V15_TARGETS + SECONDARY_V15_TARGETS


def build_v15_target_tensor(
    target_name: str,
    *,
    y_base: torch.Tensor,
    y_true: torch.Tensor,
    previous_residual_tail: torch.Tensor | None = None,
) -> torch.Tensor:
    residual = y_true.detach().float().reshape(y_true.shape[0], -1) - y_base.detach().float().reshape(y_base.shape[0], -1)
    if target_name == "raw_residual_vector":
        return residual
    if target_name == "residual_change_vector":
        if previous_residual_tail is None or previous_residual_tail.numel() == 0:
            previous = torch.cat([residual[:1], residual[:-1]], dim=0)
        else:
            tail = previous_residual_tail.detach().float().reshape(-1, residual.shape[1]).to(device=residual.device)
            previous = torch.cat([tail[-1:], residual[:-1]], dim=0)
        return residual - previous
    if target_name == "residual_sign":
        return torch.sign(residual)
    raise ValueError(f"unsupported V15 target: {target_name}")


def v15_target_to_teacher_tensors(target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    target_2d = target.detach().float().reshape(target.shape[0], -1)
    return torch.zeros_like(target_2d), target_2d
```

- [ ] **Step 2: Run target tests**

Run:

```powershell
pytest tests\test_stage3_v15_residual_target_predictability.py -q
```

Expected: target tests pass.

## Task 3: V15 Config Lock Tests

**Files:**
- Modify: `tests/test_stage3_v15_residual_target_predictability.py`
- Create: `configs/stage3_v15_residual_target_predictability_probe.yaml`

- [ ] **Step 1: Add failing config test**

Add:

```python
from pathlib import Path


def test_v15_config_locks_non_overwriting_diagnostic_protocol() -> None:
    text = Path("configs/stage3_v15_residual_target_predictability_probe.yaml").read_text(encoding="utf-8")
    assert "protocol_version: v15_residual_target_predictability_probe" in text
    assert "version: v15_residual_target_predictability_probe" in text
    assert "signal_mode: fluctuation_delta" in text
    assert "drift_source: delta_phi_v" in text
    assert "logs/s3v15target" in text
    assert "ckpt/s3v15target" in text
    assert "results/s3v15target" in text
    assert "raw_residual_vector" in text
    assert "residual_change_vector" in text
    assert "fluctuation_plus_history" in text
    assert "history_only" in text
    assert "run_generator" not in text
```

- [ ] **Step 2: Run test and confirm failure**

Run:

```powershell
pytest tests\test_stage3_v15_residual_target_predictability.py::test_v15_config_locks_non_overwriting_diagnostic_protocol -q
```

Expected: fail because config does not exist yet.

- [ ] **Step 3: Create V15 config**

Create `configs/stage3_v15_residual_target_predictability_probe.yaml` by copying V14 config semantics but changing:

```yaml
stage3_v2:
  protocol_version: v15_residual_target_predictability_probe
  log_root: logs/s3v15target
  checkpoint_root: ckpt/s3v15target
  results_root: results/s3v15target
  run_id: manual_stage3_v15_residual_target_predictability_probe
  v15_targets:
    primary:
      - raw_residual_vector
      - residual_change_vector
    secondary:
      - residual_sign
  v15_feature_modes:
    - history_only
    - fluctuation_only
    - fluctuation_plus_history
    - level_offset_only
    - level_offset_plus_history
online:
  stage3:
    version: v15_residual_target_predictability_probe
    enable: true
    allow_untrained_generator: false
    drift_source: delta_phi_v
    signal_mode: fluctuation_delta
```

- [ ] **Step 4: Run config test**

Run:

```powershell
pytest tests\test_stage3_v15_residual_target_predictability.py::test_v15_config_locks_non_overwriting_diagnostic_protocol -q
```

Expected: pass.

## Task 4: V15 Runner

**Files:**
- Modify: `train/run_stage3_v15_residual_target_predictability_probe.py`
- Test: `tests/test_stage3_v15_residual_target_predictability.py`

- [ ] **Step 1: Add runner shape and stop-rule tests**

Add tests for:

```python
from train.run_stage3_v15_residual_target_predictability_probe import (
    DEFAULT_V15_FEATURE_MODES,
    v15_directional_stop_reasons,
)


def test_v15_feature_modes_include_primary_and_auxiliary_controls() -> None:
    assert DEFAULT_V15_FEATURE_MODES == (
        "history_only",
        "fluctuation_only",
        "fluctuation_plus_history",
        "level_offset_only",
        "level_offset_plus_history",
    )


def test_v15_stop_reasons_block_generator_when_primary_targets_fail() -> None:
    rows = [
        {"target_name": "raw_residual_vector", "target_role": "primary", "primary_gate_pass": False},
        {"target_name": "residual_change_vector", "target_role": "primary", "primary_gate_pass": False},
    ]
    reasons = v15_directional_stop_reasons(rows)
    assert "no_primary_target_passed" in reasons
    assert "do_not_train_generator" in reasons
```

- [ ] **Step 2: Implement runner constants and stop reason helper**

Add:

```python
DEFAULT_V15_FEATURE_MODES = (
    "history_only",
    "fluctuation_only",
    "fluctuation_plus_history",
    "level_offset_only",
    "level_offset_plus_history",
)


def v15_directional_stop_reasons(target_gate_rows: list[dict[str, object]]) -> list[str]:
    primary_rows = [row for row in target_gate_rows if str(row.get("target_role", "")) == "primary"]
    passed = [row for row in primary_rows if bool(row.get("primary_gate_pass", False))]
    if passed:
        return []
    return ["do_not_train_generator", "no_primary_target_passed"]
```

- [ ] **Step 3: Implement actual run function**

Implement `run_stage3_v15_residual_target_predictability_probe(args)` by reusing V14 utilities:

- `_apply_dataset_override`
- `_build_dataloaders`
- `_collect_split_tensors`
- `_build_history_builder`
- `Stage3SignalBuilderV13`
- `build_stage3_v14_context_features`
- `TrainOnlyStabilityFluctuationResidualTeacherV14`
- `_rows_for_split` pattern, adapted for target tensors

The run must:

- write to V15-only roots;
- train base and aligned representation exactly as V14 does;
- collect raw drift, base predictions, and labels for train/validation/test;
- build `raw_residual_vector` and `residual_change_vector` targets;
- train one teacher per target and feature mode;
- compute validation, rolling validation, and test metrics;
- write `teacher_only_abort.json` unless a primary target passes;
- never instantiate or train a generator.

- [ ] **Step 4: Run module help**

Run:

```powershell
python train\run_stage3_v15_residual_target_predictability_probe.py --help
```

Expected: help text prints.

## Task 5: Matrix Runner

**Files:**
- Create: `train/run_stage3_v15_residual_target_predictability_matrix.py`

- [ ] **Step 1: Create thin matrix runner**

Copy V14 matrix style but point to:

```text
train/run_stage3_v15_residual_target_predictability_probe.py
configs/stage3_v15_residual_target_predictability_probe.yaml
logs/s3v15target_matrix
ckpt/s3v15target_matrix
```

- [ ] **Step 2: Run matrix help**

Run:

```powershell
python train\run_stage3_v15_residual_target_predictability_matrix.py --help
```

Expected: help text prints.

## Task 6: Audit Record

**Files:**
- Create: `docs/audit/STAGE3_V15_RESIDUAL_TARGET_PREDICTABILITY_RECORD.md`
- Modify after run: `docs/audit/EXPERIMENT_EVIDENCE_LEDGER.md`
- Modify after run: `docs/audit/RISK_REGISTER.md`

- [ ] **Step 1: Create pre-run record**

Record:

- design file path;
- plan file path;
- command template;
- stop rules;
- no-generator statement;
- pending result section.

- [ ] **Step 2: Update record after experiment**

Add exact command, output directory, manifest path, pass/fail status, and interpretation.

## Task 7: Verification And Experiment

**Files:**
- No code edits after this task unless tests fail.

- [ ] **Step 1: Run targeted tests**

Run:

```powershell
pytest tests\test_stage3_v15_residual_target_predictability.py -q
```

Expected: all V15 tests pass.

- [ ] **Step 2: Run syntax checks**

Run:

```powershell
python -m py_compile train\run_stage3_v15_residual_target_predictability_probe.py train\run_stage3_v15_residual_target_predictability_matrix.py
```

Expected: no output and zero exit code.

- [ ] **Step 3: Run real Panama b480 diagnostic**

Run:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v15_residual_target_predictability_probe.py --config configs\stage3_v15_residual_target_predictability_probe.yaml --dataset panama --run-id v15target_panama_s42_b480_01 --seed 42 --epochs 5 --invariant-epochs 5 --device auto --max-train-batches 480 --max-generator-batches 480 --max-eval-batches 480 --validation-windows 5 --out-root logs\s3v15target_b480 --checkpoint-root ckpt\s3v15target_b480
```

Expected: a V15-only output directory with `manifest.json`, target diagnostic CSV files, and either `teacher_only_abort.json` or `teacher_only_complete.json`.

- [ ] **Step 4: Report results**

Report:

- whether any primary target passed;
- target-level partial gains for `fluctuation_plus_history` versus `history_only`;
- rolling window pass counts;
- whether generator training is blocked;
- exact artifact paths.

## Self-Review

- Spec coverage: all V15 design requirements are covered by tasks.
- Placeholder scan: no placeholder tasks remain.
- Type consistency: helper names and target names are consistent across tests and plan.
- Scope: plan is limited to diagnostic implementation and one real b480 experiment.
