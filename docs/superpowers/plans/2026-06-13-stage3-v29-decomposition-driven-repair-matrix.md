# Stage III V29 Decomposition-Driven Repair Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a non-training V29 planner that converts the gate-failure decomposition into a predeclared repair matrix and safe execution plan without changing model topology or overwriting prior V29 artifacts.

**Architecture:** Add one focused planner under `train/` that reads the existing gate-failure decomposition JSON and the current V29 stability-anchor config, then emits YAML configs plus CSV/Markdown matrix artifacts. Reuse the existing safe execution planner so candidate commands stay bounded, no-overwrite, and generator-locked until preflight passes.

**Tech Stack:** Python standard library, `yaml`, existing V29 config schema, existing `train/plan_stage3_v29_tuning_execution.py`, `pytest`.

---

## File Map

- Create `train/plan_stage3_v29_decomposition_repair_matrix.py`: read decomposition evidence, derive repair candidates, merge config overrides, write CSV/Markdown/YAML artifacts.
- Create `tests/test_stage3_v29_decomposition_repair_matrix.py`: TDD coverage for candidate generation, config writing, no-overwrite behavior, and old-axis guards.
- Modify `docs/audit/STAGE3_V29_TUNING_PRIORITY_CORRECTION_REPORT.md`: append planner artifacts and decision boundary.
- Modify `docs/audit/STAGE3_V29_EXPERT_LOCAL_INVARIANT_RELEASE_IMPLEMENTATION_RECORD.md`: append implementation record item.
- Modify `docs/audit/EXPERIMENT_EVIDENCE_LEDGER.md`: add an evidence row for the planner and generated queue.
- Modify `docs/superpowers/plans/2026-06-12-stage3-v29-expert-local-invariant-release.md`: append continuation note.

## Planned Candidate Design

The planner must generate a small predeclared matrix with four mechanism-labeled candidates:

- `window0_partial_floor_rebalance`: targets window `0` partial-floor failures without increasing counterfactual pressure.
- `counterfactual_ordering_without_floor_pressure`: targets windows like `2`, where partial floor passes but counterfactual ordering fails.
- `dual_axis_floor_and_cf_coupled`: couples conservative partial-floor support with counterfactual ordering, because the decomposition shows both axes coexist.
- `mode_abstention_teacher_selection_probe`: changes teacher-selection/admission controls rather than representation topology, used to test whether stricter mode admission is safer than more correction.

Every row must include:

- `source_decomposition_protocol=stage3_v29_gate_failure_decomposition_v1`
- `architecture_policy=frozen_v29_no_model_topology_change`
- `generator_policy=preflight_required_do_not_unlock_from_matrix`
- `known_failed_axis_guard=not_one_of_prior_five_scalar_axes`
- `addresses_failure_axes` containing both relevant decomposition axes or a single explicitly isolated diagnostic axis.

## Task 1: Planner Test First

**Files:**
- Create: `tests/test_stage3_v29_decomposition_repair_matrix.py`

- [ ] **Step 1: Write failing tests**

```python
def test_decomposition_repair_matrix_uses_failure_axes_and_avoids_old_axes(tmp_path):
    decomposition = {
        "protocol_version": "stage3_v29_gate_failure_decomposition_v1",
        "matrix_decision": "stop_local_scalar_matrix",
        "dominant_failure_axes": [
            "partial_floor_violation",
            "counterfactual_ordering_failure",
            "window_level_conflict",
        ],
        "window_summaries": [
            {"validation_window_id": "0", "partial_mse_floor_failures": 27, "counterfactual_negative_rows": 25},
            {"validation_window_id": "2", "partial_mse_floor_failures": 0, "counterfactual_negative_rows": 25},
        ],
    }
    rows = build_decomposition_repair_matrix(base_config=sample_config(), decomposition_report=decomposition)
    assert len(rows) == 4
    assert {row["axis"] for row in rows}.isdisjoint({
        "history_guarded_release_floor",
        "counterfactual_margin_micro_repair",
        "late_window_partial_floor_guard",
        "delayed_change_tempered_history",
        "release_null_coupled_guard",
    })
    assert any("partial_floor_violation" in row["addresses_failure_axes"] for row in rows)
    assert any("counterfactual_ordering_failure" in row["addresses_failure_axes"] for row in rows)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests\test_stage3_v29_decomposition_repair_matrix.py -q`

Expected: FAIL with `ModuleNotFoundError` for `train.plan_stage3_v29_decomposition_repair_matrix`.

## Task 2: Planner Implementation

**Files:**
- Create: `train/plan_stage3_v29_decomposition_repair_matrix.py`
- Test: `tests/test_stage3_v29_decomposition_repair_matrix.py`

- [ ] **Step 1: Implement focused planner API**

Required public functions:

```python
def build_decomposition_repair_matrix(*, base_config: dict[str, Any], decomposition_report: dict[str, Any], run_id_prefix: str = "v29repair") -> list[dict[str, Any]]:
    ...

def write_decomposition_repair_matrix_artifacts(*, rows: list[dict[str, Any]], base_config: dict[str, Any], config_dir: Path, out_dir: Path, prefix: str, overwrite: bool = False) -> dict[str, str]:
    ...
```

- [ ] **Step 2: Run unit tests**

Run: `pytest tests\test_stage3_v29_decomposition_repair_matrix.py -q`

Expected: PASS.

## Task 3: Real Artifact Generation

**Files:**
- Generate under: `configs/stage3_v29_decomposition_repair_matrix/`
- Generate under: `artifacts/stage3_v29_teacher_gate_tuning_matrix/`

- [ ] **Step 1: Generate matrix and configs**

Run:

```powershell
C:\Users\admin\anaconda3\python.exe train\plan_stage3_v29_decomposition_repair_matrix.py --base-config configs\stage3_v29_expert_local_invariant_release_probe_tune_stability_release_deconstrained.yaml --decomposition-json artifacts\stage3_v29_teacher_gate_tuning_matrix\v29_complete_conf015_matrix_gate_failure_decomposition.json --config-dir configs\stage3_v29_decomposition_repair_matrix --out-dir artifacts\stage3_v29_teacher_gate_tuning_matrix --prefix v29_complete_conf015_decomposition_repair_matrix
```

Expected: CSV/Markdown artifacts plus four YAML configs. No training starts.

- [ ] **Step 2: Generate safe execution plan**

Run:

```powershell
C:\Users\admin\anaconda3\python.exe train\plan_stage3_v29_tuning_execution.py --matrix-csv artifacts\stage3_v29_teacher_gate_tuning_matrix\v29_complete_conf015_decomposition_repair_matrix.csv --config-dir configs\stage3_v29_decomposition_repair_matrix --out-root logs\s3v29expert_decomposition_repair_matrix --checkpoint-root ckpt\s3v29expert_decomposition_repair_matrix --out-dir artifacts\stage3_v29_teacher_gate_tuning_matrix --prefix v29_complete_conf015_decomposition_repair_execution_plan
```

Expected: execution rows are `ready_to_run` or blocked only by existing outputs, commands omit `--overwrite`, and `weather` is not passed as a dataset override.

## Task 4: Documentation

**Files:**
- Modify: `docs/audit/STAGE3_V29_TUNING_PRIORITY_CORRECTION_REPORT.md`
- Modify: `docs/audit/STAGE3_V29_EXPERT_LOCAL_INVARIANT_RELEASE_IMPLEMENTATION_RECORD.md`
- Modify: `docs/audit/EXPERIMENT_EVIDENCE_LEDGER.md`
- Modify: `docs/superpowers/plans/2026-06-12-stage3-v29-expert-local-invariant-release.md`

- [ ] **Step 1: Append evidence**

Record the planner path, generated artifacts, candidate axes, execution plan path, and explicit boundary: these are predeclared next-run candidates only, not performance evidence and not generator unlock.

## Task 5: Verification

**Files:**
- Verify: planner, execution planner, V29 tests

- [ ] **Step 1: Compile**

Run:

```powershell
C:\Users\admin\anaconda3\python.exe -m py_compile train\plan_stage3_v29_decomposition_repair_matrix.py train\plan_stage3_v29_tuning_execution.py train\analyze_stage3_v29_gate_failure_decomposition.py
```

Expected: exit code `0`.

- [ ] **Step 2: Test**

Run:

```powershell
pytest tests\test_stage3_v29_decomposition_repair_matrix.py tests\test_stage3_v29_gate_failure_decomposition.py tests\test_stage3_v29_tuning_execution_plan.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Broader V29 targeted tests**

Run the current V29 targeted test set including queue, selector, preflight, matrix, execution plan, gate-failure decomposition, and new repair planner.

Expected: no regressions.

## 2026-06-13 Execution Notes

- Original repair candidate `01`
  (`v29repair_conf015_01_window0_partial_floor_rebalance`) completed under the
  bounded weather seed42 budget and failed the teacher gate. It ranks `9/28`,
  has `2/4` positive partial windows, min partial MSE
  `-0.5455704944904651`, min partial MAE `-0.5699537539405484`, and
  counterfactual margin `-0.04551155267896673`. It must not be rerun.
- Original repair candidate `02`
  (`v29repair_conf015_02_counterfactual_ordering_without_floor_pressure`) hit a
  Windows/PyTorch checkpoint path-length failure before completion. The failing
  checkpoint path length was `263` characters, so that partial run is invalid as
  model-quality evidence.
- The planner now emits short path-safe run slugs, and the execution planner
  predicts maximum checkpoint path length before a command is treated as safe.
  Unsafe rows are blocked with `checkpoint_file_path_too_long`.
- Path-safe queue artifacts:
  `configs/stage3_v29_decomposition_repair_matrix_pathsafe/`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_matrix_pathsafe.csv`,
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_execution_plan_pathsafe.csv`.
- Path-safe repair candidate `02` (`v29rps_conf015_02_cforder`) completed and
  also failed the teacher gate. It ranks `7/29`, has `2/4` positive partial
  windows, min partial MSE `-0.4204164211957788`, min partial MAE
  `-0.500599967584385`, and counterfactual margin
  `-0.04310910574869553`.
- Path-safe repair candidate `03` (`v29rps_conf015_03_floorcf`) completed and
  also failed the teacher gate. It ranks `4/31`, has `2/4` positive partial
  windows, min partial MSE `-0.5299312143911022`, min partial MAE
  `-0.545154931341216`, and counterfactual margin
  `-0.024159312022691004`.
- Path-safe repair candidate `04` (`v29rps_conf015_04_abstain`) completed and
  also failed the teacher gate. It ranks `6/32`, has `2/4` positive partial
  windows, min partial MSE `-0.46285146776087893`, min partial MAE
  `-0.5138718209477093`, and counterfactual margin
  `-0.03642005087344469`.
- Final refreshed path-safe post-candidate plan after candidate `04`:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_execution_plan_pathsafe_after_repair04.csv`.
- Refreshed path-safe post-candidate plan after candidate `03`:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_execution_plan_pathsafe_after_repair03.csv`.
- Prior refreshed path-safe post-candidate plan after candidate `02`:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_execution_plan_pathsafe_after_repair02.csv`.
- Current safe continuation rule: do not use the original long-run-id queue,
  do not rerun path-safe candidate `01` because its long-run equivalent already
  completed negatively, and do not rerun path-safe candidates `02`, `03`, or
  `04` because they are complete and negative. The path-safe repair matrix is
  exhausted in the meaningful sense. Any next run must come from a new
  predeclared diagnosis or matrix, not from blind continuation of these four
  axes.
