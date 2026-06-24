# Stage III V29 History-Gate Recovery Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a non-overwriting V29 post-repair tuning planner that targets the observed history-dominance versus counterfactual-ordering conflict after the path-safe repair matrix exhausted.

**Architecture:** Add a focused planner that reads the final repair decomposition plus selector/window artifacts and emits a predeclared frozen-topology matrix. The matrix must not repeat old scalar or repair axes, must keep generator training blocked behind preflight, and must generate short path-safe run ids/configs.

**Tech Stack:** Python, YAML configs, CSV/Markdown artifacts, existing V29 execution-plan tooling, pytest.

---

### Task 1: Planner Tests

**Files:**
- Create: `tests/test_stage3_v29_history_gate_recovery_matrix.py`

- [ ] **Step 1: Write failing tests**

Tests must assert:

- the planner refuses non-final or gate-positive decompositions
- the planner creates mechanism-shift candidates after an exhausted repair matrix
- candidate axes do not repeat old scalar or repair axes
- generated configs preserve V29 topology and only modify invariant/train metadata
- generated commands are bounded and omit `--overwrite`

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_stage3_v29_history_gate_recovery_matrix.py -q
```

Expected: fail with missing `train.plan_stage3_v29_history_gate_recovery_matrix`.

### Task 2: Planner Implementation

**Files:**
- Create: `train/plan_stage3_v29_history_gate_recovery_matrix.py`
- Test: `tests/test_stage3_v29_history_gate_recovery_matrix.py`

- [ ] **Step 1: Implement public functions**

Required API:

```python
def build_history_gate_recovery_matrix(
    *,
    base_config: dict[str, Any],
    decomposition_report: dict[str, Any],
    selector_rows: list[dict[str, str]],
    window_rows: list[dict[str, str]],
    run_id_prefix: str = "v29hgr_conf015",
) -> list[dict[str, Any]]:
    ...

def write_history_gate_recovery_matrix_artifacts(
    *,
    rows: list[dict[str, Any]],
    base_config: dict[str, Any],
    config_dir: Path,
    out_dir: Path,
    prefix: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    ...
```

- [ ] **Step 2: Run unit tests**

Run:

```powershell
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_stage3_v29_history_gate_recovery_matrix.py -q
```

Expected: pass.

### Task 3: Real Artifact Generation

**Files:**
- Generate: `configs/stage3_v29_history_gate_recovery_matrix/`
- Generate: `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_history_gate_recovery_matrix.*`

- [ ] **Step 1: Generate matrix/configs**

Run:

```powershell
C:\Users\admin\anaconda3\python.exe train\plan_stage3_v29_history_gate_recovery_matrix.py --base-config configs\stage3_v29_expert_local_invariant_release_probe_tune_stability_release_deconstrained.yaml --decomposition-json artifacts\stage3_v29_teacher_gate_tuning_matrix\v29_complete_conf015_repair04_gate_failure_decomposition.json --selector-csv artifacts\stage3_v29_teacher_gate_tuning_matrix\v29_complete_conf015_repair04_selection.csv --windows-csv artifacts\stage3_v29_teacher_gate_tuning_matrix\v29_complete_conf015_repair04_selection_windows.csv --config-dir configs\stage3_v29_history_gate_recovery_matrix --out-dir artifacts\stage3_v29_teacher_gate_tuning_matrix --prefix v29_complete_conf015_history_gate_recovery_matrix
```

Expected: CSV/Markdown artifacts plus YAML configs. No training starts.

- [ ] **Step 2: Generate execution plan**

Run:

```powershell
C:\Users\admin\anaconda3\python.exe train\plan_stage3_v29_tuning_execution.py --matrix-csv artifacts\stage3_v29_teacher_gate_tuning_matrix\v29_complete_conf015_history_gate_recovery_matrix.csv --config-dir configs\stage3_v29_history_gate_recovery_matrix --out-root logs\s3v29hgr --checkpoint-root ckpt\s3v29hgr --out-dir artifacts\stage3_v29_teacher_gate_tuning_matrix --prefix v29_complete_conf015_history_gate_recovery_execution_plan
```

Expected: rows are ready or blocked only by existing outputs/path safety; commands omit `--overwrite`.

### Task 4: Documentation And Verification

**Files:**
- Modify: `docs/audit/STAGE3_V29_TUNING_PRIORITY_CORRECTION_REPORT.md`
- Modify: `docs/audit/STAGE3_V29_EXPERT_LOCAL_INVARIANT_RELEASE_IMPLEMENTATION_RECORD.md`
- Modify: `docs/audit/EXPERIMENT_EVIDENCE_LEDGER.md`
- Modify: `docs/superpowers/plans/2026-06-12-stage3-v29-expert-local-invariant-release.md`

- [ ] **Step 1: Document boundary**

Record that this is a new post-repair mechanism matrix, not a generator unlock or performance claim.

- [ ] **Step 2: Verify**

Run:

```powershell
C:\Users\admin\anaconda3\python.exe -m py_compile train\plan_stage3_v29_history_gate_recovery_matrix.py train\plan_stage3_v29_tuning_execution.py
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_stage3_v29_history_gate_recovery_matrix.py tests\test_stage3_v29_tuning_execution_plan.py -q
```

Expected: compile exits `0`; tests pass.

## 2026-06-13 Execution Notes

- Planner and tests were added:
  `train/plan_stage3_v29_history_gate_recovery_matrix.py` and
  `tests/test_stage3_v29_history_gate_recovery_matrix.py`.
- Real matrix artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_history_gate_recovery_matrix.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_history_gate_recovery_matrix.md`.
- Real config directory:
  `configs/stage3_v29_history_gate_recovery_matrix/`.
- Initial execution plan:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_history_gate_recovery_execution_plan.csv`,
  with four `ready_to_run` rows.
- Candidate `01` (`v29hgr_conf015_01_w0ramp`) completed and failed the teacher
  gate. It ranks `2/33`, has `2/4` positive partial windows, min partial MSE
  `-0.43209738075898796`, min partial MAE `-0.4597064241028177`, and
  counterfactual margin `-0.014453849719397449`.
- Post-candidate artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr01_selection.csv`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr01_generator_unlock_preflight.json`,
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr01_gate_failure_decomposition.json`.
- Updated execution plan:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_history_gate_recovery_execution_plan_after_hgr01.csv`.
- Current safe continuation rule: do not rerun candidate `01`. If continuing,
  run at most one of candidates `02`, `03`, or `04`, then refresh selector,
  gate-failure decomposition, and generator-unlock preflight.
- Candidate `02` (`v29hgr_conf015_02_latehist`) completed and failed the
  teacher gate. It ranks `1/34`, has `2/4` positive partial windows, min
  partial MSE `-0.5290475768525864`, min partial MAE
  `-0.5365663873681754`, and counterfactual margin
  `-0.009411902842471953`.
- Post-candidate artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr02_selection.csv`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr02_generator_unlock_preflight.json`,
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr02_gate_failure_decomposition.json`.
- Updated execution plan:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_history_gate_recovery_execution_plan_after_hgr02.csv`.
- Current safe continuation rule: do not rerun candidates `01` or `02`. If
  continuing, run at most one of candidates `03` or `04`, then refresh selector,
  gate-failure decomposition, and generator-unlock preflight.
- Candidate `03` (`v29hgr_conf015_03_cftie`) completed and failed the teacher
  gate. It ranks `1/35`, has `2/4` positive partial windows, min partial MSE
  `-0.41351019804173306`, min partial MAE `-0.466581375695994`, and
  counterfactual margin `-0.006009365114644338`.
- Post-candidate artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr03_selection.csv`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr03_generator_unlock_preflight.json`,
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr03_gate_failure_decomposition.json`.
- Updated execution plan:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_history_gate_recovery_execution_plan_after_hgr03.csv`.
- Current safe continuation rule: do not rerun candidates `01`, `02`, or `03`.
  If continuing, run candidate `04`, then refresh selector, gate-failure
  decomposition, and generator-unlock preflight.
- Candidate `04` (`v29hgr_conf015_04_histcap`) completed and failed the
  teacher gate. It ranks `9/36`, has `2/4` positive partial windows, min
  partial MSE `-0.4066591740244906`, min partial MAE
  `-0.4354709147954243`, and counterfactual margin
  `-0.030275156997368824`.
- Post-candidate artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr04_selection.csv`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr04_generator_unlock_preflight.json`,
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr04_gate_failure_decomposition.json`.
- Final execution plan:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_history_gate_recovery_execution_plan_after_hgr04.csv`.
- Final matrix decision: `stop_local_scalar_matrix`. Do not rerun candidates
  `01`-`04`; all are blocked by existing outputs and none passed the primary
  teacher gate or generator-unlock preflight.
