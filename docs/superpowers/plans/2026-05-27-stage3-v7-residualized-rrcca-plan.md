# Stage III V7 Residualized RRCCA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in Stage III V7 alignment path that makes `phi_v - phi_i` more defensible by removing invariant leakage and retaining only train-calibrated residual-drift coordinates before generator training and served-policy validation.

**Architecture:** Keep all legacy Stage III, V2/V5/V6 configs and outputs intact. Add a new train-only residualized reduced-rank CCA canonicalizer, wire it into the existing Stage III V2 runner only behind `canonicalizer_method: residualized_rrcca_v7`, and validate it with tests before running any new experiment.

**Tech Stack:** Python, PyTorch, pytest, existing Stage III V2 runner, YAML configs, CSV audit logs.

---

## Execution Status

- Task 1 completed: V7 alignment tests and implementation are in place.
- Task 2 completed: runner and artifact opt-in integration are in place.
- Task 3 completed: V7 config was added.
- Task 4 completed for the first non-downgraded canary: Panama seed42
  `g5_b480` ran and abstained under the strict served-policy claim gate.
- Diagnostic extension completed: served-policy candidate/window/mode breakdown
  files are now written before abstain, and the same-budget rerun
  `v7ps42g5b480diag1` produced `125` diagnostic rows while still abstaining.
- Task 5 completed after the latest documentation updates:
  `121 passed in 4.28s`, `py_compile` passed, and `git diff --check` passed.

Result so far: V7 improves auditability of the representation-alignment
question, but it does not improve served-policy forecasting evidence.

---

## Task 1: V7 Alignment Core

**Files:**
- Create: `models/stage3_alignment_v7.py`
- Test: `tests/test_stage3_alignment_v7.py`

- [ ] **Step 1: Write failing tests**

Add tests for:

```python
from models.stage3_alignment_v7 import TrainSetChronologicalResidualizedReducedRankCCAAligner
```

The tests must cover train-only fit, invariant leakage residualization, active-rank truncation, zero-output fail-closed behavior, and state_dict/load_state_dict round trips.

- [ ] **Step 2: Run tests and verify red**

Run:

```powershell
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_stage3_alignment_v7.py -q
```

Expected: FAIL because `models.stage3_alignment_v7` does not exist.

- [ ] **Step 3: Implement V7 aligner**

Implement `TrainSetChronologicalResidualizedReducedRankCCAAligner`:

- `fit(phi_i_train, phi_v_train, y_inv_train, y_train)`
- `transform_phi_pair(phi_i, phi_v)`
- `transform_delta(delta_phi_v)` for already residualized deltas only
- `state_dict()` / `load_state_dict()`
- `diagnostic_summary()`

Algorithm:

1. Compute raw drift `D = phi_v - phi_i`.
2. Fit ridge residualizer `D ~ [phi_i, 1]` on train only.
3. Compute residual drift `R = D - [phi_i, 1]B`.
4. Split train chronologically into inner fit/calibration sections.
5. Fit ridge-whitened CCA between `R[t]-R[t-1]` and residual forecast transition on inner fit.
6. Select active dimensions on inner calibration using fit correlation, calibration correlation/R2, bootstrap sign stability, and `max_rank`.
7. Transform validation/test by frozen residualizer and projection; inactive dimensions are exactly zero.

- [ ] **Step 4: Run tests and verify green**

Run:

```powershell
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_stage3_alignment_v7.py -q
```

Expected: PASS.

## Task 2: Runner And Artifact Integration

**Files:**
- Modify: `train/run_stage3_v2_sensitivity_audit.py`
- Modify: `train/stage3_v2_artifacts.py`
- Test: `tests/test_stage3_v2_protocol_safety.py`
- Test: `tests/test_stage3_v2_artifacts.py`

- [ ] **Step 1: Add failing protocol tests**

Add tests proving `_fit_residual_canonicalizer()` can select `residualized_rrcca_v7` and artifact loading recognizes the V7 class.

- [ ] **Step 2: Run targeted tests and verify red**

Run:

```powershell
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_stage3_v2_protocol_safety.py::test_fit_residual_canonicalizer_can_select_residualized_rrcca_v7 tests\test_stage3_v2_artifacts.py::test_stage3_v2_artifact_bundle_loads_residualized_rrcca_v7_aligner -q
```

Expected: FAIL before integration.

- [ ] **Step 3: Add opt-in integration**

Update imports and method dispatch only for:

```yaml
online:
  stage3:
    canonicalizer_method: residualized_rrcca_v7
```

Do not alter V2/V5/V6 defaults. Add V7 diagnostic summary fields into `alignment_diagnostics.csv` when available.

- [ ] **Step 4: Run targeted tests and verify green**

Run the targeted pytest command above again. Expected: PASS.

## Task 3: V7 Config

**Files:**
- Create: `configs/stage3_v7_residualized_rrcca_served_policy_gate.yaml`

- [ ] **Step 1: Create config from V6 served-policy gate**

Keep the strict served-policy protocol:

- candidate-specific risk gate retraining;
- candidate-specific threshold calibration;
- selection modes restricted to `real`, `zero`, `sign_flipped`, `amplified_2x`, `amplified_5x`;
- no offline diagnostic mode in selection;
- `abstain_if_no_pass: true`;
- no threshold relaxation.

Change only the V7 identity fields and canonicalizer method/hyperparameters.

- [ ] **Step 2: Validate config is parseable**

Run:

```powershell
C:\Users\admin\anaconda3\python.exe -m py_compile train\run_stage3_v2_sensitivity_audit.py models\stage3_alignment_v7.py train\stage3_v2_artifacts.py
```

Expected: exit 0.

## Task 4: Non-Downgraded V7 Canary

**Files:**
- Write new logs only under `logs/s3v7*`
- Write new checkpoints only under `ckpt/s3v7*`
- Update: `docs/audit/STAGE3_V7_ALIGNMENT_RECORD.md`
- Update: `docs/audit/EXPERIMENT_EVIDENCE_LEDGER.md`
- Update: `docs/audit/RISK_REGISTER.md`
- Update: `docs/audit/REVISION_ROADMAP.md`

- [ ] **Step 1: Run Panama seed42 V7 canary**

Run with short paths and no overwrite:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v2_sensitivity_audit.py --config configs\stage3_v7_residualized_rrcca_served_policy_gate.yaml --dataset panama --run-id v7ps42g5b480r1 --seed 42 --device auto --modes real zero shuffled sign_flipped amplified_2x amplified_5x random_matched_norm --validation-windows 5 --drift-source delta_phi_v --drift-polarity 1.0 --signal-mode delta_vector --normalize-vector --epochs 5 --invariant-epochs 5 --generator-epochs 5 --max-generator-batches 480 --max-eval-batches 480 --out-root logs\s3v7p --checkpoint-root ckpt\s3v7p
```

- [ ] **Step 2: Extract claim-gate evidence**

Inspect `generator_checkpoint_selection_served_policy.csv`, `alignment_diagnostics.csv`, and checkpoint directory. Confirm whether `generator_served_policy.pt` exists only if the claim gate passed.

- [ ] **Step 3: Document results in Chinese**

Record exact MSE/MAE gains, window pass counts, failure reasons, active rank, calibration diagnostics, and whether the result is manuscript-grade.

## Task 5: Full Verification

**Files:**
- All touched files

- [ ] **Step 1: Run full tests**

Run:

```powershell
C:\Users\admin\anaconda3\python.exe -m pytest tests -q
```

Expected: all tests pass.

- [ ] **Step 2: Run diff check**

Run:

```powershell
git diff --check
```

Expected: exit 0.

- [ ] **Step 3: Final claim audit**

Do not mark Stage III complete unless the V7 canary or subsequent larger validation actually passes the pre-declared served-policy gate.
