# Stage III Generator Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to execute this plan with independent implementation, testing, and review workstreams. If only one worker is available, use `superpowers:executing-plans` and complete each checkbox in order.

**Goal:** Rebuild Stage III from a weakly justified, untrained transient perturbation path into a testable drift-conditioned online adaptation module. The repaired path must satisfy: deterministic evaluation, meaningful vector drift input, identity/no-op behavior for zero drift, explicit generator training, representation alignment, and sensitivity evidence before any paper claim is upgraded.

**Architecture:** Keep the original Stage III code path intact until the repaired path passes diagnostics. Add a versioned Stage III V2 path with deterministic VAE evaluation, aligned invariant/variant embeddings, trained identity-preserving generator, and replayable sensitivity audits. Do not overwrite raw data, logs, checkpoints, existing result tables, or generated figures.

**Tech Stack:** Python, PyTorch, existing `models/`, `train/`, `configs/`, `tests/`, YAML experiment configs, CSV/JSONL logs.

---

## Current Diagnosis To Preserve

The current codebase evidence shows these defects:

1. `AdaptationGenerator` is instantiated inside `train/online_loop.py` but is not trained by any optimizer over `gen.parameters()`.
2. Zero drift produces nonzero parameter deltas because the generator uses biased linear layers and `sigmoid(...) + 1.0`.
3. Original Stage III compresses vector drift into `d_v.mean()` and repeats that scalar, losing direction information.
4. `FactorMechanism` samples VAE latent variables during `eval()`, making evaluation metrics stochastic when `use_vae=True`.
5. `encoder_i` and `encoder_v` are trained by different objectives without shared coordinate calibration, so `phi_v - phi_i` is not guaranteed to be a semantically stable drift vector.

The repair should not assume Stage III is valid. It must make validity measurable.

---

## 2026-06-01 Execution Update: V11 Fail-Closed Result And V12 Boundary

V11 has now been executed as a teacher-only safety-abstention repair. It passed
code-level checks, but failed as a performance mechanism: public-3/seeds
`42/43/44` produced `0/9` teacher gates and `0/45` rolling validation windows.
All main-matrix teachers abstained with `teacher_shrinkage_alpha=0.0`, and no
generator artifact was produced.

The key planning consequence is:

```text
Do not continue by training a larger generator.
First prove that the representation can predict residual correction
prospectively.
```

## Phase 12: V12 Outcome-Aligned Transport Evidence Chain Before Generator Training

### Goal

Convert V12 from an alignment protocol into a complete, auditable, train-only
representation-to-residual evidence chain. V12 must remain opt-in and must not
overwrite V7/V10/V11 code, logs, checkpoints, tables, or figures.

### Implemented Code-Level Foundation

- [x] Add `models/stage3_alignment_v12.py`.
- [x] Add train-only shared latent anchor before subtractive transport.
- [x] Add zero-preserving orthogonal leakage removal.
- [x] Add chronological residual-transfer active-dimension calibration.
- [x] Add fail-closed behavior when residual transfer is unstable.
- [x] Add `models/stage3_residual_teacher_v12.py` as a V12 protocol wrapper
      around the strict V11 block/reliability teacher.
- [x] Add `train/stage3_v12_artifacts.py`.
- [x] Add `train/run_stage3_v12_sensitivity_audit.py`.
- [x] Add `train/run_stage3_v12_teacher_audit_matrix.py`.
- [x] Add `configs/stage3_v12_outcome_aligned_teacher_gate.yaml`.

### Tests First

- [x] `tests/test_stage3_v12_alignment_protocol.py` passes.
- [x] V12 rejects non-train fit.
- [x] V12 preserves zero-drift identity.
- [x] V12 removes invariant leakage without destroying useful residual-transfer
      signal in the toy protocol.
- [x] V12 fails closed when residual transfer is unstable.
- [x] V12 state dict reload reproduces transformed drift.
- [x] `tests/test_stage3_v12_teacher_protocol.py` passes.
- [x] `tests/test_stage3_v12_artifacts.py` passes.

Current verification:

```text
py_compile V12 modules and runners: passed
pytest tests/test_stage3_v12_alignment_protocol.py \
       tests/test_stage3_v12_teacher_protocol.py \
       tests/test_stage3_v12_artifacts.py -q: 6 passed
git diff --check: passed
```

### Smoke Status

- [x] Direct V12 teacher-only smoke ran on Panama seed42 with tiny budget.
- [x] V12 matrix smoke ran on Panama seed42 with tiny budget.
- [x] Both smoke paths kept generator training blocked.
- [x] No `generator*.pt` artifact was produced.

Smoke is not performance evidence. The smoke result currently fail-closed:
`active_rank=0`, `teacher_alpha_source=fallback_block_gate`, `alpha=0`,
rolling teacher-only gate `0/2`.

### Acceptance Before Generator

- [ ] single-dataset teacher-only canary passes `5/5` rolling windows;
- [ ] public-3/seeds `42/43/44` pass pre-declared teacher gates;
- [ ] nonzero teacher alpha is selected prospectively, not post hoc;
- [ ] real drift beats counterfactual drift modes by pre-declared margins;
- [ ] no generator artifact is saved unless all teacher-only gates pass.

If Phase 12 fails, record it as negative evidence and keep Stage III claims
downgraded.

## 2026-06-01 Execution Update: V12 Public-3 Matrix Failed The Teacher Gate

The V12 public-3 teacher-only matrix has now been executed:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v12_teacher_audit_matrix.py --config configs\stage3_v12_outcome_aligned_teacher_gate.yaml --datasets panama tetouan spain_energy_weather --seeds 42 43 44 --epochs 5 --invariant-epochs 5 --generator-epochs 5 --max-generator-batches 480 --max-eval-batches 480 --validation-windows 5 --device auto --out-root logs\s3v12m --checkpoint-root ckpt\s3v12m --run-prefix v12tm
```

Evidence:

- Manifest:
  `logs/s3v12m/_manifests/v12tm_20260601_134623_e5_ie5_b480_ew480/matrix_manifest.json`.
- Summary:
  `logs/s3v12m/_manifests/v12tm_20260601_134623_e5_ie5_b480_ew480/matrix_summary.csv`.
- Record:
  `docs/audit/STAGE3_V12_OUTCOME_ALIGNED_TRANSPORT_RECORD.md`.

Verified result:

| Item | Result |
| --- | ---: |
| completed runs | `9/9` |
| alignment gate pass | `9/9` |
| teacher gate pass | `0/9` |
| rolling validation windows passed | `0/45` |
| nonzero teacher alpha | `0/9` |
| teacher alpha source | all `fallback_block_gate` |
| teacher-only aborts | `9/9` |
| generator artifacts | `0` |

Planning consequence:

```text
Phase 12 improved the representation audit but failed the teacher evidence
chain. Do not train a V12 generator. Do not upgrade manuscript claims.
```

The failure is now more specific than V11. V12 shows that the shared-anchor
alignment path can pass in public-3 (`alignment_gate_pass=9/9`), but the
current V12 teacher is still a V11-style strict wrapper and cannot convert the
aligned drift signal into stable residual correction. The next phase is V12b.

## Phase 13: V12b Chronological OOF Residual Teacher

### Goal

Replace the V12 wrapper teacher with a genuine train-only, chronological,
out-of-fold residual teacher that uses the same signal contract as the aligner
and produces detailed pre-gate diagnostics. This phase must remain opt-in and
must not modify V7/V10/V11/V12 historical files, logs, checkpoints, result
tables, or generated figures.

### Required New Files

- [x] `train/stage3_v12b_signal_contract.py`
- [x] `models/stage3_residual_teacher_v12b.py`
- [x] `train/run_stage3_v12b_sensitivity_audit.py`
- [x] `train/run_stage3_v12b_teacher_audit_matrix.py`
- [x] `configs/stage3_v12b_oof_teacher_gate.yaml`
- [x] `tests/test_stage3_v12b_signal_contract.py`
- [x] `tests/test_stage3_v12b_teacher_protocol.py`
- [x] `tests/test_stage3_v12b_oof_no_leakage.py`
- [x] `tests/test_stage3_v12b_gate_diagnostics.py`

### Design Requirements

- [x] Use one consistent `y_base` definition across aligner, teacher fit,
      teacher-only diagnostics, and served inference.
- [x] Build a shared `Stage3TeacherSignalBuilderV12b` so level/delta choice,
      train-only scale, zero mask, and counterfactual intervention are identical
      across fit/eval.
- [x] Use purged forward-chaining OOF folds; each fold can only fit on
      chronological prefix data before its calibration block.
- [x] Use OOF correction as the only basis for selecting ridge/rank/alpha/sign
      and per-horizon/per-output shrinkage.
- [x] Preserve zero-drift identity with no intercept and explicit zero masks.
- [x] Replace global all-or-nothing `block_scalar` shrinkage with grouped
      per-horizon/per-output shrinkage, while keeping final strict claim gates.
- [x] Write alpha-by-block and horizon-by-block pre-gate diagnostics before
      any final abstention.
- [x] Fail closed to zero correction if OOF evidence is unstable.

### Acceptance Before Generator

- [x] V12b unit tests pass.
- [x] V12b smoke writes teacher diagnostics, OOF alpha/block tables, and
      `teacher_only_abort.json` or `teacher_only_complete.json`.
- [ ] Single-dataset teacher-only canary passes `5/5` rolling windows with
      nonzero alpha.
- [ ] Public-3/seeds `42/43/44` teacher-only matrix passes pre-declared gates.
- [ ] Real drift beats zero, shuffled, sign-flipped, amplified, and
      random-matched-norm controls by pre-declared margins.
- [ ] Only after these conditions may a generator canary be trained.

### 2026-06-01 Outcome

V12b implementation and tests completed, but the scientific gate did not pass:

- V12b public-3 matrix:
  `logs/s3v12bm/_manifests/v12btm_20260601_165710_e5_ie5_b480_ew480`
- Result: `9/9` completed, `0/9` teacher gates, `0/45` rolling windows.
- V12b loose-alpha Panama seed42:
  `logs/s3v12bprobe/probe_v12b_loose_panama_s42_20260601_01/panama/seed_42`
- Result: forced nonzero alpha still failed `0/5`; `sign_flipped` was best
  in validation windows.

Conclusion: V12b is a protocol repair and negative diagnostic result. It does
not unlock generator training.

## Phase 14: V12c Multi-Policy Teacher Diagnostics

### Goal

Separate exploratory teacher signal from final claim gates. V12c must expose
`ungated`, `relaxed`, and `strict` alpha policies without relaxing the final
paper/claim gate.

### Completed Files

- [x] `models/stage3_residual_teacher_v12c.py`
- [x] `train/stage3_v12c_signal_contract.py`
- [x] `train/run_stage3_v12c_sensitivity_audit.py`
- [x] `train/run_stage3_v12c_teacher_audit_matrix.py`
- [x] `configs/stage3_v12c_multi_policy_teacher_probe.yaml`
- [x] `configs/stage3_v12c_multi_policy_teacher_probe_nonnegative.yaml`
- [x] `tests/test_stage3_v12c_teacher_policies.py`

### Verification

- [x] V12c policy tests: `4 passed`.
- [x] V12c + V12b targeted tests: `10 passed`.
- [x] V12c modules and runners compile with `py_compile`.
- [x] V12c smoke writes diagnostics and blocks generator training.
- [x] Panama seed42 8-combination probe completed.
- [x] Best-delta public-3 matrix completed.
- [ ] Any V12c matrix passes the pre-declared teacher gate.
- [ ] Any V12c run produces a served generator artifact.

### 2026-06-01 Outcome

- Panama 8-combination probe root:
  `logs/s3v12c_probe8/_manifests`
- Best-delta public-3 manifest:
  `logs/s3v12cm/_manifests/v12ctm_delta_nonneg_20260601_190259_e5_ie5_b480_ew480`
- Public-3 result: `9/9` completed, `0/9` teacher gates, `0/45` rolling
  windows, `0` generator artifacts.
- Best row: Panama seed43 passed MSE/MAE/margin thresholds but failed sign
  agreement (`0.440451 < 0.55`).

Conclusion: V12c improves diagnostic resolution and shows weak delta-vector
signal on Panama, but it still does not validate Stage III. The next phase
should compare drift-only, residual-history-only, and drift+history teachers
under the same strict gate.

## Phase 15: V12d Context Teacher Drift-Versus-History Diagnostic

### Goal

Determine whether `phi_v - phi_i` provides incremental predictive information
beyond causal residual autocorrelation. This phase remains teacher-only and
must not unlock generator training unless `drift_plus_history` beats
`history_only` under prospective rolling gates and real-drift counterfactual
tests.

### Completed Files

- [x] `train/stage3_v12d_context_features.py`
- [x] `models/stage3_residual_teacher_v12d.py`
- [x] `train/run_stage3_v12d_context_teacher_probe.py`
- [x] `train/run_stage3_v12d_context_teacher_matrix.py`
- [x] `configs/stage3_v12d_context_teacher_probe.yaml`
- [x] `tests/test_stage3_v12d_context_features.py`
- [x] `tests/test_stage3_v12d_teacher_modes.py`

### Verification

- [x] V12d RED tests failed on missing modules before implementation.
- [x] V12d tests pass: `4 passed`.
- [x] V12d modules and runners compile with `py_compile`.
- [x] V12d smoke writes teacher diagnostics and no generator artifact.
- [x] V12d public-3 matrix completed with short paths.
- [ ] Any V12d feature mode passes the pre-declared teacher gate.
- [ ] `drift_plus_history` robustly beats `history_only` across datasets and
      seeds.
- [ ] Any V12d run produces a served generator artifact.

### 2026-06-01 Outcome

- Public-3 manifest:
  `logs/s3v12dm/_manifests/v12dm_delta_20260601_203114_e5_ie5_b480_ew480`
- Record:
  `docs/audit/STAGE3_V12D_CONTEXT_TEACHER_RECORD.md`
- Result: `9/9` dataset-seed runs completed, `27/27` feature-mode rows
  completed, `0/27` teacher gates, `0` generator artifacts.
- Rolling windows:
  - `drift_only`: `3/45`
  - `history_only`: `0/45` under drift-counterfactual gate, because
    counterfactual margin is structurally zero when drift is absent.
  - `drift_plus_history`: `6/45`
- Increment over history:
  - rolling-min partial MSE positive in `3/9` runs;
  - rolling-min partial MSE average `-0.043138`;
  - test partial MSE positive in `4/9` runs;
  - test partial MSE average `-0.019044`.

Conclusion: V12d indicates that historical residual autocorrelation explains
most of the apparent correction signal. Under the current protocol,
`phi_v - phi_i` does not provide robust incremental predictive value beyond
history. Do not train a generator or upgrade Stage III manuscript claims.

---

## 2026-05-27 Execution Update

The original Phase 0-9 plan has now been extended with a V7 residualized RRCCA
alignment path. This does not replace the historical V2/V5/V6 records; it is an
additional opt-in branch created to address the `phi_v - phi_i` alignment
problem more rigorously.

Completed in the V7 branch:

- Added `models/stage3_alignment_v7.py` with
  `TrainSetChronologicalResidualizedReducedRankCCAAligner`.
- Added tests for residualization, active-rank truncation, fail-closed behavior,
  and serialization.
- Added runner/artifact opt-in integration for
  `canonicalizer_method: residualized_rrcca_v7`.
- Added `configs/stage3_v7_residualized_rrcca_served_policy_gate.yaml`.
- Ran Panama seed42 `g5_b480` with the same strict served-policy gate.

V7 result:

- alignment gate passed with active rank `8`;
- served-policy rolling validation was `0/5`;
- all generator candidates abstained;
- no `generator_served_policy.pt` was saved.

Therefore, this plan's core scientific guard remains active: Stage III should
not be promoted to a manuscript performance claim until a validation-gated run
passes the pre-declared criteria.

Detailed V7 record:

- `docs/audit/STAGE3_V7_ALIGNMENT_RECORD.md`

---

## 2026-05-28 Execution Update: V8 HyperAdapter Negative Canary

The plan was extended again with an opt-in V8 branch. V8 tested whether V7 was
limited by a weight-only generator by adding additive bias adapters:

- `models/adaptation_v2.py`
  - `LayerAwareHyperAdapterGenerator`
- `configs/stage3_v8_hyperadapter_residualized_rrcca_served_policy_gate.yaml`
  - `target_names: [weight, bias]`
- `train/stage3_generator_training.py`
  - bias-aware `target_names` training and checkpoint load
- `train/online_loop_stage3_v2.py`
  - serving-time `target_names` support

Verification:

```text
C:\Users\admin\anaconda3\python.exe -m pytest tests -q
127 passed in 5.59s

C:\Users\admin\anaconda3\python.exe -m py_compile models\adaptation_v2.py train\stage3_generator_training.py train\online_loop_stage3_v2.py train\run_stage3_v2_sensitivity_audit.py train\stage3_v2_artifacts.py
pass

git diff --check
pass
```

Same-budget Panama seed42 V8 canary:

- logs: `logs/s3v8diag/v8ha_s42_g5b480_01/panama/seed_42`
- selected candidate: epoch `3`
- rolling validation: `0/5`
- worst real MSE gain: `-0.008390`
- worst real MAE gain: `-0.005176`
- worst real-vs-best-CF margin: `-0.008390`
- final served generator: not saved

Conclusion:

V8 increases update capacity and prediction shift, but it makes the validation
result worse than V7. Therefore the next plan phase must not be "add more
capacity" or "relax the gate". It must add residual-teacher supervision and a
residual/update-aware representation alignment objective.

Detailed V8 record:

- `docs/audit/STAGE3_V8_HYPERADAPTER_RECORD.md`

---

## Phase 10: V9 Residual-Teacher Meta-Objective And Representation Alignment

### Problem

V7 cleaned the drift representation but produced weak gains. V8 increased the
generator's action space but produced harmful updates. This shows that the
missing piece is not only representation hygiene or generator capacity. The
generator lacks a supervised target for what a useful drift-conditioned update
should be.

### Intended Behavior

V9 should learn:

```text
cleaned drift -> update direction that corrects the current forecast residual
```

not merely:

```text
cleaned drift -> some parameter perturbation that might reduce batch MSE
```

### Tests First

- [x] Add a residual-teacher toy test where a closed-form ridge residual update
      lowers frozen-base residual MSE.
- [x] Add a no-leakage test proving teacher fitting uses only the train stream
      and never validation/test windows.
- [x] Add a generator-teacher alignment test: nonzero drift should learn a
      delta with positive cosine similarity to the teacher update.
- [x] Add a zero-drift identity test covering teacher loss, calibration, and
      generator output together.
- [x] Add a checkpoint round-trip test for any new V9 generator or teacher
      metadata.
- [x] Add served-policy abstention checks proving failed V9 candidates do not
      save `generator_served_policy.pt`.

### Code Changes

- [x] Add a new module, not a replacement:
      `models/stage3_residual_teacher_v9.py`.
- [x] Add train-only teacher update construction:
      - freeze base model;
      - collect `output_head` input features or residual-head features;
      - compute residual `e = y - y_base`;
      - solve a ridge residual correction/update on train-only mini-windows.
- [x] Extend `Stage3GeneratorLossConfig` with disabled-by-default V9 terms:
      - `lambda_teacher_correction`;
      - `lambda_residual_prediction`;
      - `lambda_teacher_cosine`;
      - existing shift-budget calibration terms.
- [x] Extend `stage3_generator_train_step()` with an optional teacher batch.
      Defaults must preserve V7/V8 behavior when V9 terms are absent.
- [x] Add an opt-in V9 config:
      `configs/stage3_v9_residual_teacher_rrcca_served_policy_gate.yaml`.
- [x] Keep outputs in new namespaces:
      - `logs/s3v9*`
      - `ckpt/s3v9*`
      - `results/s3v9*`

### Representation Alignment Requirement

V9 should not directly assume `phi_v - phi_i` is a valid coordinate difference.
Use the V7 residualized RRCCA path as the base, then add update-aware
calibration:

1. remove invariant-explainable structure from `D = phi_v - phi_i`;
2. learn a train-only projection between residual drift transitions and teacher
   residual-update directions;
3. keep only dimensions that are stable on an inner chronological calibration
   split;
4. report active rank, calibration correlation, update-cosine, and leakage
   diagnostics.

### Acceptance

Do not expand to public-3 until a single Panama seed42 V9 canary passes:

- rolling validation `5/5`;
- worst-window real MSE gain at least `0.005`;
- worst-window real MAE gain at least `0.002`;
- real-vs-best-counterfactual margin at least `0.002`;
- max real reject rate at most `0.2`;
- final `generator_served_policy.pt` saved only after the gate passes.

If V9 fails the same gate, record it as negative evidence and do not upgrade
paper claims.

---

## 2026-05-28 Execution Update: V9 Residual-Teacher Negative Canary

Phase 10 has been implemented as an opt-in V9 branch and tested on Panama
seed42 under the same budget used for V7/V8 comparison.

Implemented files:

- `models/stage3_residual_teacher_v9.py`
- `models/stage3_alignment_v9.py`
- `models/adaptation_v9.py`
- `configs/stage3_v9_residual_teacher_rrcca_served_policy_gate.yaml`
- `tests/test_stage3_v9_teacher_protocol.py`

Extended files:

- `train/stage3_generator_training.py`
- `train/online_loop_stage3_v2.py`
- `train/run_stage3_v2_sensitivity_audit.py`
- `train/stage3_v2_artifacts.py`

Verification before interpreting the experiment:

```text
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_stage3_v9_teacher_protocol.py -q
7 passed

C:\Users\admin\anaconda3\python.exe -m pytest tests -q
134 passed
```

V9 smoke runs showed one useful implementation failure first: the initial
high-dimensional teacher residualizer collapsed active drift rank to `0`.
That was corrected with a train-only low-rank PCA invariant residualizer and
physical zero-drift protection.

Same-budget Panama seed42 V9 canary:

- run id: `v9rt_s42_g5b480_01`
- logs: `logs/s3v9diag/v9rt_s42_g5b480_01/panama/seed_42`
- checkpoints: `ckpt/s3v9diag/panama/v9rt_s42_g5b480_01/seed_42`
- selected candidate: epoch `0`
- rolling validation: `0/5`
- worst-window real MSE gain: `-0.007082`
- worst-window real MAE gain: `-0.005109`
- worst real-vs-best-counterfactual margin: `-0.007082`
- max real reject rate: `0.0`
- final `generator_served_policy.pt`: not saved

Teacher diagnostics:

- train samples: `480`
- teacher residual MSE before: `0.1608959436`
- teacher residual MSE after: `0.1450631768`
- teacher residual R2: `0.0926146507`

Window-level diagnosis:

- real drift harms MSE/MAE in every selected validation window;
- zero drift is the best counterfactual in every selected validation window;
- train blended MSE decreases across generator epochs, but rolling validation
  worsens, indicating train-objective to served-policy mismatch.

Decision:

```text
V9 repaired several structural bugs, but it failed the scientific claim gate.
Do not upgrade the manuscript's Stage III claim.
Do not expand V9 to public-3 as a performance claim until a future
pre-declared single-dataset canary passes.
```

Detailed V9 record:

- `docs/audit/STAGE3_V9_RESIDUAL_TEACHER_RECORD.md`

Updated next step:

1. Run teacher-only validation upper-bound diagnostics.
2. Export residual sign agreement between teacher correction and true residual.
3. Redesign generator training around rolling-window/meta-objective behavior
   rather than train mini-batch blended MSE.
4. Keep all V9 evidence in the audit layer unless the strict gate passes.

---

## 2026-05-28 Execution Update: V9 Teacher-Only Diagnostic

The teacher-only diagnostic requested by the V9 negative canary has now been
implemented and run.

Implementation:

- `train/stage3_v9_teacher_diagnostics.py`
- `train/run_stage3_v2_sensitivity_audit.py`
- `tests/test_stage3_v9_teacher_protocol.py`

The diagnostic evaluates the train-only residual teacher directly:

```text
y_teacher = y_base + teacher(z)
```

It does not call the generator, does not build parameter deltas, and writes
separate CSV files:

- `teacher_only_validation_summary.csv`
- `teacher_only_rolling_validation_summary.csv`
- `teacher_only_rolling_validation_window_summary.csv`
- `teacher_only_rolling_validation_aggregate.csv`
- `teacher_only_test_summary.csv`

Same-budget Panama seed42 run:

- run id: `v9teacher_s42_g5b480_01`
- logs: `logs/s3v9teacher_diag/v9teacher_s42_g5b480_01/panama/seed_42`
- checkpoints: `ckpt/s3v9teacher_diag/panama/v9teacher_s42_g5b480_01/seed_42`
- strict served-policy result: abstain, no `generator_served_policy.pt`

Teacher-only rolling validation:

- pass rate: `0/5`
- worst MSE gain: `-0.197784`
- worst MAE gain: `-0.118167`
- worst real-vs-best-counterfactual margin: `-0.197784`
- worst sign agreement: `0.515625`
- best counterfactual in every window: `zero`

Teacher-only test:

- MSE gain: `-0.097074`
- MAE gain: `-0.052800`
- sign agreement: `0.513368`
- residual-correction cosine: `0.057671`
- best counterfactual: `zero`

Decision:

```text
The current V9 residual teacher itself does not transfer to rolling validation.
Do not train larger generators against this teacher direction.
Do not expand V9 to public-3 as a performance claim.
```

New next step:

1. Add train-calibrated shrinkage for teacher corrections.
2. Add residual-sign-aware teacher fitting and diagnostics.
3. Replace additive `D - f(phi_i)` residualization with cross-fitted
   leakage-subspace projection that preserves zero identity.
4. Only if teacher-only rolling validation becomes non-harmful should a V10
   generator/meta-objective be trained.

Detailed record:

- `docs/audit/STAGE3_V9_TEACHER_ONLY_DIAGNOSTIC_RECORD.md`

---

## 2026-05-29 Execution Update: V10 Cross-Fitted Teacher Matrix

Phase 11 has now been executed as a teacher-only V10 repair and public-3
matrix. It does not overwrite V9 or legacy paths.

Implemented V10 files:

- `models/stage3_residual_teacher_v10.py`
- `models/stage3_alignment_v10.py`
- `models/adaptation_v10.py`
- `train/stage3_generator_training_v10.py`
- `train/stage3_v10_artifacts.py`
- `train/run_stage3_v10_sensitivity_audit.py`
- `train/run_stage3_v10_teacher_audit_matrix.py`
- `configs/stage3_v10_crossfit_teacher_projection_served_policy_gate.yaml`
- `configs/stage3_v10_crossfit_teacher_projection_signed_alpha_gate.yaml`
- `tests/test_stage3_v10_teacher_protocol.py`

Verification:

```text
C:\Users\admin\anaconda3\python.exe -m py_compile train\run_stage3_v10_sensitivity_audit.py train\run_stage3_v10_teacher_audit_matrix.py
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_stage3_v10_teacher_protocol.py tests\test_stage3_v9_teacher_protocol.py -q
13 passed in 1.50s
```

Key behavior:

- `--teacher-only` stops after teacher diagnostics and prevents accidental
  generator training.
- V10 alignment uses cross-fitted drift-space projection, not additive
  `D - f(phi_i)`, preserving `delta=0 -> 0`.
- V10 teacher uses train-only shrinkage calibration and can abstain.

Matrix:

- datasets: `panama`, `tetouan`, `spain_energy_weather`;
- seeds: `42`, `43`, `44`;
- budget: `5 epochs`, `480` teacher train batches, `480` eval batches;
- rolling windows: `5`;
- effective runs: `9`.

Result:

| Dataset | Teacher Gate | Rolling Windows Passed | Mean Worst MSE Gain | Mean Worst MAE Gain |
| --- | ---: | ---: | ---: | ---: |
| Panama | `0/3` | `7/15` | `+0.007158` | `+0.000809` |
| Tetouan | `0/3` | `1/15` | `-0.144104` | `-0.096893` |
| Spain | `0/3` | `0/15` | `-0.025551` | `-0.018493` |
| All | `0/9` | `8/45` | `-0.054166` | `-0.038192` |

Decision:

```text
V10 is an auditability and Panama-mechanism improvement, not a validated
Stage III method. Do not train or report a V10 generator as a paper
contribution until teacher-only rolling validation passes prospectively.
```

Detailed records:

- `docs/audit/STAGE3_V10_CROSSFIT_TEACHER_RECORD.md`
- `docs/audit/STAGE3_V10_TEACHER_MATRIX_RECORD.md`
- `docs/audit/STAGE3_V10_TEACHER_MATRIX_SUMMARY.csv`

Updated next step:

1. Build V11 around worst-block chronological calibration and selective
   reliability-gated residual correction.
2. Keep exact zero-drift identity and cross-fitted leakage projection.
3. Do not resume generator training until teacher-only passes the pre-declared
   rolling gate.

---

## Phase 0: Safety And Evidence Freeze

- [ ] Run `git status --short` and save the output in the execution note before editing code.
- [ ] Confirm no command will overwrite:
  - raw datasets;
  - original logs;
  - model checkpoints;
  - existing tables/figures;
  - previously generated audit documents.
- [ ] Create a new experiment namespace for repaired Stage III:
  - `logs/load_benchmarks/stage3_v2/`
  - `checkpoints/stage3_v2/`
  - `results/stage3_v2/`
- [ ] Keep original entry points unchanged until tests pass:
  - `train/online_loop.py`
  - `train/run_experiment.py`
  - `models/adaptation.py`
  - `models/factor_mechanism.py`

Acceptance:

- Original Stage III remains reproducible.
- New implementation is opt-in through config flags.

---

## Phase 1: Make VAE Evaluation Deterministic By Default

### Problem

In forecasting papers, validation/test metrics must usually be deterministic unless the paper explicitly reports Monte Carlo uncertainty. The current `FactorMechanism` samples latent noise in `eval()` because `_reparameterize()` always calls `torch.randn_like(std)`.

Training-time VAE sampling is correct. Evaluation-time random sampling is inappropriate for ordinary MSE/MAE comparison because the same checkpoint and input can produce different predictions.

### Intended Behavior

- During `model.train()`: keep stochastic reparameterization.
- During `model.eval()` with default config: use `mu_z` directly.
- During `model.eval()` with explicit Monte Carlo config: allow sampling and require fixed seed/repeat reporting.

### Tests First

- [ ] Add `tests/test_factor_mechanism_eval.py`.
- [ ] Add `test_vae_eval_uses_mu_by_default_for_deterministic_predictions`.
- [ ] Add `test_vae_train_mode_keeps_sampling`.
- [ ] Add `test_vae_eval_sample_mode_remains_stochastic_when_explicitly_enabled`.

Test intent:

```python
def test_vae_eval_uses_mu_by_default_for_deterministic_predictions():
    torch.manual_seed(0)
    cfg = FactorMechanismConfig(use_vae=True, vae_eval_mode="mean")
    model = FactorMechanism(cfg).eval()
    x = torch.randn(4, cfg.input_window, cfg.input_dim)
    y1 = model(x)["y_hat"]
    y2 = model(x)["y_hat"]
    assert torch.allclose(y1, y2)
```

### Code Changes

- [ ] Extend `models/factor_mechanism.py`.
- [ ] Add `vae_eval_mode: str = "mean"` to `FactorMechanismConfig`.
- [ ] Store `self.vae_eval_mode`.
- [ ] Replace unconditional eval sampling with a selector:

```python
def _select_latent(self, mu_z: torch.Tensor, log_var_z: torch.Tensor) -> torch.Tensor:
    if self.training:
        return self._reparameterize(mu_z, log_var_z)
    if self.vae_eval_mode == "mean":
        return mu_z
    if self.vae_eval_mode == "sample":
        return self._reparameterize(mu_z, log_var_z)
    raise ValueError(f"Unsupported vae_eval_mode: {self.vae_eval_mode}")
```

- [ ] Update model construction in `train/run_experiment.py` to pass:

```python
vae_eval_mode=cfg["model"].get("vae_eval_mode", "mean")
```

- [ ] Add this field to repaired configs:

```yaml
model:
  use_vae: true
  vae_eval_mode: mean
```

Acceptance:

- Same model, same input, same eval mode gives identical `y_hat`.
- Training mode still samples.
- Old behavior remains accessible only with `vae_eval_mode: sample`.

---

## Phase 2: Replace Random Non-Identity Generator With A Trainable Identity-Preserving Generator

### Problem

The current generator:

```python
h = torch.sigmoid(self.w1(drift))
alpha_full = torch.sigmoid(self.w2_alpha(h)) + 1.0
beta_full = torch.sigmoid(self.w2_beta(h)) + 1.0
delta = (alpha.unsqueeze(1) * beta.unsqueeze(0) - 1.0) * p
```

has three major design problems:

1. It is random at online inference time because it is never trained.
2. Zero drift does not map to zero update.
3. `sigmoid + 1.0` only creates positive scale factors from roughly 1 to 2, so the generated update mostly rescales existing weights instead of learning signed, layer-aware corrections.

### Intended Behavior

Generator V2 must satisfy:

- zero drift produces exactly zero or near-zero delta;
- real drift can produce signed positive/negative deltas;
- generated deltas are bounded by a config-controlled scale;
- generator parameters are trained before online test evaluation;
- generator outputs diagnostic values for sensitivity analysis.

### Tests First

- [ ] Add `tests/test_stage3_generator_v2.py`.
- [ ] Add `test_zero_drift_produces_zero_delta_at_initialization`.
- [ ] Add `test_nonzero_drift_changes_delta_after_training_step`.
- [ ] Add `test_sign_flip_changes_delta_direction_after_training_step`.
- [ ] Add `test_delta_shape_matches_target_parameters`.

Test intent:

```python
def test_zero_drift_produces_zero_delta_at_initialization():
    gen = IdentityLowRankAdaptationGenerator(drift_dim=16, hidden_dim=32, max_dim=64, max_scale=0.05)
    layer = nn.Linear(8, 4, bias=False)
    drift = torch.zeros(16)
    delta = build_identity_param_delta(layer, drift, gen)
    assert torch.allclose(delta["weight"], torch.zeros_like(layer.weight), atol=1e-8)
```

### Code Changes

- [ ] Add a new class instead of mutating the old one:

```python
class IdentityLowRankAdaptationGenerator(nn.Module):
    def __init__(
        self,
        drift_dim: int,
        hidden_dim: int,
        max_dim: int,
        max_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.max_dim = max_dim
        self.max_scale = max_scale
        self.net = nn.Sequential(
            nn.Linear(drift_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * max_dim),
        )
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, drift: torch.Tensor, out_dim: int, in_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(drift)
        alpha_raw = raw[:out_dim]
        beta_raw = raw[self.max_dim : self.max_dim + in_dim]
        alpha = self.max_scale * torch.tanh(alpha_raw)
        beta = self.max_scale * torch.tanh(beta_raw)
        return alpha, beta
```

- [ ] Add a V2 delta builder:

```python
def build_identity_param_delta(
    module: nn.Module,
    drift: torch.Tensor,
    generator: IdentityLowRankAdaptationGenerator,
    target_names: tuple[str, ...] = ("weight",),
) -> dict[str, torch.Tensor]:
    deltas = {}
    for name, param in module.named_parameters():
        if name not in target_names or param.ndim != 2:
            continue
        out_dim, in_dim = param.shape
        alpha, beta = generator(drift, out_dim=out_dim, in_dim=in_dim)
        low_rank_scale = alpha.unsqueeze(1) * beta.unsqueeze(0)
        deltas[name] = low_rank_scale * param
    return deltas
```

Important design note:

- This initialization gives zero adaptation before training. That is intentional.
- If the trained generator cannot beat zero adaptation, Stage III should not be claimed as a working innovation.

Acceptance:

- Zero drift no longer causes large nonzero deltas.
- Sign-flipped drift can generate different signed deltas after training.
- The old `AdaptationGenerator` remains available for historical comparison.

---

## Phase 3: Stop Compressing Vector Drift Into A Scalar

### Problem

The original Stage III code computes:

```python
d_v = phi_v_cur - phi_i_cur
drift_scalar = d_v.mean()
drift_vec = drift_scalar.repeat(lrp_bottleneck)
```

This discards which embedding dimensions changed. If the paper describes vector interpolation or vector drift, the scalar implementation is not faithful.

### Intended Behavior

Use one of these explicit modes:

1. `same_time_vector`: `phi_v(t) - phi_i(t)`.
2. `temporal_delta_vector`: `(phi_v(t) - phi_i(t)) - (phi_v(t-1) - phi_i(t-1))`.
3. `variant_temporal_vector`: `phi_v(t) - phi_v(t-1)`.
4. `projected_temporal_delta`: temporal delta followed by a learned or fixed projection into generator input dimension.

For the user's concern, `temporal_delta_vector` should be the primary experimental candidate because it removes slow stable offsets and emphasizes fluctuations.

### Tests First

- [ ] Extend `tests/test_stage3_signals.py`.
- [ ] Add test for `temporal_delta_vector`.
- [ ] Add test for dimension projection.
- [ ] Add test that first step uses zero drift or configured warm-start behavior.

Expected helper behavior:

```python
def compute_stage3_drift(
    phi_i_cur: torch.Tensor,
    phi_v_cur: torch.Tensor,
    phi_i_prev: torch.Tensor | None,
    phi_v_prev: torch.Tensor | None,
    mode: str,
) -> torch.Tensor:
    gap_cur = phi_v_cur - phi_i_cur
    if mode == "same_time_vector":
        return gap_cur
    if mode == "temporal_delta_vector":
        if phi_i_prev is None or phi_v_prev is None:
            return torch.zeros_like(gap_cur)
        gap_prev = phi_v_prev - phi_i_prev
        return gap_cur - gap_prev
    if mode == "variant_temporal_vector":
        if phi_v_prev is None:
            return torch.zeros_like(phi_v_cur)
        return phi_v_cur - phi_v_prev
    raise ValueError(f"Unsupported drift mode: {mode}")
```

### Code Changes

- [ ] Use `train/stage3_signals.py` as the canonical drift signal module.
- [ ] Update the repaired Stage III path to consume full vectors.
- [ ] Add config:

```yaml
online:
  stage3:
    enabled: true
    version: v2
    drift_signal_mode: temporal_delta_vector
    first_step_drift: zero
```

Acceptance:

- No `d_v.mean().repeat(...)` exists in the V2 path.
- Drift signal mode is logged in every result file.
- First online step behavior is explicit and reproducible.

---

## Phase 4: Align `phi_i` And `phi_v` Before Subtracting Them

### Problem

`encoder_i` and `encoder_v` are currently separate networks:

- `encoder_i` is trained in the invariant learning path.
- `encoder_v` is trained on residual targets.

Because they do not share a coordinate system, the raw subtraction `phi_v - phi_i` may be mathematically defined but semantically weak. Dimension 3 in `phi_i` is not guaranteed to mean the same latent factor as dimension 3 in `phi_v`.

### Minimum Safe Fix

Use train-set calibration before subtraction:

- estimate mean/std of `phi_i` and `phi_v` on train stream only;
- optionally learn a small projection from `phi_v` into `phi_i` coordinates;
- freeze the calibration before validation/test.

Config:

```yaml
invariant:
  alignment:
    enabled: true
    method: train_set_affine_projection
    projection_dim: 16
    fit_split: train
```

### Stronger Fix

Replace two unrelated encoders with a shared trunk and two heads:

```text
x
  -> shared_encoder(x) = h
  -> invariant_head(h) = phi_i
  -> variant_residual_head(h) = delta_phi_v
  -> phi_v = phi_i + delta_phi_v
  -> drift = delta_phi_v or temporal_delta(delta_phi_v)
```

This makes subtraction meaningful because `phi_v` is anchored to `phi_i`.

### Loss Design

Use a combined objective:

```text
L = L_forecast
  + lambda_inv * L_invariance
  + lambda_var * L_residual
  + lambda_orth * ||corr(phi_i, delta_phi_v)||_F
  + lambda_smooth * ||delta_phi_v(t) - delta_phi_v(t-1)||_2
  + lambda_norm * ||delta_phi_v||_2
```

Interpretation:

- `L_forecast`: keeps representations predictive.
- `L_invariance`: protects stable features.
- `L_residual`: encourages variant part to explain residual changes.
- `L_orth`: reduces leakage of invariant information into variant residual.
- `L_smooth`: avoids treating noise as concept drift.
- `L_norm`: prevents arbitrary large residual embeddings.

### Tests First

- [ ] Add `tests/test_aligned_factor_representations.py`.
- [ ] Add shape tests for shared trunk and heads.
- [ ] Add test that `phi_v = phi_i + delta_phi_v`.
- [ ] Add test that drift is zero when `delta_phi_v` is zero.
- [ ] Add test that calibration statistics are fit only on train split.

### Code Changes

- [ ] Add `models/aligned_factor_representation.py`.
- [ ] Add `train/offline_aligned_invariant.py`.
- [ ] Add an opt-in config:

```yaml
invariant:
  representation_version: aligned_v1
```

Acceptance:

- V2 Stage III never subtracts uncalibrated independent embeddings by default.
- The code can still run legacy `encoder_i`/`encoder_v` for ablation.
- Any paper claim about drift representation uses only aligned/calibrated results.

---

## Phase 5: Add Explicit Generator Training

### Problem

A generator that maps drift to parameter delta is a neural network. It cannot be treated as valid if random weights are used at test time.

### Intended Training Protocol

Freeze the already trained base forecaster and representation modules, then train only the generator on a training stream:

```text
offline trained base model
  + aligned representation modules
  + train stream replay
  -> compute drift(t)
  -> generator(drift(t)) creates delta(t)
  -> temporary adapted prediction
  -> optimize generator loss
```

Do not update the test stream during generator training.

### Loss

```text
L_gen = MSE(y_adapted, y_true)
      + lambda_delta * ||delta||_2
      + lambda_zero * ||G(0)||_2
      + lambda_shift * ||y_adapted - y_base||_2
      + lambda_smooth * ||delta(t) - delta(t-1)||_2
```

Meaning:

- `MSE`: generator must improve prediction.
- `lambda_delta`: prevents wild parameter changes.
- `lambda_zero`: enforces no drift -> no adaptation.
- `lambda_shift`: avoids large prediction jumps.
- `lambda_smooth`: discourages unstable online updates.

### Tests First

- [ ] Add `tests/test_stage3_generator_training.py`.
- [ ] Add toy regression test where generator training decreases loss.
- [ ] Add test that base model parameters remain frozen.
- [ ] Add test that only generator parameters receive gradients.
- [ ] Add test that zero-drift regularization lowers `G(0)` norm.

### Code Changes

- [ ] Add `train/stage3_generator_training.py`.
- [ ] Add CLI/config entry point:

```bash
python -m train.stage3_generator_training --config configs/stage3_v2.yaml
```

- [ ] Save generator checkpoints separately:

```text
checkpoints/stage3_v2/<dataset>/<run_id>/generator.pt
checkpoints/stage3_v2/<dataset>/<run_id>/calibration.json
checkpoints/stage3_v2/<dataset>/<run_id>/train_metrics.csv
```

- [ ] Log these per step:
  - `drift_norm`;
  - `delta_norm`;
  - `zero_delta_norm`;
  - `candidate_shift`;
  - `base_mse`;
  - `adapted_mse`;
  - `loss_delta_reg`;
  - `loss_zero_reg`.

Acceptance:

- There is an optimizer with `gen.parameters()`.
- Generator checkpoint is loaded for test-time Stage III V2.
- Repaired test path refuses to run if `stage3.version: v2` and no trained generator checkpoint is supplied, unless `allow_untrained_generator: true` is explicitly set for diagnostics.

---

## Phase 6: Rebuild Online Stage III V2 As Opt-In

### Problem

The legacy `train/online_loop.py` mixes Stage I, Stage II, Stage III and makes it easy for Stage III behavior to be hidden by clamps, blending, or random VAE noise.

### Intended Behavior

Add a clean V2 branch with explicit data flow:

```text
base prediction
  -> compute aligned drift vector
  -> load trained generator
  -> build bounded delta
  -> temporary functional forward
  -> compute candidate prediction
  -> apply configured blend/reject rule
  -> log every intermediate value
```

### Code Changes

- [ ] Add `train/online_loop_stage3_v2.py` first, rather than replacing `train/online_loop.py`.
- [ ] Make V2 path reject unsafe config combinations:
  - `vae_eval_mode: sample` without `mc_eval_repeats`;
  - `generator_checkpoint` missing;
  - unaligned embeddings when `require_aligned_embeddings: true`;
  - scalar drift mode unless `legacy_scalar_ablation` is explicitly set.
- [ ] Log:
  - `stage3_version`;
  - `drift_signal_mode`;
  - `alignment_method`;
  - `generator_checkpoint`;
  - `vae_eval_mode`;
  - `delta_norm_pre_clamp`;
  - `delta_norm_post_clamp`;
  - `clamp_fraction`;
  - `candidate_shift`;
  - `blend_weight`;
  - `reject_reason`.

Acceptance:

- Legacy Stage III can still be run.
- Stage III V2 is fully traceable from each prediction to its drift and delta diagnostics.

---

## Phase 7: Add Sensitivity Diagnostics As A Required Gate

### Problem

Even a trained generator might ignore drift if clamps, blending, or architecture bottlenecks dominate.

### Diagnostic Modes

Run the same offline model and same online stream under:

1. `real`: real drift signal.
2. `zero`: all-zero drift.
3. `shuffled`: real drift values shuffled across time.
4. `sign_flipped`: `-drift`.
5. `amplified_2x`: `2 * drift`.
6. `amplified_5x`: `5 * drift`.
7. `random_matched_norm`: random direction with same norm distribution.

### Code Changes

- [ ] Add or extend `train/run_stage3_sensitivity_audit.py`.
- [ ] Save output to:

```text
logs/load_benchmarks/stage3_v2_sensitivity/<run_id>/
```

- [ ] Produce:
  - `step_metrics.csv`;
  - `summary_by_mode.csv`;
  - `paired_mode_differences.csv`;
  - `audit_manifest.json`.

### Acceptance Criteria For Claim Safety

Stage III V2 can only be described as drift-conditioned if:

- `real` is measurably different from `zero` in `delta_norm` and/or `candidate_shift`;
- `real` is not identical to `shuffled`;
- `sign_flipped` changes generated deltas or predictions;
- amplification changes delta magnitude before clamps;
- real-drift metrics are not worse than zero/shuffled under paired comparison.

If these conditions fail, the paper must not claim that Stage III learns concept drift. It may only claim that the current implementation includes a tested but ineffective transient adaptation module.

---

## Phase 8: Re-run Minimal Experiments Before Any Paper Revision

### Minimal Safe Experiment Set

Do not run expensive full training until the small protocol passes.

- [ ] One dataset.
- [ ] One seed.
- [ ] Reduced epochs but more than one epoch.
- [ ] Compare:
  - base offline model;
  - Stage I only;
  - legacy Stage III;
  - Stage III V2 zero drift;
  - Stage III V2 real drift;
  - Stage III V2 shuffled drift.

### Expanded Experiment Set

Only after minimal diagnostics pass:

- [ ] All three datasets.
- [ ] At least three seeds.
- [ ] All prediction horizons used in the paper.
- [ ] Paired significance or confidence intervals, if enough repeated runs are available.
- [ ] Report MSE and MAE for every method/dataset/horizon cell.

### Paper Update Gate

Update the LaTeX only if evidence supports the wording:

- If V2 beats zero/shuffled and improves efficiency:
  - claim online drift-conditioned adaptation with efficiency/stability framing.
- If V2 beats baseline but not zero/shuffled:
  - claim online adaptation but not drift-conditioned learning.
- If V2 does not beat Stage I or base:
  - remove Stage III as a claimed contribution and report it as an ablation/negative finding.

---

## Phase 9: Documentation Updates

- [ ] Update `docs/audit/STAGE3_GENERATOR_BEGINNER_EXPLANATION.md` with:
  - deterministic VAE explanation;
  - why untrained generator is not scientifically valid;
  - why scalar drift contradicts vector adaptation language;
  - why alignment is needed before subtraction.
- [ ] Keep `docs/audit/STAGE3_EXPERT_DIAGNOSIS_AND_REPAIR_STRATEGY.md` as the expert diagnosis.
- [ ] Add implementation results after code changes to:
  - `docs/audit/STAGE3_V2_IMPLEMENTATION_LOG.md`;
  - `docs/audit/STAGE3_V2_SENSITIVITY_LEDGER.md`.

---

## Verification Commands

Run in this order after implementation:

```powershell
python -m pytest tests\test_factor_mechanism_eval.py -q
python -m pytest tests\test_stage3_generator_v2.py -q
python -m pytest tests\test_stage3_signals.py -q
python -m pytest tests\test_aligned_factor_representations.py -q
python -m pytest tests\test_stage3_generator_training.py -q
python -m pytest tests\test_stage3_ablation_benchmark.py -q
```

Then run the smallest diagnostic:

```powershell
python -m train.run_stage3_sensitivity_audit --config configs\stage3_v2_minimal.yaml
```

Do not promote results into the paper until the audit CSVs show that real drift behaves differently from zero and shuffled drift.

---

## Final Acceptance Checklist

- [ ] Eval metrics are deterministic by default.
- [ ] Stage III V2 refuses untrained generator checkpoints unless explicitly in diagnostic mode.
- [ ] Zero drift produces zero or near-zero generator delta.
- [ ] Vector drift is not reduced to a scalar in the V2 path.
- [ ] Representation subtraction is calibrated or replaced by aligned residual representation.
- [ ] Sensitivity diagnostics show whether drift is actually used.
- [ ] Paper claims are downgraded or upgraded strictly according to evidence.

---

## Recommended Minimal Next Step

Implement only Phase 1 first.

Reason: deterministic evaluation is a prerequisite for every later comparison. If `eval()` remains stochastic, generator sensitivity, Stage III ablation, and MSE/MAE comparisons can all be contaminated by latent sampling noise.

---

## 2026-05-25 Execution Status Addendum

This plan has now been executed beyond the original minimal next step in an opt-in V2/V2.5 path without overwriting the legacy Stage III files.

Current implementation status:

| Plan requirement | Current status | Evidence |
| --- | --- | --- |
| Deterministic VAE evaluation | Implemented in V2 path | `models/factor_mechanism_v2.py`; `configs/stage3_v2_strong.yaml` uses `vae_eval_mode: mean` |
| Identity-preserving generator | Implemented in V2 path | `models/adaptation_v2.py` uses bias-free identity-centered generators |
| Full-vector drift instead of scalar drift | Implemented in V2 strong path | `configs/stage3_v2_strong.yaml` uses vector drift settings; legacy scalar path remains only for compatibility/ablation |
| Explicit generator training | Implemented in V2 runner | `train/run_stage3_v2_sensitivity_audit.py`; generator checkpoints are saved and reloaded |
| Representation alignment / calibration | Partially implemented | `models/aligned_representation_v2.py` uses residual `phi_v = phi_i + delta_phi_v` plus train-only residual drift canonicalizer |
| Counterfactual sensitivity gate | Implemented | `train/stage3_sensitivity_gate_v2.py` and `sensitivity_claim_gate.csv` outputs |
| Rolling-validation generator checkpoint selection | Implemented in V2.5 | `train/stage3_generator_selection_v2.py`; `generator_checkpoint_selection.csv` outputs |
| Paper claim upgrade | Not allowed by evidence | Panama seed42/43 b60/b120 all fail rolling/test claim gates |

Latest experiment status:

| seed | budget | test MSE gain | test MAE gain | rolling validation pass rate | test gate |
| ---: | --- | ---: | ---: | ---: | --- |
| 42 | b60 | -0.0938% | -0.3505% | 0.3333 | fail |
| 42 | b120 | +1.2656% | about -0.4907% | 0.3333 | fail |
| 43 | b60 | -0.3250% | +0.1074% | 0.3333 | fail |
| 43 | b120 | +1.6361% | +1.1517% | 0.3333 | fail |

Decision:

```text
The repair plan improved auditability but did not produce validated significant performance improvement.
Do not upgrade the manuscript's Stage III claim.
Stage III V2.5 should be treated as audited diagnostic evidence unless future pre-declared rolling validation and counterfactual gates pass.
```

Updated next step:

1. Do not run another larger training sweep before low-cost replay diagnostics.
2. First separate training-budget effects from evaluation-window effects by prefix-matched replay of existing checkpoints.
3. Run risk-gate-disabled / reject-rate-capped replay for the seed43 b120 checkpoint.
4. Export per-sample losses for paired uncertainty analysis.
5. Only after these diagnostics pass should any broader multi-dataset retraining be considered.

---

## 2026-06-01 V12e Execution Status Addendum

The plan has now advanced through V12e, still in isolated opt-in paths and without overwriting legacy Stage III files, original result logs, generated tables, or figures.

Current V12e implementation status:

| Requirement | Current status | Evidence |
| --- | --- | --- |
| Per-horizon label availability | Implemented | `train/stage3_v12e_label_availability.py`; `tests/test_stage3_v12e_label_availability.py` |
| Fixed-history conditional drift controls | Implemented | `train/stage3_v12e_conditional_permutation.py`; `tests/test_stage3_v12e_conditional_permutation.py` |
| V12d default compatibility | Preserved | V12e activates only with `--history-protocol v12e_per_horizon_label_availability` |
| Public-3 conditional matrix | Completed | `logs/s3v12emc/_manifests/v12emc_delta_20260601_214834_e5_ie5_b480_ew480` |
| Generator training | Still blocked | V12e teacher gates passed `0/27`; no `generator*.pt` artifact exists under `ckpt/s3v12emc` |
| Paper claim upgrade | Not allowed by evidence | `docs/audit/STAGE3_V12E_LABEL_AVAILABILITY_CONTEXT_TEACHER_RECORD.md` |

Latest V12e result:

| Feature mode | Gate pass | Rolling window pass | Avg worst MSE gain | Avg worst counterfactual margin |
| --- | ---: | ---: | ---: | ---: |
| `drift_only` | `0/9` | `2/45` | `-0.004451` | `-0.010971` |
| `history_only` | `0/9` | `0/45` | `-0.000911` | `0.000000` |
| `drift_plus_history` | `0/9` | `0/45` | `0.001000` | `-0.005649` |

Decision:

```text
The strictest delayed-feedback Stage III diagnostic is still negative.
Do not train or report a V12e generator.
The next Stage III design must change the representation/teacher objective, not merely scale the current generator training.
```

Updated next step:

1. Design a new teacher objective that predicts delayed residual-release events or per-horizon residual corrections under the V12e availability mask.
2. Keep `history_only`, `drift_only`, `drift_plus_history`, and fixed-history conditional controls as non-negotiable gates.
3. Require public-3 teacher-only success before any generator canary.
4. Only after teacher-only public-3 passes should a generator be trained and evaluated.
