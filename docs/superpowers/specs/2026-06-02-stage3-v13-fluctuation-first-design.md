# Stage III V13 Fluctuation-First Design

## 1. Purpose

This specification defines a new Stage III V13 algorithm path for the load-forecasting online-learning project. V13 must follow the user's corrected theory:

1. `phi_i` represents stable, long-term forecasting structure.
2. `phi_v` may represent stable but time-varying long-term state, not necessarily short-term fluctuation.
3. `phi_v - phi_i` should be treated as a long-term variable-state offset candidate, not as direct evidence of variant unobserved factor driven fluctuation.
4. Stage III should primarily respond to temporal fluctuations, represented by a transition signal such as `delta_phi_v(t) - delta_phi_v(t-1)`, where `delta_phi_v = phi_v - phi_i` in the aligned residual representation.

The V13 goal is not to force a positive result. The goal is to create a versioned, auditable algorithm and experiment path that can test whether the fluctuation signal provides stable predictive value beyond causal residual-history controls.

## 2. Non-Overwrite Boundary

V13 must be added as a new version. It must not overwrite V2-V12 code paths, configs, logs, checkpoints, or experiment summaries.

V13 will create new files:

- `train/stage3_v13_signal_contract.py`
- `models/stage3_residual_teacher_v13.py`
- `train/run_stage3_v13_fluctuation_teacher_probe.py`
- `train/run_stage3_v13_fluctuation_teacher_matrix.py`
- `configs/stage3_v13_fluctuation_teacher_probe.yaml`
- `tests/test_stage3_v13_signal_contract.py`
- `tests/test_stage3_v13_teacher_protocol.py`
- `docs/audit/STAGE3_V13_FLUCTUATION_DESIGN_AND_EXPERIMENT_RECORD.md`

V13 may import reusable utilities from V12 files when doing so does not change old behavior. Good reuse candidates are:

- `train/stage3_v12e_label_availability.py`
- `train/stage3_v12d_context_features.py`
- `models/stage3_residual_teacher_v12d.py`
- shared loader and training helpers from existing runner files

V13 experiment outputs must use new roots:

- `logs/s3v13fluct/...`
- `ckpt/s3v13fluct/...`
- `results/s3v13fluct/...` if a results root is needed

All V13 run directories must use fresh run IDs and the existing no-overwrite behavior. If an output directory or checkpoint directory already exists, the runner must fail unless an explicit `--overwrite` flag is provided. Normal V13 experiment commands should not use `--overwrite`.

## 3. Selective Inheritance From V2-V12

V13 must selectively inherit the audited strengths of V2-V12. It must not restart from a simplified toy implementation.

### 3.1 Mechanisms To Inherit

V13 should inherit these mechanisms unless implementation evidence shows a direct conflict with the fluctuation-first theory:

| Source | Inherit | Reason |
| --- | --- | --- |
| V2 | deterministic evaluation, zero-drift identity behavior, full-vector drift handling, explicit generator-training prohibition before evidence | These fixed basic scientific invalidity risks in the old Stage III path. |
| V2.1 | split-safe preprocessing, shared residual representation, projected residual alignment, invariant-branch forecast responsibility | These make `delta_phi_v = phi_v - phi_i` auditable as a residual offset. |
| V3 | generator-input counterfactual intervention location and resolved-config manifests | Counterfactuals must be applied where the served adaptation actually consumes the signal. |
| V4 | amplified hard-negative philosophy for future generator canary | If generator training becomes justified, amplified fluctuation counterfactuals must remain part of the loss and diagnostics. |
| V5 | served-policy selection and fail-closed abstention | Failed candidates must not be saved as deployable models. |
| V7-V12 | train-only alignment/canonicalization diagnostics where compatible | These identify invariant leakage and coordinate instability, but must be used as diagnostics unless V13 gates prove predictive value. |
| V12c | train-only robust scaling, zero-preserving signal transform, explicit counterfactual modes | This directly fits V13's signal-contract needs. |
| V12d | drift-vs-history feature comparisons and partial gain versus history-only | This is central to proving that fluctuation adds information beyond residual autocorrelation. |
| V12e | per-horizon label availability for residual-history controls | This prevents future-horizon labels from leaking into online history features. |

### 3.2 Mechanisms Not To Inherit Blindly

V13 must not inherit these old assumptions as default claims:

- raw `phi_v - phi_i` as direct concept-drift fluctuation;
- scalar compression of embedding drift;
- untrained generator use;
- test-set based model selection;
- toy synthetic data as evidence for algorithm effectiveness;
- relaxed gates chosen after seeing results;
- a generator canary before teacher-only fluctuation evidence passes;
- old V12 dispatch branches modified in place.

V13 must also not inherit the V12 canonicalized served-signal path for its main fluctuation signal. The V13 primary signal must be computed from raw aligned-residual `delta_phi_v` or exact raw `phi_v - phi_i` before any residual canonicalizer, CCA/projection transport, or delta standardizer is applied. Canonicalizer and standardizer diagnostics may be reported separately, but if they transform the main V13 signal path the run is invalid and must stop before pilot or matrix execution.

### 3.3 Anti-Toy Implementation Requirement

V13 implementation must be production-grade within the existing research codebase. This means:

- use the real project data loaders, split-safe preprocessing, model training, aligned representation training, and chronological validation utilities;
- use real datasets for experiments, not synthetic or mock data except inside unit tests;
- use the same base model and aligned representation training path as the audited V12 family unless the implementation plan explicitly documents a scientific reason to differ;
- write real manifests, resolved configs, diagnostics, CSV summaries, abort files, and checkpoint metadata;
- run with non-overwriting real output roots;
- keep any protocol-connectivity checks clearly labeled as implementation checks, not algorithm evidence;
- evaluate algorithm claims only on real-data, multi-window experiments with pre-declared gates.

A minimal protocol-connectivity command is allowed only to prove that the V13 pipeline imports, parses config, and writes audit artifacts. It must not be described as an experiment restart or as evidence that the algorithm works. The first meaningful V13 experiment is the Panama real-data pilot in Stage B.

## 4. Signal Contract

V13 must make the semantic distinction between long-term offset and short-term fluctuation explicit.

### 4.1 Raw Residual Offset

The raw aligned residual offset is:

```text
level_offset(t) = delta_phi_v(t)
                = phi_v(t) - phi_i(t)
```

In the V2.1 aligned residual implementation, `phi_v = phi_i + delta_phi_v`, so `phi_v - phi_i` equals `delta_phi_v`.

V13 must not call this signal a verified concept-drift fluctuation. It is a candidate long-term variable-state offset.

### 4.2 Fluctuation Signal

The V13 primary drift signal is:

```text
fluctuation_delta(t) = level_offset(t) - level_offset(t-1)
                     = delta_phi_v(t) - delta_phi_v(t-1)
```

The first sample in a stream or validation window must use one of these explicit behaviors:

- If a previous split tail is available, use the previous tail as `t-1`.
- If no previous tail is available, emit exact zero fluctuation for the first sample.

This behavior must be deterministic and logged.

### 4.3 Optional Diagnostic Signal

V13 may also compute:

```text
variant_temporal_delta(t) = phi_v(t) - phi_v(t-1)
```

This is diagnostic only. It should not be the default served signal because it can include changes in `phi_i` through `phi_v = phi_i + delta_phi_v`.

### 4.4 Train-Only Scaling

V13 must fit robust per-dimension scaling on the training split only. Validation and test splits must reuse the frozen train-fitted scale.

Zero fluctuation must map to exact zero after scaling.

This train-only scaling is the V13 signal-contract scaling implemented by `Stage3SignalBuilderV13`. It is not the V12 `delta_standardizer`, and it must not be preceded by residual canonicalization or latent-space transport in the served V13 fluctuation signal.

The signal contract must report diagnostics:

- protocol version
- mode name
- fit split
- fit sample count
- scale statistics
- zero mask fraction
- level offset norm summary
- fluctuation norm summary

## 5. Context Features And Teacher Probe

V13 must begin with a teacher-only probe rather than immediate generator training.

The teacher feature modes are:

1. `fluctuation_only`
2. `history_only`
3. `fluctuation_plus_history`
4. `level_offset_only` as a diagnostic mode
5. `level_offset_plus_history` as a diagnostic mode

The main scientific gate is based on `fluctuation_plus_history` versus `history_only`. The level-offset modes are diagnostic and must not be used to claim that Stage III has learned short-term fluctuation.

Residual-history features must use causal label availability. The preferred history builder is the V12e per-horizon label-availability builder. It prevents a residual from entering history before its true label would have arrived for that prediction horizon.

## 6. Teacher Model Design

The first V13 model should be a train-only chronological out-of-fold residual teacher. It predicts residual correction targets from the V13 context feature matrix.

The teacher must keep these safety properties:

- fit only on training split
- chronological folds
- purge gap between fit and calibration blocks
- validation-only gate for model acceptance
- report-only test metrics
- shrinkage alpha selected without using test data
- fail-closed behavior when validation gates fail

The teacher may reuse the V12d teacher implementation through a V13 wrapper if the wrapper records V13-specific protocol names and feature semantics. A new `models/stage3_residual_teacher_v13.py` file should own V13 metadata and checkpoint save/load functions.

The V13 teacher must not be a toy regressor or dummy correction. It must reuse the chronological out-of-fold residual teacher policy, ridge/alpha grid calibration, block diagnostics, and fail-closed gate behavior from V12c/V12d unless a later implementation plan documents a stricter replacement.

## 7. Generator Policy

V13 must not train a generator by default. Generator training is allowed only after the teacher-only V13 gate passes.

The generator canary entrypoint is deliberately out of scope for the first implementation batch unless the teacher-only probe passes. The first implementation must still leave a clear extension point:

- accepted teacher checkpoint path
- V13 signal builder state
- V13 history builder state
- V13 context feature mode
- diagnostic manifest explaining why generator training was or was not allowed

If the teacher-only gate fails, the runner must write an abort file explaining that generator training is not scientifically justified.

If generator training becomes allowed later, it must also be production-grade within this repository: no untrained generator, no identity-only dummy generator, no synthetic target-only demonstration, and no saved deployable artifact unless served-policy validation passes.

## 8. Counterfactuals And Claim Gates

V13 must evaluate these intervention modes:

- `real`
- `zero`
- `shuffled`
- `sign_flipped`
- `amplified_2x`
- `amplified_5x`
- `random_matched_norm`
- `conditional_shuffled`
- `conditional_random_matched_norm`

The strict V13 teacher-only gate for `fluctuation_plus_history` should require:

- positive real MSE gain on every rolling validation window
- positive real MAE gain on every rolling validation window
- real fluctuation beats the best counterfactual by the configured margin
- `fluctuation_plus_history` beats `history_only` by a positive partial MSE gain
- `fluctuation_plus_history` beats `history_only` by a positive partial MAE gain
- sign agreement is at or above the configured threshold

All listed counterfactual modes, including `shuffled`, `random_matched_norm`, `conditional_shuffled`, and `conditional_random_matched_norm`, must participate in the teacher acceptance gate. V13 must not inherit the V12 default that excludes offline diagnostic counterfactuals from the gate. If the runner cannot verify that the full counterfactual set participated in the gate, the run must write an abort record rather than a complete record.

Default proposed thresholds:

```yaml
diagnostic_min_real_mse_gain: 0.005
diagnostic_min_real_mae_gain: 0.002
diagnostic_counterfactual_margin: 0.002
diagnostic_min_sign_agreement: 0.55
diagnostic_min_partial_mse_gain_vs_history: 0.001
diagnostic_min_partial_mae_gain_vs_history: 0.0005
rolling_min_window_pass_rate: 1.0
```

These thresholds are pre-declared gate values. They must not be relaxed after seeing test results.

## 9. Experiment Restart Plan

V13 experiments restart in stages.

### Directional Stop Rule

If V13 evidence shows that the new idea is conceptually wrong or experimentally unsafe, experiments must stop immediately rather than continue scaling. The runner or matrix wrapper must write an abort or diagnostic record and must not launch later-stage experiments.

Stop conditions include:

- `fluctuation_delta` collapses to near-zero for most real samples after train-only scaling;
- real `fluctuation_delta` is consistently worse than `zero`, `shuffled`, `sign_flipped`, or conditional counterfactuals in validation;
- `fluctuation_plus_history` fails to beat `history_only` on rolling validation partial gains;
- `fluctuation_plus_history` fails to beat `history_only` on either rolling validation MSE partial gain or rolling validation MAE partial gain;
- the full real/zero/shuffled/sign-flipped/amplified/random/conditional counterfactual set is not included in the acceptance gate;
- the main V13 signal path uses residual canonicalization, latent-space transport, or V12 delta standardization before computing `fluctuation_delta`;
- rolling validation gate artifacts are missing;
- per-horizon label availability or previous-tail logic is violated;
- V13 accidentally uses test results for selection;
- V13 uses raw `phi_v - phi_i` as the served fluctuation signal;
- any implementation path falls back to a toy teacher, toy data, toy gate, or dummy generator.

When any stop condition is triggered, the next action is documentation and diagnosis, not larger experiments.

### Stage A: Unit And Protocol Tests

Run:

```powershell
pytest tests\test_stage3_v13_signal_contract.py tests\test_stage3_v13_teacher_protocol.py -q
```

This stage validates signal semantics, zero preservation, train-only scaling, previous-tail behavior, feature-mode construction, and checkpoint metadata.

### Stage B: Panama Real-Data Pilot

Run a real-data teacher-only pilot on Panama. This is still not publication evidence, but it must use the real model, real data pipeline, real aligned representation path, real V13 teacher, and real V13 audit artifacts.

```powershell
python train\run_stage3_v13_fluctuation_teacher_probe.py --config configs\stage3_v13_fluctuation_teacher_probe.yaml --dataset panama --run-id v13fluct_pilot_panama_s42_01 --seed 42 --epochs 3 --invariant-epochs 3 --device auto --max-train-batches 120 --max-generator-batches 120 --max-eval-batches 120 --validation-windows 5 --out-root logs\s3v13fluct_pilot --checkpoint-root ckpt\s3v13fluct_pilot
```

This stage checks whether the V13 pipeline can run at a meaningful pilot scale. It does not prove general performance, but it must be large enough to expose obvious failures in chronological folds, history controls, counterfactual gates, and no-overwrite manifests.

### Stage C: Public-3 Teacher Matrix

After the Panama pilot works, run a real public-3 matrix:

```powershell
python train\run_stage3_v13_fluctuation_teacher_matrix.py --config configs\stage3_v13_fluctuation_teacher_probe.yaml --datasets panama tetouan spain_energy_weather --seeds 42 43 44 --epochs 5 --invariant-epochs 5 --device auto --max-generator-batches 480 --max-eval-batches 480 --validation-windows 5 --out-root logs\s3v13fluct_matrix --checkpoint-root ckpt\s3v13fluct_matrix
```

This stage decides whether V13 has enough teacher-only evidence to justify a generator canary.

### Stage D: Generator Canary

Generator canary is allowed only if Stage C passes the V13 teacher-only gate. If Stage C fails, the correct scientific result is an abort record and a negative diagnostic report.

## 10. Documentation And Evidence Ledger

V13 implementation must update or create:

- `docs/audit/STAGE3_V13_FLUCTUATION_DESIGN_AND_EXPERIMENT_RECORD.md`
- `docs/audit/EXPERIMENT_EVIDENCE_LEDGER.md`
- `docs/audit/RISK_REGISTER.md`
- `docs/audit/PAPER_CLAIMS_MATRIX.md` only after experiments produce evidence relevant to manuscript claims

The V13 record must distinguish:

- verified implementation facts
- teacher-only pilot results
- matrix results
- failed gates
- passed gates
- remaining risks

## 11. Claims That Remain Forbidden Until Evidence Exists

The manuscript must not claim:

- V13 proves concept drift
- V13 significantly improves forecasting accuracy
- V13 is state-of-the-art
- the generator is validated
- `phi_v - phi_i` is a short-term fluctuation signal

Allowed wording before positive evidence:

```text
V13 introduces an auditable fluctuation-first Stage III protocol that separates long-term representation offsets from temporal fluctuation signals and tests their incremental value against causal residual-history controls.
```

Allowed wording after a passing teacher-only gate but before generator success:

```text
The V13 teacher-only probe indicates that the temporal fluctuation signal can provide incremental residual-correction information beyond causal history controls under the pre-declared validation gate.
```

Allowed wording after a passing generator canary and public-3 validation:

```text
The V13 generator path is supported by validation-gated teacher-only evidence and served-policy generator canary evidence under counterfactual fluctuation interventions.
```

## 12. Acceptance Criteria For The First V13 Implementation Batch

The first V13 implementation batch is complete only when:

1. V13 signal contract tests pass.
2. V13 teacher protocol tests pass.
3. V13 config exists and uses new logs/checkpoint roots.
4. V13 runner writes resolved config, manifest, diagnostics, and abort or complete status.
5. V13 Panama real-data pilot runs without overwriting old outputs.
6. The V13 audit record documents the command, output directory, gate status, and interpretation.
7. The V13 design and implementation explicitly document which V2-V12 mechanisms were inherited and which were rejected.
8. No V13 algorithm or experiment artifact uses a toy implementation as evidence of scientific performance.

The full thread goal is not complete until V13 code exists, experiments are restarted, evidence records are updated, and remaining scientific risks are explicitly reported.
