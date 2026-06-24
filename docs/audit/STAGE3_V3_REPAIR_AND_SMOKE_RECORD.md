# Stage III V3 Repair And Smoke Record

Date: 2026-05-26

This record documents the first V3-style repair pass after the V2.5 large-scale audit failed all claim gates.

## Why V3 Was Needed

The V2.5 b240/full-eval matrix showed:

- 9/9 runs failed the test claim gate.
- 9/9 runs failed rolling validation.
- Mean MSE gain was only about 0.3172%.
- Mean MAE gain was only about 0.1969%.
- Mean reject rate was about 82.94%.
- `amplified_2x` or `amplified_5x` often tied or beat real drift.

The last point exposed a protocol bug: counterfactual amplified drift was built in raw-drift space and then passed through `normalize_vector=True`, which could divide away the amplification before the generator saw it.

## Code Changes

All changes remain in the opt-in Stage III V2/V3 path. Legacy Stage III files are not replaced.

| Area | Change | Files |
| --- | --- | --- |
| Train-only alignment | Added `TrainSetTemporalResidualCCAAligner`, a ridge-whitened train-only CCA aligner for `delta_phi_v[t]-delta_phi_v[t-1]` versus forecast residual transitions. | `models/aligned_representation_v2.py` |
| Alignment diagnostics | Added `temporal_residual_alignment_diagnostics()` and runner output `alignment_diagnostics.csv`. | `models/aligned_representation_v2.py`, `train/run_stage3_v2_sensitivity_audit.py` |
| Artifact reload | Artifact bundle can reload both old PLS-style canonicalizer and new temporal residual CCA aligner. | `train/stage3_v2_artifacts.py` |
| No-overwrite protection | Runner refuses existing output/checkpoint directories by default; generator/risk-gate checkpoint savers refuse existing `.pt` files unless `overwrite=True`. | `train/run_stage3_v2_sensitivity_audit.py`, `train/stage3_generator_training.py`, `train/stage3_risk_gate_v2.py` |
| Resolved config evidence | Runner writes `resolved_config.json` and records its SHA256 in `run_manifest.json`. | `train/run_stage3_v2_sensitivity_audit.py` |
| Counterfactual intervention space | Non-real sensitivity modes now default to generator-input space, so `amplified_2x/5x` preserve magnitude after the real signal pipeline. | `train/run_stage3_v2_sensitivity_audit.py` |
| V3 config | Added strict V3 config with temporal residual CCA, generator-input counterfactuals, five validation windows, and separate stage3_v3 roots. | `configs/stage3_v3_strict.yaml` |

## Verification

Targeted tests:

```powershell
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_factor_mechanism_v2.py tests\test_adaptation_v2.py tests\test_hyperlora_generator_v2.py tests\test_aligned_representation_v2.py tests\test_offline_aligned_representation_v2.py tests\test_stage3_generator_training_v2.py tests\test_stage3_generator_selection_v2.py tests\test_stage3_functional_v2.py tests\test_stage3_risk_gate_v2.py tests\test_stage3_sensitivity_gate_v2.py tests\test_stage3_sensitivity_v2.py tests\test_online_loop_stage3_v2.py tests\test_stage3_v2_protocol_safety.py tests\test_stage3_v2_split_safe_preprocessing.py tests\test_stage3_v2_artifacts.py tests\test_stage3_signals.py -q
```

Result:

```text
88 passed in 3.75s
```

## Smoke Run

Command:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v2_sensitivity_audit.py --config configs\stage3_v3_strict.yaml --dataset panama --run-id codex_stage3_v3_smoke_20260526_01 --seed 42 --epochs 1 --invariant-epochs 1 --generator-epochs 2 --device cpu --modes real zero amplified_2x random_matched_norm --max-train-batches 5 --max-generator-batches 5 --max-eval-batches 10 --validation-windows 2 --drift-source delta_phi_v --drift-polarity 1.0 --signal-mode delta_vector --normalize-vector --out-root logs\load_benchmarks\stage3_v3\smoke_20260526 --checkpoint-root checkpoints\stage3_v3\smoke_20260526
```

Output:

```text
logs/load_benchmarks/stage3_v3/smoke_20260526/codex_stage3_v3_smoke_20260526_01/panama/seed_42
checkpoints/stage3_v3/smoke_20260526/panama/codex_stage3_v3_smoke_20260526_01/seed_42
```

Manifest confirms:

- `stage3_protocol_version = v3_strict_temporal_residual_cca`
- `stage3_residual_canonicalizer_method = temporal_residual_cca_v3`
- `stage3_counterfactual_intervention_space = generator_input`
- `stage3_artifacts_reloaded_for_eval = true`
- `test_gate_report_only = true`
- `resolved_config_sha256` matches the saved `resolved_config.json`.

## Smoke Result

This smoke is not a performance result. It is only a protocol check.

| Metric | Value |
| --- | ---: |
| test real MSE gain | 0.001976 |
| test real MAE gain | 0.000978 |
| best counterfactual | `amplified_2x` |
| real-vs-best-counterfactual MSE margin | 0.0000918 |
| test reject rate | 0.0 |
| rolling validation pass rate | 0.0 |
| test claim gate | fail |

Interpretation:

V3 smoke validates the repaired audit mechanics, but it still does not validate a performance claim. The fact that `amplified_2x` remains the best counterfactual means the generator still has not proven real-drift-specific conditioning, even though amplification is no longer normalized away by the old raw-drift intervention path.

## Current Scientific Status

V3 has improved the evidence protocol and alignment diagnostics. It has not yet shown significant forecasting improvement.

Allowed wording:

```text
Stage III V3 adds train-only temporal residual CCA alignment, generator-input counterfactual interventions, no-overwrite provenance safeguards, and resolved-config manifests for stricter auditability.
```

Not allowed:

```text
Stage III V3 significantly improves online forecasting accuracy.
Stage III V3 proves reliable concept-drift-conditioned adaptation.
```

## Next Probe

Run one b120/full-eval V3 probe before any multi-dataset expansion:

```text
dataset: panama
seed: 42
epochs: 3 / 3 / 6
train/generator budget: 120 batches
test: full eval, report-only
```

If this probe still fails rolling validation or counterfactual margin, do not proceed directly to b480/b960. Repair the generator/risk-gate training objective first.

## b120 Full-Eval Probe

Command:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v2_sensitivity_audit.py --config configs\stage3_v3_strict.yaml --dataset panama --run-id codex_stage3_v3_probe_b120_panama_seed42_20260526_01 --seed 42 --epochs 3 --invariant-epochs 3 --generator-epochs 6 --device auto --modes real zero shuffled sign_flipped amplified_2x amplified_5x random_matched_norm --max-train-batches 120 --max-generator-batches 120 --validation-windows 5 --drift-source delta_phi_v --drift-polarity 1.0 --signal-mode delta_vector --normalize-vector --out-root logs\load_benchmarks\stage3_v3\probe_20260526 --checkpoint-root checkpoints\stage3_v3\probe_20260526
```

Output:

```text
logs/load_benchmarks/stage3_v3/probe_20260526/codex_stage3_v3_probe_b120_panama_seed42_20260526_01/panama/seed_42
checkpoints/stage3_v3/probe_20260526/panama/codex_stage3_v3_probe_b120_panama_seed42_20260526_01/seed_42
```

The first foreground command timed out after one hour, but the Python process continued and eventually wrote `run_manifest.json`. The run is complete because the manifest exists and test sensitivity files were written.

### b120 Result

| Metric | Value |
| --- | ---: |
| test real MSE gain | 0.009338 |
| test real MAE gain | 0.006660 |
| test reject rate | 0.168563 |
| test use rate | 0.831437 |
| best counterfactual | `amplified_2x` |
| best counterfactual MSE gain | 0.010273 |
| real-vs-best-counterfactual margin | -0.000935 |
| rolling validation pass rate | 0.0 |
| test claim gate | fail |

Full test rows:

| mode | MSE | MAE | use rate |
| --- | ---: | ---: | ---: |
| real | 0.289382 | 0.375730 | 0.8314 |
| zero | 0.292109 | 0.378249 | 0.0000 |
| shuffled | 0.294831 | 0.381470 | 0.7775 |
| sign_flipped | 0.301125 | 0.388104 | 0.8401 |
| amplified_2x | 0.289108 | 0.375696 | 0.9808 |
| amplified_5x | 0.289109 | 0.375680 | 0.9802 |
| random_matched_norm | 0.295031 | 0.381660 | 0.7990 |

Rolling validation:

```text
rolling_claim_gate_pass = false
validation_window_pass_rate = 0.0
rolling_failure_reasons = window_pass_rate;counterfactual
```

### Interpretation

V3 improved some mechanics relative to V2.5:

- reject rate dropped to 16.86%, below the 20% claim-gate threshold;
- real drift beat zero, shuffled, sign-flipped, and random matched-norm interventions on test;
- MSE and MAE gains were both positive on Panama seed42 b120/full-eval;
- artifact reload, split-safe preprocessing, resolved config, and temporal residual CCA were recorded in manifest.

But V3 still failed the scientific claim:

- `amplified_2x` was better than real drift;
- 5/5 rolling validation windows failed by counterfactual margin;
- generator selection itself did not pass rolling validation;
- one dataset/seed cannot establish robustness or statistical significance.

Conclusion:

```text
V3 has stronger auditability and a better served-output profile than V2.5 on this probe, but it still does not prove real-drift-specific adaptation. Do not update the manuscript to claim significant Stage III performance improvement.
```

## Next Required Repair

The next repair should target generator amplitude calibration:

1. Add a monotonicity regularizer so small amplification changes delta norm without making amplified drift systematically better than real drift.
2. Add a served-policy generator selection mode that includes the trained risk gate, or disable risk gate for selection and final serving consistently.
3. Add per-step trace rows for real and amplified modes to locate whether amplification wins because of delta magnitude, risk-gate acceptance, or blend behavior.
4. Re-run Panama seed42 b120 after those repairs before expanding to three datasets.
