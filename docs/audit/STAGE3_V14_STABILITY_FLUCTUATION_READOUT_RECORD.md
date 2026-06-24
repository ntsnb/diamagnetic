# Stage III V14 Stability-First Fluctuation Readout Record

## Scope

V14 is a non-overwriting successor to V13. It keeps the corrected Stage III signal:

```text
fluctuation_delta(t) = delta_phi_v(t) - delta_phi_v(t-1)
```

where `delta_phi_v = phi_v - phi_i`.

V14 does not use `phi_v(t)-phi_v(t-1)` as the served Stage III signal. That variant temporal delta remains diagnostic only.

## Why V14 Was Created

V13 corrected the signal semantics and full counterfactual gate, but both corrected Panama probes failed:

- `logs/s3v13fluct_pilot/v13fluct_pilot_panama_s42_01/panama/seed_42/teacher_only_abort.json`
- `logs/s3v13fluct_b480/v13fluct_panama_s42_b480_01/panama/seed_42/teacher_only_abort.json`

The larger V13 b480 probe showed that failure was not explained by the small b120 sample alone. The inherited V12b readout expanded the context feature vector through `[x, abs(x), x^2, norm(x)]`, and relaxed policies could preserve exploratory corrections that strict gates would reject.

V14 therefore tests a narrower falsification:

- keep raw `fluctuation_delta`;
- keep V12e per-horizon label-availability history;
- replace the default V12b quadratic readout with train-only scaled linear features;
- use `selected_policy=strict` as the default prediction path;
- stop before public-3 or generator training if the single Panama pilot fails.

## New Versioned Files

- `models/stage3_residual_teacher_v14.py`
- `train/run_stage3_v14_stability_teacher_probe.py`
- `train/run_stage3_v14_stability_teacher_matrix.py`
- `train/analyze_stage3_v14_failure_diagnostics.py`
- `configs/stage3_v14_stability_teacher_probe.yaml`
- `tests/test_stage3_v14_teacher_protocol.py`
- `docs/superpowers/specs/2026-06-02-stage3-v14-stability-fluctuation-readout-design.md`
- `docs/superpowers/plans/2026-06-02-stage3-v14-stability-fluctuation-readout.md`

V14 did not overwrite V2-V13 code, raw logs, raw data, checkpoints, tables, or figures.

## Implementation Checks

- `C:\Users\admin\anaconda3\python.exe -m pytest tests\test_stage3_v14_teacher_protocol.py -q`
  - Result: `8 passed`.
- `C:\Users\admin\anaconda3\python.exe -m pytest tests\test_stage3_v14_teacher_protocol.py tests\test_stage3_v13_failure_diagnostics.py tests\test_stage3_v13_signal_contract.py tests\test_stage3_v13_teacher_protocol.py tests\test_stage3_v12e_label_availability.py tests\test_stage3_v12d_context_features.py tests\test_stage3_v12b_signal_contract.py -q`
  - Result: `36 passed`.
- `C:\Users\admin\anaconda3\python.exe -m py_compile models\stage3_residual_teacher_v14.py train\run_stage3_v14_stability_teacher_probe.py train\run_stage3_v14_stability_teacher_matrix.py`
  - Result: exited successfully.
- `C:\Users\admin\anaconda3\python.exe train\run_stage3_v14_stability_teacher_probe.py --help`
  - Result: exited successfully and showed the V14 probe CLI.
- `C:\Users\admin\anaconda3\python.exe train\run_stage3_v14_stability_teacher_matrix.py --help`
  - Result: exited successfully and showed the V14 matrix CLI.
- `C:\Users\admin\anaconda3\python.exe -m py_compile train\analyze_stage3_v14_failure_diagnostics.py`
  - Result: exited successfully.

## Panama b480 Pilot

Command:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v14_stability_teacher_probe.py --config configs\stage3_v14_stability_teacher_probe.yaml --dataset panama --run-id v14stable_panama_s42_b480_01 --seed 42 --epochs 5 --invariant-epochs 5 --device auto --max-train-batches 480 --max-generator-batches 480 --max-eval-batches 480 --validation-windows 5 --out-root logs\s3v14stability_b480 --checkpoint-root ckpt\s3v14stability_b480
```

Primary output:

`logs/s3v14stability_b480/v14stable_panama_s42_b480_01/panama/seed_42/teacher_only_abort.json`

Stop reasons:

- `fluctuation_plus_history_teacher_gate_failed`
- `partial_mae_gain_vs_history_nonpositive`
- `partial_mse_gain_vs_history_nonpositive`

Key evidence:

- V14 used `teacher_readout_mode=linear`.
- V14 used `teacher_selected_policy=strict` and `teacher_claim_policy=strict`.
- `fluctuation_plus_history` had `feature_dim=208`, not the inherited V12b quadratic `625`-feature readout.
- `fluctuation_plus_history` strict output pass count was `0/12`, and strict alpha nonzero fraction was `0.0`.
- `history_only` strict output pass count was `1/12`, and strict alpha nonzero fraction was `0.083333`.
- `fluctuation_plus_history` rolling validation passed `0/5` windows. All five windows failed by real behavior, attribution, and incremental-value criteria.
- Validation partial gain versus `history_only` was negative: MSE `-0.014151`, MAE `-0.016179`.
- Test partial gain versus `history_only` was also negative: MSE `-0.012672`, MAE `-0.011822`.
- `zero` was the best counterfactual for `fluctuation_plus_history` because the strict policy made the real correction exactly zero.

Derived diagnostics:

- `docs/audit/stage3_v14_pilot_failure_diagnostics/v14stable_panama_s42_b480_01/v14_window_failure_diagnostics.csv`
- `docs/audit/stage3_v14_pilot_failure_diagnostics/v14stable_panama_s42_b480_01/v14_split_generalization_diagnostics.csv`
- `docs/audit/stage3_v14_pilot_failure_diagnostics/v14stable_panama_s42_b480_01/v14_alpha_candidate_stability_fh.csv`
- `docs/audit/stage3_v14_pilot_failure_diagnostics/v14stable_panama_s42_b480_01/v14_failure_diagnostics_report.md`

Derived diagnostic summary:

- Diagnosed windows: `5`.
- Failed windows: `5`.
- Alpha candidate rows: `108`.
- Rows with stability warnings: `108`.

## Current Decision

Stop V14 expansion.

Do not run public-3 or generator canaries from V14. The V14 result is useful negative evidence: lowering readout capacity and enforcing strict OOF policy prevents unsupported relaxed corrections, but it also shows that the current `fluctuation_delta + history` representation does not produce stable incremental residual-correction value beyond `history_only` on the Panama b480 pilot.

## Scientific Interpretation

V14 supports a safety conclusion, not a performance conclusion.

The defensible statement is:

> Under a stricter stability-first readout, the current fluctuation signal is rejected rather than served. This suggests the previous nonzero relaxed corrections were not stable enough to support a concept-drift adaptation claim.

The non-defensible statements are:

- V14 validates Stage III.
- V14 improves online forecasting.
- V14 proves short-term fluctuation-driven adaptation.
- V14 justifies generator training.
- V14 should be added to manuscript performance tables.

