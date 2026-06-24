# Stage III V13 Fluctuation-First Design And Experiment Record

## Scope

V13 is a new non-overwriting Stage III version. It tests whether the temporal change of the aligned residual offset, `delta_phi_v(t)-delta_phi_v(t-1)`, provides incremental residual-correction value beyond causal residual-history controls.

## Pre-Pilot Stop Event

Status: experiments stopped before Panama pilot.

Reason: a read-only ML protocol review found two direction-level risks that would make V13 pilot results uninterpretable if left unresolved.

1. The copied V12d collection path could apply residual canonicalization or V12 delta standardization before V13 computes the fluctuation signal. That would change the meaning of the V13 signal from raw `delta_phi_v` transition to a transformed latent transition.
2. The copied V12 teacher gate helper could exclude offline and conditional counterfactuals from the acceptance gate. That would allow a false V13 pass even when `real` does not beat the strongest counterfactual modes.

The initial V13 pilot was not launched until these two risks were corrected.

## Corrected V13 Protocol

- The main V13 signal path must read raw `representation.delta_phi_v`, or exact `representation.phi_v - representation.phi_i` only as a fallback.
- Residual canonicalizers, latent-space transports, and V12 delta standardizers are excluded from the main V13 signal path.
- V13 train-only scaling is performed by `Stage3SignalBuilderV13`.
- The V13 teacher gate includes `real`, `zero`, `shuffled`, `sign_flipped`, `amplified_2x`, `amplified_5x`, `random_matched_norm`, `conditional_shuffled`, and `conditional_random_matched_norm`.
- `fluctuation_plus_history` must beat `history_only` on both rolling MSE partial gain and rolling MAE partial gain before any larger experiment or generator canary is allowed.

## Current Status

Implementation-level protocol fixes have been applied. Two corrected Panama teacher-only probes have now been executed: the first bounded probe with `120` train/generator/eval batches, and a larger single-dataset probe with `480` train/generator/eval batches. Both wrote `teacher_only_abort.json`, so V13 is stopped before any public-3 matrix or generator canary.

## Verification Commands

- `C:\Users\admin\anaconda3\python.exe -m pytest tests\test_stage3_v13_signal_contract.py tests\test_stage3_v13_teacher_protocol.py -q`
  - Result: `13 passed`.
- `C:\Users\admin\anaconda3\python.exe -m pytest tests\test_stage3_v12e_label_availability.py tests\test_stage3_v12d_context_features.py tests\test_stage3_v12b_signal_contract.py -q`
  - Result: `10 passed`.
- `C:\Users\admin\anaconda3\python.exe train\run_stage3_v13_fluctuation_teacher_probe.py --help`
  - Result: exited successfully and showed V13 probe CLI.
- `C:\Users\admin\anaconda3\python.exe train\run_stage3_v13_fluctuation_teacher_matrix.py --help`
  - Result: exited successfully and showed V13 matrix CLI.
- `C:\Users\admin\anaconda3\python.exe -m py_compile train\stage3_v13_signal_contract.py models\stage3_residual_teacher_v13.py train\run_stage3_v13_fluctuation_teacher_probe.py train\run_stage3_v13_fluctuation_teacher_matrix.py`
  - Result: exited successfully.
- `C:\Users\admin\anaconda3\python.exe -m pytest tests\test_stage3_v13_failure_diagnostics.py tests\test_stage3_v13_signal_contract.py tests\test_stage3_v13_teacher_protocol.py -q`
  - Result: `18 passed`.
- `C:\Users\admin\anaconda3\python.exe -m py_compile train\analyze_stage3_v13_failure_diagnostics.py`
  - Result: exited successfully.

## Experiment Status

- Panama `b120` pilot: run and stopped by the pre-declared V13 direction gate.
- Panama `b480` probe: run and stopped by the pre-declared V13 direction gate.
- Public-3 matrix: not run.
- Generator canary: blocked.

## Corrected Panama Pilot

Command:

`C:\Users\admin\anaconda3\python.exe train\run_stage3_v13_fluctuation_teacher_probe.py --config configs\stage3_v13_fluctuation_teacher_probe.yaml --dataset panama --run-id v13fluct_pilot_panama_s42_01 --seed 42 --epochs 3 --invariant-epochs 3 --device auto --max-train-batches 120 --max-generator-batches 120 --max-eval-batches 120 --validation-windows 5 --out-root logs\s3v13fluct_pilot --checkpoint-root ckpt\s3v13fluct_pilot`

Primary output:

`logs/s3v13fluct_pilot/v13fluct_pilot_panama_s42_01/panama/seed_42/teacher_only_abort.json`

Stop reasons:

- `fluctuation_plus_history_teacher_gate_failed`
- `partial_mae_gain_vs_history_nonpositive`
- `partial_mse_gain_vs_history_nonpositive`

Key evidence:

- V13 signal contract used `signal_mode=fluctuation_delta`, fit on the training split only, with `signal_fit_sample_count=120`.
- The rolling validation gate for `fluctuation_plus_history` failed: pass rate `3/5`, failed windows `1;3`, worst MAE gain `-0.008035`, worst real-vs-best-counterfactual MSE margin `0.001857 < 0.002`, and worst sign agreement `0.506944 < 0.55`.
- The aggregate validation partial gain over `history_only` was positive: partial MSE gain `0.151253`, partial MAE gain `0.097218`, and partial sign agreement gain `0.077083`.
- The test partial gain over `history_only` was negative: partial MSE gain `-0.039385`, partial MAE gain `-0.009873`, with almost no sign-agreement improvement (`0.000694`).
- `fluctuation_only` was harmful on validation and test. Its validation MSE/MAE gains were `-0.065867` and `-0.052023`; its test MSE/MAE gains were `-0.482847` and `-0.179868`.
- Level-offset diagnostic modes did not rescue the pilot. `level_offset_only` was harmful, and `level_offset_plus_history` did not pass the rolling gate.

Interpretation:

The corrected V13 pilot shows that short-term fluctuation information may add value on some validation windows when combined with causal residual history, but it is not stable enough to justify expanding the experiment. The negative test partial gain means V13 must not be used as manuscript performance evidence or as a basis for generator training.

Derived diagnostics:

- `docs/audit/stage3_v13_pilot_failure_diagnostics/v13fluct_pilot_panama_s42_01/v13_window_failure_diagnostics.csv`
- `docs/audit/stage3_v13_pilot_failure_diagnostics/v13fluct_pilot_panama_s42_01/v13_split_generalization_diagnostics.csv`
- `docs/audit/stage3_v13_pilot_failure_diagnostics/v13fluct_pilot_panama_s42_01/v13_alpha_candidate_stability_fh.csv`
- `docs/audit/stage3_v13_pilot_failure_diagnostics/v13fluct_pilot_panama_s42_01/v13_failure_diagnostics_report.md`

The derived window diagnostics separate two failure types:

- Window `1` is primarily an attribution failure: real MSE gain is high, but `zero` nearly ties real, with margin `0.001857 < 0.002`.
- Window `3` is a real-behavior and incremental-value failure: real MAE gain is `-0.008035`, sign agreement is `0.506944 < 0.55`, and partial MSE/MAE gains over `history_only` are negative.

## Larger Panama `b480` Probe

Command:

`C:\Users\admin\anaconda3\python.exe train\run_stage3_v13_fluctuation_teacher_probe.py --config configs\stage3_v13_fluctuation_teacher_probe.yaml --dataset panama --run-id v13fluct_panama_s42_b480_01 --seed 42 --epochs 5 --invariant-epochs 5 --device auto --max-train-batches 480 --max-generator-batches 480 --max-eval-batches 480 --validation-windows 5 --out-root logs\s3v13fluct_b480 --checkpoint-root ckpt\s3v13fluct_b480`

Primary output:

`logs/s3v13fluct_b480/v13fluct_panama_s42_b480_01/panama/seed_42/teacher_only_abort.json`

Stop reasons:

- `fluctuation_plus_history_teacher_gate_failed`
- `partial_mae_gain_vs_history_nonpositive`
- `partial_mse_gain_vs_history_nonpositive`

Key evidence:

- V13 signal contract again used `signal_mode=fluctuation_delta`.
- `fluctuation_plus_history` rolling validation failed more clearly than in the bounded pilot: pass rate `2/5`, failed windows `1;2;3`.
- Aggregate validation partial gain over `history_only` became negative: partial MSE gain `-0.004330`, partial MAE gain `-0.000359`.
- Test partial gain over `history_only` became positive (`MSE +0.001248`, `MAE +0.005806`), but this is not usable because the prospective rolling validation gate failed.
- The teacher was no longer as sample-starved as the `b120` pilot: fit samples increased from `120` to `480`, OOF-covered samples from `56` to `416`. Despite that, the `fluctuation_plus_history` teacher still had negative worst-block OOF evidence: worst block MSE gain `-0.268927`, worst block MAE gain `-0.089480`, worst block sign agreement `0.361446`.

Derived diagnostics:

- `docs/audit/stage3_v13_pilot_failure_diagnostics/v13fluct_panama_s42_b480_01/v13_window_failure_diagnostics.csv`
- `docs/audit/stage3_v13_pilot_failure_diagnostics/v13fluct_panama_s42_b480_01/v13_split_generalization_diagnostics.csv`
- `docs/audit/stage3_v13_pilot_failure_diagnostics/v13fluct_panama_s42_b480_01/v13_alpha_candidate_stability_fh.csv`
- `docs/audit/stage3_v13_pilot_failure_diagnostics/v13fluct_panama_s42_b480_01/v13_failure_diagnostics_report.md`

Interpretation:

The larger probe weakens the hypothesis that the `b120` failure was only caused by a too-small pilot. Increasing the train/generator/eval cap to `480` did not make `fluctuation_plus_history` stable beyond `history_only`; instead, the validation partial gain became negative and four of five derived window diagnostics failed at least one gate or incremental condition.

## Current Decision

Stop V13 expansion. Public-3 expansion and generator canaries remain blocked. The next allowed work is algorithm redesign or diagnostic readout, not larger experiments under the same relaxed teacher path. A defensible V13 successor should first address teacher dimensionality, policy aggressiveness, and per-output stability before another training probe is launched.
