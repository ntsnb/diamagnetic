# Stage III V15 Residual-Target Predictability Diagnostic Design

## Purpose

V15 is a non-overwriting diagnostic version of Stage III. Its purpose is to test whether the corrected Stage III fluctuation signal contains stable residual-prediction information beyond causal residual-history controls.

V15 does not train a Stage III generator. V15 does not update the manuscript with positive Stage III claims unless the diagnostic produces stable, pre-declared evidence.

## Background

The V14 Panama b480 pilot produced `teacher_only_abort.json` and stopped before generator training. The recorded stop reasons were:

- `fluctuation_plus_history_teacher_gate_failed`
- `partial_mae_gain_vs_history_nonpositive`
- `partial_mse_gain_vs_history_nonpositive`

The rolling validation evidence showed that `fluctuation_plus_history` failed all five validation windows and had negative partial MSE and MAE gains relative to `history_only`. This means expanding V14 or launching a V14 generator would violate the direction-level stop rule.

The next safe question is narrower:

Does `fluctuation_delta` contain any stable, train-only, chronologically validated information about pre-declared residual targets that is not already captured by residual history?

## Approved Direction

V15 will be a target-predictability diagnostic. It will compare different input feature groups against pre-declared residual targets under chronological validation.

The primary comparison is:

```text
fluctuation_plus_history > history_only
```

The comparison must be measured on rolling validation windows. It is not enough for an ungated model or a single validation segment to look favorable.

## Non-Goals

V15 will not:

- Train `AdaptationGenerator` or any Stage III generator.
- Run public test-set generator evaluation.
- Modify V12, V13, or V14 implementation files.
- Replace existing V12-V14 logs, checkpoints, or diagnostic tables.
- Claim manuscript success from secondary targets alone.
- Treat `level_offset_plus_history` success as proof that fluctuation-driven adaptation works.
- Tune target definitions after seeing the diagnostic output.

## Input Feature Groups

V15 will evaluate the following feature groups.

### Primary Groups

- `history_only`: causal residual-history features constructed with the existing V12e per-horizon label-availability protocol.
- `fluctuation_only`: the V13/V14 corrected fluctuation signal.
- `fluctuation_plus_history`: concatenation of the corrected fluctuation signal and causal residual-history features.

### Auxiliary Mechanism Controls

- `level_offset_only`: the undifferenced representation offset.
- `level_offset_plus_history`: the undifferenced representation offset plus causal residual-history features.

The auxiliary controls are diagnostic only. If `level_offset_plus_history` succeeds while `fluctuation_plus_history` fails, the conclusion is that temporal differencing may be losing useful information. That outcome does not justify generator training.

## Signal Contract

V15 uses the same corrected signal contract as V13 and V14:

```text
delta_phi_v(t) = phi_v(t) - phi_i(t)
fluctuation_delta(t) = delta_phi_v(t) - delta_phi_v(t-1)
```

The raw `delta_phi_v` source must come from the aligned representation model. V15 must reject configurations that request `phi_gap`, scalar mean-compressed drift, residual canonicalizers, or delta standardizers in the main path.

The signal builder must be fit only on the train split and then applied to validation and test splits with causal previous-tail handling.

## Residual Targets

V15 uses two primary targets. These targets are fixed before running the diagnostic.

### Primary Target 1: `raw_residual_vector`

Definition:

```text
r(t, h) = y(t, h) - y_base(t, h)
```

This target asks whether the context can predict the correction that would directly improve the base forecast.

### Primary Target 2: `residual_change_vector`

Definition:

```text
delta_r(t, h) = r(t, h) - r(t-1, h)
```

This target asks whether `fluctuation_delta` is aligned with temporal changes in forecasting error.

### Secondary Targets

Secondary targets may be reported, but they cannot unlock generator training:

- `residual_sign`
- pre-declared horizon blocks, such as short, medium, and long horizons

Secondary targets are exploratory diagnostics. They can motivate a future design only if they are documented as secondary evidence.

## Model Class

V15 will use a train-only chronological readout. The preferred implementation is to reuse the V14 stability teacher and adapt the `y_base_train` / `y_train` tensors so that the teacher predicts the selected residual target.

The readout must remain low-capacity:

- linear readout
- train-only robust scaling
- chronological out-of-fold policy selection
- no MLP readout
- no validation-set target redefinition

This avoids turning V15 into a high-capacity probe that can fit noise.

## Evaluation Metrics

For `raw_residual_vector` and `residual_change_vector`, the primary metrics are:

- partial MSE gain of `fluctuation_plus_history` relative to `history_only`
- partial MAE gain of `fluctuation_plus_history` relative to `history_only`
- rolling validation window pass count
- worst-window MSE and MAE gain

For `residual_sign`, the metrics are:

- balanced accuracy gain relative to `history_only`
- Brier skill relative to `history_only`

The raw sign target is secondary. It does not replace MSE/MAE evidence on the primary residual targets.

## Counterfactual Checks

V15 must retain real-versus-counterfactual checks for the main fluctuation path. The required modes are:

- `real`
- `zero`
- `shuffled`
- `sign_flipped`
- `amplified_2x`
- `amplified_5x`
- `random_matched_norm`
- `conditional_shuffled`
- `conditional_random_matched_norm`

If `real` does not outperform the relevant shuffled or random-matched controls, the diagnostic must not be interpreted as evidence of a meaningful fluctuation signal.

## Success Gate

V15 passes the primary diagnostic only if at least one primary target satisfies all of the following conditions on rolling validation:

- `fluctuation_plus_history` has positive partial MSE gain relative to `history_only`.
- `fluctuation_plus_history` has positive partial MAE gain relative to `history_only`.
- At least four out of five rolling validation windows have positive partial MSE and MAE gains.
- No rolling validation window shows a large harmful reversal.
- The `real` fluctuation signal beats the pre-declared counterfactual controls.
- The run manifest confirms chronological split, train-only scaling, V12e label availability, and no generator training.

If these conditions fail, V15 stops and does not unlock representation retraining or generator training.

## Stop Rules

V15 must stop before any representation redesign or generator experiment when any of the following occurs:

- neither primary target passes the success gate;
- `fluctuation_plus_history` does not beat `history_only`;
- only auxiliary `level_offset` features succeed;
- counterfactual controls perform as well as or better than the real signal;
- any leakage or target redefinition is detected;
- output files would overwrite existing V12, V13, V14, or V15 artifacts.

## Output Artifacts

V15 will write only new versioned artifacts:

- `configs/stage3_v15_residual_target_predictability_probe.yaml`
- `train/run_stage3_v15_residual_target_predictability_probe.py`
- `train/run_stage3_v15_residual_target_predictability_matrix.py`
- `tests/test_stage3_v15_residual_target_predictability.py`
- `docs/audit/STAGE3_V15_RESIDUAL_TARGET_PREDICTABILITY_RECORD.md`

Run outputs must use new roots:

- `logs/s3v15target`
- `ckpt/s3v15target`
- `results/s3v15target`

## Expected Interpretation

If V15 fails, the correct conclusion is:

The current fluctuation representation does not provide stable residual-target predictability beyond causal residual history under the tested protocol.

If V15 passes on a primary target, the correct conclusion is narrower:

The current fluctuation representation contains some chronologically validated residual-target information beyond causal residual history. This may justify a later representation-training redesign or generator experiment, but it does not by itself prove a deployable Stage III adaptation mechanism.

## Spec Self-Review

- Placeholder scan: no placeholder requirements are present.
- Internal consistency: the design consistently treats V15 as a diagnostic, not a generator experiment.
- Scope check: the design is focused on one implementation plan and does not include representation retraining.
- Ambiguity check: primary targets, auxiliary controls, output paths, success gates, and stop rules are explicitly defined.
