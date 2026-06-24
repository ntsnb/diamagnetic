# Stage III V18 Structured-Env Release Implementation Record

## Scope

This record covers the first implementation pass of a new `V18` line that
extends `V17` rather than overwriting it.

Objective of `V18`:

- keep the expert-local dual-stream invariant idea;
- keep environment inference causal and online-available;
- make `phi_i` less dependent on environment-conditioned direct construction;
- separate `invariant anchor`, `environment suppression`, and `residual-release state`;
- create a new official next-generation version line without breaking `V16` or
  `V17`.

Old `V2.x`, `V16`, and `V17` files were not removed or renamed.

## Why V18 exists

The two Chinese audit reports argued that `V17` was closer to the user's idea
than older Stage III lines, but it still preserved a conceptual weakness:

- environment inference still directly participated in defining `phi_i`;
- residual-release state and uncertainty/confidence were not explicitly
  separated;
- the code still treated the expert-local variant mostly as a better-behaved
  `delta_phi_v`, rather than as a structured release state.

`V18` changes the ML system boundary to:

`expert state -> shared post representation -> invariant anchor -> environment-conditioned suppression -> phi_i -> causal residual-release state -> expert-local predictions -> alpha fusion -> Stage III teacher-only probe`

This is a stronger realization of the user's oral idea and is also more
consistent with the critique in:

- `docs/audit/2026-06-09_EXPERT_INVARIANT_DUAL_STREAM_REVIEW_CN.md`
- `docs/audit/2026-06-10_EXPERT_LOCAL_INVARIANT_DUAL_STREAM_ALGORITHM_CORRECTION_REPORT_CN.md`

## New files

Core V18 version line:

- `models/factor_mechanism_v18.py`
- `train/offline_factor_v18.py`
- `train/run_stage3_v18_structured_env_release_probe.py`
- `train/online_loop_stage3_v18.py`
- `train/stage3_generator_training_v18.py`
- `train/stage3_v18_artifacts.py`
- `configs/stage3_v18_structured_env_release_probe.yaml`
- `configs/stage3_v18_structured_env_release_probe_release_friendly.yaml`
- `configs/stage3_v18_structured_env_release_probe_balanced.yaml`

Tests:

- `tests/test_factor_mechanism_v18.py`
- `tests/test_offline_factor_v18.py`
- `tests/test_stage3_v18_structured_env_release.py`
- `tests/test_stage3_generator_training_v18.py`

## What V18 adds beyond V17

### 1. `phi_i` is built from anchor minus suppression

Each expert-local head now produces:

- `anchor_seq`
- `suppression_seq`
- `phi_i_seq = anchor_seq - suppression_seq`

The environment posterior is used as a suppression operator for
environment-explainable nuisance, rather than directly as the definition of the
invariant representation itself.

### 2. Residual-release state is explicit

Each expert-local head also maintains a recurrent causal release state driven by:

- shared expert state,
- current `phi_i`,
- environment posterior shift.

This state produces:

- `raw_release_seq`
- `delta_phi_v_seq`
- `phi_v_seq = phi_i_seq + delta_phi_v_seq`

### 3. Uncertainty/confidence is explicit

Each expert-local head now also exposes:

- `release_confidence_seq`
- pooled `expert_release_confidence`

This confidence is available to:

- the local variant readout,
- the routing context,
- training-time confidence alignment losses.

### 4. Router depends on structured context

The router now uses:

- pooled environment shift,
- pooled release confidence,
- pooled suppression norm

instead of only `z` plus a simpler environment bias.

### 5. New losses

`V18` keeps the existing representation losses but adds:

- `loss_anchor_preservation`
- `loss_suppression_sparsity`
- `loss_release_confidence`
- `loss_anchor_diversity`
- `loss_release_temporal`

These act on the new decomposition directly.

## TDD and deterministic validation

The `V18` line was added with test-first workflow.

Initial expected failure:

- `ModuleNotFoundError: No module named 'models.factor_mechanism_v18'`

After implementation:

- targeted V18 tests passed:
  - `pytest tests\test_factor_mechanism_v18.py tests\test_offline_factor_v18.py tests\test_stage3_v18_structured_env_release.py tests\test_stage3_generator_training_v18.py -q`
  - result: `13 passed`
- combined V16+V17+V18 regression passed:
  - `pytest tests\test_factor_mechanism_v17.py tests\test_offline_factor_v17.py tests\test_stage3_v17_causal_env_dual_stream.py tests\test_stage3_generator_training_v17.py tests\test_factor_mechanism_v18.py tests\test_offline_factor_v18.py tests\test_stage3_v18_structured_env_release.py tests\test_stage3_generator_training_v18.py -q`
  - result: `26 passed`
- syntax verification passed:
  - `python -m py_compile models\factor_mechanism_v18.py train\offline_factor_v18.py train\run_stage3_v18_structured_env_release_probe.py train\stage3_generator_training_v18.py train\stage3_v18_artifacts.py train\online_loop_stage3_v18.py`

## V18 smoke run

Command:

```bash
C:\Users\admin\anaconda3\python.exe train\run_stage3_v18_structured_env_release_probe.py --config configs\stage3_v18_structured_env_release_probe.yaml --device cpu --epochs 1 --max-train-batches 1 --max-generator-batches 1 --max-eval-batches 1 --validation-windows 2 --run-id v18_smoke_01 --out-root logs\s3v18expert_smoke --checkpoint-root ckpt\s3v18expert_smoke --overwrite
```

Output directory:

- `logs/s3v18expert_smoke/v18_smoke_01/weather/seed_42`

Observed artifacts include:

- `offline_metrics_v18.csv`
- `teacher_only_v18_target_gate.csv`
- `teacher_only_abort.json`
- `run_manifest_v18.json`

Stop file:

```json
{
  "stop_reasons": [
    "do_not_train_generator",
    "no_primary_target_passed"
  ],
  "protocol_version": "v18_structured_env_release_probe"
}
```

Interpretation:

- the V18 base/teacher path is operational;
- versioned output roots are correct;
- the evidence gate still protects the back-half path.

## V18 tuning-screen configs

Two additional version-safe configs were added after the current-code `V18`
weather run showed that the corrected semantics were cleaner but too weak under
the base loss mix.

### Release-friendly config

File:

- `configs/stage3_v18_structured_env_release_probe_release_friendly.yaml`

Intent:

- reduce anchor/suppression regularization pressure;
- reduce confidence regularization pressure;
- increase local residual supervision;
- see whether the structured release state can regain signal without
  re-entangling confidence and drift.

### Balanced config

File:

- `configs/stage3_v18_structured_env_release_probe_balanced.yaml`

Intent:

- keep the corrected V18 semantics;
- ease regularization only moderately;
- test whether `raw_residual_vector` and `residual_change_vector` can both stay
  non-degenerate under a more conservative compromise.

## Medium-budget formal experiments

### Weather seed42 b480, first V18 implementation pass

Important note:

- This run was produced by the **first** V18 implementation pass, where
  `delta_phi_v` was confidence-gated inside the representation itself.
- That code is not the current repo state anymore.
- This run remains part of the scientific evidence trail and is retained as a
  historical V18 result.

Command:

```bash
C:\Users\admin\anaconda3\python.exe train\run_stage3_v18_structured_env_release_probe.py --config configs\stage3_v18_structured_env_release_probe.yaml --epochs 5 --max-train-batches 480 --max-generator-batches 480 --max-eval-batches 480 --validation-windows 5 --run-id v18_weather_s42_b480_01 --out-root logs\s3v18expert_b480 --checkpoint-root ckpt\s3v18expert_b480 --overwrite
```

Output directory:

- `logs/s3v18expert_b480/v18_weather_s42_b480_01/weather/seed_42`

Primary gate file:

- `teacher_only_v18_target_gate.csv`

Key results:

- `raw_residual_vector`
  - `positive_partial_window_count = 1 / 5`
  - `min_partial_mse_gain_vs_history = -0.2752165313934257`
  - `counterfactual_margin = -0.04646438925803101`
  - gate failed
- `residual_change_vector`
  - `positive_partial_window_count = 4 / 5`
  - `min_partial_mse_gain_vs_history = -0.0009630599729170295`
  - `min_partial_mae_gain_vs_history = -0.0002712354193387456`
  - `counterfactual_margin = -0.002058955804167442`
  - gate failed

Validation partial-gain summary:

- `residual_change_vector`
  - `partial_mse_gain_vs_history = 0.002437869686027323`
  - `partial_mae_gain_vs_history = 0.0030236033828292285`

Interpretation:

- the first V18 pass preserved a weak `residual_change_vector` signal;
- but it was weaker than `V17 weather seed42 b480`;
- generator training remained correctly blocked.

### Semantic correction inside V18

After the first V18 formal run, the representation semantics were tightened:

- `release_confidence` was removed from the direct definition of
  `delta_phi_v`;
- `delta_phi_v` was returned to the explicit release state itself;
- confidence remained available only to routing, local readout input, and
  confidence-alignment losses.

Reason for this correction:

- the audit reports explicitly argued that drift state and uncertainty should
  not be collapsed into the same vector;
- keeping them entangled was mathematically and semantically weaker, even if it
  could sometimes look numerically more stable.

Current repo code corresponds to this **corrected** V18 semantics.

### Weather seed42 b480, current V18 code

Command:

```bash
C:\Users\admin\anaconda3\python.exe train\run_stage3_v18_structured_env_release_probe.py --config configs\stage3_v18_structured_env_release_probe.yaml --epochs 5 --max-train-batches 480 --max-generator-batches 480 --max-eval-batches 480 --validation-windows 5 --run-id v18_weather_s42_b480_02 --out-root logs\s3v18expert_b480 --checkpoint-root ckpt\s3v18expert_b480 --overwrite
```

Output directory:

- `logs/s3v18expert_b480/v18_weather_s42_b480_02/weather/seed_42`

Verified gate outcome:

- `teacher_only_abort.json` again reports:
  - `do_not_train_generator`
  - `no_primary_target_passed`

Primary gate file:

- `teacher_only_v18_target_gate.csv`

Key results:

- `raw_residual_vector`
  - `positive_partial_window_count = 0 / 5`
  - `min_partial_mse_gain_vs_history = -0.44045578705553023`
  - `min_partial_mae_gain_vs_history = -0.35821057924669786`
  - `counterfactual_margin = -0.023176171045770307`
  - gate failed
- `residual_change_vector`
  - `positive_partial_window_count = 0 / 5`
  - all partial-gain gate statistics effectively zero
  - gate failed

Validation partial-gain summary:

- `raw_residual_vector`
  - `partial_mse_gain_vs_history = -0.16103206194785716`
  - `partial_mae_gain_vs_history = -0.16300878827535406`
- `residual_change_vector`
  - all zeros

Interpretation:

- the semantic correction made the representation conceptually cleaner;
- but under the current loss/configuration mix, it made the teacher-only signal
  worse on `weather seed42 b480`;
- therefore the current repo state of `V18` is **more principled but less
  effective** than the first V18 pass, and clearly weaker than `V17` on this
  benchmark.

### Weather seed42 b240, release-friendly screening

Command:

```bash
C:\Users\admin\anaconda3\python.exe train\run_stage3_v18_structured_env_release_probe.py --config configs\stage3_v18_structured_env_release_probe_release_friendly.yaml --epochs 5 --max-train-batches 240 --max-generator-batches 240 --max-eval-batches 240 --validation-windows 5 --run-id v18rf_weather_s42_b240_01 --out-root logs\s3v18expert_screen --checkpoint-root ckpt\s3v18expert_screen --overwrite
```

Output directory:

- `logs/s3v18expert_screen/v18rf_weather_s42_b240_01/weather/seed_42`

Verified gate outcome:

- `teacher_only_abort.json` again reports:
  - `do_not_train_generator`
  - `no_primary_target_passed`

Primary gate file:

- `teacher_only_v18_target_gate.csv`

Key results:

- `raw_residual_vector`
  - `positive_partial_window_count = 3 / 5`
  - `min_partial_mse_gain_vs_history = -0.08816522718259813`
  - `min_partial_mae_gain_vs_history = -0.05937871648104068`
  - `counterfactual_margin = -0.007118749233544431`
  - gate failed
- `residual_change_vector`
  - all primary target gate statistics effectively zero
  - gate failed
- `residual_sign`
  - `positive_partial_window_count = 5 / 5`
  - `min_partial_mse_gain_vs_history = 0.0016841888427734375`
  - `min_partial_mae_gain_vs_history = 0.003961145877838135`
  - `counterfactual_margin = -0.03752708435058594`
  - gate failed

Validation partial-gain summary:

- `raw_residual_vector`
  - `partial_mse_gain_vs_history = 0.0006063157551076115`
  - `partial_mae_gain_vs_history = -0.0017796009897636036`
- `residual_sign`
  - `partial_mse_gain_vs_history = 0.022557854652404785`
  - `partial_mae_gain_vs_history = 0.01890397071838379`

Interpretation:

- the release-friendly sweep did not recover a viable primary target;
- it shifted signal toward `residual_sign`, which is secondary in this probe;
- therefore it was not a candidate to replace the current base V18 config.

### Weather seed42 b240, balanced screening

Command:

```bash
C:\Users\admin\anaconda3\python.exe train\run_stage3_v18_structured_env_release_probe.py --config configs\stage3_v18_structured_env_release_probe_balanced.yaml --epochs 5 --max-train-batches 240 --max-generator-batches 240 --max-eval-batches 240 --validation-windows 5 --run-id v18bal_weather_s42_b240_01 --out-root logs\s3v18expert_screen --checkpoint-root ckpt\s3v18expert_screen --overwrite
```

Output directory:

- `logs/s3v18expert_screen/v18bal_weather_s42_b240_01/weather/seed_42`

Verified gate outcome:

- `teacher_only_abort.json` again reports:
  - `do_not_train_generator`
  - `no_primary_target_passed`

Primary gate file:

- `teacher_only_v18_target_gate.csv`

Key results:

- `raw_residual_vector`
  - `positive_partial_window_count = 2 / 5`
  - `min_partial_mse_gain_vs_history = -0.01921071360350978`
  - `min_partial_mae_gain_vs_history = -0.02038728756320622`
  - `counterfactual_margin = -0.007588815800448869`
  - gate failed
- `residual_change_vector`
  - all primary target gate statistics effectively zero
  - gate failed
- `residual_sign`
  - `positive_partial_window_count = 4 / 5`
  - `min_partial_mse_gain_vs_history = -0.00716710090637207`
  - `min_partial_mae_gain_vs_history = 0.03591102361679077`
  - `counterfactual_margin = -0.1454160213470459`
  - gate failed

Validation partial-gain summary:

- `raw_residual_vector`
  - `partial_mse_gain_vs_history = 0.0070728888592978476`
  - `partial_mae_gain_vs_history = 0.00308686616356736`
- `residual_sign`
  - `partial_mse_gain_vs_history = 0.0971139669418335`
  - `partial_mae_gain_vs_history = 0.08557945489883423`

Interpretation:

- this was a better screening direction than the release-friendly config;
- it recovered a modest positive validation partial gain on
  `raw_residual_vector`;
- however, it still did not produce enough cross-window stability or
  counterfactual margin to unlock the generator path.

### Weather seed42 b480, balanced config

Command:

```bash
C:\Users\admin\anaconda3\python.exe train\run_stage3_v18_structured_env_release_probe.py --config configs\stage3_v18_structured_env_release_probe_balanced.yaml --epochs 5 --max-train-batches 480 --max-generator-batches 480 --max-eval-batches 480 --validation-windows 5 --run-id v18bal_weather_s42_b480_01 --out-root logs\s3v18expert_b480 --checkpoint-root ckpt\s3v18expert_b480 --overwrite
```

Output directory:

- `logs/s3v18expert_b480/v18bal_weather_s42_b480_01/weather/seed_42`

Verified gate outcome:

- `teacher_only_abort.json` again reports:
  - `do_not_train_generator`
  - `no_primary_target_passed`

Primary gate file:

- `teacher_only_v18_target_gate.csv`

Key results:

- `raw_residual_vector`
  - `positive_partial_window_count = 0 / 5`
  - `min_partial_mse_gain_vs_history = -0.4099081456841217`
  - `min_partial_mae_gain_vs_history = -0.3159576234045408`
  - `counterfactual_margin = -0.032774923477839474`
  - gate failed
- `residual_change_vector`
  - `positive_partial_window_count = 4 / 5`
  - `min_partial_mse_gain_vs_history = -1.7441654701635097e-05`
  - `min_partial_mae_gain_vs_history = 0.00019673292765146587`
  - `counterfactual_margin = -0.00042808255423042255`
  - gate failed

Validation partial-gain summary:

- `raw_residual_vector`
  - `partial_mse_gain_vs_history = -0.13528990320940273`
  - `partial_mae_gain_vs_history = -0.13455636720904537`
- `residual_change_vector`
  - `partial_mse_gain_vs_history = 0.0013954950827064847`
  - `partial_mae_gain_vs_history = 0.0015963487436164072`

Interpretation:

- the balanced screening direction did not survive promotion to `b480`;
- its weak `raw_residual_vector` recovery at `b240` collapsed at formal budget;
- it retained only a very weak `residual_change_vector` signal, still weaker
  than `V17 weather seed42 b480`.

## Comparative interpretation: V17 vs V18

Current defensible statement:

> V18 is a more faithful implementation of the user's stated idea and the audit
> reports' mathematical critique, because it separates invariant anchor,
> environment suppression, residual-release state, and confidence more
> explicitly than V17. However, the first paper-relevant weather benchmark shows
> that this conceptual improvement has not yet translated into stronger
> residual-target evidence. The current V18 code is not ready to replace V17 as
> the strongest empirical line.

What V18 improves structurally:

- `phi_i` is no longer directly defined by environment-conditioned construction;
- suppression and release are disentangled in the architecture;
- confidence is explicit and available for downstream control;
- routing context is richer and more expert-local.

What V18 has not yet achieved:

- stronger teacher-only evidence than `V17`;
- stable target-level gains under the strict gate;
- counterfactual specificity;
- generator unlock;
- served online improvement.

## Current limitations

1. No V18 formal run has passed the strict teacher gate.
2. The current repo state of V18 underperforms V17 on `weather seed42 b480`.
3. The cleaner semantic split between drift state and confidence did not, by
   itself, improve residual-target predictability.
4. The `release-friendly` and `balanced` sweeps did not convert the cleaner
   V18 structure into a stronger formal `b480` result.
5. Cross-seed and cross-dataset behavior remain unverified for V18.

## Recommended next actions

1. Keep V18 as the official next-generation structured line, but do **not**
   claim it is the strongest empirical successor yet.
2. Do not promote either current `release-friendly` or `balanced` settings to
   the default V18 config.
3. If V18 continues, the next sweep should be narrower and more diagnostic than
   the two configs already tried:
   - keep the corrected semantics fixed,
   - target `residual_change_vector` specifically,
   - adjust only one or two coefficients at a time,
   - inspect whether raw-target harm is being introduced by routing or by local
     release supervision.
4. V17 remains the strongest empirical line on the current weather benchmark,
   even though V18 is the more principled architecture.
5. Only after V18 regains a positive teacher-only signal should the V18
   generator/risk-gate path be considered scientifically meaningful.

## Superpowers and subagent note

The implementation followed the superpowers workflow:

- design before edits,
- TDD for the new version line,
- non-overwriting versioned paths,
- deterministic validation before experiment claims,
- experiment results written back to audit documentation.

The repository-level `AGENTS.md` also requested subagent use when possible.
This session still did not expose an executable child-agent dispatch tool, so
the work used subagent-style decomposition but not actual runnable subagent
execution.
