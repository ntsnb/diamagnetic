# STAGE3 V25 CHANGE QUALITY IMPLEMENTATION RECORD

## Scope

This record documents the `V25` official version line added on `2026-06-11`.
`V25` does not overwrite `V16`-`V24`. It adds a new base-model and Stage III
probe line under a dedicated `s3v25expert` namespace.

`V25` was created after the `V23` and `V24` evidence converged on a narrower
diagnosis:

- `V23 weather seed42 b240` recovered a weak non-zero
  `residual_change_vector` signal.
- `V24 weather seed42 b240` lost that signal again.

The working conclusion was that the next bottleneck was no longer observable
aggregation alone. The more likely failure mode was:

- no dedicated change-target readout
- no delayed-change supervision
- router visibility of drift magnitude but not drift quality

`V25` therefore tries to turn the user's newer idea into a more operational
version line:

- expert-local invariant / correction backbone remains
- but `residual_change` gets an explicit readout path
- that readout is quality-gated
- and the served drift source becomes quality-aware

## ML System Boundary

The changed ML boundary for this version is:

`V25 expert-local correction backbone -> explicit change readout -> change-quality gate -> quality-aware served drift -> Stage III teacher gate -> generator/risk-gate/online wrappers`

This is a training-serving consistency change centered on:

- residual-change observability
- target-aligned readout
- quality-aware routing

## Core V25 Mechanism

`V25` keeps the `V24` expert-local backbone:

- shared expert-local representation
- invariant anchor
- environment-conditioned correction
- release increment
- accumulated release state
- history branch after release state

The new structural changes are:

1. The variant head becomes an explicit **change readout** head:
   - it now consumes
     - release state
     - budgeted release
     - observable change
     - budgeted history
     - release confidence
     - release budget
     - environment shift
2. A new per-expert **change quality head** produces
   - `expert_change_quality_score in [0, 1]`
3. The final expert change contribution becomes
   - `expert_y_var = expert_change_quality_score * expert_y_change_raw`
4. A served drift package is built explicitly as
   - base change-observable package from the correction backbone
   - quality-weighted observable change
   - change quality score
   - raw change readout
   - delayed change readout
5. Router context is expanded so each expert router view now includes:
   - environment shift
   - env-correction norm
   - release budget
   - observable-change norm
   - change-quality score
   - change-delta norm

The authoritative model-native drift source for `V25` is:

- `pooled_servable_change`

This is the signal consumed by:

- split-tensor collection
- teacher-only gate
- generator wrapper
- online adaptation wrapper

## File Boundary

Files added for `V25`:

- `models/factor_mechanism_v25.py`
- `train/offline_factor_v25.py`
- `train/stage3_generator_training_v25.py`
- `train/stage3_v25_artifacts.py`
- `train/online_loop_stage3_v25.py`
- `train/run_stage3_v25_change_quality_probe.py`
- `configs/stage3_v25_change_quality_probe.yaml`
- `tests/test_factor_mechanism_v25.py`
- `tests/test_offline_factor_v25.py`
- `tests/test_stage3_v25_change_quality.py`
- `tests/test_stage3_generator_training_v25.py`
- `docs/superpowers/specs/2026-06-11-stage3-v25-change-quality-design.md`
- `docs/superpowers/plans/2026-06-11-stage3-v25-change-quality.md`

Key implementation locations:

- explicit change-quality-aware served drift construction
  - [factor_mechanism_v25.py](/C:/Users/admin/Desktop/最近/1.28/IEEE-Transactions-LaTeX2e-templates-and-instructions/models/factor_mechanism_v25.py:521)
- router quality features
  - [factor_mechanism_v25.py](/C:/Users/admin/Desktop/最近/1.28/IEEE-Transactions-LaTeX2e-templates-and-instructions/models/factor_mechanism_v25.py:557)
- delayed-change / change-readout / quality-alignment losses
  - [offline_factor_v25.py](/C:/Users/admin/Desktop/最近/1.28/IEEE-Transactions-LaTeX2e-templates-and-instructions/train/offline_factor_v25.py:160)
- V25 model-native drift source collection
  - [run_stage3_v25_change_quality_probe.py](/C:/Users/admin/Desktop/最近/1.28/IEEE-Transactions-LaTeX2e-templates-and-instructions/train/run_stage3_v25_change_quality_probe.py:201)

## Deterministic Validation

Completed checks:

- `python -m py_compile models\factor_mechanism_v25.py train\offline_factor_v25.py train\stage3_generator_training_v25.py train\stage3_v25_artifacts.py train\online_loop_stage3_v25.py train\run_stage3_v25_change_quality_probe.py tests\test_factor_mechanism_v25.py tests\test_offline_factor_v25.py tests\test_stage3_v25_change_quality.py tests\test_stage3_generator_training_v25.py`
- `pytest tests\test_factor_mechanism_v25.py tests\test_offline_factor_v25.py tests\test_stage3_v25_change_quality.py tests\test_stage3_generator_training_v25.py -q`
  - result: `11 passed`
- `pytest tests\test_factor_mechanism_v24.py tests\test_offline_factor_v24.py tests\test_stage3_v24_tail_observable_change.py tests\test_stage3_generator_training_v24.py tests\test_factor_mechanism_v25.py tests\test_offline_factor_v25.py tests\test_stage3_v25_change_quality.py tests\test_stage3_generator_training_v25.py -q`
  - result: `22 passed`

These checks verify:

- V25 schema consistency
- change-quality head registration
- generator drift-dimension compatibility
- no immediate regression against the `V24` deterministic boundary

## Smoke Run

Smoke command completed:

- run id: `v25_smoke_01`
- output:
  [v25_smoke_01](/C:/Users/admin/Desktop/最近/1.28/IEEE-Transactions-LaTeX2e-templates-and-instructions/logs/s3v25expert_smoke/v25_smoke_01/weather/seed_42)

The smoke run completed the protocol and produced the expected output path.

## Formal Screen 1: Weather seed42 b240

Formal screen output:

- run id: `v25_weather_s42_b240_01`
- directory:
  [v25_weather_s42_b240_01](/C:/Users/admin/Desktop/最近/1.28/IEEE-Transactions-LaTeX2e-templates-and-instructions/logs/s3v25expert_screen/v25_weather_s42_b240_01/weather/seed_42)

Key manifest facts:

- `protocol_version = v25_change_quality_probe`
- `v25_model_native_drift_source = pooled_servable_change`

Teacher gate summary:

- `raw_residual_vector`
  - `positive_partial_window_count = 1 / 5`
  - `min_partial_mse_gain_vs_history = -0.3738528598242493`
  - `min_partial_mae_gain_vs_history = -0.46000044410984053`
  - `counterfactual_margin = -0.05060432856565171`
  - gate: fail
- `residual_change_vector`
  - `positive_partial_window_count = 0 / 5`
  - `min_partial_mse_gain_vs_history = -0.14410186851300474`
  - `min_partial_mae_gain_vs_history = -0.12279341770571621`
  - `counterfactual_margin = 0.0`
  - gate: fail
- `residual_sign`
  - gate: fail

Validation partial gain vs history:

- `raw_residual_vector`
  - `history_only_real_mse_gain = 0.21330607304532806`
  - `fluctuation_plus_history_real_mse_gain = 0.1449236457482711`
  - `partial_mse_gain_vs_history = -0.06838242729705696`
  - `history_only_real_mae_gain = 0.30048344966359064`
  - `fluctuation_plus_history_real_mae_gain = 0.15612629288581817`
  - `partial_mae_gain_vs_history = -0.14435715677777247`
- `residual_change_vector`
  - `history_only_real_mse_gain = 0.08948986500984651`
  - `fluctuation_plus_history_real_mse_gain = 0.0`
  - `partial_mse_gain_vs_history = -0.08948986500984651`
  - `history_only_real_mae_gain = 0.06625071421531976`
  - `fluctuation_plus_history_real_mae_gain = 0.0`
  - `partial_mae_gain_vs_history = -0.06625071421531976`

Abort file:

- `do_not_train_generator`
- `no_primary_target_passed`

## Interpretation of the Formal Screen

`V25` is a clear negative result.

This is stronger than the `V24` negative result because `V25` was not merely
neutral or weak. It actively regressed the key target relative to both
`history_only` and the earlier `V23 seed42` hint.

The intended thesis of `V25` was:

- explicit change readout
- delayed-change supervision
- quality-aware drift packaging

The actual result was the opposite of the intended direction:

- `raw_residual_vector` degraded materially
- `residual_change_vector` was not stabilized and instead became clearly
  negative

So `V25` does not support the hypothesis that stronger target-aligned readout
and quality gating are sufficient on top of the `V24` backbone.

## Comparison Against V23 and V24

Reference points on `weather seed42 b240`:

- `V23`
  - `raw partial_mse_gain_vs_history = -0.022176874611375547`
  - `residual_change_vector partial_mse_gain_vs_history = 0.0004171525744068808`
  - `raw counterfactual_margin = -0.07100568366234472`
- `V24`
  - `raw partial_mse_gain_vs_history = -0.022873772360176736`
  - `residual_change_vector partial_mse_gain_vs_history = 0.0`
  - `raw counterfactual_margin = -0.04975623733051171`
- `V25`
  - `raw partial_mse_gain_vs_history = -0.06838242729705696`
  - `residual_change_vector partial_mse_gain_vs_history = -0.08948986500984651`
  - `raw counterfactual_margin = -0.05060432856565171`

Precise ranking on this slice:

1. `V23` remains the strongest observability-oriented line inside the `V20+`
   family.
2. `V24` is weaker than `V23`, but still less harmful than `V25`.
3. `V25` is the worst of the three on `residual_change_vector`, which was the
   very target it was designed to help.

## Current Interpretation

The strongest defensible reading is:

1. The `V23` hint was real enough to motivate targeted follow-up.
2. But simply adding
   - explicit change readout
   - delayed-change auxiliary loss
   - quality-aware gating
   on top of the `V24` backbone is not sufficient.
3. The likely remaining bottleneck is more structural:
   - the local correction state may still not carry the right causal signal
   - the change target may not be identifiable from the present expert-local
     state under the current training decomposition
   - quality gating may be suppressing signal rather than separating useful
     change from noise

So the precise conclusion is:

> `V25` falsifies the simpler next hypothesis after `V24`: stronger
> residual-change readout and quality-aware gating alone do not rescue the
> expert-local correction line into a useful Stage III signal on
> `weather seed42 b240`.

## Why No Second Seed Was Run

A second seed was not run for `V25` in this turn because the first formal screen
already gave a clear negative answer:

- it did not weakly improve and then need stability confirmation
- it regressed the primary target directly

Given that evidence, spending a second seed would not change the current
version ordering.

## Current Limitations

1. `V25` has only been screened on `weather seed42 b240`.
2. Generator training remains blocked.
3. No paper-grade positive claim is justified.
4. The full user objective remains incomplete because no new official line has
   yet surpassed the empirical reference behavior strongly enough to support a
   manuscript-grade explanation.

## Recommended Next Actions

1. Do not promote `V25` as the new mainline.
2. Keep `V23` as the strongest `V20+` observability reference.
3. If continuing to `V26`, do not keep stacking readout losses on the same
   correction state. Move to a more structural change, such as:
   - explicit multi-timescale change state
   - separate causal change memory from instantaneous release proposal
   - or reframe the expert-local drift state around identifiable delayed error
     release rather than direct target-space readout

## Tooling Note

This work continued to use the superpowers workflow style: isolated version
lines, deterministic validation, smoke before formal screen, and evidence-first
writeback.

The current tool surface still does not expose a real child-agent dispatch
capability, so no actual executable subagent was spawned. That should not be
misreported.
