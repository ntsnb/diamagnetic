# STAGE3 V23 OBSERVABLE DRIFT IMPLEMENTATION RECORD

## Scope

This record documents the `V23` official version line added on `2026-06-11`.
`V23` does not overwrite `V16`-`V22`. It adds a new base-model and Stage III
probe line under a dedicated `s3v23expert` namespace.

The reason for `V23` was the `V22` negative result:

- `V22` changed the internal release dynamics, but the teacher-only evidence
  still did not become usable.
- This suggested the next bottleneck was not only internal state construction,
  but **teacher-facing observability**.

`V23` therefore introduces a serving-aligned `observable drift` signal and
propagates it through the model, teacher probe, generator wrapper, and online
wrapper.

## ML System Boundary

The changed ML boundary for this version is:

`V23 base model internal state -> teacher-facing observable drift -> Stage III teacher gate -> generator/risk-gate/online wrappers`

Unlike `V22`, this version explicitly tries to reduce training-serving mismatch
at the drift-interface level.

## Core V23 Mechanism

`V23` keeps the `V22` internal release-state construction:

- shared expert-local representation
- invariant anchor
- environment-conditioned correction
- release increment
- accumulated release state
- history branch after release state

The new element is a dedicated **observable drift** channel:

- geometric state:
  - `pooled_delta_phi_v`
  - still serves as the internal release-state representation
- teacher/serve drift source:
  - `pooled_observable_drift`
  - defined as a concatenation of:
    - release state
    - current budgeted release increment
    - release confidence

So `V23` explicitly separates:

- the geometric semantics of `phi_i / phi_v`
- the operational semantics of “what Stage III should actually read”

This is closer to the criticism in the Chinese audit reports:

> do not force all downstream semantics through a single `phi_v - phi_i`
> quantity when the model actually needs a richer, more observable drift signal

## Files Added

- `models/factor_mechanism_v23.py`
- `train/offline_factor_v23.py`
- `train/stage3_generator_training_v23.py`
- `train/stage3_v23_artifacts.py`
- `train/online_loop_stage3_v23.py`
- `train/run_stage3_v23_observable_drift_probe.py`
- `configs/stage3_v23_observable_drift_probe.yaml`
- `tests/test_factor_mechanism_v23.py`
- `tests/test_offline_factor_v23.py`
- `tests/test_stage3_v23_observable_drift.py`
- `tests/test_stage3_generator_training_v23.py`
- `docs/superpowers/specs/2026-06-10-stage3-v23-observable-drift-design.md`
- `docs/superpowers/plans/2026-06-10-stage3-v23-observable-drift.md`

## Deterministic Validation

Completed checks:

- `python -m py_compile models\factor_mechanism_v23.py train\offline_factor_v23.py train\stage3_generator_training_v23.py train\stage3_v23_artifacts.py train\online_loop_stage3_v23.py train\run_stage3_v23_observable_drift_probe.py`
- `pytest tests\test_factor_mechanism_v23.py tests\test_offline_factor_v23.py tests\test_stage3_v23_observable_drift.py tests\test_stage3_generator_training_v23.py -q`
  - result: `11 passed`
- `pytest tests\test_factor_mechanism_v22.py tests\test_offline_factor_v22.py tests\test_stage3_v22_release_driven.py tests\test_stage3_generator_training_v22.py tests\test_factor_mechanism_v23.py tests\test_offline_factor_v23.py tests\test_stage3_v23_observable_drift.py tests\test_stage3_generator_training_v23.py -q`
  - result: `22 passed`

## Smoke Run

Smoke command completed:

- run id: `v23_smoke_01`
- output:
  [v23_smoke_01](/C:/Users/admin/Desktop/最近/1.28/IEEE-Transactions-LaTeX2e-templates-and-instructions/logs/s3v23expert_smoke/v23_smoke_01/weather/seed_42)

Smoke still stopped at:

- `do_not_train_generator`
- `no_primary_target_passed`

That is acceptable for smoke; the purpose was protocol validation.

## Formal Screen 1: Weather seed42 b240

Formal screen output:

- run id: `v23_weather_s42_b240_01`
- directory:
  [v23_weather_s42_b240_01](/C:/Users/admin/Desktop/最近/1.28/IEEE-Transactions-LaTeX2e-templates-and-instructions/logs/s3v23expert_screen/v23_weather_s42_b240_01/weather/seed_42)

Key manifest:

- `v23_model_native_drift_source = pooled_observable_drift`

Teacher gate:

- `raw_residual_vector`
  - `positive_partial_window_count = 2 / 5`
  - `min_partial_mse_gain_vs_history = -0.14469283260833854`
  - `min_partial_mae_gain_vs_history = -0.10905875722050733`
  - `counterfactual_margin = -0.07100568366234472`
  - gate: fail
- `residual_change_vector`
  - `positive_partial_window_count = 3 / 5`
  - `min_partial_mse_gain_vs_history = -0.004449183927011115`
  - `min_partial_mae_gain_vs_history = -0.003771134644297128`
  - `counterfactual_margin = -0.004449183927011115`
  - gate: fail
- `residual_sign`
  - gate: fail

Validation partial gain vs history:

- `raw_residual_vector`
  - `partial_mse_gain_vs_history = -0.022176874611375547`
  - `partial_mae_gain_vs_history = 0.00961867832583091`
- `residual_change_vector`
  - `partial_mse_gain_vs_history = 0.0004171525744068808`
  - `partial_mae_gain_vs_history = -0.0002831498884708877`
  - sign agreement delta: `+0.04236111044883728`

Interpretation of seed42:

- `V23` did not pass the gate.
- But unlike `V22`, `residual_change_vector` is no longer identically zero.
- This is the first correction-operator line after `V17` that recovers a weak
  non-zero `residual_change_vector` signal on `weather seed42`.

## Formal Screen 2: Weather seed43 b240

Formal screen output:

- run id: `v23_weather_s43_b240_01`
- directory:
  [v23_weather_s43_b240_01](/C:/Users/admin/Desktop/最近/1.28/IEEE-Transactions-LaTeX2e-templates-and-instructions/logs/s3v23expert_screen/v23_weather_s43_b240_01/weather/seed_43)

Teacher gate:

- `raw_residual_vector`
  - `positive_partial_window_count = 1 / 5`
  - `min_partial_mse_gain_vs_history = -0.13737434667582848`
  - `min_partial_mae_gain_vs_history = -0.10155957410200772`
  - `counterfactual_margin = -0.1464207183358452`
  - gate: fail
- `residual_change_vector`
  - `positive_partial_window_count = 0 / 5`
  - `min_partial_mse_gain_vs_history = -0.13319367092047277`
  - `min_partial_mae_gain_vs_history = -0.07554401097545978`
  - gate: fail

Validation partial gain vs history:

- `raw_residual_vector`
  - `partial_mse_gain_vs_history = -0.025438677413398586`
  - `partial_mae_gain_vs_history = -0.02205583454521587`
- `residual_change_vector`
  - `partial_mse_gain_vs_history = -0.05641983706372909`
  - `partial_mae_gain_vs_history = -0.045432580005648954`

Interpretation of seed43:

- The weak `seed42` `residual_change_vector` recovery is not yet stable.
- Under a second seed, the same target falls back to clearly negative.

## Comparison Against V22, V21, and V17

Key reference points:

- `V22 weather seed42 b240`
  - `raw partial_mse_gain_vs_history = -0.03921956662575807`
  - `residual_change_vector = 0.0`
  - `counterfactual_margin(raw) = -0.20234545087930134`
- `V21 weather seed42 b240`
  - `raw partial_mse_gain_vs_history = -0.00536056546860289`
  - `residual_change_vector = 0.0`
  - `counterfactual_margin(raw) = -0.0492276391028037`
- `V17 weather seed42 b480`
  - stronger `residual_change_vector` evidence than `V20+`
  - still not sufficient to pass the final gate, but remains the empirical
    reference line on `weather`

`V23` relative position:

- better than `V22` on `residual_change_vector` observability in `seed42`
- worse than `V21` on raw-target stability
- still weaker and less stable than the best `V17` weather behavior

## Current Interpretation

The strongest current reading is:

1. The `V22` negative result was useful because it isolated the bottleneck away
   from internal release-state construction alone.
2. `V23` confirms that teacher-facing observability matters.
3. But the first `seed42` positive hint does not survive a second seed.

So the precise conclusion is:

> `V23` is the first `V20+` line that revives a weak non-zero
> `residual_change_vector` signal on `weather seed42`, which supports the
> observable-drift hypothesis. However, that gain is not yet seed-stable, so
> `V23` cannot be promoted as a validated new mainline.

## Current Limitations

1. Generator training is still blocked in both `V23` formal runs.
2. `V23` has only been screened on `weather` so far.
3. The `residual_change_vector` recovery appears in `seed42` but not `seed43`.
4. No paper-grade positive claim is justified yet.

## Recommended Next Actions

1. Keep `V23` as the current best observability-oriented correction-operator
   line.
2. Do not replace `V17` as the empirical reference line yet.
3. If continuing, the next version should focus on stabilizing the observable
   signal across seeds, likely via:
   - a dedicated residual-change readout head
   - a delayed change auxiliary target
   - routing features that explicitly see observable-drift quality, not just
     magnitude

## Tooling Note

This turn continued to use the superpowers workflow style: version isolation,
deterministic validation, smoke before formal screens, and evidence-first
writeback.

The current tool surface still does not expose a real child-agent dispatch
capability, so no actual executable subagent was spawned. That should not be
misreported.
