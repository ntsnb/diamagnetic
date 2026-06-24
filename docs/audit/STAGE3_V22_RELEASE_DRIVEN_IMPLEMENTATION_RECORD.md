# STAGE3 V22 RELEASE-DRIVEN IMPLEMENTATION RECORD

## Scope

This record documents the `V22` official version line added on `2026-06-10`.
`V22` does not overwrite `V16`-`V21`. It adds a new base-model and Stage III
probe line under a dedicated `s3v22expert` namespace.

The design target was narrow and evidence-driven:

- `V21` improved the `raw_residual_vector` side relative to `V20`, but
  `residual_change_vector` remained absent.
- The next structural hypothesis was that the model still treated release as a
  mostly static correction level, not as a dynamically accumulated release
  state.
- `V22` therefore changes the internal expert-local mechanism from
  release-first budgeting to **release-driven state accumulation**.

## ML System Boundary

The changed boundary for this version is:

`V22 base model internal expert representation -> teacher-only target gate -> generator unlock decision -> artifact compatibility`

No old version line was modified in-place.

## Core V22 Mechanism

Relative to `V21`, `V22` keeps the same high-level correction-operator framing:

- shared expert-local representation
- invariant anchor
- environment-conditioned correction
- release budget and confidence
- history branch after release

The main structural change is:

1. The release branch predicts a **budgeted release increment**.
2. The increment is integrated into an explicit **release state** over the
   expert sequence.
3. `delta_phi_v` exposed to Stage III becomes the pooled release state, not the
   one-step increment.
4. The history branch only sees the remainder after subtracting the accumulated
   release state.

This was intended to make `fluctuation_delta(delta_phi_v)` more semantically
aligned with a meaningful release-change process.

## Files Added

- `models/factor_mechanism_v22.py`
- `train/offline_factor_v22.py`
- `train/stage3_generator_training_v22.py`
- `train/stage3_v22_artifacts.py`
- `train/online_loop_stage3_v22.py`
- `train/run_stage3_v22_release_driven_probe.py`
- `configs/stage3_v22_release_driven_probe.yaml`
- `tests/test_factor_mechanism_v22.py`
- `tests/test_offline_factor_v22.py`
- `tests/test_stage3_v22_release_driven.py`
- `tests/test_stage3_generator_training_v22.py`
- `docs/superpowers/specs/2026-06-10-stage3-v22-release-driven-design.md`
- `docs/superpowers/plans/2026-06-10-stage3-v22-release-driven.md`

## Deterministic Validation

Completed checks:

- `python -m py_compile models\factor_mechanism_v22.py train\offline_factor_v22.py train\stage3_generator_training_v22.py train\stage3_v22_artifacts.py train\online_loop_stage3_v22.py train\run_stage3_v22_release_driven_probe.py tests\test_factor_mechanism_v22.py tests\test_offline_factor_v22.py tests\test_stage3_v22_release_driven.py tests\test_stage3_generator_training_v22.py`
- `pytest tests\test_factor_mechanism_v22.py tests\test_offline_factor_v22.py tests\test_stage3_v22_release_driven.py tests\test_stage3_generator_training_v22.py -q`
  - result: `11 passed`
- `pytest tests\test_factor_mechanism_v21.py tests\test_offline_factor_v21.py tests\test_stage3_v21_release_first_budgeted.py tests\test_stage3_generator_training_v21.py tests\test_factor_mechanism_v22.py tests\test_offline_factor_v22.py tests\test_stage3_v22_release_driven.py tests\test_stage3_generator_training_v22.py -q`
  - result: `22 passed`

## Smoke Run

Smoke command completed:

- run id: `v22_smoke_01`
- output:
  [v22_smoke_01](/C:/Users/admin/Desktop/最近/1.28/IEEE-Transactions-LaTeX2e-templates-and-instructions/logs/s3v22expert_smoke/v22_smoke_01/weather/seed_42)

Smoke still stopped at:

- `do_not_train_generator`
- `no_primary_target_passed`

This is acceptable for a smoke; the purpose was execution-path validation.

## Formal Screen: Weather seed42 b240

Formal screen output:

- run id: `v22_weather_s42_b240_01`
- directory:
  [v22_weather_s42_b240_01](/C:/Users/admin/Desktop/最近/1.28/IEEE-Transactions-LaTeX2e-templates-and-instructions/logs/s3v22expert_screen/v22_weather_s42_b240_01/weather/seed_42)

Key manifest:

- `v22_model_native_drift_source = pooled_delta_phi_v`

Teacher gate:

- `raw_residual_vector`
  - `positive_partial_window_count = 1 / 5`
  - `min_partial_mse_gain_vs_history = -0.25135754879636873`
  - `min_partial_mae_gain_vs_history = -0.07209216461037526`
  - `counterfactual_margin = -0.20234545087930134`
  - gate: fail
- `residual_change_vector`
  - `positive_partial_window_count = 0 / 5`
  - `min_partial_mse_gain_vs_history = 0.0`
  - `min_partial_mae_gain_vs_history = 0.0`
  - gate: fail
- `residual_sign`
  - gate: fail

Validation partial gain vs history:

- `raw_residual_vector`
  - `partial_mse_gain_vs_history = -0.03921956662575807`
  - `partial_mae_gain_vs_history = 0.012320914275385508`
- `residual_change_vector`
  - `partial_mse_gain_vs_history = 0.0`
  - `partial_mae_gain_vs_history = 0.0`
- `residual_sign`
  - all zero

Abort:

- `do_not_train_generator`
- `no_primary_target_passed`

## Comparison Against V21 and V20

Reference points:

- `V20 base weather seed42 b240`
  - `raw partial_mse_gain_vs_history = -0.01972691923212508`
  - `counterfactual_margin = -0.09156975486390212`
- `V20 release-priority weather seed42 b240`
  - `raw partial_mse_gain_vs_history = -0.006590608917462576`
  - `counterfactual_margin = -0.09155519219640097`
- `V21 weather seed42 b240`
  - `raw partial_mse_gain_vs_history = -0.00536056546860289`
  - `counterfactual_margin = -0.0492276391028037`
  - `residual_change_vector` still absent
- `V22 weather seed42 b240`
  - `raw partial_mse_gain_vs_history = -0.03921956662575807`
  - `counterfactual_margin = -0.20234545087930134`
  - `residual_change_vector` still absent

Interpretation:

- `V22` is not an empty code change; the structural semantics are genuinely
  different.
- But empirically, the first `b240` result is worse than `V21`.
- The attempted increment-to-state redesign did **not** recover
  `residual_change_vector`.
- It also degraded the raw-target attribution pattern under strict gate.

## Current Interpretation

The strongest current reading is:

1. The `V20 -> V21` move improved the correction-operator family by reducing raw
   harm and counterfactual penalty.
2. The `V21 -> V22` move shows that simply making release explicitly
   state-driven is not sufficient.
3. The bottleneck is likely no longer only “what is the release state?”, but
   “how the training objective makes that state observable to the teacher-only
   residual targets”.

In other words:

> `V22` made the internal state more dynamic, but the current readout/training
> combination did not convert that added structure into a useful Stage III
> signal.

## Current Limitations

1. Only one meaningful `V22` formal screen has been run so far.
2. `residual_change_vector` remains completely absent.
3. `raw_residual_vector` became worse than `V21`.
4. Generator training is still blocked, so no paper-grade online-serving result
   can be claimed for `V22`.

## Recommended Next Actions

1. Do not promote `V22` as the new best empirical line.
2. Keep `V21` as the strongest current correction-operator evidence line.
3. If continuing, the next version should target **observability/readout**, not
   just internal release dynamics. Good candidates include:
   - a dedicated residual-change readout head
   - a teacher-facing delayed-change auxiliary target
   - router features that explicitly expose release observability, not only
     release budget

## Tooling Note

This turn continued to use the superpowers workflow style: design isolation,
TDD-like validation, smoke before formal screen, and evidence-first writeback.

However, the current tool surface still does not expose a real child-agent
dispatch capability, so no actual executable subagent was spawned. That should
not be misreported.
