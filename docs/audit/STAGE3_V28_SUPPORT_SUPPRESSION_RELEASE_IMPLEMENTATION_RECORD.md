# STAGE3 V28 SUPPORT-SUPPRESSION RELEASE IMPLEMENTATION RECORD

This record documents the `V28` official version line started on `2026-06-12`.
`V28` does not overwrite `V16`-`V27`.

## Why V28 Was Started

`V27` established that raw-side signal was reproducible, but its core weakness
remained:

- the change-target was still unstable across seed and dataset
- a narrow change-priority retune did not fix it

The two Chinese algorithm reports argued that environment inference should not
directly define invariant representation. It should act as a suppression or
routing condition, while the main learning target should move toward an
expert-local residual-release state.

## Changed ML Boundary

`V28` changes the ML boundary to:

`V26 correction backbone -> suppression conditioner + support conditioner -> corrected release package -> Stage III teacher/generator/online path`

## Core V28 Files

- `models/factor_mechanism_v28.py`
- `train/offline_factor_v28.py`
- `train/run_stage3_v28_support_suppression_release_probe.py`
- `train/stage3_generator_training_v28.py`
- `train/stage3_v28_artifacts.py`
- `train/online_loop_stage3_v28.py`
- `configs/stage3_v28_support_suppression_release_probe.yaml`

Tests:

- `tests/test_factor_mechanism_v28.py`
- `tests/test_offline_factor_v28.py`
- `tests/test_stage3_v28_support_suppression_release.py`
- `tests/test_stage3_generator_training_v28.py`

## Mechanism Summary

For each expert, `V28`:

1. reuses the `V26` anchor and causal env filter
2. estimates a suppressible raw-change component
3. estimates a supported delayed-change component
4. forms:

`expert_y_var = raw_change - suppression_score * suppression_vector + supported_delta`

This is more faithful to the audit recommendation than `V27`, because the
environment-conditioned branch no longer directly defines invariant output. It
instead governs:

- what should be suppressed
- what delayed signal should be supported
- what should be exposed to the served drift package

## Deterministic Validation

Completed deterministic checks:

- `python -m py_compile models\factor_mechanism_v28.py train\offline_factor_v28.py train\run_stage3_v28_support_suppression_release_probe.py train\stage3_generator_training_v28.py train\stage3_v28_artifacts.py train\online_loop_stage3_v28.py tests\test_factor_mechanism_v28.py tests\test_offline_factor_v28.py tests\test_stage3_v28_support_suppression_release.py tests\test_stage3_generator_training_v28.py`
- `pytest tests\test_factor_mechanism_v28.py tests\test_offline_factor_v28.py tests\test_stage3_v28_support_suppression_release.py tests\test_stage3_generator_training_v28.py -q`
  - `11 passed`
- `pytest tests\test_factor_mechanism_v27.py tests\test_offline_factor_v27.py tests\test_stage3_v27_quality_conditioned_dual_stream.py tests\test_stage3_generator_training_v27.py tests\test_factor_mechanism_v28.py tests\test_offline_factor_v28.py tests\test_stage3_v28_support_suppression_release.py tests\test_stage3_generator_training_v28.py -q`
  - `22 passed`

## Experiment Results So Far

### Smoke

Completed smoke run:

- `logs/s3v28expert_smoke/v28_smoke_01/weather/seed_42`

Result:

- `teacher_only_abort.json`:
  - `do_not_train_generator`
  - `no_primary_target_passed`
- all three validation partial gains were `0.0`

Interpretation:

- the full `V28` protocol path is operational
- the smoke budget is too small to establish any signal

### First Formal `weather seed42 b60` Screen

Completed first formal screen:

- `logs/s3v28expert_screen/v28_weather_s42_b60_01/weather/seed_42`

Validation partial gain summary:

- `raw_residual_vector partial_mse_gain_vs_history = -0.15099822794284545`
- `raw_residual_vector partial_mae_gain_vs_history = -0.0285762229166428`
- `residual_change_vector partial_mse_gain_vs_history = 0.0`
- `residual_sign partial_mse_gain_vs_history = 0.5301952809095383`
- `residual_sign partial_mae_gain_vs_history = 0.47581037878990173`

Gate result:

- `teacher_only_abort.json`:
  - `do_not_train_generator`
  - `no_primary_target_passed`
- `raw_residual_vector positive_partial_window_count = 3/5`
- `raw_residual_vector counterfactual_margin = -1.4917062058146413`
- `residual_change_vector positive_partial_window_count = 0/5`
- `residual_sign positive_partial_window_count = 5/5`

Interpretation:

- `V28` is not a null implementation; it produces a strong secondary `residual_sign`
  signal on the first formal weather slice
- but it fails more important targets than `V27` on this slice:
  - raw-side primary gain turns negative
  - change-target remains exactly zero
- so the first formal evidence is mixed-to-negative, not a promotion result

### Cross-Seed `weather seed43 b60` Screen

Completed cross-seed weather screen:

- `logs/s3v28expert_screen/v28_weather_s43_b60_01/weather/seed_43`

Validation partial gain summary:

- `raw_residual_vector partial_mse_gain_vs_history = 0.08864664802310351`
- `raw_residual_vector partial_mae_gain_vs_history = 0.16308540735651073`
- `residual_change_vector partial_mse_gain_vs_history = 0.0`
- `residual_sign partial_mse_gain_vs_history = 0.2184801697731018`
- `residual_sign partial_mae_gain_vs_history = 0.3544214069843292`

Gate result:

- `teacher_only_abort.json`:
  - `do_not_train_generator`
  - `no_primary_target_passed`
- `raw_residual_vector positive_partial_window_count = 3/5`
- `raw_residual_vector counterfactual_margin = -1.1718262383582514`
- `residual_change_vector positive_partial_window_count = 0/5`
- `residual_sign positive_partial_window_count = 3/5`

Interpretation:

- `V28` does recover a positive raw-side signal on `weather seed43`
- `residual_sign` remains positive, though weaker than `seed42`
- the primary change target remains exactly zero
- strict gate still rejects the run

### Public Load Transfer: `panama seed42 b60`

Completed public-load transfer screen:

- `logs/s3v28expert_screen/v28_panama_s42_b60_01/panama/seed_42`

Validation partial gain summary:

- `raw_residual_vector partial_mse_gain_vs_history = -0.0885877386872736`
- `raw_residual_vector partial_mae_gain_vs_history = -0.09805435846215761`
- `residual_change_vector partial_mse_gain_vs_history = 0.004641127665773605`
- `residual_change_vector partial_mae_gain_vs_history = -0.002969357117626279`
- `residual_sign partial_mse_gain_vs_history = 0.00022989511489868164`

Gate result:

- `teacher_only_abort.json`:
  - `do_not_train_generator`
  - `no_primary_target_passed`
- `raw_residual_vector positive_partial_window_count = 1/5`
- `residual_change_vector positive_partial_window_count = 2/5`
- `residual_change_vector min_partial_mse_gain_vs_history = -0.09811592463328736`
- `residual_change_vector min_partial_mae_gain_vs_history = -0.0623860504716188`
- `residual_change_vector counterfactual_margin = -0.0321665381759643`

Interpretation:

- transfer does not support a raw-side claim for `V28`
- the primary change target becomes slightly positive in aggregate MSE, but not
  in MAE and not across enough windows
- the counterfactual gate remains negative
- this is directionally interesting but not paper-grade evidence

### Narrow Tuning: `support_push`

The user explicitly requested that the project should stop opening new model
versions too quickly and should prioritize careful tuning before further
structure changes. Following that constraint, this pass changed only the `V28`
configuration and did not modify the model code.

Tuning config:

- `configs/stage3_v28_support_suppression_release_probe_tune_support_push.yaml`

Purpose:

- increase the usable `residual_change_vector` signal
- keep the support/suppression architecture fixed
- avoid treating a new structure as the answer before the current version is
  tuned rigorously

Main tuning changes relative to the base `V28` config:

- lower generator/probe learning rate to `0.00045`
- increase delayed-change support pressure:
  - `lambda_delayed_change = 0.44`
  - `lambda_support_alignment = 0.32`
  - `lambda_change_readout = 0.2`
- reduce over-aggressive suppression pressure:
  - `lambda_suppression_alignment = 0.1`
  - `lambda_suppression_gate_sparsity = 0.035`
- keep observable-change and budget-alignment terms nonzero but moderate:
  - `lambda_observable_change_alignment = 0.1`
  - `lambda_budget_alignment = 0.18`

Completed `panama seed42 b60` support-push screen:

- `logs/s3v28expert_tuning_screen/v28sup_panama_s42_b60_01/panama/seed_42`

Validation partial gain summary:

- `raw_residual_vector partial_mse_gain_vs_history = 0.016891664820294097`
- `raw_residual_vector partial_mae_gain_vs_history = 0.0053734945154286645`
- `residual_change_vector partial_mse_gain_vs_history = 0.0416912078672488`
- `residual_change_vector partial_mae_gain_vs_history = 0.02530271018320987`
- `residual_change_vector partial_sign_agreement_vs_history = 0.04444444552063942`
- `residual_sign partial_mse_gain_vs_history = 0.021044433116912842`

Gate result:

- `teacher_only_abort.json`:
  - `do_not_train_generator`
  - `no_primary_target_passed`
- `raw_residual_vector positive_partial_window_count = 2/5`
- `raw_residual_vector min_partial_mse_gain_vs_history = -0.17430540478113962`
- `raw_residual_vector counterfactual_margin = -0.028133440280876623`
- `residual_change_vector positive_partial_window_count = 3/5`
- `residual_change_vector min_partial_mse_gain_vs_history = -0.029099270088399836`
- `residual_change_vector min_partial_mae_gain_vs_history = -0.014788669493788856`
- `residual_change_vector counterfactual_margin = -0.04168162823048081`
- `residual_sign positive_partial_window_count = 4/5`

Interpretation:

- this is the strongest current evidence that `V28` is tunable rather than
  simply structurally wrong
- compared with base `panama seed42 b60`, `residual_change_vector` improves
  from `0.004641127665773605` to `0.0416912078672488` in partial MSE gain
- `residual_change_vector` MAE also changes from negative
  `-0.002969357117626279` to positive `0.02530271018320987`
- raw-side aggregate MSE also changes from negative
  `-0.0885877386872736` to positive `0.016891664820294097`
- however, the strict gate still rejects the run because the change target is
  positive in only `3/5` windows and the counterfactual margin remains negative

Completed `weather seed43 b60` support-push screen:

- `logs/s3v28expert_tuning_screen/v28sup_weather_s43_b60_01/weather/seed_43`

Validation partial gain summary:

- `raw_residual_vector partial_mse_gain_vs_history = -0.07366123276356079`
- `raw_residual_vector partial_mae_gain_vs_history = 0.021715380330724565`
- `raw_residual_vector partial_sign_agreement_vs_history = 0.23472219705581665`
- `residual_change_vector partial_mse_gain_vs_history = 0.0`
- `residual_change_vector partial_mae_gain_vs_history = 0.0`
- `residual_sign partial_mse_gain_vs_history = 0.2630879282951355`
- `residual_sign partial_mae_gain_vs_history = 0.28129076957702637`
- `residual_sign partial_sign_agreement_vs_history = 0.3402777910232544`

Gate result:

- `teacher_only_abort.json`:
  - `do_not_train_generator`
  - `no_primary_target_passed`
- `raw_residual_vector positive_partial_window_count = 4/5`
- `raw_residual_vector min_partial_mse_gain_vs_history = -0.509984377000866`
- `raw_residual_vector min_partial_mae_gain_vs_history = -0.4078569322059277`
- `raw_residual_vector counterfactual_margin = -0.2799629518064148`
- `residual_change_vector positive_partial_window_count = 0/5`
- `residual_sign positive_partial_window_count = 4/5`
- `residual_sign min_partial_mse_gain_vs_history = -0.167854905128479`
- `residual_sign counterfactual_margin = -0.4949782192707062`

Interpretation:

- `support_push` does not provide a robust cross-dataset improvement
- compared with base `weather seed43 b60`, raw MSE regresses from
  `0.08864664802310351` to `-0.07366123276356079`
- the secondary `residual_sign` target remains positive and slightly stronger
  than base seed43, but this is not enough to unlock the teacher gate
- `residual_change_vector` remains exactly zero on this weather slice
- therefore this tuning is useful evidence about sensitivity, but it is not a
  validated configuration

## Current Scientific Interpretation

The current evidence says:

- `V28` is a complete official version line with model, train, probe,
  generator/artifact, online wrapper, config, tests, and logs
- it is structurally closer than `V27` to the two Chinese reports because it
  separates environment-conditioned suppression from supported delayed release
- it changes target behavior in a real way:
  - `weather seed42`: strong secondary sign signal, primary raw negative,
    change zero
  - `weather seed43`: primary raw positive, sign positive, change zero
  - `panama seed42`: weak positive change MSE, raw negative
  - `support_push panama seed42`: materially stronger change MSE and positive
    change MAE, but still not gate-passing
  - `support_push weather seed43`: secondary sign remains positive, but raw MSE
    regresses and change remains zero
- no completed `V28` run passes the strict teacher gate
- generator is still not unlocked

So `V28` is not a toy implementation, but it is also not a validated replacement
for `V27`. The most defensible statement is:

> `V28` provides a cleaner support/suppression formulation of the user's idea
> and produces non-null teacher behavior on multiple formal slices, but its
> primary change-target evidence remains too weak and unstable for a paper-grade
> positive claim. A narrow support-push tuning pass shows the change target can
> be improved on `panama`, but the same setting damages weather raw stability
> and still fails the strict gate.

## Current Risks

1. `V28` is implemented and deterministic validation passed.
2. The support/suppression split is structurally stronger than `V27`, but the
   first formal `weather b60` result does not improve the primary change target.
3. Completed formal runs suggest a target shift rather than clear superiority:
   - stronger `residual_sign`
   - seed-sensitive `raw_residual_vector`
   - weak and unstable `residual_change_vector`
4. The first narrow tuning pass confirms that `residual_change_vector` is
   adjustable, but it also exposes a tradeoff between public-load change gain
   and weather raw stability.
5. Generator and online wrappers are versioned and protocol-valid, but still
   unverified in generator-unlocked conditions.

## Recommended Next Actions

1. Keep `V28` as an active official line for now.
2. Do not promote `V28` above `V27` empirically; current evidence is non-null
   but not gate-passing.
3. Prioritize rigorous `V28` tuning before any `V29` structure work.
4. The next useful tuning pass should explicitly preserve weather raw stability
   while retaining the panama change-target gain; do not optimize only one slice.
5. If another narrow `V28` tuning pass still leaves `residual_change_vector`
   weak, treat this branch as structurally cleaner but empirically secondary.
