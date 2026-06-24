# STAGE3 V27 QUALITY-CONDITIONED DUAL-STREAM IMPLEMENTATION RECORD

This record documents the `V27` official version line added on `2026-06-11`.
`V27` does not overwrite `V16`-`V26`. It adds a new base-model wrapper,
training path, probe path, generator/artifact/online wrappers, config, tests,
and supporting design documents.

## Why V27 Was Started

The new version was started after three evidence-backed constraints became
clear:

1. `V17` still remained the strongest empirical weather reference.
2. `V23` remained the best `V20+` observability-oriented reference.
3. `V26` tuning showed local movement but also strong saturation:
   - raw-side gains were possible
   - primary change-target evidence remained unstable or null

So `V27` was not started as another broad tuning family. It was started as a
new structural version line that still remains faithful to the user's original
idea and the two Chinese audit reports.

## Changed ML Boundary

The changed ML boundary for `V27` is:

`V26 expert-local correction backbone -> expert quality adapter -> quality-conditioned variant aggregation -> quality-aware router bias -> served drift package -> Stage III teacher/generator/online path`

This is different from `V26` in one crucial sense:

- `V26` asked the backbone itself to provide a directly useful multiscale score
- `V27` asks a separate quality-conditioning layer to decide when delayed
  change should be trusted and served

## Core V27 Files

- `models/factor_mechanism_v27.py`
- `train/offline_factor_v27.py`
- `train/run_stage3_v27_quality_conditioned_dual_stream_probe.py`
- `train/stage3_generator_training_v27.py`
- `train/stage3_v27_artifacts.py`
- `train/online_loop_stage3_v27.py`
- `configs/stage3_v27_quality_conditioned_dual_stream_probe.yaml`
- `configs/stage3_v27_quality_conditioned_dual_stream_probe_tune_change_priority.yaml`

Tests:

- `tests/test_factor_mechanism_v27.py`
- `tests/test_offline_factor_v27.py`
- `tests/test_stage3_v27_quality_conditioned_dual_stream.py`
- `tests/test_stage3_generator_training_v27.py`

Design documents:

- `docs/superpowers/specs/2026-06-11-stage3-v27-quality-conditioned-dual-stream-design.md`
- `docs/superpowers/plans/2026-06-11-stage3-v27-quality-conditioned-dual-stream.md`

## Mechanism Summary

`V27` wraps the verified `V26` backbone and adds:

- expert-local `quality_state`
- expert-local `quality_score`
- quality-conditioned variant fusion:
  - `expert_y_var = expert_y_change_raw + quality_score * expert_change_delta`
- a second router bias driven by quality-aware features
- a new `pooled_servable_change` package for Stage III

This keeps the user's preferred late-correction structure:

- most layers remain shared in the expert branch
- environment inference still acts as a correction operator
- invariant and variant components remain easier to align than in the older
  fully decoupled paths

## Deterministic Validation

Completed deterministic checks:

- `python -m py_compile models\factor_mechanism_v27.py train\offline_factor_v27.py train\run_stage3_v27_quality_conditioned_dual_stream_probe.py train\stage3_generator_training_v27.py train\stage3_v27_artifacts.py train\online_loop_stage3_v27.py`
- `pytest tests\test_factor_mechanism_v27.py tests\test_offline_factor_v27.py tests\test_stage3_v27_quality_conditioned_dual_stream.py tests\test_stage3_generator_training_v27.py -q`
  - `11 passed`
- `pytest tests\test_factor_mechanism_v26.py tests\test_offline_factor_v26.py tests\test_stage3_v26_multiscale_change.py tests\test_stage3_generator_training_v26.py tests\test_factor_mechanism_v27.py tests\test_offline_factor_v27.py tests\test_stage3_v27_quality_conditioned_dual_stream.py tests\test_stage3_generator_training_v27.py -q`
  - `22 passed`

## Experiment Results So Far

### Smoke

Smoke run:

- `logs/s3v27expert_smoke/v27_smoke_01/weather/seed_42`

Result:

- protocol completed
- output files were generated
- teacher path stopped at:
  - `do_not_train_generator`
  - `no_primary_target_passed`

Validation partial gain summary:

- all three targets were `0.0`

Interpretation:

- the code path is operational
- but the smoke budget is too small to claim any signal

### First Completed Initial Screen

Completed initial screen:

- `logs/s3v27expert_screen/v27_weather_s42_b20_01/weather/seed_42`

Result:

- `teacher_only_abort.json`:
  - `do_not_train_generator`
  - `no_primary_target_passed`
- `teacher_only_validation_partial_gain_vs_history_v27.csv`:
  - `raw_residual_vector partial_mse_gain_vs_history = 0.0`
  - `residual_change_vector partial_mse_gain_vs_history = 0.0`
  - `residual_sign partial_mse_gain_vs_history = 0.0`

Interpretation:

- `V27` is not a broken implementation
- but the first completed screen is still a null result
- at the currently completed budget, `V27` has not yet shown non-zero teacher
  evidence

### Completed Formal `weather b60` Screens

Two matched-budget `weather b60` formal screens were completed:

- `logs/s3v27expert_screen/v27_weather_s42_b60_02/weather/seed_42`
- `logs/s3v27expert_screen/v27_weather_s43_b60_01/weather/seed_43`

#### `seed42`

- `raw_residual_vector partial_mse_gain_vs_history = 0.19515517310349917`
- `raw_residual_vector partial_mae_gain_vs_history = 0.14359122589171935`
- `residual_change_vector partial_mse_gain_vs_history = -0.04661352437424025`
- `residual_sign partial_mse_gain_vs_history = 0.2657555863261223`
- gate result:
  - `do_not_train_generator`
  - `no_primary_target_passed`
  - `raw_residual_vector positive_partial_window_count = 2/5`
  - `raw_residual_vector counterfactual_margin = -0.07107470796418469`

#### `seed43`

- `raw_residual_vector partial_mse_gain_vs_history = 0.11035856277349587`
- `raw_residual_vector partial_mae_gain_vs_history = 0.08019061966267282`
- `residual_change_vector partial_mse_gain_vs_history = -0.04450259745524796`
- `residual_sign partial_mse_gain_vs_history = 0.1661827564239502`
- gate result:
  - `do_not_train_generator`
  - `no_primary_target_passed`
  - `raw_residual_vector positive_partial_window_count = 2/5`
  - `raw_residual_vector counterfactual_margin = -0.1369925879159959`

Interpretation:

- `V27` is no longer a null line once the budget is large enough
- the positive raw-side signal is not a single-seed accident; it appears on both
  `seed42` and `seed43`
- `residual_change_vector` still remains negative on both seeds
- the strict gate still rejects the version because:
  - not enough validation windows are positive
  - harmful windows still exceed the tolerance
  - counterfactual attribution remains negative

### Public Load Transfer: `panama seed42 b60`

Completed transfer screen:

- `logs/s3v27expert_screen/v27_panama_s42_b60_01/panama/seed_42`

Result:

- `raw_residual_vector partial_mse_gain_vs_history = -0.030597802643083878`
- `residual_change_vector partial_mse_gain_vs_history = 0.05138822577169395`
- `residual_sign partial_mse_gain_vs_history = 0.046341776847839355`
- gate result:
  - `do_not_train_generator`
  - `no_primary_target_passed`
  - `residual_change_vector positive_partial_window_count = 4/5`
  - `residual_change_vector min_partial_mse_gain_vs_history = 0.014263601343710382`
  - `residual_change_vector min_partial_mae_gain_vs_history = -0.003580153889150358`
  - `residual_change_vector counterfactual_margin = -0.042183278763293655`

Interpretation:

- this is the first verified `V27` run where the strongest primary target
  becomes directionally positive
- but the gain is not yet gate-safe:
  - MAE still has a slightly harmful window
  - counterfactual attribution remains negative
- `V27` therefore still cannot unlock generator training, but it is no longer
  accurate to describe it as lacking change-target signal everywhere

### Public Load Transfer: `panama seed43 b60`

Completed transfer confirmation:

- `logs/s3v27expert_screen/v27_panama_s43_b60_01/panama/seed_43`

Result:

- `raw_residual_vector partial_mse_gain_vs_history = 0.11582781158577896`
- `raw_residual_vector partial_mae_gain_vs_history = 0.07239284257595771`
- `residual_change_vector partial_mse_gain_vs_history = 0.0`
- `residual_sign partial_mse_gain_vs_history = 0.0`
- gate result:
  - `do_not_train_generator`
  - `no_primary_target_passed`
  - `raw_residual_vector positive_partial_window_count = 4/5`
  - `raw_residual_vector min_partial_mse_gain_vs_history = -0.017010839527087873`
  - `raw_residual_vector counterfactual_margin = -0.024852551309569842`

Interpretation:

- the positive `residual_change_vector` signal from `panama seed42` did not
  replicate on `seed43`
- instead, `V27` again behaves like a raw-side improvement version
- this weakens the claim that `V27` has a stable transfer-side change-target
  capability
- but it strengthens a narrower claim:
  - `V27` can repeatedly produce raw-side signal across both `weather` and
    `panama`

### Narrow Change-Priority Tuning: `panama seed43 b60`

A narrow tuning pass was run to test whether the unstable change-target behavior
could be stabilized without changing the `V27` structure:

- config:
  - `configs/stage3_v27_quality_conditioned_dual_stream_probe_tune_change_priority.yaml`
- run:
  - `logs/s3v27expert_tuning_screen/v27chg_panama_s43_b60_01/panama/seed_43`

Key tuning intent:

- reduce direct raw/readout dominance
- strengthen delayed-change emphasis
- strengthen quality-conditioned change alignment

Result:

- `raw_residual_vector partial_mse_gain_vs_history = 0.060555941988040936`
- `raw_residual_vector partial_mae_gain_vs_history = 0.013343536698259112`
- `residual_change_vector partial_mse_gain_vs_history = -0.01346211091000404`
- `residual_sign partial_mse_gain_vs_history = -0.010883152484893799`
- gate result:
  - `do_not_train_generator`
  - `no_primary_target_passed`
  - `raw_residual_vector positive_partial_window_count = 2/5`
  - `residual_change_vector positive_partial_window_count = 1/5`

Interpretation:

- this targeted tuning pass is a negative result
- it did not stabilize the primary change target
- it also weakened the untuned `panama seed43 b60` raw-side gain
- this suggests the remaining `V27` weakness is not just a missing local
  coefficient setting on the change branch

## Scientific Interpretation

The defensible current interpretation is:

- `V27` is a full official version line, not a toy branch
- it is closer than `V26` to the algorithmic intent in the user's reports
- it is not a null implementation:
  - it consistently improves `raw_residual_vector` and `residual_sign` at the
    completed `weather b60` level
- it also shows that the change-target signal can become directionally positive
  on at least one public-load transfer run
- but it still fails on the strongest primary target:
  - on `weather`, `residual_change_vector` remains negative
  - on `panama`, `residual_change_vector` is positive on `seed42` but collapses
    back to zero on untuned `seed43` and becomes negative under the first
    narrow change-priority tuning pass
- therefore `V27` cannot yet replace `V17` as the empirical reference or `V23`
  as the strongest `V20+` observability reference

The more precise scientific statement is:

> `V27` successfully implements a quality-conditioned expert-local correction
> version of the user's dual-stream idea. At matched `weather b60` budget it
> produces reproducible raw-side teacher signal across `seed42/43`, and on
> `panama b60` it can make `residual_change_vector` directionally positive on
> one seed.
> However, the strict teacher gate still rejects all completed runs, so `V27`
> cannot yet be promoted as a validated new mainline.

## Current Risks

1. `V27` still cannot pass the strict teacher gate on any completed run.
2. The target preference is both dataset-sensitive and seed-sensitive:
   - `weather`: raw-side signal stronger than change-target signal
   - `panama seed42`: change-target signal appears but fails attribution safety
   - `panama seed43`: untuned raw-side gain remains while the change target
     collapses; the first targeted tuning pass weakens raw-side gain and still
     does not recover the change target
3. The first narrow change-priority retune is negative, so there is no evidence
   yet that the remaining gap can be closed by a simple local coefficient
   adjustment.
4. No completed `b120` or higher formal run exists yet.
5. No generator-unlocked run exists yet, so the full online correction value is
   still unverified.

## Recommended Next Actions

1. Keep `V27` as an active official line; it has now shown non-zero,
   cross-seed raw-side teacher signal.
2. The next decision point is no longer "does transfer help at all?" but
   "should `V27` be tuned toward the raw-side signal it reproduces, or should it
   be redesigned again around the unstable change target?"
3. The first narrow change-priority tuning pass did not help, so the next
   iteration should be either:
   - one more very narrow, hypothesis-driven retune with explicit stop criteria,
     or
   - a decision to stop local tuning and revisit the structural source of
     change-target instability
4. Do not claim `V27` is successful until at least one completed run improves
   the primary change target or passes the strict teacher gate.
