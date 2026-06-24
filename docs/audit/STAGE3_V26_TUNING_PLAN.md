# STAGE3 V26 TUNING PLAN

## Scope

This note freezes the `V26` structure and defines a tuning-first workflow.
No model-code changes are part of this plan.

## Baseline Diagnosis

Baseline run:

- `logs/s3v26expert_screen/v26_weather_s42_b240_01/weather/seed_42`

Observed failure pattern:

- `raw_residual_vector` was weak and mostly harmful against `history_only`
- `residual_change_vector` collapsed to zero real gain on validation
- strict teacher gate stopped before generator training

Interpretation:

- the problem is not implementation breakage
- the current loss and budget balance likely suppresses the change branch too
  strongly relative to history and quality gating

## Tuning Principle

Tune only parameters that can plausibly change:

1. how much correction budget is left for the change branch
2. how strongly delayed change is supervised
3. how conservative the quality gate is
4. whether optimization is too sharp for a fragile branch

Do not change:

- architecture
- feature semantics
- serving contract
- teacher gate criteria

## Tuning Families

### 1. Release balanced

Goal:

- give the change branch more room without fully relaxing the gate

Main changes:

- increase `lambda_change_readout`
- increase `lambda_delayed_change`
- slightly reduce `lambda_history_after_release`
- slightly reduce `lambda_history_budget_cap`
- modestly relax `lambda_release_confidence`

Config:

- `configs/stage3_v26_multiscale_change_probe_tune_release_balanced.yaml`

### 2. Release aggressive

Goal:

- test whether the main failure is over-conservative gating and underfunded
  change allocation

Main changes:

- strongly increase change supervision
- strongly relax gate-related regularization
- reduce history capture and budget constraints

Config:

- `configs/stage3_v26_multiscale_change_probe_tune_release_aggressive.yaml`

### 3. Stability

Goal:

- test whether the branch is present but training dynamics are too unstable

Main changes:

- lower learning rate
- lower weight decay
- slightly increase epochs
- increase temporal and router stability terms

Config:

- `configs/stage3_v26_multiscale_change_probe_tune_stability.yaml`

## Evaluation Order

1. Run all three configs at a lower but non-trivial budget screen.
2. Compare:
   - `teacher_only_v26_target_gate.csv`
   - `teacher_only_validation_partial_gain_vs_history_v26.csv`
3. Promote only the best config to a matched formal budget.

## Success Criteria

The first success criterion is not generator unlock.
The first success criterion is:

- recover non-zero `residual_change_vector`
- reduce harm on `raw_residual_vector`
- improve against the baseline `V26` run

Generator training should remain blocked unless the teacher gate actually
passes.

## Current Outcome

Completed tuning runs:

- `v26bal_weather_s42_b120_01`
- `v26agg_weather_s42_b120_01`
- `v26stab_weather_s42_b120_01`
- `v26hyb_weather_s42_b120_01`
- promoted run: `v26hyb_weather_s42_b240_01`
- matched-budget confirmation: `v26hyb_weather_s43_b240_01`
- transfer confirmation: `v26hyb_panama_s42_b240_01`
- narrow follow-up: `v26rob_weather_s42_b120_01`
- narrow follow-up: `v26chg_weather_s42_b120_01`
- narrow follow-up: `v26gchg_weather_s42_b120_01`
- promoted follow-up: `v26chg_weather_s42_b240_01`

Observed ranking:

1. `hybrid`
   - strongest overall tuning candidate
   - recovered a non-zero `residual_change_vector` at `b120`
   - preserved positive validation MSE partial gain on `raw_residual_vector`
2. `stability`
   - helped `raw_residual_vector`
   - did not recover change
3. `aggressive`
   - did not recover change
4. `balanced`
   - did not recover change and worsened harmful windows

Promoted-run takeaway:

- `hybrid` kept a small positive validation partial gain on
  `raw_residual_vector` at `b240`
- but `residual_change_vector` fell back to `0.0`
- strict teacher gate still failed

Confirmation takeaway:

- `weather seed43` did not preserve the `seed42` raw MSE improvement
  - `raw partial_mse_gain_vs_history = -0.009271226065472732`
  - `residual_change partial_mse_gain_vs_history = -0.04162774022097448`
- `panama seed42` showed clear negative transfer
  - `raw partial_mse_gain_vs_history = -0.06543549955725775`
  - `residual_change partial_mse_gain_vs_history = -0.2779851797273738`

This changes the interpretation of `hybrid`:

- it remains the best local tuning family found so far on `V26`
- but it is not yet a robust stability improvement
- current evidence supports "worth tuning further with caution", not
  "ready to replace earlier references"

Narrow follow-up takeaway:

- `hybrid_robust` regressed on both primary targets
- `hybrid_guarded_change` stayed positive on raw MSE but still fully lost the
  change branch
- `hybrid_change_persist` slightly improved raw screen MSE, but the gain did
  not survive promotion to `b240`

This means the tuning space has been probed more carefully than before, and the
evidence now points more toward local saturation than toward a missing obvious
hyperparameter setting.

So the current decision is:

- keep `V26` frozen
- treat `hybrid` as the only tuning family worth continuing
- do not claim tuning has solved the version
- if tuning continues, prioritize one-variable local probes around `hybrid`
  rather than introducing new tuning families

## One-Variable Probe Follow-Up

Three narrow probes were then run around the original `hybrid` setting so that
the next step would still respect the "tune first, do not change structure"
constraint.

### Probe A: `lambda_change_readout` up

- config:
  `configs/stage3_v26_multiscale_change_probe_tune_hybrid_readout_up.yaml`
- run:
  `logs/s3v26expert_tuning_screen/v26rdup_weather_s42_b120_01/weather/seed_42`
- result:
  - `raw_residual_vector partial_mse_gain_vs_history = -0.5378334782331711`
  - `residual_change_vector partial_mse_gain_vs_history = -0.03932522178603953`
- conclusion:
  the readout coefficient is not the missing easy fix; increasing it alone is
  strongly harmful

### Probe B: `lambda_history_after_release` down

- config:
  `configs/stage3_v26_multiscale_change_probe_tune_hybrid_history_down.yaml`
- run:
  `logs/s3v26expert_tuning_screen/v26hdown_weather_s42_b120_01/weather/seed_42`
- result:
  - `raw_residual_vector partial_mse_gain_vs_history = 0.027909430870031282`
  - `residual_change_vector partial_mse_gain_vs_history = -0.052292246387956924`
- conclusion:
  reducing the history penalty keeps a weak raw-side gain but still does not
  revive the primary change target

### Probe C: `lambda_delayed_change` up

- config:
  `configs/stage3_v26_multiscale_change_probe_tune_hybrid_delayed_up.yaml`
- run:
  `logs/s3v26expert_tuning_screen/v26dup_weather_s42_b120_01/weather/seed_42`
- result:
  - `raw_residual_vector partial_mse_gain_vs_history = 0.0732831485470159`
  - `residual_change_vector partial_mse_gain_vs_history = -0.07927276826205717`
  - `raw_residual_vector positive_partial_window_count = 3/5`
  - `raw_residual_vector counterfactual_margin = -0.11616261179160026`
- conclusion:
  delayed-change supervision is the only one-variable move that materially
  improves raw-side validation gain, but it still fails strict attribution and
  does not rescue `residual_change_vector`

## Updated Tuning Decision

The one-variable follow-up sharpens the previous tuning conclusion:

- tuning has been detailed enough to rule out several obvious local fixes
- the original `hybrid` family remains the only defensible tuning neighborhood
- even the strongest local one-variable probe improves only part of the raw
  screen metrics and still fails the full teacher gate
- therefore the current blocker should not be described as "we just need more
  routine hyperparameter tuning"
