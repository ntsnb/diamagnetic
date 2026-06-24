# STAGE3 V26 MULTISCALE CHANGE IMPLEMENTATION RECORD

## Scope

This record documents the `V26` official version line added on `2026-06-11`.
`V26` does not overwrite older version lines. It adds a dedicated base-model
and Stage III probe path under `s3v26expert`.

`V26` was implemented as a tuning-worthy official line, not a toy branch. In
this turn, the structure was explicitly frozen and the work shifted to a
tuning-first evaluation protocol.

## ML System Boundary

The changed ML boundary for `V26` is:

`expert-local correction backbone -> multiscale change readout -> served drift package -> teacher-only gate -> generator/risk-gate/online wrappers`

The model-native served drift source is:

- `pooled_servable_change`

That source is consumed consistently by:

- split-tensor collection
- teacher-only residual-target gate
- generator wrapper
- online adaptation wrapper

## File Boundary

Core `V26` files:

- `models/factor_mechanism_v26.py`
- `train/offline_factor_v26.py`
- `train/stage3_generator_training_v26.py`
- `train/stage3_v26_artifacts.py`
- `train/online_loop_stage3_v26.py`
- `train/run_stage3_v26_multiscale_change_probe.py`
- `configs/stage3_v26_multiscale_change_probe.yaml`
- `tests/test_factor_mechanism_v26.py`
- `tests/test_offline_factor_v26.py`
- `tests/test_stage3_v26_multiscale_change.py`
- `tests/test_stage3_generator_training_v26.py`
- `docs/superpowers/specs/2026-06-11-stage3-v26-multiscale-change-design.md`
- `docs/superpowers/plans/2026-06-11-stage3-v26-multiscale-change.md`

Tuning-only configs added after structure freeze:

- `configs/stage3_v26_multiscale_change_probe_tune_release_balanced.yaml`
- `configs/stage3_v26_multiscale_change_probe_tune_release_aggressive.yaml`
- `configs/stage3_v26_multiscale_change_probe_tune_stability.yaml`
- `configs/stage3_v26_multiscale_change_probe_tune_hybrid.yaml`
- `configs/stage3_v26_multiscale_change_probe_tune_hybrid_robust.yaml`
- `configs/stage3_v26_multiscale_change_probe_tune_hybrid_change_persist.yaml`
- `configs/stage3_v26_multiscale_change_probe_tune_hybrid_guarded_change.yaml`

Tuning note:

- `docs/audit/STAGE3_V26_TUNING_PLAN.md`

## Deterministic Validation

Completed checks:

- `python -m py_compile models\factor_mechanism_v26.py train\offline_factor_v26.py train\stage3_generator_training_v26.py train\stage3_v26_artifacts.py train\online_loop_stage3_v26.py train\run_stage3_v26_multiscale_change_probe.py tests\test_factor_mechanism_v26.py tests\test_offline_factor_v26.py tests\test_stage3_v26_multiscale_change.py tests\test_stage3_generator_training_v26.py`
- `pytest tests\test_factor_mechanism_v26.py tests\test_offline_factor_v26.py tests\test_stage3_v26_multiscale_change.py tests\test_stage3_generator_training_v26.py -q`
  - result: `11 passed`
- `pytest tests\test_factor_mechanism_v25.py tests\test_offline_factor_v25.py tests\test_stage3_v25_change_quality.py tests\test_stage3_generator_training_v25.py tests\test_factor_mechanism_v26.py tests\test_offline_factor_v26.py tests\test_stage3_v26_multiscale_change.py tests\test_stage3_generator_training_v26.py -q`
  - result: `22 passed`

These checks verify:

- `V26` schema consistency
- generator drift-dimension compatibility
- no immediate deterministic regression against the `V25` boundary

## Smoke Run

Smoke run:

- run:
  `logs/s3v26expert_smoke/v26_smoke_01/weather/seed_42`

Smoke result:

- protocol completed
- `teacher_only_abort.json`
  - `do_not_train_generator`
  - `no_primary_target_passed`

This confirms the path is wired correctly end-to-end before tuning.

## Baseline Formal Screen

Baseline formal run:

- run:
  `logs/s3v26expert_screen/v26_weather_s42_b240_01/weather/seed_42`

Teacher gate summary:

- `raw_residual_vector`
  - `positive_partial_window_count = 1 / 5`
  - `min_partial_mse_gain_vs_history = -0.11504783087697545`
  - `min_partial_mae_gain_vs_history = -0.18791909487040626`
  - `counterfactual_margin = -0.10133722508498194`
  - gate: fail
- `residual_change_vector`
  - `positive_partial_window_count = 0 / 5`
  - `min_partial_mse_gain_vs_history = -0.20679189790668973`
  - `min_partial_mae_gain_vs_history = -0.19889048206302282`
  - `counterfactual_margin = 0.0`
  - gate: fail

Validation partial gain vs history:

- `raw_residual_vector`
  - `history_only_real_mse_gain = 0.23770030396878988`
  - `fluctuation_plus_history_real_mse_gain = 0.22930224944374625`
  - `partial_mse_gain_vs_history = -0.008398054525043636`
  - `partial_mae_gain_vs_history = -0.047518803014525546`
- `residual_change_vector`
  - `history_only_real_mse_gain = 0.13396690932313343`
  - `fluctuation_plus_history_real_mse_gain = 0.0`
  - `partial_mse_gain_vs_history = -0.13396690932313343`
  - `partial_mae_gain_vs_history = -0.10688730365589363`

Baseline conclusion:

- `V26` baseline was not a broken implementation
- `raw_residual_vector` was near-neutral on validation MSE
- `residual_change_vector` collapsed to zero real gain
- the strict gate correctly blocked generator training

This justified a tuning-first response rather than yet another immediate
architectural jump.

## Tuning Sweep

All tuning runs kept the `V26` structure fixed and changed only configuration.

### Release Balanced

- config:
  `configs/stage3_v26_multiscale_change_probe_tune_release_balanced.yaml`
- run:
  `logs/s3v26expert_tuning_screen/v26bal_weather_s42_b120_01/weather/seed_42`
- key result:
  - `raw_residual_vector validation partial_mse_gain_vs_history = -0.05045760890969253`
  - `residual_change_vector validation partial_mse_gain_vs_history = 0.0`

Conclusion:

- slightly increasing change emphasis without changing optimization dynamics did
  not recover the change branch
- raw harmful windows became worse than the baseline

### Release Aggressive

- config:
  `configs/stage3_v26_multiscale_change_probe_tune_release_aggressive.yaml`
- run:
  `logs/s3v26expert_tuning_screen/v26agg_weather_s42_b120_01/weather/seed_42`
- key result:
  - `raw_residual_vector validation partial_mse_gain_vs_history = -0.03163758605387279`
  - `residual_change_vector validation partial_mse_gain_vs_history = 0.0`

Conclusion:

- relaxing the gate and pushing release harder still did not produce a non-zero
  `residual_change_vector`
- this argues against a simple "the branch is only over-gated" story

### Stability

- config:
  `configs/stage3_v26_multiscale_change_probe_tune_stability.yaml`
- run:
  `logs/s3v26expert_tuning_screen/v26stab_weather_s42_b120_01/weather/seed_42`
- key result:
  - `raw_residual_vector validation partial_mse_gain_vs_history = 0.02891955328464957`
  - `residual_change_vector validation partial_mse_gain_vs_history = -0.0822897624582415`

Conclusion:

- better optimization dynamics can help `raw_residual_vector`
- but they still do not stabilize the change target

### Hybrid

- config:
  `configs/stage3_v26_multiscale_change_probe_tune_hybrid.yaml`
- run:
  `logs/s3v26expert_tuning_screen/v26hyb_weather_s42_b120_01/weather/seed_42`
- key result:
  - `raw_residual_vector validation partial_mse_gain_vs_history = 0.02989281159288515`
  - `residual_change_vector validation partial_mse_gain_vs_history = -0.049557947262597914`
  - `residual_change_vector fluctuation_plus_history_real_mse_gain = 0.0029920045658275037`

Conclusion:

- this was the strongest tuning candidate
- it preserved positive validation MSE partial gain on `raw_residual_vector`
- and it recovered a non-zero `residual_change_vector` readout at screen budget

## Promoted Hybrid Formal Run

Promoted hybrid run:

- run:
  `logs/s3v26expert_tuning_screen/v26hyb_weather_s42_b240_01/weather/seed_42`

Teacher gate summary:

- `raw_residual_vector`
  - `positive_partial_window_count = 2 / 5`
  - `min_partial_mse_gain_vs_history = -0.12092321147815666`
  - `min_partial_mae_gain_vs_history = -0.09545738290951514`
  - `counterfactual_margin = -0.008465084336597192`
  - gate: fail
- `residual_change_vector`
  - `positive_partial_window_count = 0 / 5`
  - `min_partial_mse_gain_vs_history = 0.0`
  - `min_partial_mae_gain_vs_history = 0.0`
  - `counterfactual_margin = 0.0`
  - gate: fail

Validation partial gain vs history:

- `raw_residual_vector`
  - `history_only_real_mse_gain = 0.23356506963346008`
  - `fluctuation_plus_history_real_mse_gain = 0.24381028669023133`
  - `partial_mse_gain_vs_history = 0.010245217056771255`
  - `partial_mae_gain_vs_history = 0.020706872427290912`
- `residual_change_vector`
  - `history_only_real_mse_gain = 0.0`
  - `fluctuation_plus_history_real_mse_gain = 0.0`
  - `partial_mse_gain_vs_history = 0.0`

Promoted-run conclusion:

- the hybrid tuning preserved a small positive validation advantage on
  `raw_residual_vector`
- but the weak `residual_change_vector` recovery seen at `b120` did not survive
  promotion to `b240`
- the generator remained correctly blocked

## Matched-Budget Stability Check: Weather Seed43

Matched-budget confirmation run:

- run:
  `logs/s3v26expert_tuning_screen/v26hyb_weather_s43_b240_01/weather/seed_43`

Teacher gate summary:

- `raw_residual_vector`
  - `positive_partial_window_count = 1 / 5`
  - `min_partial_mse_gain_vs_history = -0.07712087982221905`
  - `min_partial_mae_gain_vs_history = -0.052532780651564415`
  - `counterfactual_margin = -0.08037262886117175`
  - gate: fail
- `residual_change_vector`
  - `positive_partial_window_count = 0 / 5`
  - `min_partial_mse_gain_vs_history = -0.08135447843177034`
  - `min_partial_mae_gain_vs_history = -0.05829244938470129`
  - `counterfactual_margin = 0.0`
  - gate: fail

Validation partial gain vs history:

- `raw_residual_vector`
  - `history_only_real_mse_gain = 0.23044901192007378`
  - `fluctuation_plus_history_real_mse_gain = 0.22117778585460104`
  - `partial_mse_gain_vs_history = -0.009271226065472732`
  - `partial_mae_gain_vs_history = 0.0020188246712380398`
- `residual_change_vector`
  - `history_only_real_mse_gain = 0.04162774022097448`
  - `fluctuation_plus_history_real_mse_gain = 0.0`
  - `partial_mse_gain_vs_history = -0.04162774022097448`
  - `partial_mae_gain_vs_history = -0.025647998451540775`

Seed43 conclusion:

- the `weather seed42` raw gain does not carry over cleanly to a second seed
- `residual_change_vector` did not stabilize and fell back to zero real gain
- this weakens the claim that the hybrid family is already a robust fix even
  within the same dataset

## Transfer Check: Panama Seed42

Cross-dataset transfer run:

- run:
  `logs/s3v26expert_tuning_screen/v26hyb_panama_s42_b240_01/panama/seed_42`

Teacher gate summary:

- `raw_residual_vector`
  - `positive_partial_window_count = 1 / 5`
  - `min_partial_mse_gain_vs_history = -0.2668424608231396`
  - `min_partial_mae_gain_vs_history = -0.27520125740805546`
  - `counterfactual_margin = -0.0762740647556493`
  - gate: fail
- `residual_change_vector`
  - `positive_partial_window_count = 0 / 5`
  - `min_partial_mse_gain_vs_history = -0.5113365794621808`
  - `min_partial_mae_gain_vs_history = -0.3774511498489487`
  - `counterfactual_margin = -0.02661169565001278`
  - gate: fail

Validation partial gain vs history:

- `raw_residual_vector`
  - `history_only_real_mse_gain = 0.37966960329241417`
  - `fluctuation_plus_history_real_mse_gain = 0.3142341037351564`
  - `partial_mse_gain_vs_history = -0.06543549955725775`
  - `partial_mae_gain_vs_history = -0.09240933309076138`
- `residual_change_vector`
  - `history_only_real_mse_gain = 0.3044972818375131`
  - `fluctuation_plus_history_real_mse_gain = 0.02651210211013931`
  - `partial_mse_gain_vs_history = -0.2779851797273738`
  - `partial_mae_gain_vs_history = -0.22558171087845114`

Panama conclusion:

- the hybrid tuning does not transfer as a weak positive signal
- on a public load dataset, both primary targets are clearly worse than
  `history_only`
- this is evidence against interpreting the `weather seed42` improvement as a
  generally stable tuning win

## Narrow Hybrid-Family Follow-Up Sweep

After the seed and transfer checks weakened confidence in the original hybrid
setting, a narrower follow-up sweep was run without changing the `V26`
structure. The goal was to determine whether the remaining gap was still likely
to be solved by disciplined hyperparameter tuning.

### Hybrid Robust

- config:
  `configs/stage3_v26_multiscale_change_probe_tune_hybrid_robust.yaml`
- run:
  `logs/s3v26expert_tuning_screen/v26rob_weather_s42_b120_01/weather/seed_42`
- validation:
  - `raw_residual_vector partial_mse_gain_vs_history = -0.12208023709263949`
  - `residual_change_vector partial_mse_gain_vs_history = -0.1039044210000171`
- gate:
  - `raw_residual_vector positive_partial_window_count = 1 / 5`
  - `raw_residual_vector counterfactual_margin = -0.12539062509193422`
  - `residual_change_vector positive_partial_window_count = 0 / 5`

Conclusion:

- pushing harder on robustness/quality terms did not stabilize the version
- both primary targets regressed, so this is a clearly worse branch than the
  original hybrid

### Hybrid Change Persist

- config:
  `configs/stage3_v26_multiscale_change_probe_tune_hybrid_change_persist.yaml`
- screen run:
  `logs/s3v26expert_tuning_screen/v26chg_weather_s42_b120_01/weather/seed_42`
- screen validation:
  - `raw_residual_vector partial_mse_gain_vs_history = 0.032477790911450866`
  - `residual_change_vector partial_mse_gain_vs_history = -0.08130450934661411`
- screen gate:
  - `raw_residual_vector positive_partial_window_count = 3 / 5`
  - `raw_residual_vector counterfactual_margin = -0.0842167869120416`
  - `residual_change_vector positive_partial_window_count = 0 / 5`

Screen conclusion:

- this variant slightly improved the raw validation MSE over the original
  hybrid screen
- but it achieved that by letting `residual_change_vector` collapse fully to
  zero real gain
- therefore it was a "better raw / worse change" tradeoff, not a clear overall
  improvement

Promoted formal run:

- run:
  `logs/s3v26expert_tuning_screen/v26chg_weather_s42_b240_01/weather/seed_42`
- validation:
  - `raw_residual_vector partial_mse_gain_vs_history = -0.1761043763038767`
  - `residual_change_vector partial_mse_gain_vs_history = -0.014120859180188515`
- gate:
  - `raw_residual_vector positive_partial_window_count = 1 / 5`
  - `raw_residual_vector min_partial_mse_gain_vs_history = -0.7894855736321952`
  - `raw_residual_vector counterfactual_margin = -0.02431167755184782`
  - `residual_change_vector positive_partial_window_count = 2 / 5`
  - `residual_change_vector min_partial_mse_gain_vs_history = -0.02563946605782718`

Promoted-run conclusion:

- the apparent screen advantage did not survive promotion
- at formal budget, this branch became sharply worse than the original hybrid
- this is strong evidence that the raw-only screen improvement was not a
  stable tuning win

### Hybrid Guarded Change

- config:
  `configs/stage3_v26_multiscale_change_probe_tune_hybrid_guarded_change.yaml`
- run:
  `logs/s3v26expert_tuning_screen/v26gchg_weather_s42_b120_01/weather/seed_42`
- validation:
  - `raw_residual_vector partial_mse_gain_vs_history = 0.01642296865707349`
  - `residual_change_vector partial_mse_gain_vs_history = -0.09088596141828674`
- gate:
  - `raw_residual_vector positive_partial_window_count = 2 / 5`
  - `raw_residual_vector counterfactual_margin = -0.15765154026288042`
  - `residual_change_vector positive_partial_window_count = 0 / 5`

Conclusion:

- adding extra quality pressure while retaining release emphasis did not rescue
  the change branch
- this branch is weaker than the original hybrid on both gate shape and change
  behavior

### Follow-Up Sweep Conclusion

This narrow follow-up sweep sharpens the earlier diagnosis:

- the original `hybrid` setting remains the best local `V26` tuning family
- but every nearby attempt still trades off the same two failures:
  - better raw slices usually come with complete collapse of
    `residual_change_vector`
  - stronger robustness/quality terms degrade both primary targets
- the follow-up sweep therefore supports a "tuning saturation" diagnosis more
  than a "we simply had not tuned carefully enough" diagnosis

### Narrow One-Variable Hybrid Probes

To test whether the remaining gap was just one missing scalar hyperparameter,
three matched-budget local probes were run around the original `hybrid` setting
at `weather seed42 b120`:

- `hybrid_readout_up`
  - config:
    `configs/stage3_v26_multiscale_change_probe_tune_hybrid_readout_up.yaml`
  - run:
    `logs/s3v26expert_tuning_screen/v26rdup_weather_s42_b120_01/weather/seed_42`
  - change: raise `lambda_change_readout` from `0.55` to `0.62`
  - result:
    - `raw_residual_vector partial_mse_gain_vs_history = -0.5378334782331711`
    - `residual_change_vector partial_mse_gain_vs_history = -0.03932522178603953`
  - interpretation:
    increasing the readout pressure alone is clearly harmful and destroys even
    the weak raw-side local gain

- `hybrid_history_down`
  - config:
    `configs/stage3_v26_multiscale_change_probe_tune_hybrid_history_down.yaml`
  - run:
    `logs/s3v26expert_tuning_screen/v26hdown_weather_s42_b120_01/weather/seed_42`
  - change: lower `lambda_history_after_release` from `0.07` to `0.05`
  - result:
    - `raw_residual_vector partial_mse_gain_vs_history = 0.027909430870031282`
    - `residual_change_vector partial_mse_gain_vs_history = -0.052292246387956924`
  - interpretation:
    weakening the history-after-release penalty preserves a weak raw MSE
    advantage but does not recover the change branch

- `hybrid_delayed_up`
  - config:
    `configs/stage3_v26_multiscale_change_probe_tune_hybrid_delayed_up.yaml`
  - run:
    `logs/s3v26expert_tuning_screen/v26dup_weather_s42_b120_01/weather/seed_42`
  - change: raise `lambda_delayed_change` from `0.35` to `0.42`
  - result:
    - `raw_residual_vector partial_mse_gain_vs_history = 0.0732831485470159`
    - `residual_change_vector partial_mse_gain_vs_history = -0.07927276826205717`
    - `raw_residual_vector positive_partial_window_count = 3/5`
    - `raw_residual_vector counterfactual_margin = -0.11616261179160026`
  - interpretation:
    stronger delayed-change supervision can amplify raw-side validation gain,
    but it still fails the counterfactual gate and leaves
    `residual_change_vector` completely negative

Taken together, these probes strengthen the same diagnosis:

- there is still some sensitivity inside the original `hybrid` neighborhood
- but none of the one-variable moves restored a viable primary change target
- the best local move improved only the raw-side screen metric, not the full
  gate logic
- this makes it harder to argue that `V26` is blocked by one overlooked scalar
  coefficient

## Current Scientific Conclusion

The right conclusion after tuning is narrower than either "V26 failed
completely" or "tuning solved it":

- tuning is materially relevant on `V26`
  - it can move `raw_residual_vector`
  - it can briefly recover a non-zero change signal at screen budget
- but tuning did not produce a teacher-gate pass
- the strongest target remains unstable at the promoted budget
- the best `weather seed42` slice does not hold up cleanly across seed or
  transfer dataset
- the nearest follow-up tuning variants confirm local saturation rather than a
  missing easy hyperparameter fix

So the most defensible statement is:

> The user's direction was worth implementing and worth tuning in depth, but the
> current `V26` structure still does not have enough stable change-signal
> headroom to replace the earlier empirical references, and the current tuning
> gains appear local rather than robust.

Current ranking:

- `V26 hybrid` is stronger than the untuned `V26` baseline
- `V26 hybrid` is not yet a stable cross-seed or cross-dataset improvement
- `V26` still does not exceed the earlier empirical references such as `V17`
- this is still not a paper-grade positive result
- the new `robust`, `change_persist`, and `guarded_change` follow-ups did not
  surpass the original `hybrid` branch
- the later one-variable probes also failed to break the same saturation
  pattern, even when one of them materially improved raw-side validation gain

## Recommended Next Action

Before any new structural version:

1. keep `V26` frozen
2. if more tuning budget is available, stay inside this parameter family:
   - `lambda_change_readout`
   - `lambda_delayed_change`
   - `lambda_release_confidence`
   - `lambda_multiscale_change_alignment`
   - `lambda_history_after_release`
   - `lambda_budget_alignment`
   - `lr`
   - `weight_decay`
3. if more tuning is attempted, treat it as a narrow robustness search rather
   than a presumed-success refinement
4. do not switch structure again until the tuned family either:
   - shows at least weak cross-seed stability on `weather`, or
   - avoids clear negative transfer on a public load benchmark
