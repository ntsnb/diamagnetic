# STAGE3 V29 EXPERT-LOCAL INVARIANT RELEASE IMPLEMENTATION RECORD

This record documents the `V29` official version line started on `2026-06-12`.
`V29` does not overwrite `V16`-`V28`.

## Why V29 Was Started

The user requested a full implementation of the expert-local invariant-learning
idea:

- each expert should start from a shared expert-local representation
- each expert should infer environment locally
- invariant learning should correct that shared representation into `phi_i`
- Stage III should not depend on a bare `phi_v - phi_i` subtraction as the
  semantic definition of drift
- the implementation should be a proper new version line, not a toy patch

The two Chinese reports support this direction structurally, but also warn that
better representation geometry does not automatically solve the harder problem:
the served signal still needs stable conditional residual-target information.

## Changed ML Boundary

`V29` changes the model boundary to:

`V26 correction backbone -> expert-local shared anchor -> local environment inference -> invariant correction -> residual-release state -> V29 served package -> Stage III teacher/generator/online path`

The native V29 Stage III drift source is:

`pooled_servable_change`

The native V29 state is:

`expert_local_invariant_release`

Compatibility aliases such as `expert_delta_phi_v`, `pooled_delta_phi_v`, and
`expert_phi_v = expert_phi_i + expert_residual_release_state` remain available
only for inherited loss/generator interfaces. They should not be interpreted as
the mathematical claim that `phi_v - phi_i` is the true drift mechanism.

## Core V29 Files

- `models/factor_mechanism_v29.py`
- `train/offline_factor_v29.py`
- `train/run_stage3_v29_expert_local_invariant_release_probe.py`
- `train/stage3_generator_training_v29.py`
- `train/stage3_v29_artifacts.py`
- `train/stage3_v29_online_gate_features.py`
- `train/stage3_v29_online_mode_gate.py`
- `train/online_loop_stage3_v29.py`
- `configs/stage3_v29_expert_local_invariant_release_probe.yaml`
- `docs/superpowers/plans/2026-06-12-stage3-v29-expert-local-invariant-release.md`

Tests:

- `tests/test_factor_mechanism_v29.py`
- `tests/test_offline_factor_v29.py`
- `tests/test_stage3_v29_expert_local_invariant_release.py`
- `tests/test_stage3_generator_training_v29.py`
- `tests/test_stage3_v29_online_gate_features.py`
- `tests/test_stage3_v29_online_mode_gate.py`

## Mechanism Summary

For each expert, `V29` constructs:

1. `shared_anchor = expert_base + expert_invariant_adapter`
2. `local_env_posterior = softmax(env_head(shared_anchor, expert state package))`
3. `phi_i = shared_anchor + invariant_residual - correction_gate * invariant_correction`
4. `residual_release_state = release_state_head(phi_i, local_env_posterior, expert state package)`
5. `expert_y_var = raw_change + support_score * release_delta_y - suppression_score * release_suppression_y`
6. `expert_servable_change = concat(phi_i, residual_release_state, invariant correction, local env posterior, release scores, release deltas, raw/change diagnostics)`

This is more faithful to the user's original idea than `V28` because the new
state package explicitly starts from the expert-local shared anchor and performs
environment-conditioned invariant correction before forming the residual-release
state.

It is also more conservative than a pure `phi_v - phi_i` design because the
served signal is the residual-release package itself.

## Added V29 Losses

`train/offline_factor_v29.py` inherits the V28/V26 losses and adds:

- `loss_local_env_consistency`
- `loss_invariant_correction_budget`
- `loss_residual_release_alignment`
- `loss_residual_release_advantage`
- `loss_expert_release_orthogonal`
- `loss_residual_state_noncollapse`

These losses are intended to make the implementation non-toy:

- local environment inference is trained against the inherited causal
  environment posterior
- invariant correction is budgeted rather than unconstrained
- residual-release state is aligned with existing observable/change-memory
  evidence
- release state is explicitly penalized when it fails to improve residual
  prediction over raw change
- expert-local release state is kept geometrically separated from `phi_i`
- collapse of the pooled release state is penalized

## Deterministic Validation

Completed deterministic checks:

- `C:\Users\admin\anaconda3\python.exe -m py_compile models\factor_mechanism_v29.py train\offline_factor_v29.py train\run_stage3_v29_expert_local_invariant_release_probe.py train\stage3_generator_training_v29.py train\stage3_v29_artifacts.py train\online_loop_stage3_v29.py tests\test_factor_mechanism_v29.py tests\test_offline_factor_v29.py tests\test_stage3_v29_expert_local_invariant_release.py tests\test_stage3_generator_training_v29.py`
- `C:\Users\admin\anaconda3\python.exe -m pytest tests\test_factor_mechanism_v29.py tests\test_offline_factor_v29.py tests\test_stage3_v29_expert_local_invariant_release.py tests\test_stage3_generator_training_v29.py -q`
  - `11 passed`
- `C:\Users\admin\anaconda3\python.exe -m pytest tests\test_factor_mechanism_v28.py tests\test_offline_factor_v28.py tests\test_stage3_v28_support_suppression_release.py tests\test_stage3_generator_training_v28.py tests\test_factor_mechanism_v29.py tests\test_offline_factor_v29.py tests\test_stage3_v29_expert_local_invariant_release.py tests\test_stage3_generator_training_v29.py -q`
  - `22 passed`
- After tightening the V29 drift-source contract and adding one contract test:
  - `C:\Users\admin\anaconda3\python.exe -m py_compile train\run_stage3_v29_expert_local_invariant_release_probe.py tests\test_stage3_v29_expert_local_invariant_release.py`
  - `C:\Users\admin\anaconda3\python.exe -m pytest tests\test_factor_mechanism_v28.py tests\test_offline_factor_v28.py tests\test_stage3_v28_support_suppression_release.py tests\test_stage3_generator_training_v28.py tests\test_factor_mechanism_v29.py tests\test_offline_factor_v29.py tests\test_stage3_v29_expert_local_invariant_release.py tests\test_stage3_generator_training_v29.py -q`
    - `23 passed`

## Runtime Contract Fix

The inherited V18 runner forcibly sets `online.stage3.drift_source =
delta_phi_v` before writing `resolved_config.json`. That was acceptable for
V18, but it made V29 configuration snapshots semantically inconsistent with the
actual V29 split tensor source.

V29 now uses two checks:

- strict user-YAML validation requires `online.stage3.drift_source =
  pooled_servable_change`
- runtime compatibility validation tolerates the inherited V18 mutation, then
  `_rename_v18_outputs_to_v29` rewrites the final `resolved_config.json` and
  manifest SHA so the recorded V29 artifact states the native drift source
  correctly

Evidence:

- `tests/test_stage3_v29_expert_local_invariant_release.py` now checks that a
  user config with `drift_source = delta_phi_v` is rejected
- new runs such as
  `logs/s3v29expert_tuning_screen/v29guard_weather_s42_b60_01/weather/seed_42`
  record `drift_source = pooled_servable_change` in `resolved_config.json`

## Experiment Results So Far

### Smoke: `weather seed42`

Completed smoke run:

- `logs/s3v29expert_smoke/v29_smoke_01/weather/seed_42`

Manifest evidence:

- `run_manifest_v29.json`
  - `protocol_version = v29_expert_local_invariant_release_probe`
  - `v29_model_native_drift_source = pooled_servable_change`
  - `v29_native_state = expert_local_invariant_release`

Teacher result:

- `teacher_only_abort.json`
  - `do_not_train_generator`
  - `no_primary_target_passed`

Validation partial gain summary:

- all validation partial gains were `0.0`

Interpretation:

- the V29 full protocol path is executable
- the smoke budget provides no positive signal evidence
- this smoke must not be used as an empirical improvement claim

### Formal Screen: `weather seed42 b60`

Completed formal screen:

- `logs/s3v29expert_screen/v29_weather_s42_b60_01/weather/seed_42`

Validation partial gain summary:

- `raw_residual_vector partial_mse_gain_vs_history = -0.3287200828542485`
- `raw_residual_vector partial_mae_gain_vs_history = -0.23311392059796301`
- `residual_change_vector partial_mse_gain_vs_history = 0.0`
- `residual_change_vector partial_mae_gain_vs_history = 0.0`
- `residual_sign partial_mse_gain_vs_history = -0.17321443557739258`
- `residual_sign partial_mae_gain_vs_history = -0.15279662609100342`
- `residual_sign partial_sign_agreement_vs_history = 0.034722208976745605`

Gate result:

- `teacher_only_abort.json`
  - `do_not_train_generator`
  - `no_primary_target_passed`
- `raw_residual_vector positive_partial_window_count = 2/5`
- `raw_residual_vector counterfactual_margin = -0.2986872140884438`
- `residual_change_vector positive_partial_window_count = 0/5`
- `residual_sign positive_partial_window_count = 1/5`

Interpretation:

- `V29` does not improve the `weather seed42 b60` slice
- raw and secondary sign targets regress versus history
- the primary change target remains zero
- compared with base `V28 weather seed42 b60`, this is worse on the most
  important partial-gain metrics

### Formal Screen: `panama seed42 b60`

Completed public-load transfer screen:

- `logs/s3v29expert_screen/v29_panama_s42_b60_01/panama/seed_42`

Validation partial gain summary:

- `raw_residual_vector partial_mse_gain_vs_history = -0.05012521758881246`
- `raw_residual_vector partial_mae_gain_vs_history = -0.04713466322530834`
- `residual_change_vector partial_mse_gain_vs_history = -0.1455648117660742`
- `residual_change_vector partial_mae_gain_vs_history = -0.0994119303934433`
- `residual_sign partial_mse_gain_vs_history = 0.017972350120544434`
- `residual_sign partial_mae_gain_vs_history = -0.004154205322265625`

Gate result:

- `teacher_only_abort.json`
  - `do_not_train_generator`
  - `no_primary_target_passed`
- `raw_residual_vector positive_partial_window_count = 0/5`
- `raw_residual_vector counterfactual_margin = -0.014770617729002086`
- `residual_change_vector positive_partial_window_count = 0/5`
- `residual_change_vector counterfactual_margin = 0.0`
- `residual_sign positive_partial_window_count = 3/5`

Interpretation:

- `V29` does not reproduce the useful `panama` change-target direction seen in
  the V28 `support_push` tuning run
- the primary change target is negative in aggregate MSE and MAE partial gain
- secondary `residual_sign` has a small positive MSE partial gain, but MAE is
  slightly negative and the strict gate still rejects the run

### Narrow Tuning: `stability_release_deconstrained`

The first `V29` screens showed that the new residual-release state was likely
over-constrained. The tuning pass changed only configuration weights and did
not modify the V29 architecture.

Tuning config:

- `configs/stage3_v29_expert_local_invariant_release_probe_tune_stability_release_deconstrained.yaml`

Purpose:

- reduce overbinding of residual-release state to weak observable/change-memory
  signals
- reduce expert release orthogonality pressure that may suppress useful
  residual directions
- restore raw/sign stability before increasing release-state advantage
- keep the V29 architecture fixed

Key tuning changes:

- `lr = 0.00038`
- `lambda_residual_release_alignment = 0.03`
- `lambda_expert_release_orthogonal = 0.01`
- `lambda_residual_state_noncollapse = 0.01`
- `lambda_local_env_consistency = 0.04`
- `lambda_invariant_correction_budget = 0.02`
- `lambda_change_readout = 0.3`
- `lambda_delayed_change = 0.28`
- `lambda_router_stability = 0.12`
- `lambda_anchor_preservation = 0.12`

Completed tuned `weather seed42 b60` screen:

- `logs/s3v29expert_tuning_screen/v29stable_weather_s42_b60_01/weather/seed_42`

Validation partial gain summary:

- `raw_residual_vector partial_mse_gain_vs_history = 0.34745194792605627`
- `raw_residual_vector partial_mae_gain_vs_history = 0.26141223387647555`
- `raw_residual_vector partial_sign_agreement_vs_history = 0.45694447308778763`
- `residual_change_vector partial_mse_gain_vs_history = 0.002388139577353788`
- `residual_change_vector partial_mae_gain_vs_history = 0.006411211314488097`
- `residual_sign partial_mse_gain_vs_history = 0.6950467228889465`
- `residual_sign partial_mae_gain_vs_history = 0.6325697600841522`
- `residual_sign partial_sign_agreement_vs_history = 0.7138888910412788`

Gate result:

- `teacher_only_abort.json`
  - `do_not_train_generator`
  - `no_primary_target_passed`
- `raw_residual_vector positive_partial_window_count = 4/5`
- `raw_residual_vector counterfactual_margin = -0.7246091209150117`
- `residual_change_vector positive_partial_window_count = 2/5`
- `residual_sign positive_partial_window_count = 5/5`
- `residual_sign counterfactual_margin = -0.09309427440166473`

Interpretation:

- this tuning pass successfully reverses the negative base `V29 weather seed42`
  result on aggregate raw/sign metrics
- compared with base `V29 weather seed42 b60`, raw MSE changes from
  `-0.3287200828542485` to `0.34745194792605627`
- secondary sign MSE changes from `-0.17321443557739258` to
  `0.6950467228889465`
- this is also stronger than base `V28 weather seed42 b60` on raw aggregate MSE,
  but not on gate safety because the counterfactual margin remains strongly
  negative
- the strict teacher gate still rejects the run, so this is not a promotion
  result

Completed tuned `panama seed42 b60` screen:

- `logs/s3v29expert_tuning_screen/v29stable_panama_s42_b60_01/panama/seed_42`

Validation partial gain summary:

- `raw_residual_vector partial_mse_gain_vs_history = -0.033578837325045394`
- `raw_residual_vector partial_mae_gain_vs_history = -0.018688653143735363`
- `residual_change_vector partial_mse_gain_vs_history = 0.0`
- `residual_change_vector partial_mae_gain_vs_history = 0.0`
- `residual_sign partial_mse_gain_vs_history = 0.003960847854614258`
- `residual_sign partial_mae_gain_vs_history = 0.006877124309539795`

Gate result:

- `teacher_only_abort.json`
  - `do_not_train_generator`
  - `no_primary_target_passed`
- `raw_residual_vector positive_partial_window_count = 2/5`
- `raw_residual_vector counterfactual_margin = -0.05145275393231989`
- `residual_change_vector positive_partial_window_count = 0/5`
- `residual_sign positive_partial_window_count = 3/5`
- `residual_sign counterfactual_margin = -0.031080126762390137`

Interpretation:

- the tuned config improves over base `V29 panama seed42 b60` by removing the
  large negative `residual_change_vector` aggregate MSE (`-0.1455648117660742`
  to `0.0`)
- it does not reproduce the positive panama change-target evidence of V28
  `support_push`, where `residual_change_vector` MSE reached
  `0.0416912078672488`
- raw remains slightly negative and no primary target reaches the strict gate
- this is partial stabilization, not cross-dataset validation

### Narrow Tuning: `counterfactual_guarded`

The `stability_release_deconstrained` run recovered strong weather aggregate
raw/sign signal, but the strict gate still failed because the worst-window
counterfactual margin remained negative. A second tuning pass tested whether
stronger counterfactual/release-null pressure could improve gate safety.

Tuning config:

- `configs/stage3_v29_expert_local_invariant_release_probe_tune_counterfactual_guarded.yaml`

Purpose:

- increase `lambda_counterfactual_shuffle`
- increase `lambda_release_null`
- reduce release/readout amplitude pressure
- keep the V29 architecture fixed

Completed tuned `weather seed42 b60` screen:

- `logs/s3v29expert_tuning_screen/v29guard_weather_s42_b60_01/weather/seed_42`

Validation partial gain summary:

- `raw_residual_vector partial_mse_gain_vs_history = 0.35570934066691234`
- `raw_residual_vector partial_mae_gain_vs_history = 0.2638440583324483`
- `raw_residual_vector partial_sign_agreement_vs_history = 0.5680555775761604`
- `residual_change_vector partial_mse_gain_vs_history = 0.0`
- `residual_change_vector partial_mae_gain_vs_history = 0.0`
- `residual_sign partial_mse_gain_vs_history = 0.378975510597229`
- `residual_sign partial_mae_gain_vs_history = 0.34554940462112427`

Gate result:

- `raw_residual_vector positive_partial_window_count = 4/5`
- `raw_residual_vector counterfactual_margin = -2.5328747236838742`
- `residual_change_vector positive_partial_window_count = 0/5`
- `residual_sign positive_partial_window_count = 4/5`
- `residual_sign counterfactual_margin = -0.24945169687271118`

Interpretation:

- `counterfactual_guarded` preserves aggregate raw gain but makes worst-window
  raw counterfactual safety much worse than `stability_release_deconstrained`
- it also weakens secondary sign aggregate MSE from `0.6950467228889465` to
  `0.378975510597229`
- this setting is a negative tuning result and should not be carried forward
- the panama screen was intentionally skipped because the weather gate target
  moved in the wrong direction

### Narrow Tuning: `early_window_stability`

After the user explicitly requested more rigorous tuning before further model
changes, this pass kept the V29 architecture fixed and tested whether more
temporal/release smoothing, stronger anchor preservation, and a mild
counterfactual increase could repair the first validation window.

Tuning config:

- `configs/stage3_v29_expert_local_invariant_release_probe_tune_early_window_stability.yaml`

Completed tuned `weather seed42 b60` screen:

- `logs/s3v29expert_tuning_screen/v29earlystab_weather_s42_b60_01/weather/seed_42`

Validation partial gain summary:

- `raw_residual_vector partial_mse_gain_vs_history = 0.6487557143040037`
- `raw_residual_vector partial_mae_gain_vs_history = 0.4907870429021493`
- `raw_residual_vector partial_sign_agreement_vs_history = 0.7236111238598824`
- `residual_change_vector partial_mse_gain_vs_history = 0.0`
- `residual_change_vector partial_mae_gain_vs_history = 0.0`
- `residual_sign partial_mse_gain_vs_history = 0.5103973746299744`
- `residual_sign partial_mae_gain_vs_history = 0.5049261450767517`

Gate result:

- `teacher_only_abort.json`
  - `do_not_train_generator`
  - `no_primary_target_passed`
- `raw_residual_vector positive_partial_window_count = 4/5`
- `raw_residual_vector min_partial_mse_gain_vs_history = -3.2306852535381294`
- `raw_residual_vector counterfactual_margin = -5.456856716067559`
- `residual_change_vector positive_partial_window_count = 0/5`
- `residual_sign positive_partial_window_count = 4/5`
- `residual_sign counterfactual_margin = -0.2489793300628662`

Interpretation:

- this pass improves aggregate raw MSE versus `stability_release_deconstrained`
  (`0.6487557143040037` vs `0.34745194792605627`) but catastrophically worsens
  the first validation window
- window `0` raw partial MSE falls to `-3.2306852535381294`, compared with
  `-0.1465061551651995` in `stability_release_deconstrained`
- the worst raw counterfactual margin falls to `-5.456856716067559`
- this rejects the idea that broad increases to temporal/anchor/budget
  regularization are sufficient for safe V29 tuning
- the panama screen was skipped because the weather worst-window failure was
  much worse than the current best V29 tuning anchor

### Narrow Tuning: `stability_micro_cf`

This pass returned close to `stability_release_deconstrained` and changed only a
small set of amplitudes and counterfactual-related weights. It tested whether
small-step tuning could improve counterfactual safety without the large
early-window collapse observed in `early_window_stability`.

Tuning config:

- `configs/stage3_v29_expert_local_invariant_release_probe_tune_stability_micro_cf.yaml`

Completed tuned `weather seed42 b60` screen:

- `logs/s3v29expert_tuning_screen/v29microcf_weather_s42_b60_01/weather/seed_42`

Validation partial gain summary:

- `raw_residual_vector partial_mse_gain_vs_history = 0.2107932743377172`
- `raw_residual_vector partial_mae_gain_vs_history = 0.14838422745752536`
- `raw_residual_vector partial_sign_agreement_vs_history = 0.3513888865709305`
- `residual_change_vector partial_mse_gain_vs_history = 0.0`
- `residual_change_vector partial_mae_gain_vs_history = 0.0`
- `residual_sign partial_mse_gain_vs_history = 0.33958321809768677`
- `residual_sign partial_mae_gain_vs_history = 0.3522756099700928`

Gate result:

- `teacher_only_abort.json`
  - `do_not_train_generator`
  - `no_primary_target_passed`
- `raw_residual_vector positive_partial_window_count = 3/5`
- `raw_residual_vector min_partial_mse_gain_vs_history = -0.5858395508423936`
- `raw_residual_vector counterfactual_margin = -0.5858395508423936`
- `residual_change_vector positive_partial_window_count = 0/5`
- `residual_sign positive_partial_window_count = 4/5`
- `residual_sign counterfactual_margin = -0.26667869091033936`

Interpretation:

- this is less catastrophic than `early_window_stability`, but still worse than
  `stability_release_deconstrained`
- aggregate raw MSE falls from `0.34745194792605627` to
  `0.2107932743377172`
- raw positive partial windows fall from `4/5` to `3/5`
- window `0` raw partial MSE worsens from `-0.1465061551651995` to
  `-0.5858395508423936`
- the current best V29 tuning anchor remains
  `stability_release_deconstrained`
- future tuning should use single-factor or very small-grid sweeps around the
  stability config instead of bundled regularization changes

### Single-Factor Sweep: `lambda_counterfactual_shuffle`

After the bundled tuning failures, the user requested more careful tuning before
additional model changes. This sweep changed only one weight around the current
best anchor `stability_release_deconstrained`:

- anchor: `lambda_counterfactual_shuffle = 0.08`
- low side: `lambda_counterfactual_shuffle = 0.07`
- high side: `lambda_counterfactual_shuffle = 0.09`

All other model, train, invariant, Stage III, and online fields were kept equal
to the stability anchor.

Low-side tuning config:

- `configs/stage3_v29_expert_local_invariant_release_probe_tune_cf_shuffle_007_only.yaml`

Low-side tuned `weather seed42 b60` screen:

- `logs/s3v29expert_tuning_screen/v29cf007_weather_s42_b60_01/weather/seed_42`

Low-side validation partial gain summary:

- `raw_residual_vector partial_mse_gain_vs_history = 0.3656415117996853`
- `raw_residual_vector partial_mae_gain_vs_history = 0.2818368642950102`
- `residual_change_vector partial_mse_gain_vs_history = 0.006659980640750224`
- `residual_change_vector partial_mae_gain_vs_history = 0.00408486381885984`
- `residual_sign partial_mse_gain_vs_history = 0.6250559985637665`
- `residual_sign partial_mae_gain_vs_history = 0.5982058346271515`

Low-side gate result:

- `teacher_only_abort.json`
  - `do_not_train_generator`
  - `no_primary_target_passed`
- `raw_residual_vector positive_partial_window_count = 4/5`
- `raw_residual_vector min_partial_mse_gain_vs_history = -3.093504781268863`
- `raw_residual_vector counterfactual_margin = -3.4914044381611973`
- `residual_change_vector positive_partial_window_count = 3/5`
- `residual_sign positive_partial_window_count = 5/5`
- `residual_sign counterfactual_margin = -0.08216537535190582`

High-side tuning config:

- `configs/stage3_v29_expert_local_invariant_release_probe_tune_cf_shuffle_009_only.yaml`

High-side tuned `weather seed42 b60` screen:

- `logs/s3v29expert_tuning_screen/v29cf009_weather_s42_b60_01/weather/seed_42`

High-side validation partial gain summary:

- `raw_residual_vector partial_mse_gain_vs_history = 0.5549855579529909`
- `raw_residual_vector partial_mae_gain_vs_history = 0.43667150919668096`
- `residual_change_vector partial_mse_gain_vs_history = 0.0`
- `residual_change_vector partial_mae_gain_vs_history = 0.0`
- `residual_sign partial_mse_gain_vs_history = 0.47422459721565247`
- `residual_sign partial_mae_gain_vs_history = 0.4501376450061798`

High-side gate result:

- `teacher_only_abort.json`
  - `do_not_train_generator`
  - `no_primary_target_passed`
- `raw_residual_vector positive_partial_window_count = 4/5`
- `raw_residual_vector min_partial_mse_gain_vs_history = -2.304469675805639`
- `raw_residual_vector counterfactual_margin = -2.304469675805639`
- `residual_change_vector positive_partial_window_count = 0/5`
- `residual_sign positive_partial_window_count = 5/5`
- `residual_sign counterfactual_margin = -0.16054761409759521`

Interpretation:

- both sides of the local sweep fail to improve the strict teacher gate
- lowering to `0.07` helps `residual_change_vector` and keeps strong
  `residual_sign`, but it catastrophically harms raw window `0`
- increasing to `0.09` raises aggregate raw MSE versus the anchor, but also
  catastrophically harms raw window `0`
- the anchor `0.08` remains the best observed value for raw worst-window
  stability in this local neighborhood
- this factor should not be swept further until a separate mechanism addresses
  window `0` raw failure

### Single-Factor Sweep: `lambda_release_null`

This sweep changed only `lambda_release_null` around the current best anchor
`stability_release_deconstrained`:

- anchor: `lambda_release_null = 0.05`
- low side: `lambda_release_null = 0.045`
- high side: `lambda_release_null = 0.055`

All other model, train, invariant, Stage III, and online fields were kept equal
to the stability anchor.

Low-side tuning config:

- `configs/stage3_v29_expert_local_invariant_release_probe_tune_release_null_0045_only.yaml`

Low-side tuned `weather seed42 b60` screen:

- `logs/s3v29expert_tuning_screen/v29null0045_weather_s42_b60_01/weather/seed_42`

Low-side validation partial gain summary:

- `raw_residual_vector partial_mse_gain_vs_history = 0.28996881937757046`
- `raw_residual_vector partial_mae_gain_vs_history = 0.2268456821889368`
- `residual_change_vector partial_mse_gain_vs_history = 0.0031415870650911355`
- `residual_change_vector partial_mae_gain_vs_history = 0.003833954794371326`
- `residual_sign partial_mse_gain_vs_history = 0.5695108473300934`
- `residual_sign partial_mae_gain_vs_history = 0.5540281534194946`

Low-side gate result:

- `raw_residual_vector positive_partial_window_count = 4/5`
- `raw_residual_vector min_partial_mse_gain_vs_history = -3.5556763517029277`
- `raw_residual_vector counterfactual_margin = -3.5556763517029277`
- `residual_change_vector positive_partial_window_count = 2/5`
- `residual_sign positive_partial_window_count = 5/5`
- `residual_sign counterfactual_margin = -0.33146932721138`

High-side tuning config:

- `configs/stage3_v29_expert_local_invariant_release_probe_tune_release_null_0055_only.yaml`

High-side tuned `weather seed42 b60` screen:

- `logs/s3v29expert_tuning_screen/v29null0055_weather_s42_b60_01/weather/seed_42`

High-side validation partial gain summary:

- `raw_residual_vector partial_mse_gain_vs_history = 0.5222828881075632`
- `raw_residual_vector partial_mae_gain_vs_history = 0.39975286480841377`
- `residual_change_vector partial_mse_gain_vs_history = 0.001206289769733205`
- `residual_change_vector partial_mae_gain_vs_history = 0.0020458585732038423`
- `residual_sign partial_mse_gain_vs_history = 0.5026110410690308`
- `residual_sign partial_mae_gain_vs_history = 0.47933268547058105`

High-side gate result:

- `raw_residual_vector positive_partial_window_count = 4/5`
- `raw_residual_vector min_partial_mse_gain_vs_history = -0.44838369064419603`
- `raw_residual_vector counterfactual_margin = -0.823874883701122`
- `residual_change_vector positive_partial_window_count = 3/5`
- `residual_sign positive_partial_window_count = 4/5`
- `residual_sign counterfactual_margin = -0.16620111465454102`

Interpretation:

- neither side passes the strict teacher gate
- lowering to `0.045` catastrophically harms raw window `0`
- increasing to `0.055` improves aggregate raw gain versus the anchor, but
  worsens the raw worst window and counterfactual margin and weakens the sign
  target
- `0.055` has a small `residual_change_vector` window-count improvement
  (`3/5` vs `2/5`), but aggregate change gain is lower than the anchor and the
  raw/sign regressions block promotion
- the anchor `0.05` remains the best observed release-null value for balanced
  raw/sign safety in this local neighborhood

### Single-Factor Sweep: `lambda_release_first_increment`

This sweep changed only `lambda_release_first_increment` around the current best
anchor `stability_release_deconstrained`:

- anchor: `lambda_release_first_increment = 0.40`
- low side: `lambda_release_first_increment = 0.38`
- midpoint: `lambda_release_first_increment = 0.41`
- high side: `lambda_release_first_increment = 0.42`

All other model, train, invariant, Stage III, and online fields were kept equal
to the stability anchor.

Low-side tuning config:

- `configs/stage3_v29_expert_local_invariant_release_probe_tune_release_first_038_only.yaml`

Low-side tuned `weather seed42 b60` screen:

- `logs/s3v29expert_tuning_screen/v29first038_weather_s42_b60_01/weather/seed_42`

Low-side validation partial gain summary:

- `raw_residual_vector partial_mse_gain_vs_history = 0.46718376446062737`
- `raw_residual_vector partial_mae_gain_vs_history = 0.3312979410475856`
- `residual_change_vector partial_mse_gain_vs_history = 0.0`
- `residual_change_vector partial_mae_gain_vs_history = 0.0`
- `residual_sign partial_mse_gain_vs_history = 0.40803998708724976`
- `residual_sign partial_mae_gain_vs_history = 0.41544824838638306`

Low-side gate result:

- `raw_residual_vector positive_partial_window_count = 4/5`
- `raw_residual_vector min_partial_mse_gain_vs_history = -1.805349288341766`
- `raw_residual_vector counterfactual_margin = -1.805349288341766`
- `residual_change_vector positive_partial_window_count = 0/5`
- `residual_sign positive_partial_window_count = 4/5`
- `residual_sign counterfactual_margin = -0.16586098074913025`

High-side tuning config:

- `configs/stage3_v29_expert_local_invariant_release_probe_tune_release_first_042_only.yaml`

High-side tuned `weather seed42 b60` screen:

- `logs/s3v29expert_tuning_screen/v29first042_weather_s42_b60_01/weather/seed_42`

High-side validation partial gain summary:

- `raw_residual_vector partial_mse_gain_vs_history = 0.2843195154550431`
- `raw_residual_vector partial_mae_gain_vs_history = 0.1888567188290113`
- `residual_change_vector partial_mse_gain_vs_history = 0.0`
- `residual_change_vector partial_mae_gain_vs_history = 0.0`
- `residual_sign partial_mse_gain_vs_history = 0.6246607899665833`
- `residual_sign partial_mae_gain_vs_history = 0.581342488527298`

High-side gate result:

- `raw_residual_vector positive_partial_window_count = 4/5`
- `raw_residual_vector min_partial_mse_gain_vs_history = -0.42083107598996183`
- `raw_residual_vector counterfactual_margin = -0.42083107598996183`
- `residual_change_vector positive_partial_window_count = 0/5`
- `residual_sign positive_partial_window_count = 5/5`
- `residual_sign counterfactual_margin = -0.0507509708404541`

Midpoint tuning config:

- `configs/stage3_v29_expert_local_invariant_release_probe_tune_release_first_041_only.yaml`

Midpoint tuned `weather seed42 b60` screen:

- `logs/s3v29expert_tuning_screen/v29first041_weather_s42_b60_01/weather/seed_42`

Midpoint validation partial gain summary:

- `raw_residual_vector partial_mse_gain_vs_history = 0.566033730128696`
- `raw_residual_vector partial_mae_gain_vs_history = 0.44005444560676393`
- `residual_change_vector partial_mse_gain_vs_history = 0.0`
- `residual_change_vector partial_mae_gain_vs_history = 0.0`
- `residual_sign partial_mse_gain_vs_history = 0.5793227255344391`
- `residual_sign partial_mae_gain_vs_history = 0.5344200730323792`

Midpoint gate result:

- `raw_residual_vector positive_partial_window_count = 4/5`
- `raw_residual_vector min_partial_mse_gain_vs_history = -0.39785273932262527`
- `raw_residual_vector counterfactual_margin = -0.8293233156592138`
- `residual_change_vector positive_partial_window_count = 0/5`
- `residual_sign positive_partial_window_count = 5/5`
- `residual_sign counterfactual_margin = -0.31375235319137573`

Interpretation:

- none of the tested values passes the strict teacher gate
- lowering to `0.38` increases aggregate raw gain relative to the anchor but
  makes window `0` and the counterfactual margin much worse
- increasing to `0.42` improves the raw and sign counterfactual margins relative
  to the anchor, but weakens aggregate raw gain and still leaves window `0`
  below the harmful threshold
- the midpoint `0.41` gives the strongest aggregate raw gain in this local
  sweep, but its raw counterfactual margin is worse than both `0.40` and `0.42`,
  and the `residual_change_vector` remains fully collapsed
- unlike the previous `lambda_counterfactual_shuffle` and
  `lambda_release_null` sweeps, `lambda_release_first_increment` shows a
  meaningful trade-off rather than pure degradation
- the next tuning step should not promote `0.41` or `0.42`; the useful lesson is
  that amplitude tuning alone cannot fix the first-window/counterfactual failure

### Single-Factor Sweep: `train.lr`

This sweep changed only `train.lr` around the current best anchor
`stability_release_deconstrained`:

- anchor: `train.lr = 0.00038`
- low side: `train.lr = 0.00034`
- high side: `train.lr = 0.00042`

All model, invariant, Stage III, and online fields were kept equal to the
stability anchor.

Low-side tuning config:

- `configs/stage3_v29_expert_local_invariant_release_probe_tune_lr_00034_only.yaml`

Low-side tuned `weather seed42 b60` screen:

- `logs/s3v29expert_tuning_screen/v29lr00034_weather_s42_b60_01/weather/seed_42`

Low-side validation partial gain summary:

- `raw_residual_vector partial_mse_gain_vs_history = 0.43066202825842365`
- `raw_residual_vector partial_mae_gain_vs_history = 0.34073262799962667`
- `residual_change_vector partial_mse_gain_vs_history = 0.0`
- `residual_change_vector partial_mae_gain_vs_history = 0.0`
- `residual_sign partial_mse_gain_vs_history = 0.5830316543579102`
- `residual_sign partial_mae_gain_vs_history = 0.5360222458839417`

Low-side gate result:

- `raw_residual_vector positive_partial_window_count = 4/5`
- `raw_residual_vector min_partial_mse_gain_vs_history = -3.7506495003938225`
- `raw_residual_vector counterfactual_margin = -4.664620134611712`
- `residual_change_vector positive_partial_window_count = 0/5`
- `residual_sign positive_partial_window_count = 5/5`
- `residual_sign counterfactual_margin = -0.33076202869415283`

High-side tuning config:

- `configs/stage3_v29_expert_local_invariant_release_probe_tune_lr_00042_only.yaml`

High-side tuned `weather seed42 b60` screen:

- `logs/s3v29expert_tuning_screen/v29lr00042_weather_s42_b60_01/weather/seed_42`

High-side validation partial gain summary:

- `raw_residual_vector partial_mse_gain_vs_history = 0.5935039309515732`
- `raw_residual_vector partial_mae_gain_vs_history = 0.45041361929254287`
- `residual_change_vector partial_mse_gain_vs_history = 0.0`
- `residual_change_vector partial_mae_gain_vs_history = 0.0`
- `residual_sign partial_mse_gain_vs_history = 0.7864210605621338`
- `residual_sign partial_mae_gain_vs_history = 0.7489124089479446`

High-side gate result:

- `raw_residual_vector positive_partial_window_count = 4/5`
- `raw_residual_vector min_partial_mse_gain_vs_history = -0.7459058197140563`
- `raw_residual_vector counterfactual_margin = -0.7459058197140563`
- `residual_change_vector positive_partial_window_count = 0/5`
- `residual_sign positive_partial_window_count = 5/5`
- `residual_sign counterfactual_margin = -0.08257308113388717`

Interpretation:

- neither side passes the strict teacher gate
- lowering the learning rate to `0.00034` is clearly negative: it preserves
  positive aggregate raw/sign metrics but catastrophically harms raw window `0`
  and the counterfactual margin
- increasing the learning rate to `0.00042` improves aggregate raw/sign metrics
  and slightly improves the sign counterfactual margin versus the anchor, but it
  worsens raw worst-window safety and leaves `residual_change_vector` collapsed
- learning rate alone is not the missing lever for V29; subsequent tuning should
  move to signal-shaping weights such as readout/delayed-change amplitudes while
  keeping the architecture fixed

### Single-Factor Sweep: `lambda_change_readout`

This sweep changed only `lambda_change_readout` around the current best anchor
`stability_release_deconstrained`:

- anchor: `lambda_change_readout = 0.30`
- low side: `lambda_change_readout = 0.26`
- high side: `lambda_change_readout = 0.34`

All model, train, other invariant, Stage III, and online fields were kept equal
to the stability anchor.

Low-side tuning config:

- `configs/stage3_v29_expert_local_invariant_release_probe_tune_change_readout_026_only.yaml`

Low-side tuned `weather seed42 b60` screen:

- `logs/s3v29expert_tuning_screen/v29cr026_weather_s42_b60_01/weather/seed_42`

Low-side validation partial gain summary:

- `raw_residual_vector partial_mse_gain_vs_history = 0.36031556562316`
- `raw_residual_vector partial_mae_gain_vs_history = 0.2844912794303554`
- `residual_change_vector partial_mse_gain_vs_history = 0.0007235077687528137`
- `residual_change_vector partial_mae_gain_vs_history = 0.0011029297049069816`
- `residual_sign partial_mse_gain_vs_history = 0.5691874921321869`
- `residual_sign partial_mae_gain_vs_history = 0.5646478235721588`

Low-side gate result:

- `raw_residual_vector positive_partial_window_count = 4/5`
- `raw_residual_vector min_partial_mse_gain_vs_history = -1.9118969332201075`
- `raw_residual_vector counterfactual_margin = -1.9118969332201075`
- `residual_change_vector positive_partial_window_count = 1/5`
- `residual_sign positive_partial_window_count = 5/5`
- `residual_sign counterfactual_margin = -0.12734347581863403`

High-side tuning config:

- `configs/stage3_v29_expert_local_invariant_release_probe_tune_change_readout_034_only.yaml`

High-side tuned `weather seed42 b60` screen:

- `logs/s3v29expert_tuning_screen/v29cr034_weather_s42_b60_01/weather/seed_42`

High-side validation partial gain summary:

- `raw_residual_vector partial_mse_gain_vs_history = 0.2034717094000943`
- `raw_residual_vector partial_mae_gain_vs_history = 0.12241154687785447`
- `residual_change_vector partial_mse_gain_vs_history = 0.0`
- `residual_change_vector partial_mae_gain_vs_history = 0.0`
- `residual_sign partial_mse_gain_vs_history = 0.2965698838233948`
- `residual_sign partial_mae_gain_vs_history = 0.319587767124176`

High-side gate result:

- `raw_residual_vector positive_partial_window_count = 3/5`
- `raw_residual_vector min_partial_mse_gain_vs_history = -1.1054253750406193`
- `raw_residual_vector counterfactual_margin = -1.1054253750406193`
- `residual_change_vector positive_partial_window_count = 0/5`
- `residual_sign positive_partial_window_count = 4/5`
- `residual_sign counterfactual_margin = -0.5392091274261475`

Interpretation:

- neither side passes the strict teacher gate
- lowering `lambda_change_readout` to `0.26` weakly reactivates the
  `residual_change_vector` target (`1/5` positive windows), but it badly worsens
  raw window `0` and the raw counterfactual margin
- increasing `lambda_change_readout` to `0.34` weakens aggregate raw/sign
  metrics, reduces raw positive windows to `3/5`, and makes the sign
  counterfactual margin much worse
- the anchor `0.30` remains the best observed value in this local neighborhood;
  next tuning should move to `lambda_delayed_change`, not continue readout-only
  sweeps

### Single-Factor Sweep: `lambda_delayed_change`

This sweep changed only `lambda_delayed_change` around the current best anchor
`stability_release_deconstrained`:

- anchor: `lambda_delayed_change = 0.28`
- low side: `lambda_delayed_change = 0.24`
- high side: `lambda_delayed_change = 0.32`

All model, train, other invariant, Stage III, and online fields were kept equal
to the stability anchor.

Low-side tuning config:

- `configs/stage3_v29_expert_local_invariant_release_probe_tune_delayed_change_024_only.yaml`

Low-side tuned `weather seed42 b60` screen:

- `logs/s3v29expert_tuning_screen/v29dc024_weather_s42_b60_01/weather/seed_42`

Low-side validation partial gain summary:

- `raw_residual_vector partial_mse_gain_vs_history = 0.2778015450214807`
- `raw_residual_vector partial_mae_gain_vs_history = 0.21024832722187548`
- `residual_change_vector partial_mse_gain_vs_history = 0.0`
- `residual_change_vector partial_mae_gain_vs_history = 0.0`
- `residual_sign partial_mse_gain_vs_history = 0.2842310667037964`
- `residual_sign partial_mae_gain_vs_history = 0.2967662811279297`

Low-side gate result:

- `raw_residual_vector positive_partial_window_count = 4/5`
- `raw_residual_vector min_partial_mse_gain_vs_history = -0.998103525805619`
- `raw_residual_vector counterfactual_margin = -0.998103525805619`
- `residual_change_vector positive_partial_window_count = 0/5`
- `residual_sign positive_partial_window_count = 4/5`
- `residual_sign counterfactual_margin = -0.32214784622192383`

High-side tuning config:

- `configs/stage3_v29_expert_local_invariant_release_probe_tune_delayed_change_032_only.yaml`

High-side tuned `weather seed42 b60` screen:

- `logs/s3v29expert_tuning_screen/v29dc032_weather_s42_b60_01/weather/seed_42`

High-side validation partial gain summary:

- `raw_residual_vector partial_mse_gain_vs_history = 0.48845339421202494`
- `raw_residual_vector partial_mae_gain_vs_history = 0.3558175390086717`
- `residual_change_vector partial_mse_gain_vs_history = 0.0`
- `residual_change_vector partial_mae_gain_vs_history = 0.0`
- `residual_sign partial_mse_gain_vs_history = 0.6566083133220673`
- `residual_sign partial_mae_gain_vs_history = 0.6184916794300079`

High-side gate result:

- `raw_residual_vector positive_partial_window_count = 4/5`
- `raw_residual_vector min_partial_mse_gain_vs_history = -1.4773854050120248`
- `raw_residual_vector counterfactual_margin = -1.4773854050120248`
- `residual_change_vector positive_partial_window_count = 0/5`
- `residual_sign positive_partial_window_count = 5/5`
- `residual_sign counterfactual_margin = -0.08246718347072601`

Interpretation:

- neither side passes the strict teacher gate
- lowering `lambda_delayed_change` to `0.24` weakens aggregate raw/sign metrics
  and worsens raw window `0` relative to the stability anchor
- increasing `lambda_delayed_change` to `0.32` improves aggregate raw/sign
  metrics but materially worsens raw window `0` and the raw counterfactual
  margin; the `residual_change_vector` remains collapsed
- `lambda_delayed_change = 0.28` remains the best local value under the strict
  window/counterfactual gate
- repeated single-axis sweeps now show that simple amplitude retuning is not
  fixing the first-window failure; the next step should be diagnostic, not
  another blind scalar sweep

### Tuning-Priority Correction Audit

After the delayed-change sweep, the user explicitly requested that the project
stop repeatedly changing model structure and focus on rigorous tuning first.
That request is now part of the V29 working constraint.

Detailed tuning-priority report:

- `docs/audit/STAGE3_V29_TUNING_PRIORITY_CORRECTION_REPORT.md`

Key finding:

- the V29 architecture direction remains more defensible than direct
  `phi_v - phi_i` subtraction because the native served state is
  `pooled_servable_change`
- the current training evidence does not yet prove deployable improvement:
  aggregate weather raw/sign gains exist, but strict teacher gates still fail
  because first-window raw correction is harmful and counterfactual margins are
  negative
- the current failure is a tuning/serving-safety problem before it is a new
  architecture problem

Window-level evidence behind this decision:

- best anchor:
  `logs/s3v29expert_tuning_screen/v29stable_weather_s42_b60_01/weather/seed_42`
- `teacher_only_validation_diagnostic_summary_v29.csv` shows aggregate
  `raw_residual_vector`/`fluctuation_plus_history` validation MSE gain
  `0.4800888639411266`
- `teacher_only_rolling_validation_window_summary_v29.csv` shows the same
  target/mode has first-window MSE gain `-0.7246091209150117`; the best
  counterfactual is `zero`
- `teacher_only_v29_target_gate.csv` shows `raw_residual_vector` fails the
  primary gate with positive windows `4/5`, min partial MSE
  `-0.1465061551651995`, and counterfactual margin `-0.7246091209150117`
- `residual_sign` has useful positive windows but remains secondary and still
  fails counterfactual safety

Practical consequence:

- freeze `models/factor_mechanism_v29.py`, `train/offline_factor_v29.py`, and
  the V29 Stage III boundary for the next block
- keep `stability_release_deconstrained` as the tuning anchor
- treat future changes as pre-registered tuning sweeps unless there is a new,
  documented root-cause finding that requires a narrow structural patch

### Constrained Two-Factor Candidate: `first042_null0055`

Following the tuning-priority audit, a non-structural two-factor candidate was
added:

- `configs/stage3_v29_expert_local_invariant_release_probe_tune_first042_null0055.yaml`

Only two loss weights differ from the current anchor:

- `lambda_release_first_increment = 0.42`
- `lambda_release_null = 0.055`

Rationale:

- `lambda_release_first_increment = 0.42` was the single-factor setting with the
  mildest observed raw first-window harm among recent release-first sweeps
- `lambda_release_null = 0.055` preserved stronger aggregate raw signal than
  the low-side null sweep, although it still failed the gate on its own
- combining these two factors is a constrained test of whether release amplitude
  and null regularization can improve first-window safety without changing
  architecture

Attempted run:

- command class: V29 weather seed42 matched-budget b60 screen
- run id: `v29first042null0055_weather_s42_b60_01`
- output path:
  `logs/s3v29expert_tuning_screen/v29first042null0055_weather_s42_b60_01/weather/seed_42`
- result: timed out before teacher-only diagnostics were written
- files observed after timeout:
  - `resolved_config.json`
  - `offline_metrics_v18.csv`
- missing files:
  - `teacher_only_validation_diagnostic_summary_v29.csv`
  - `teacher_only_rolling_validation_window_summary_v29.csv`
  - `teacher_only_v29_target_gate.csv`

Interpretation:

- this attempt is not a completed experiment
- it must not be used as positive or negative teacher-gate evidence
- the candidate remains queued for a full matched-budget run or a separately
  labeled short diagnostic run
- the timeout reinforces that future tuning should either use a controlled
  background run window or a clearly labeled diagnostic budget before expensive
  b60 screens

Execution root cause found:

- `--max-train-batches` limits offline training only
- Stage III train split tensor collection is controlled by
  `--max-generator-batches`
- the original `v29first042null0055_weather_s42_b60_01` timeout happened because
  `--max-generator-batches` was omitted, so the runner attempted to collect the
  full training split with `online_batch_size=1`
- subsequent runs must set both `--max-train-batches` and
  `--max-generator-batches` for matched-budget screens

Runtime triage:

- `v29first042null0055_weather_s42_e0g1_runtime02`
  - command class: no-training output-chain triage
  - limits: `epochs=0`, `max_generator_batches=1`, `max_eval_batches=1`,
    `validation_windows=1`
  - result: completed and wrote V29 teacher/gate/manifest files
  - purpose: validates the output chain only; no performance interpretation

Short diagnostic:

- `v29first042null0055_weather_s42_b60g60e1_diag04_hist`
  - limits: `epochs=1`, `max_train_batches=60`,
    `max_generator_batches=60`, `max_eval_batches=60`
  - feature modes: `fluctuation_plus_history`, `history_only`
  - result: useful diagnostic but not matched 5-epoch b60 evidence

Key diagnostic observations:

- aggregate raw `fluctuation_plus_history` validation MSE gain:
  `0.7254056242654647`
- aggregate raw partial MSE gain over `history_only`:
  `0.06249680328378038`
- rolling raw windows were real-MSE positive, including window `0`:
  `0.39300387720720026`
- the strict gate still failed because raw partial-vs-history was positive in
  only `2/5` windows and counterfactual margin remained negative
  (`-0.037411800179940835`)
- this diagnostic suggested the candidate was worth one formal matched-budget
  run, but not worth promotion

Completed matched-budget screen:

- `v29first042null0055_weather_s42_b60_02`
  - limits: `max_train_batches=60`, `max_generator_batches=60`,
    `max_eval_batches=60`, `validation_windows=5`
  - default targets and feature modes were used
  - output:
    `logs/s3v29expert_tuning_screen/v29first042null0055_weather_s42_b60_02/weather/seed_42`

Formal result:

- `raw_residual_vector` primary gate: fail
- `raw_residual_vector` positive partial windows: `1/5`
- `raw_residual_vector` min partial MSE gain over history:
  `-0.34518940793525116`
- `raw_residual_vector` counterfactual margin:
  `-0.48101357731444827`
- aggregate raw `fluctuation_plus_history` MSE gain:
  `0.2744594875034728`
- aggregate raw `history_only` MSE gain:
  `0.37503164388636623`
- `residual_change_vector`: still collapsed at `0.0`
- `residual_sign` secondary gate: fail, counterfactual margin
  `-0.17833828926086426`
- generator remained locked with `do_not_train_generator` and
  `no_primary_target_passed`

Interpretation:

- the two-factor candidate is not an improvement over the
  `stability_release_deconstrained` anchor under formal b60 criteria
- the diagnostic e1 signal does not transfer to the formal 5-epoch b60 screen
- combining `lambda_release_first_increment=0.42` and
  `lambda_release_null=0.055` should not be promoted
- the next tuning step should not continue this pair; it should instead target
  why longer training shifts raw correction back behind `history_only` and
  worsens first-window counterfactual safety

### Stability Anchor Epoch-Sensitivity Diagnostic

After the formal `first042_null0055` rejection, the next diagnostic returned to
the current anchor and changed only the number of offline training epochs.
The goal was to test whether the harmful first-window/counterfactual behavior is
mainly an overtraining or schedule-selection issue before changing model
structure.

Fixed anchor:

- `configs/stage3_v29_expert_local_invariant_release_probe_tune_stability_release_deconstrained.yaml`

Controlled budget:

- `max_train_batches=60`
- `max_generator_batches=60`
- `max_eval_batches=60`
- `validation_windows=5`
- default V29 targets and feature modes

Diagnostic outputs:

- `logs/s3v29expert_tuning_diag/v29stable_weather_s42_b60g60e1_epochdiag01/weather/seed_42`
- `logs/s3v29expert_tuning_diag/v29stable_weather_s42_b60g60e2_epochdiag01/weather/seed_42`
- formal 5-epoch comparator:
  `logs/s3v29expert_tuning_screen/v29stable_weather_s42_b60_01/weather/seed_42`

One-epoch result:

- `raw_residual_vector` primary gate: fail
- positive partial windows: `2/5`
- min partial MSE gain over history: `-0.34678779319025144`
- min partial MAE gain over history: `-0.43605292871442536`
- counterfactual margin: `-0.011714003971613818`
- aggregate raw `fluctuation_plus_history` MSE gain:
  `0.7365256503202612`
- aggregate raw `history_only` MSE gain:
  `0.6634583147010287`
- aggregate raw partial MSE gain over history:
  `0.07306733561923251`
- aggregate raw partial MAE gain over history:
  `-0.023886260849879992`
- aggregate raw partial sign gain over history:
  `0.10416662693023682`
- raw window `0` MSE gain:
  `0.47259663917691164`
- raw window `0` counterfactual margin:
  `0.001773456637841675`
- `residual_change_vector` still collapses to `0.0`

Two-epoch result:

- `raw_residual_vector` primary gate: fail
- positive partial windows: `2/5`
- min partial MSE gain over history: `-0.5914179524177163`
- min partial MAE gain over history: `-0.6223088508899188`
- counterfactual margin: `-0.31489352634826306`
- aggregate raw `fluctuation_plus_history` MSE gain:
  `0.6324920846738479`
- aggregate raw `history_only` MSE gain:
  `0.6444789182331099`
- aggregate raw partial MSE gain over history:
  `-0.011986833559262`
- aggregate raw partial MAE gain over history:
  `-0.0833125710728263`
- aggregate raw partial sign gain over history:
  `-0.005555570125579834`
- raw window `0` MSE gain:
  `0.2926844096148406`
- raw window `0` counterfactual margin:
  `0.0016081947232749472`
- raw window `4` counterfactual margin:
  `-0.31489352634826306`
- `residual_change_vector` still collapses to `0.0`

Comparison with the formal 5-epoch anchor:

- the formal 5-epoch anchor has stronger aggregate raw partial gain over
  history (`0.34745194792605627`) and reaches `4/5` positive raw partial
  windows
- the same 5-epoch anchor has a harmful first validation window:
  raw window `0` MSE gain `-0.7246091209150117`
- the 5-epoch raw counterfactual margin is also `-0.7246091209150117`, with
  `zero` as the best counterfactual in window `0`
- the one-epoch run is safer on window `0` and counterfactual margin, but it
  does not have enough positive partial windows and it loses on MAE
- the two-epoch run is dominated for this diagnostic purpose: it loses the
  one-epoch aggregate partial-MSE advantage while also worsening the
  counterfactual margin

Interpretation:

- epoch count is now a first-class tuning dimension for V29
- this supports the user's instruction to tune rigorously before changing
  structure again
- the issue is not "more epochs are better"; longer training can trade aggregate
  partial gain for first-window and counterfactual failure
- do not run an unconstrained epoch escalation as the next step
- if epoch tuning continues, it should use a predeclared gate-aware selection
  rule that scores first-window raw non-harm, counterfactual margin, positive
  partial-window count, and aggregate gain in that order
- no epoch diagnostic has unlocked generator training or produced a paper-grade
  improvement claim

### Gate-Aware Tuning Selector

A read-only selector was added to make V29 tuning decisions reproducible instead
of relying on manual comparison of CSV files.

Implementation:

- `train/analyze_stage3_v29_tuning_selection.py`

Test:

- `tests/test_stage3_v29_tuning_selection.py`

Validation:

- `pytest tests\test_stage3_v29_tuning_selection.py -q`
- result: `3 passed`

Generated artifacts:

- `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection.csv`
- `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection.md`
- `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection_windows.csv`
- `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection_windows.md`
- `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection_feature_mode_policy.csv`
- `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection_feature_mode_policy.md`
- `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection_mode_gate_labels.csv`
- `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection_mode_gate_labels.md`
- `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection_online_gate_features.csv`
- `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection_online_gate_features.md`
- `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection_skipped.csv`

Selector policy:

- require V29 teacher diagnostics before ranking a run
- skip incomplete run directories instead of treating them as negative evidence
- rank by a lexicographic safety-first tuple:
  1. strict target gate pass
  2. generator unlock
  3. non-harmful raw window `0`
  4. counterfactual margin
  5. positive partial-window count
  6. min partial MSE
  7. min partial MAE
  8. aggregate partial MSE
  9. aggregate partial MAE
  10. aggregate raw MSE

Current selection input:

- `v29stable_weather_s42_b60g60e1_epochdiag01`
- `v29stable_weather_s42_b60g60e2_epochdiag01`
- `v29stable_weather_s42_b60_01`
- `v29first042null0055_weather_s42_b60_02`

Current selection result:

- ranked runs: `5`
- skipped run directories: `0`
- rank `1`: `v29stable_weather_s42_b60g60e1_epochdiag01`
- label: `gate_near_diagnostic`
- target gate pass: `False`
- generator unlocked: `False`
- raw window `0` MSE gain: `0.472596639177`
- counterfactual margin: `-0.0117140039716`
- positive partial windows: `2/4`
- validation partial MSE gain over history: `0.0730673356192`

Interpretation:

- the selector agrees with the manual audit that the one-epoch stability anchor
  is the safest current diagnostic point
- it also makes the non-promotion decision explicit: the top-ranked run remains
  gate-negative and generator-locked
- this tool is a tuning-control artifact, not a model-performance result
- future V29 tuning should add runs to the selector before deciding whether to
  spend budget on larger screens or cross-seed checks

Window-level selector extension:

- the selector now writes a per-window breakdown for each ranked run
- this table joins `teacher_only_rolling_validation_window_summary_v29.csv` and
  `teacher_only_rolling_partial_gain_vs_history_v29.csv`
- diagnostic flags include:
  - `fh_mse_harmful`
  - `partial_mse_negative`
  - `partial_mae_negative`
  - `partial_mse_below_gate_floor`
  - `partial_mae_below_gate_floor`
  - `counterfactual_negative`

Current window-level diagnosis:

- rank `1` (`v29stable_weather_s42_b60g60e1_epochdiag01`) fails windows `0`,
  `1`, and `4` because `history_only` is stronger than
  `fluctuation_plus_history`
- the same rank `1` run has positive partial MSE on windows `2` and `3`, but
  both windows still have negative counterfactual margins
- rank `2` (`v29first042only_weather_s42_b60g60e1_diag01`) keeps the same
  failure shape: windows `0`, `1`, and `4` remain partial-negative, and windows
  `2` and `3` remain counterfactual-negative
- this explains why `lambda_release_first_increment=0.42` improves aggregate
  partial MSE but does not improve the gate: it strengthens the already-positive
  middle windows while leaving the history-dominated windows unresolved

Tuning implication:

- do not continue blind single-axis amplitude sweeps from this point
- the next candidate must explicitly address conditional use of
  `fluctuation_plus_history` versus `history_only`, or otherwise reduce damage
  in history-dominated windows without weakening windows `2` and `3`
- this is still a tuning/selection problem, not permission to create a new model
  structure

Conditional feature-mode policy diagnostic:

- the selector now also writes an oracle policy diagnostic across feature modes
- policies:
  - `fh_only`: current `fluctuation_plus_history` path
  - `history_protected_oracle`: choose `history_only` whenever it beats
    `fluctuation_plus_history`
  - `best_available_feature_oracle`: choose the best validation-window feature
    mode by MSE
  - `cf_safe_best_feature_oracle`: choose the best feature mode with
    nonnegative counterfactual margin
- all four policies are validation-oracle diagnostics, not online-available
  policies

Current policy finding:

- for rank `1` and rank `2`, even `history_protected_oracle` and
  `best_available_feature_oracle` remain `2/4` positive windows, so FH/history
  selection alone cannot pass the gate
- the formal 5-epoch anchor has a notable upper-bound signal:
  `cf_safe_best_feature_oracle` is gate-like passing with `4/4` positive
  windows, min partial MSE `0`, and min counterfactual margin `0`
- that passing oracle uses modes:
  `0:fluctuation_only;1:level_offset_plus_history;2:level_offset_plus_history;3:level_offset_plus_history;4:history_only`
- this suggests the next implementable tuning direction should be an
  online-available mode-gating or abstention policy across the existing V29
  teacher feature modes, not another scalar loss-weight sweep
- because the current policy is validation-oracle, it cannot be used as a
  manuscript result or deployment policy without a causal/online selection rule

Mode-gate oracle labels:

- the selector now writes a per-window oracle label table for future mode-gate
  design
- output:
  `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection_mode_gate_labels.md`
- rows: `25`
- key fields:
  - `best_available_feature_label`
  - `cf_safe_feature_label`
  - `history_protected_label`
  - `needs_non_fh_mode`
  - `label_source`
  - `online_label_available`
  - `online_gate_feature_status`

Important online boundary:

- every current label has `label_source = validation_window_outcome_oracle`
- every current label has `online_label_available = False`
- every current label has
  `online_gate_feature_status = not_defined_in_current_v29_artifact`
- therefore, V29 now has a clear supervision target for a possible mode gate,
  but not yet a deployable mode gate

Online mode-gate feature matrix:

- the selector now also writes a leakage-guarded feature matrix for a future
  online mode gate
- output:
  `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection_online_gate_features.md`
- rows: `25`
- the `feature_` namespace is restricted to pre-outcome metadata:
  - run/dataset/seed and validation-window schedule
  - available feature-mode set
  - train-fit signal contract diagnostics
  - train-fit residual-history contract diagnostics
  - resolved config fields such as lookback, horizon, and
    `lambda_release_first_increment`
- oracle targets stay under `oracle_label_`, including
  `oracle_label_best_available_feature`,
  `oracle_label_cf_safe_feature`, and
  `oracle_label_history_protected_feature`
- the regenerated artifact was checked for banned feature-column tokens
  (`gain`, `counterfactual`, `label`, `partial`, `teacher_real`, `outcome`,
  `best`) and found no leaking `feature_` columns
- every row currently records
  `feature_dynamic_signal_status = missing_current_serving_window_statistics`

Forward-dynamic feature collection hook:

- `train/stage3_v29_online_gate_features.py` defines
  `build_online_gate_dynamic_feature_row`
- `train/run_stage3_v29_expert_local_invariant_release_probe.py` exposes
  `collect_stage3_v29_online_gate_dynamic_feature_rows`
- the V18-compatible main runner now has a default no-op
  `write_online_gate_dynamic_features` hook, and the V29 wrapper installs
  `write_stage3_v29_online_gate_dynamic_features`
- therefore new V29 bounded probes write
  `stage3_v29_online_gate_dynamic_features.csv` after split forward collection
  and before teacher/validation outcome evaluation
- the V29 output renamer records that CSV in `run_manifest_v29.json` under
  `stage3_v29_online_gate_dynamic_features_csv`, including path, sha256,
  row count, pre-outcome scope, and the no-ground-truth/no-oracle leakage
  contract
- these utilities summarize model-forward state only:
  - router entropy/top-1/margin from `alpha`
  - local-environment entropy from `expert_local_env_posteriors`
  - release confidence/support/suppression
  - norm statistics for `pooled_servable_change`,
    `pooled_residual_release_state`, invariant correction, and prediction
- they do not accept ground-truth `y`, teacher gains, counterfactual outcomes,
  partial-vs-history results, or oracle labels
- the selector merges
  `stage3_v29_online_gate_dynamic_features.csv` into the online-gate feature
  matrix, but only for safe `feature_dynamic_` columns that avoid outcome-like
  names

Validation:

- `C:\Users\admin\anaconda3\python.exe -m pytest tests\test_stage3_v29_expert_local_invariant_release.py tests\test_stage3_v29_online_gate_features.py tests\test_stage3_v29_tuning_selection.py tests\test_stage3_v29_online_mode_gate.py -q`
- result: `27 passed`

Next implementation requirement:

- completed as a bounded smoke, not as performance evidence:
  `logs/s3v29expert_dynamic_smoke/v29dynamic_manifest_smoke_s42_b1w2_20260612_03/weather/seed_42`
- command scope: `epochs=1`, `max_train_batches=1`,
  `max_generator_batches=1`, `max_eval_batches=1`,
  `validation_windows=2`, target `raw_residual_vector`, feature mode
  `fluctuation_plus_history`, CPU
- the run writes `stage3_v29_online_gate_dynamic_features.csv`
  and `run_manifest_v29.json.stage3_v29_online_gate_dynamic_features_csv`
  with `row_count=1`, sha256, pre-outcome scope, and leakage contract
- selector smoke artifact:
  `artifacts/stage3_v29_dynamic_smoke_validation/v29dynamic_manifest_smoke_w2_selection_online_gate_features.md`
- selector result: `Feature rows: 1` and
  `feature_dynamic_signal_status=available_from_model_forward`
- interpretation guard: the smoke run remains gate-negative
  (`teacher_only_v29_target_gate.csv` reports
  `primary_gate_pass=False`) and generator-locked; it validates the artifact
  chain only
- next real implementation requirement:
  train/evaluate an online mode gate only on past/inner validation windows, then
  test on held-out future windows
- keep the oracle labels out of manuscript performance claims until an online
  selection rule reproduces useful behavior without future information

Online mode-gate v1:

- implementation: `train/stage3_v29_online_mode_gate.py`
- test: `tests/test_stage3_v29_online_mode_gate.py`
- model: standardized nearest-centroid classifier with confidence fallback
- inputs: selector `feature_` columns only, with outcome-like names removed by
  the same banned-token policy as the selector
- labels: `oracle_label_cf_safe_feature` is used only as supervision/evaluation
- split: current V29 tuning matrix uses validation windows `0-2` for training
  and windows `3-4` as held-out future windows
- current 25-row input:
  `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection_online_gate_features.csv`
- dynamic-smoke input:
  `artifacts/stage3_v29_dynamic_smoke_validation/v29dynamic_manifest_smoke_w2_selection_online_gate_features.csv`
- dynamic-smoke output:
  `artifacts/stage3_v29_dynamic_smoke_validation/v29dynamic_manifest_smoke_w2_online_mode_gate_summary.json`
  records `status=insufficient_data`, which is correct for a one-row smoke
- threshold sweep on the current 25-row matrix:
  - `min_confidence=0.00`: held-out `1/10`, fallback `0`
  - `min_confidence=0.15`: held-out `3/10`, fallback `8`
  - `min_confidence=0.25`: held-out `7/10`, fallback `12`
  - `min_confidence=0.35`: held-out `10/10`, fallback `15`
  - `min_confidence=0.45`: held-out `9/10`, fallback `21`
- best conservative artifact:
  `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection_online_mode_gate_conf035_summary.json`
- serving-path helpers:
  - `train.stage3_v29_online_mode_gate.load_online_mode_gate_policy`
  - `train.stage3_v29_online_mode_gate.apply_online_mode_gate_policy`
  - `train.online_loop_stage3_v29.select_stage3_v29_serving_mode`
- real artifact serving check:
  loading `current_v29_tuning_selection_online_mode_gate_conf035` and applying
  it to held-out
  `v29stable_weather_s42_b60g60e1_epochdiag01/window=4` without any
  `oracle_label_` columns selects `history_only`, records
  `confidence_below_threshold`, and reports `oracle_label_columns_seen=[]`
- rolling outcome replay artifact:
  `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection_online_mode_gate_conf035_outcome_replay_summary.json`
- replay scope: the same current `25` V29 weather tuning rows, scored against
  the corresponding `teacher_only_rolling_validation_window_summary_v29.csv`
  rows after serving-mode selection
- replay result:
  - selected mean MSE gain: `0.5506004468595593`
  - fixed `fluctuation_plus_history` candidate mean MSE gain:
    `0.5030690635309452`
  - fixed `history_only` mean MSE gain: `0.45593240636333965`
  - oracle cf-safe mean MSE gain: `0.5545271305345248`
  - selected harmful windows: `0`
  - fixed `fluctuation_plus_history` harmful windows: `2`
  - fixed `history_only` harmful windows: `1`
- served policy package:
  `artifacts/stage3_v29_served_mode_gate/v29_conf035_weather_served_mode_gate_manifest.json`
- package scope: `mode_selection_only_no_generator_unlock`
- admission result: `admission_pass=true`
- admission criteria:
  - `min_evaluated_rows=25`
  - `max_selected_harmful_windows=0`
  - selected mean MSE gain must be no worse than fixed
    `fluctuation_plus_history`
  - selected mean MSE gain must be no worse than fixed `history_only`
- manifest locks sha256 hashes for policy summary/model/predictions/report and
  outcome replay summary/rows/report
- interpretation: this supports a conservative fallback/abstention gate over
  current features, not an aggressive multi-mode correction policy and not a
  generator-unlock result

### Gate-Aware E1 Diagnostic: `release_first_042_only`

Following the selector result, one bounded single-factor diagnostic was run.
The goal was to test whether the `lambda_release_first_increment=0.42`
single-factor setting could preserve the one-epoch anchor's first-window safety
while improving partial-vs-history windows.

Config:

- `configs/stage3_v29_expert_local_invariant_release_probe_tune_release_first_042_only.yaml`

Command profile:

- `epochs=1`
- `max_train_batches=60`
- `max_generator_batches=60`
- `max_eval_batches=60`
- `validation_windows=5`

Output:

- `logs/s3v29expert_tuning_diag/v29first042only_weather_s42_b60g60e1_diag01/weather/seed_42`

Result:

- `raw_residual_vector` primary gate: fail
- positive partial windows: `2/4`
- min partial MSE gain over history: `-0.3471599666919558`
- min partial MAE gain over history: `-0.4361023426001065`
- counterfactual margin: `-0.01632314949103475`
- aggregate raw `fluctuation_plus_history` MSE gain:
  `0.7490209970538897`
- aggregate raw partial MSE gain over history:
  `0.0860759770012467`
- aggregate raw partial MAE gain over history:
  `-0.01657983364551907`
- raw window `0` MSE gain:
  `0.46851635401983394`
- raw window `0` counterfactual margin:
  `-0.008011704781347995`
- worst raw partial window moved to window `4`:
  `-0.3471599666919558`
- generator remained locked with `do_not_train_generator` and
  `no_primary_target_passed`

Updated selector result:

- `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection.md`
- ranked runs: `5`
- rank `1`: `v29stable_weather_s42_b60g60e1_epochdiag01`
- rank `2`: `v29first042only_weather_s42_b60g60e1_diag01`

Interpretation:

- `release_first_042_only` at one epoch is mixed but not an improvement over the
  selector anchor
- it slightly improves aggregate raw/partial MSE over the one-epoch stability
  anchor, but loses the safety-first ranking because counterfactual margin is
  worse and positive partial-window count is unchanged
- the run shows that simply increasing `lambda_release_first_increment` does not
  solve the partial-window problem; it shifts failure mass rather than unlocking
  the strict gate
- do not promote this candidate or spend larger-budget runs on it unless a
  future paired tuning rule specifically addresses late-window partial failure

## Current Scientific Interpretation

The current evidence says:

- `V29` is a complete, versioned implementation line, not a one-file toy
- it implements the user's structural idea more directly than `V28`
- deterministic tests and V28 regression tests pass
- the full Stage III probe path runs and writes V29 manifests/logs
- no V29 experiment has passed the strict teacher gate
- generator training remains locked because no primary target passed
- the first matched-budget evidence is empirically negative:
  - `weather seed42 b60` regresses on raw and sign targets
  - `panama seed42 b60` regresses on the change target
- the first narrow tuning pass changes the evidence:
  - `weather seed42 b60` becomes strongly positive on aggregate raw/sign
    targets but still fails counterfactual gate
  - `panama seed42 b60` is stabilized from negative change to zero change, but
    does not become positive or gate-passing
- the second guarded tuning pass is negative:
  - it keeps aggregate weather raw positive
  - it makes worst-window counterfactual margin materially worse
  - it should not replace `stability_release_deconstrained`
- the next two tuning passes are also negative:
  - `early_window_stability` improves aggregate raw gain but makes window `0`
    and counterfactual safety much worse
  - `stability_micro_cf` avoids catastrophic collapse but weakens aggregate
    raw/sign and worsens the first validation window versus
    `stability_release_deconstrained`
- the single-factor `lambda_counterfactual_shuffle` sweep around `0.08` is
  negative:
  - `0.07` improves some change/sign metrics but makes raw window `0`
    catastrophic
  - `0.09` improves aggregate raw gain but also makes raw window `0`
    catastrophic
  - `0.08` remains the best local value for raw worst-window stability
- the single-factor `lambda_release_null` sweep around `0.05` is negative:
  - `0.045` catastrophically harms raw window `0`
  - `0.055` improves aggregate raw gain but worsens raw worst-window safety and
    weakens the sign target
  - `0.05` remains the best local balance
- the single-factor `lambda_release_first_increment` sweep around `0.40` is
  mixed:
  - `0.38` improves aggregate raw gain but worsens raw window `0`
  - `0.41` gives the strongest aggregate raw gain in the local sweep but worsens
    the raw counterfactual margin to `-0.8293233156592138` and leaves
    `residual_change_vector` at `0/5`
  - `0.42` improves counterfactual margins but weakens aggregate raw and still
    fails the window `0` threshold
  - this factor is more informative than the previous two sweeps but still not
    gate-passing
- the single-factor `train.lr` sweep around `0.00038` is negative for strict
  promotion:
  - `0.00034` catastrophically worsens raw window `0` and counterfactual safety
  - `0.00042` strengthens aggregate raw/sign metrics but still fails the raw
    window `0` and counterfactual gates
  - this points away from further global learning-rate sweeps and toward
    signal-shaping weight tuning
- the single-factor `lambda_change_readout` sweep around `0.30` is also
  gate-negative:
  - `0.26` weakly reactivates `residual_change_vector` but severely worsens raw
    window `0` and counterfactual margin
  - `0.34` reduces raw positive windows to `3/5` and worsens sign safety
  - `0.30` remains the best local readout value
- the single-factor `lambda_delayed_change` sweep around `0.28` is also
  gate-negative:
  - `0.24` weakens aggregate raw/sign metrics and worsens raw window `0`
  - `0.32` improves aggregate raw/sign but worsens raw window `0` and
    counterfactual margin
  - `0.28` remains the best local delayed-change value
- the best current V29 tuning anchor is still
  `stability_release_deconstrained`
- the latest epoch-sensitivity diagnostic shows that a one-epoch stability
  anchor is safer on raw window `0` and counterfactual margin than the formal
  5-epoch anchor, but it still fails the strict gate; a two-epoch diagnostic is
  worse than the one-epoch diagnostic on aggregate partial gain and
  counterfactual safety
- the new gate-aware tuning selector ranks the one-epoch stability anchor first
  among the current five evidence-bearing weather runs, but labels it
  `gate_near_diagnostic`, not a promotable model
- the first post-selector single-factor e1 run,
  `release_first_042_only`, improves aggregate partial MSE but does not improve
  positive partial-window count and worsens counterfactual safety relative to
  the one-epoch stability anchor
- the new window-level selector artifact shows the current gate failure is
  window-conditional: windows `0`, `1`, and `4` are dominated by `history_only`,
  while windows `2` and `3` are mainly counterfactual-attribution failures
- the new feature-mode policy oracle shows FH/history switching is insufficient
  for the current top e1 diagnostics, but a counterfactual-safe all-mode oracle
  on the formal 5-epoch anchor has a gate-like pass; this is a design clue for
  online mode gating, not performance evidence
- the new mode-gate label artifact makes the next gap explicit: labels exist as
  validation oracles, but serving-time gate features are not yet defined in V29
- the newest online-gate feature matrix defines leakage-guarded static and
  contract-level `feature_` columns plus separate oracle labels, but it also
  records that the historical ranked runs do not yet contain dynamic
  serving-window CSVs
- the new forward-dynamic feature hook defines those dynamic rows without
  accepting ground truth, gains, counterfactual outcomes, or oracle labels; it
  is now wired into the V29 probe wrapper, but still needs a new bounded run
  before it can support an online gate experiment
- the newest tuning-priority audit says the model correction direction is
  structurally reasonable, but current gains are not effective enough for a
  teacher-passing, generator-unlocked, or paper-facing improvement claim

The defensible paper-facing statement is:

> `V29` successfully implements the expert-local invariant-release architecture
> requested by the user and removes the need to treat `phi_v - phi_i` as the
> native served drift definition. However, the first matched-budget experiments
> do not show gate-passing empirical improvement. A stability/deconstraint
> tuning pass recovers strong aggregate weather raw/sign signal, but the
> counterfactual gate remains negative and panama change-target evidence remains
> weaker than V28 support-push. Three later tuning attempts did not improve this
> anchor: stronger counterfactual/release-null pressure, broad
> early-window-stability regularization, and small counterfactual micro-tuning
> all worsened at least one required gate. A later single-factor sweep of
> `lambda_counterfactual_shuffle` also failed on both sides of the current
> anchor. A later single-factor sweep of `lambda_release_null` also failed to
> improve the strict gate. A later `lambda_release_first_increment` sweep found
> a trade-off, and the `0.41` midpoint improved aggregate raw gain, but no value
> fixed the first validation window or counterfactual gate. A later
> single-factor learning-rate sweep also failed to unlock the strict gate:
> lower learning rate was catastrophic and higher learning rate improved
> aggregate signal but not worst-window safety. A later
> `lambda_change_readout` sweep was also gate-negative on both sides of the
> anchor. A later `lambda_delayed_change` sweep was also gate-negative on both
> sides of the anchor. A later epoch-sensitivity diagnostic showed that
> training schedule matters: the one-epoch stability anchor reduces the
> first-window and counterfactual failure relative to the formal 5-epoch anchor,
> but still fails the strict teacher gate, while the two-epoch diagnostic is
> weaker than the one-epoch diagnostic. Current evidence supports V29 as a
> structurally cleaner and tunable research branch, not as a validated
> performance upgrade.

## Current Risks

1. The architecture is more semantically aligned with the user's proposal, but
   the first configuration appears over-constrained.
2. Direct residual-release state learning can still collapse to a non-useful
   signal, as shown by zero or negative primary partial gains on panama.
3. Local environment inference is now explicit, but the environment posterior
   may still be a nuisance proxy rather than a residual-causal condition.
4. Orthogonality and noncollapse losses enforce geometry, but they do not prove
   residual-target conditional increment.
5. The tuned weather result shows aggregate signal can be recovered, but the
   negative counterfactual margin means the served signal may be exploiting
   non-causal or unstable correlation.
6. Simply increasing counterfactual/release-null pressure can worsen the
   worst-window gate even when aggregate gain remains high.
7. Bundled temporal/anchor/budget regularization can inflate aggregate metrics
   while catastrophically harming the first validation window.
8. Micro counterfactual tuning can still degrade positive-window count and
   worst-window safety, so the search should move to single-factor sweeps.
9. The first single-factor sweep shows `lambda_counterfactual_shuffle` is not
   the immediate lever for fixing raw counterfactual safety; both `0.07` and
   `0.09` worsen raw window `0` versus the `0.08` anchor.
10. The second single-factor sweep shows `lambda_release_null` also is not the
    immediate lever; the high side improves aggregate raw but weakens safety,
    while the low side collapses the first validation window.
11. The `lambda_release_first_increment` sweep identifies a trade-off:
    increasing the weight can improve counterfactual margins but reduces raw
    aggregate gain, while the `0.41` midpoint improves raw aggregate gain but
    worsens counterfactual safety; none fixes the first validation window.
12. The learning-rate sweep rules out a simple optimizer-speed explanation:
    both `0.00034` and `0.00042` remain gate-failing, and neither recovers
    `residual_change_vector`.
13. The `lambda_change_readout` sweep shows readout pressure alone is not the
    missing lever: lower readout slightly reactivates change but harms raw
    safety, while higher readout harms raw/sign window consistency.
14. The `lambda_delayed_change` sweep shows delayed-change pressure alone is not
    the missing lever: higher delayed-change strengthens aggregate signal but
    worsens first-window safety, and lower delayed-change weakens signal without
    fixing the gate.
15. Repeated single-factor tuning now shows the main failure is not one scalar
    amplitude. The dominant current risk is first-window correction safety:
    aggregate validation gain can coexist with a harmful first served window.
16. Epoch count is a real tuning axis, but selecting epochs by aggregate
    validation gain alone can prefer a checkpoint with harmful first-window
    counterfactual behavior.
17. Online availability is implemented through the V26-compatible wrapper, but no
   generator-unlocked online run exists yet.
18. The current selector/replay artifacts are not yet synchronized with all
   completed V29 tune configs. The tuning queue records `18` V29 tune config
   rows: `15` are `completed_needs_selector_refresh`, and only `3` are
   `covered_by_selector`.
19. Latest bounded V29 regression after adding the tuning queue, full-queue
   selector guards, complete-only selector path, generator-unlock preflight,
   teacher-gate tuning matrix, and safe execution plan passes with `48 passed`:
   `tests/test_stage3_v29_tuning_queue.py`,
   `tests/test_stage3_v29_expert_local_invariant_release.py`,
   `tests/test_stage3_v29_online_gate_features.py`,
   `tests/test_stage3_v29_tuning_selection.py`, and
   `tests/test_stage3_v29_online_mode_gate.py`,
   `tests/test_stage3_v29_generator_unlock_preflight.py`, and
   `tests/test_stage3_v29_teacher_gate_tuning_matrix.py`, and
   `tests/test_stage3_v29_tuning_execution_plan.py`.
20. The refreshed full-queue selector now covers `24` weather V29 tune
   candidates and skips `1` incomplete runtime-triage directory. The top
   complete candidate remains `v29stable_weather_s42_b60g60e1_epochdiag01`,
   and it is still gate-negative and generator-locked.
21. Full-queue online mode-gate threshold tuning identifies `min_confidence=0.15`
   as the harm-first replay setting: selected mean MSE gain
   `0.2270443942984707`, fixed `fluctuation_plus_history`
   `0.22015735047664628`, fixed `history_only` `0.16830160575541872`, and
   selected harmful windows `0` versus `18`/`6` for fixed FH/history. The replay
   is still not promotable because `7` outcome rows are missing.
22. Complete-only selector filtering removes incomplete diagnostics before
   mode-gate replay. The resulting complete replay evaluates `110/110` rows
   with `0` missing outcomes. `min_confidence=0.15` remains harm-first best:
   selected mean MSE gain `0.25546113326914255`, fixed
   `fluctuation_plus_history` `0.2018502975357777`, fixed `history_only`
   `0.16830160575541872`, and selected harmful windows `0` versus `18`/`6`.
23. A new hash-locked complete-only served package exists at
   `artifacts/stage3_v29_served_mode_gate/v29_complete_conf015_weather_served_mode_gate_manifest.json`
   with `admission_pass=true`, but the scope remains
   `mode_selection_only_no_generator_unlock`.
24. A read-only generator-unlock preflight now exists at
   `train/stage3_v29_generator_unlock_preflight.py`, with artifacts
   `artifacts/stage3_v29_generator_unlock_preflight/v29_complete_conf015_generator_unlock_preflight.json`
   and
   `artifacts/stage3_v29_generator_unlock_preflight/v29_complete_conf015_generator_unlock_preflight.md`.
   The complete-only mode gate passes as a serving prerequisite, but the top
   selector candidate `v29stable_weather_s42_b60g60e1_epochdiag01` still fails
   generator unlock with `target_gate_not_passed`,
   `no_primary_gate_passed`, `teacher_abort_written`,
   `generator_not_unlocked`, `do_not_train_generator`,
   `no_primary_target_passed`, insufficient positive partial windows, partial
   MSE/MAE floor failures, and negative counterfactual margin. The decision is
   `do_not_train_generator`.
25. A non-training teacher-gate tuning matrix now converts those blockers into
   five frozen-architecture V29 config candidates. The planner is
   `train/plan_stage3_v29_teacher_gate_tuning_matrix.py`; artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_teacher_gate_tuning_matrix.csv`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_teacher_gate_tuning_matrix.md`;
   generated configs live under
   `configs/stage3_v29_teacher_gate_tuning_matrix/`. The five axes are
   `history_guarded_release_floor`, `counterfactual_margin_micro_repair`,
   `late_window_partial_floor_guard`, `delayed_change_tempered_history`, and
   `release_null_coupled_guard`. They are coupled scalar tuning probes around
   `stability_release_deconstrained`, not model-topology changes and not
   performance evidence.
26. A non-executing tuning execution plan now exists at
   `train/plan_stage3_v29_tuning_execution.py`. It writes
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_tuning_execution_plan.csv`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_tuning_execution_plan.md`.
   The plan assigns all five candidates to the new log root
   `logs/s3v29expert_teacher_gate_matrix` and checkpoint root
   `ckpt/s3v29expert_teacher_gate_matrix`, marks all five as `ready_to_run`,
   and omits `--overwrite` from every planned command. The plan now also omits
   `--dataset weather`, because `weather` is loaded from the config path rather
   than from the public-dataset override registry.
27. The first bounded teacher-gate matrix candidate completed at
   `logs/s3v29expert_teacher_gate_matrix/v29tg_conf015_01_history_guarded_release_floor/weather/seed_42`.
   It is a negative tuning result. Refreshed selector artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate01_selection.csv`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate01_selection.md`;
   refreshed preflight artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate01_generator_unlock_preflight.json`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate01_generator_unlock_preflight.md`.
   Candidate `01` ranks `3/23`, remains `primary_gate_pass=False`, and keeps
   the generator locked. Raw primary metrics are `2/4` positive partial windows,
   min partial MSE `-0.45713857481802944`, min partial MAE
   `-0.5075939740964291`, and counterfactual margin
   `-0.016909116092234222`. The preflight decision remains
   `do_not_train_generator`.
28. The second bounded teacher-gate matrix candidate completed at
   `logs/s3v29expert_teacher_gate_matrix/v29tg_conf015_02_counterfactual_margin_micro_repair/weather/seed_42`.
   It is also a negative tuning result. Refreshed selector artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate02_selection.csv`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate02_selection.md`;
   refreshed preflight artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate02_generator_unlock_preflight.json`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate02_generator_unlock_preflight.md`.
   Candidate `02` ranks `5/24`, remains `primary_gate_pass=False`, and keeps
   the generator locked. Raw primary metrics are `2/4` positive partial windows,
   min partial MSE `-0.46744894881173493`, min partial MAE
   `-0.5296254452103905`, and counterfactual margin
   `-0.038892373912445266`. The preflight decision remains
   `do_not_train_generator`.
29. The third bounded teacher-gate matrix candidate completed at
   `logs/s3v29expert_teacher_gate_matrix/v29tg_conf015_03_late_window_partial_floor_guard/weather/seed_42`.
   It is also a negative tuning result. Refreshed selector artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate03_selection.csv`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate03_selection.md`;
   refreshed preflight artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate03_generator_unlock_preflight.json`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate03_generator_unlock_preflight.md`.
   Candidate `03` ranks `4/25`, remains `primary_gate_pass=False`, and keeps
   the generator locked. Raw primary metrics are `2/4` positive partial windows,
   min partial MSE `-0.4671504578747007`, min partial MAE
   `-0.49524774880538697`, and counterfactual margin
   `-0.02420296088538021`. The preflight decision remains
   `do_not_train_generator`.
30. The fourth bounded teacher-gate matrix candidate completed at
   `logs/s3v29expert_teacher_gate_matrix/v29tg_conf015_04_delayed_change_tempered_history/weather/seed_42`.
   It is also a negative tuning result. Refreshed selector artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate04_selection.csv`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate04_selection.md`;
   refreshed preflight artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate04_generator_unlock_preflight.json`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate04_generator_unlock_preflight.md`.
   Candidate `04` ranks `7/26`, remains `primary_gate_pass=False`, and keeps
   the generator locked. Raw primary metrics are `2/4` positive partial windows,
   min partial MSE `-0.422131539589501`, min partial MAE
   `-0.500774945598131`, and counterfactual margin
   `-0.043278617034422484`. The preflight decision remains
   `do_not_train_generator`.
31. The fifth bounded teacher-gate matrix candidate completed at
   `logs/s3v29expert_teacher_gate_matrix/v29tg_conf015_05_release_null_coupled_guard/weather/seed_42`.
   It is also a negative tuning result. Refreshed selector artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate05_selection.csv`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate05_selection.md`;
   refreshed preflight artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate05_generator_unlock_preflight.json`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate05_generator_unlock_preflight.md`.
   Candidate `05` ranks `8/27`, remains `primary_gate_pass=False`, and keeps
   the generator locked. Raw primary metrics are `2/4` positive partial windows,
   min partial MSE `-0.4669403209264789`, min partial MAE
   `-0.4851098794798296`, and counterfactual margin
   `-0.043559999700157115`. The preflight decision remains
   `do_not_train_generator`.
32. The five-candidate frozen-architecture matrix is now exhausted. None of
   the predeclared coupled scalar candidates passed the primary teacher gate,
   and none changed the generator-unlock preflight decision. The repeated
   blockers remain positive partial-window insufficiency, partial MSE/MAE floor
   failures, and negative counterfactual margin. This is evidence to stop this
   local scalar matrix, not evidence to change the V29 structure again.
33. A read-only gate-failure decomposition tool now records why the matrix
   should stop before another tuning block is invented. The tool is
   `train/analyze_stage3_v29_gate_failure_decomposition.py`, with tests in
   `tests/test_stage3_v29_gate_failure_decomposition.py`. Real artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_matrix_gate_failure_decomposition.json`,
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_matrix_gate_failure_decomposition.csv`,
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_matrix_gate_failure_decomposition_windows.csv`,
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_matrix_gate_failure_decomposition.md`.
   The decomposition covers `27` complete weather candidates and reports
   `gate_pass_count=0`, `generator_unlocked_count=0`, and
   `matrix_decision=stop_local_scalar_matrix`. Window `0` fails partial MSE/MAE
   floors in `27/27` rows and is counterfactual-negative in `25/27`; window `2`
   has `0/27` partial-floor failures but still has `25/27`
   counterfactual-negative rows. This separates partial-floor instability from
   counterfactual-ordering failure and blocks generator training.
34. A decomposition-driven non-training repair matrix now converts the
   gate-failure diagnosis into the next predeclared candidate queue. The planner
   is `train/plan_stage3_v29_decomposition_repair_matrix.py`, with tests in
   `tests/test_stage3_v29_decomposition_repair_matrix.py` and implementation
   plan
   `docs/superpowers/plans/2026-06-13-stage3-v29-decomposition-driven-repair-matrix.md`.
   Generated configs live under
   `configs/stage3_v29_decomposition_repair_matrix/`. Matrix artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_matrix.csv`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_matrix.md`.
   The safe execution plan is
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_execution_plan.csv`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_execution_plan.md`.
   The four candidate axes are `window0_partial_floor_rebalance`,
   `counterfactual_ordering_without_floor_pressure`,
   `dual_axis_floor_and_cf_coupled`, and
   `mode_abstention_teacher_selection_probe`; all four are `ready_to_run` under
   non-overwriting bounded commands. These are next-run controls only, not
   training evidence. The YAML `teacher_selection_overrides` block is diagnostic
   metadata; current V29 training consumes the invariant overrides, not that
   metadata, as active loss changes.
35. The first decomposition-repair candidate completed at
   `logs/s3v29expert_decomposition_repair_matrix/v29repair_conf015_01_window0_partial_floor_rebalance/weather/seed_42`.
   It is a negative tuning result. Refreshed selector artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair01_selection.csv`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair01_selection.md`;
   refreshed preflight artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair01_generator_unlock_preflight.json`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair01_generator_unlock_preflight.md`;
   refreshed decomposition artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair01_gate_failure_decomposition.json`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair01_gate_failure_decomposition.md`.
   Candidate `01` ranks `9/28`, remains `primary_gate_pass=False`, and keeps
   the generator locked. Raw primary metrics are `2/4` positive partial windows,
   min partial MSE `-0.5455704944904651`, min partial MAE
   `-0.5699537539405484`, counterfactual margin `-0.04551155267896673`, and
   window0 FH MSE gain `0.2653257120709129`. The refreshed decomposition still
   reports `gate_pass_count=0`, `generator_unlocked_count=0`, and
   `matrix_decision=stop_local_scalar_matrix`. Window `0` floor failures are
   now `28/28`. The refreshed execution plan
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_execution_plan_after_repair01.csv`
   marks candidate `01` as `blocked_existing_output` and candidates `02`-`04`
   as `ready_to_run`.
36. The original long-run-id attempt for decomposition-repair candidate `02`
   failed before completion because of a Windows/PyTorch checkpoint path-length
   boundary. The failed path was the
   `teacher_v18_raw_residual_vector_fluctuationonly.pt` checkpoint under
   `ckpt/s3v29expert_decomposition_repair_matrix/weather/v29repair_conf015_02_counterfactual_ordering_without_floor_pressure/seed_42/`.
   The absolute path length was `263` characters; a shorter sibling checkpoint
   at `259` characters saved successfully, and a dummy `torch.save` to the
   exact failing path reproduced the failure. This partial run is invalid as
   model-quality evidence. The repair is now in the planners: short run slugs
   in `train/plan_stage3_v29_decomposition_repair_matrix.py`, plus checkpoint
   path-length prediction and `checkpoint_file_path_too_long` blocking in
   `train/plan_stage3_v29_tuning_execution.py`.
37. A path-safe decomposition-repair queue now supersedes the original long
   queue for future bounded repair runs. Generated configs live under
   `configs/stage3_v29_decomposition_repair_matrix_pathsafe/`; matrix artifacts
   are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_matrix_pathsafe.csv`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_matrix_pathsafe.md`.
   The path-safe execution plan is
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_execution_plan_pathsafe.csv`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_execution_plan_pathsafe.md`.
   The four run ids are `v29rps_conf015_01_w0floor`,
   `v29rps_conf015_02_cforder`, `v29rps_conf015_03_floorcf`, and
   `v29rps_conf015_04_abstain`. The execution planner reports a maximum
   expected checkpoint path length of `200`, below the enforced `259`-character
   limit.
38. The path-safe second decomposition-repair candidate completed at
   `logs/s3v29rps/v29rps_conf015_02_cforder/weather/seed_42`.
   It is also a negative tuning result. Refreshed selector artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair02_selection.csv`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair02_selection.md`;
   refreshed preflight artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair02_generator_unlock_preflight.json`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair02_generator_unlock_preflight.md`;
   refreshed decomposition artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair02_gate_failure_decomposition.json`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair02_gate_failure_decomposition.md`.
   Candidate `02` ranks `7/29`, remains `primary_gate_pass=False`, and keeps
   the generator locked. Raw primary metrics are `2/4` positive partial windows,
   min partial MSE `-0.4204164211957788`, min partial MAE
   `-0.500599967584385`, counterfactual margin `-0.04310910574869553`, and
   window0 FH MSE gain `0.3989539074568899`. The refreshed decomposition still
   reports `gate_pass_count=0`, `generator_unlocked_count=0`, and
   `matrix_decision=stop_local_scalar_matrix`. The updated path-safe execution
   plan
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_execution_plan_pathsafe_after_repair02.csv`
   marks candidate `02` as `blocked_existing_output`. Candidate `01` in that
   path-safe plan is equivalent to the already completed negative candidate
   `01`, so the meaningful remaining path-safe rows are `03` and `04`.
39. The path-safe third decomposition-repair candidate completed at
   `logs/s3v29rps/v29rps_conf015_03_floorcf/weather/seed_42`.
   It is also a negative tuning result. Refreshed selector artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair03_selection.csv`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair03_selection.md`;
   refreshed preflight artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair03_generator_unlock_preflight.json`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair03_generator_unlock_preflight.md`;
   refreshed decomposition artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair03_gate_failure_decomposition.json`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair03_gate_failure_decomposition.md`.
   Candidate `03` ranks `4/31`, remains `primary_gate_pass=False`, and keeps
   the generator locked. Raw primary metrics are `2/4` positive partial windows,
   min partial MSE `-0.5299312143911022`, min partial MAE
   `-0.545154931341216`, counterfactual margin `-0.024159312022691004`, and
   window0 FH MSE gain `0.2764126358705603`. The refreshed decomposition still
   reports `gate_pass_count=0`, `generator_unlocked_count=0`, and
   `matrix_decision=stop_local_scalar_matrix`. Window `0` partial MSE floor
   failures are now `31/31`; window `2` still has `0/31` partial-floor failures
   but `29/31` counterfactual-negative rows. The updated path-safe execution
   plan
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_execution_plan_pathsafe_after_repair03.csv`
   marks candidate `03` as `blocked_existing_output`. Candidate `04` is the
   only meaningful remaining path-safe row; candidate `01` is technically
   ready but should not be rerun because its long-run equivalent already
   completed negatively.
40. The path-safe fourth decomposition-repair candidate completed at
   `logs/s3v29rps/v29rps_conf015_04_abstain/weather/seed_42`.
   It is also a negative tuning result. Refreshed selector artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair04_selection.csv`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair04_selection.md`;
   refreshed preflight artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair04_generator_unlock_preflight.json`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair04_generator_unlock_preflight.md`;
   refreshed decomposition artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair04_gate_failure_decomposition.json`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair04_gate_failure_decomposition.md`.
   Candidate `04` ranks `6/32`, remains `primary_gate_pass=False`, and keeps
   the generator locked. Raw primary metrics are `2/4` positive partial windows,
   min partial MSE `-0.46285146776087893`, min partial MAE
   `-0.5138718209477093`, counterfactual margin `-0.03642005087344469`, and
   window0 FH MSE gain `0.35833191100094924`. The final refreshed
   decomposition reports `candidate_count=32`, `gate_pass_count=0`,
   `generator_unlocked_count=0`, and
   `matrix_decision=stop_local_scalar_matrix`. The final path-safe execution
   plan
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_execution_plan_pathsafe_after_repair04.csv`
   marks candidates `02`, `03`, and `04` as `blocked_existing_output`.
   Candidate `01` is technically ready only because its path-safe equivalent
   was not rerun; its original long-run equivalent is already negative. The
   path-safe repair matrix is now exhausted in the meaningful sense.
41. A post-repair history-gate recovery matrix now converts the exhausted
   repair diagnosis into a new predeclared mechanism test. The planner is
   `train/plan_stage3_v29_history_gate_recovery_matrix.py`, with tests in
   `tests/test_stage3_v29_history_gate_recovery_matrix.py` and implementation
   plan
   `docs/superpowers/plans/2026-06-13-stage3-v29-history-gate-recovery-matrix.md`.
   Generated configs live under
   `configs/stage3_v29_history_gate_recovery_matrix/`. Matrix artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_history_gate_recovery_matrix.csv`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_history_gate_recovery_matrix.md`.
   The safe execution plan is
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_history_gate_recovery_execution_plan.csv`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_history_gate_recovery_execution_plan.md`.
   The four axes are `window0_safe_release_ramp`,
   `late_history_dominance_bridge`,
   `counterfactual_tie_breaker_floor_safe`, and
   `history_cap_release_balance`. This matrix targets the observed
   history-dominance/partial-floor conflict and near-tie counterfactual margins;
   it does not change V29 topology and does not unlock generator training.
42. The first history-gate recovery candidate completed at
   `logs/s3v29hgr/v29hgr_conf015_01_w0ramp/weather/seed_42`.
   It is a negative tuning result. Refreshed selector artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr01_selection.csv`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr01_selection.md`;
   refreshed preflight artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr01_generator_unlock_preflight.json`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr01_generator_unlock_preflight.md`;
   refreshed decomposition artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr01_gate_failure_decomposition.json`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr01_gate_failure_decomposition.md`.
   Candidate `01` ranks `2/33`, remains `primary_gate_pass=False`, and keeps
   the generator locked. Raw primary metrics are `2/4` positive partial windows,
   min partial MSE `-0.43209738075898796`, min partial MAE
   `-0.4597064241028177`, counterfactual margin `-0.014453849719397449`, and
   window0 FH MSE gain `0.36675840890386036`. The refreshed decomposition
   still reports `gate_pass_count=0`, `generator_unlocked_count=0`, and
   `matrix_decision=stop_local_scalar_matrix`. The refreshed execution plan
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_history_gate_recovery_execution_plan_after_hgr01.csv`
   marks candidate `01` as `blocked_existing_output` and candidates `02`-`04`
   as `ready_to_run`.
43. The second history-gate recovery candidate completed at
   `logs/s3v29hgr/v29hgr_conf015_02_latehist/weather/seed_42`.
   It is also a negative tuning result. Refreshed selector artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr02_selection.csv`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr02_selection.md`;
   refreshed preflight artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr02_generator_unlock_preflight.json`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr02_generator_unlock_preflight.md`;
   refreshed decomposition artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr02_gate_failure_decomposition.json`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr02_gate_failure_decomposition.md`.
   Candidate `02` ranks `1/34`, but remains `primary_gate_pass=False` and
   keeps the generator locked. Raw primary metrics are `2/4` positive partial
   windows, min partial MSE `-0.5290475768525864`, min partial MAE
   `-0.5365663873681754`, counterfactual margin
   `-0.009411902842471953`, and window0 FH MSE gain
   `0.2906420799259992`. The refreshed decomposition reports
   `candidate_count=34`, `gate_pass_count=0`,
   `generator_unlocked_count=0`, and
   `matrix_decision=stop_local_scalar_matrix`. The refreshed execution plan
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_history_gate_recovery_execution_plan_after_hgr02.csv`
   marks candidates `01` and `02` as `blocked_existing_output` and candidates
   `03` and `04` as `ready_to_run`. Candidate `02` should not be promoted or
   used for generator training: its rank improves because the counterfactual
   margin is less negative, but its minimum partial floors are worse and the
   primary gate still fails.
44. The third history-gate recovery candidate completed at
   `logs/s3v29hgr/v29hgr_conf015_03_cftie/weather/seed_42`.
   It is also a negative tuning result. Refreshed selector artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr03_selection.csv`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr03_selection.md`;
   refreshed preflight artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr03_generator_unlock_preflight.json`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr03_generator_unlock_preflight.md`;
   refreshed decomposition artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr03_gate_failure_decomposition.json`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr03_gate_failure_decomposition.md`.
   Candidate `03` ranks `1/35`, but remains `primary_gate_pass=False` and
   keeps the generator locked. Raw primary metrics are `2/4` positive partial
   windows, min partial MSE `-0.41351019804173306`, min partial MAE
   `-0.466581375695994`, counterfactual margin `-0.006009365114644338`,
   and window0 FH MSE gain `0.3719791529173322`. The refreshed decomposition
   reports `candidate_count=35`, `gate_pass_count=0`,
   `generator_unlocked_count=0`, and
   `matrix_decision=stop_local_scalar_matrix`. The refreshed execution plan
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_history_gate_recovery_execution_plan_after_hgr03.csv`
   marks candidates `01`, `02`, and `03` as `blocked_existing_output`; only
   candidate `04` remains `ready_to_run`. Candidate `03` should not be promoted
   or used for generator training: it is the best selector-ranked row so far,
   but the primary gate still fails on positive-window count, partial floors,
   and counterfactual ordering.
45. The fourth history-gate recovery candidate completed at
   `logs/s3v29hgr/v29hgr_conf015_04_histcap/weather/seed_42`.
   It is also a negative tuning result and exhausts the predeclared HGR matrix.
   Refreshed selector artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr04_selection.csv`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr04_selection.md`;
   refreshed preflight artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr04_generator_unlock_preflight.json`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr04_generator_unlock_preflight.md`;
   refreshed decomposition artifacts are
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr04_gate_failure_decomposition.json`
   and
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr04_gate_failure_decomposition.md`.
   Candidate `04` ranks `9/36`, remains `primary_gate_pass=False`, and keeps
   the generator locked. Raw primary metrics are `2/4` positive partial
   windows, min partial MSE `-0.4066591740244906`, min partial MAE
   `-0.4354709147954243`, counterfactual margin
   `-0.030275156997368824`, and window0 FH MSE gain
   `0.3889744979968982`. The final refreshed decomposition reports
   `candidate_count=36`, `gate_pass_count=0`,
   `generator_unlocked_count=0`, and
   `matrix_decision=stop_local_scalar_matrix`. The final execution plan
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_history_gate_recovery_execution_plan_after_hgr04.csv`
   marks all four HGR candidates as `blocked_existing_output`. Candidate `03`
   remains the best selector-ranked HGR row, but no HGR row passes the primary
   teacher gate, and none may unlock generator training.

## Recommended Next Actions

1. Do not promote `V29` above `V28` empirically.
2. Keep `V29` as the official implementation of the user's structural idea.
3. Continue tuning `V29` before changing structure again; this is now the
   explicit working priority.
4. The next tuning target should keep the weather raw/sign recovery while
   improving counterfactual margin:
   - do not continue the `counterfactual_guarded` direction as-is
   - do not continue the `early_window_stability` bundled-regularization
     direction as-is
   - do not treat `stability_micro_cf` as an improvement over
     `stability_release_deconstrained`
   - focus on worst-window stabilization rather than global counterfactual
     pressure
   - inspect early validation-window behavior before increasing any amplitude
     or shuffle penalty
   - retune panama separately only if the same config can keep weather stable
5. Keep `stability_release_deconstrained` as the current V29 tuning anchor.
6. Stop the completed five-candidate teacher-gate scalar matrix:
   - do not rerun `history_guarded_release_floor`,
     `counterfactual_margin_micro_repair`, `late_window_partial_floor_guard`,
     `delayed_change_tempered_history`, or `release_null_coupled_guard` as
     blind scalar variants
   - keep the complete-only `conf015` served package only as a conservative
     mode-selection gate; do not treat it as generator-unlocked evidence
   - run the generator-unlock preflight before any generator experiment; if it
     returns `do_not_train_generator`, do not start generator training
   - next perform a higher-level diagnostic that explicitly explains why
     partial validation-window gains and counterfactual ordering conflict
   - score window-level raw non-harm, late-window partial floors, and
     counterfactual margin before aggregate gain
   - treat epoch count and early-stop selection as controlled tuning variables,
     not as a default increase from `1` to `5`
   - do not run an unconstrained `epochs=3` escalation unless it is part of a
     predeclared gate-aware epoch-sensitivity matrix
   - inspect per-window release diagnostics before changing representation
     losses
   - run seed checks only after a weather setting becomes gate-near
7. Use the new gate-failure decomposition artifact before designing the next
   matrix. A valid next tuning plan must state which mechanism is intended to
   reduce window `0` partial-floor failures and which mechanism is intended to
   make real signal beat counterfactuals in windows where partial floors are
   already satisfied.
8. Stop the path-safe decomposition-repair matrix. Use
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_execution_plan_pathsafe_after_repair04.csv`
   only as the final run-control record, not as a source for blind reruns. Do
   not use the original long-run-id queue. Do not rerun candidate `01` because
   its long-run equivalent already completed negatively, and do not rerun
   candidates `02`, `03`, or `04` because the path-safe runs are complete and
   negative. Any next run must come from a new predeclared diagnosis or matrix
   that changes the tested mechanism rather than repeating these axes.
9. Stop the history-gate recovery matrix. Use
   `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_history_gate_recovery_execution_plan_after_hgr04.csv`
   only as the final run-control record. Do not rerun candidates `01`-`04`,
   and do not train the generator from this matrix. Any next experiment must be
   a new predeclared diagnosis or matrix that changes the tested mechanism,
   because this matrix did not resolve positive-window shortfall, partial-floor
   violations, or counterfactual ordering.
10. Require at minimum:
   - one weather slice without raw/sign regression
   - one public-load slice with positive `residual_change_vector`
   - at least `4/5` positive validation windows for a primary target
   - nonnegative counterfactual margin
   - generator unlock

Until those conditions are met, V29 should be described as a correct structural
implementation with negative first empirical evidence, not as an accuracy
improvement.
