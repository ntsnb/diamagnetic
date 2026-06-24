# Stage III V29 Tuning-Priority Correction Report

Date: 2026-06-12

Scope: read-only audit and tuning guidance for the current `V29` expert-local
invariant-release line. This report intentionally does not propose another
model-structure change before stricter tuning and diagnostics are exhausted.

## Bottom Line

The current correction direction is structurally reasonable, but the current
evidence does not prove an effective deployable improvement.

The reason is specific:

- `V29` correctly implements the user's intended boundary: each expert starts
  from a shared anchor, performs local environment inference, constructs
  `phi_i`, learns a residual-release state, and exposes native
  `pooled_servable_change` instead of relying on a direct semantic claim that
  `phi_v - phi_i` is a valid drift vector.
- The best current tuning anchor restores strong aggregate weather signal
  compared with base `V29`, so the direction is not random noise.
- The strict Stage III teacher gate still fails because the served raw
  correction is harmful in the first validation window and loses to simple
  counterfactual controls. Therefore the improvement is not yet valid as a
  paper-grade result or an online-generator unlock result.

Operational decision:

- Freeze the V29 structure for now.
- Keep `stability_release_deconstrained` as the tuning anchor.
- Continue rigorous tuning and diagnostic sweeps before changing model
  architecture again.

## Code Boundary Audited

Model boundary:

- `models/factor_mechanism_v29.py:53` defines
  `ExpertLocalInvariantReleaseCorrector`.
- `models/factor_mechanism_v29.py:164` computes `local_env_posterior`.
- `models/factor_mechanism_v29.py:169` constructs `phi_i` from
  `shared_anchor`, invariant residual, correction gate, and invariant
  correction.
- `models/factor_mechanism_v29.py:172` computes `residual_release_state`.
- `models/factor_mechanism_v29.py:300` builds `expert_servable_change`.
- `models/factor_mechanism_v29.py:345` pools to `pooled_servable_change`.
- `models/factor_mechanism_v29.py:390` and `models/factor_mechanism_v29.py:391`
  keep legacy `delta_phi_v` aliases for compatibility, but the native served
  V29 signal remains `pooled_servable_change`.

Stage III boundary:

- `train/run_stage3_v29_expert_local_invariant_release_probe.py:40` validates
  the V29 config contract.
- `train/run_stage3_v29_expert_local_invariant_release_probe.py:47` requires
  `online.stage3.drift_source=pooled_servable_change`.
- `train/run_stage3_v29_expert_local_invariant_release_probe.py:187` collects
  the native `pooled_servable_change` tensor.
- `train/run_stage3_v29_expert_local_invariant_release_probe.py:245` delegates
  execution to the inherited V18 Stage III probe after adapting the V29
  artifacts.

Teacher-only diagnostics:

- `train/run_stage3_v18_structured_env_release_probe.py:544` writes validation
  summary rows.
- `train/run_stage3_v18_structured_env_release_probe.py:545` writes validation
  diagnostic summary rows.
- `train/run_stage3_v18_structured_env_release_probe.py:578` to
  `train/run_stage3_v18_structured_env_release_probe.py:581` write rolling
  validation diagnostics.
- `train/run_stage3_v18_structured_env_release_probe.py:611` computes target
  gate rows.
- `train/run_stage3_v18_structured_env_release_probe.py:622` writes
  `teacher_only_abort.json` when no primary teacher target passes.

Current tuning anchor:

- `configs/stage3_v29_expert_local_invariant_release_probe_tune_stability_release_deconstrained.yaml:48`
  sets `lr=0.00038`.
- `configs/stage3_v29_expert_local_invariant_release_probe_tune_stability_release_deconstrained.yaml:85`
  sets `lambda_release_first_increment=0.4`.
- `configs/stage3_v29_expert_local_invariant_release_probe_tune_stability_release_deconstrained.yaml:94`
  sets `lambda_change_readout=0.3`.
- `configs/stage3_v29_expert_local_invariant_release_probe_tune_stability_release_deconstrained.yaml:95`
  sets `lambda_delayed_change=0.28`.
- `configs/stage3_v29_expert_local_invariant_release_probe_tune_stability_release_deconstrained.yaml:104`
  sets `lambda_counterfactual_shuffle=0.08`.
- `configs/stage3_v29_expert_local_invariant_release_probe_tune_stability_release_deconstrained.yaml:158`
  keeps the online Stage III drift source at `pooled_servable_change`.

## Current Data Flow

1. The temporal encoder and V26-compatible backbone produce expert-local base
   representations.
2. For each expert, V29 forms `shared_anchor = expert_base +
   expert_invariant_adapter`.
3. Each expert infers a local environment posterior from the expert feature and
   anchor.
4. The invariant branch constructs `phi_i` by combining the anchor, invariant
   residual, and gated invariant correction.
5. The release branch learns `residual_release_state` conditioned on the local
   feature, local environment posterior, and `phi_i`.
6. V29 concatenates corrected invariant state, residual-release state, local
   environment posterior, support/suppression scores, and release diagnostics
   into `expert_servable_change`.
7. Router weights pool expert states into `pooled_servable_change`.
8. Stage III teacher diagnostics evaluate whether this signal improves residual
   prediction over history and whether the real signal beats counterfactual
   interventions.
9. If no primary target passes the strict teacher gate, generator training is
   aborted and no online improvement claim is available.

This is the right implementation boundary for the user's idea. The remaining
problem is not whether the code is still using direct `phi_v - phi_i` as the
native concept. It is not. The problem is that the learned release signal is
not yet stable enough across windows and counterfactual controls.

## Evidence From Current Best Anchor

Best anchor run:

`logs/s3v29expert_tuning_screen/v29stable_weather_s42_b60_01/weather/seed_42`

Signal/history scale:

- `stage3_v29_signal_contract_diagnostics.csv:2` shows
  `signal_scale_abs_mean=0.005911679472774267`.
- `stage3_v29_history_contract_diagnostics.csv:2` shows
  `history_scale_abs_mean=1.2024359703063965`.

This scale gap matters. The history feature block is much larger than the
normalized fluctuation signal, so tuning can easily produce a correction that
looks good in aggregate but is dominated by history context in early windows.

Aggregate validation diagnostic:

- `teacher_only_validation_diagnostic_summary_v29.csv:3` reports
  `raw_residual_vector` with `fluctuation_plus_history` has validation MSE gain
  `0.4800888639411266`, MAE gain `0.35939743700746857`, and real-vs-best
  counterfactual margin `0.0033674472025697066`.

Gate result:

- `teacher_only_v29_target_gate.csv:2` reports `raw_residual_vector` fails the
  primary gate: positive windows `4/5`, min partial MSE
  `-0.1465061551651995`, min partial MAE `-0.058653165834468446`, and
  counterfactual margin `-0.7246091209150117`.
- `teacher_only_v29_target_gate.csv:3` reports `residual_change_vector` fails
  with only `2/5` positive windows.
- `teacher_only_v29_target_gate.csv:4` reports `residual_sign` has `5/5`
  positive windows but remains secondary and fails counterfactual margin
  `-0.09309427440166473`.

First validation window:

- `teacher_only_rolling_validation_window_summary_v29.csv:3` shows
  `raw_residual_vector` with `fluctuation_plus_history` has first-window MSE
  gain `-0.7246091209150117`, MAE gain `-0.27465505576055543`, best
  counterfactual mode `zero`, and real-vs-best margin
  `-0.7246091209150117`.
- `teacher_only_rolling_validation_window_summary_v29.csv:4` shows
  `history_only` is also harmful in the same first window with MSE gain
  `-0.5781029657498122`.
- `teacher_only_rolling_validation_window_summary_v29.csv:13` shows
  `residual_sign` has useful first-window real MSE gain
  `0.664833277463913`, but it loses to `conditional_shuffled` with margin
  `-0.09309427440166473`.

Interpretation:

- The model has learned a useful aggregate direction.
- The raw served correction is unsafe at the first validation window.
- The safest counterfactual in the failing first window is often zero, meaning
  the model should sometimes abstain rather than always apply a correction.
- The sign branch contains useful information, but it is not yet robust enough
  to justify generator training because counterfactual variants can do better.

## What The Recent Tuning Sweeps Prove

The current sweeps are valuable because they rule out simple explanations.

Counterfactual loss only:

- Lower and higher `lambda_counterfactual_shuffle` did not fix the first-window
  raw failure. Both sides worsened the raw first-window/counterfactual margin
  relative to the anchor.

Release-null only:

- `lambda_release_null=0.045` became catastrophically unsafe in the first
  window.
- `lambda_release_null=0.055` improved some aggregate values but still failed
  first-window and counterfactual safety.

Release-first only:

- `lambda_release_first_increment=0.42` produced the mildest observed raw
  first-window loss among recent sweeps, but `raw_residual_vector` still failed
  with first-window MSE gain `-0.42083107598996183`.
- This suggests amplitude helps but does not solve the root mechanism.

Learning rate only:

- `lr=0.00034` and `lr=0.00042` both remained gate-negative.
- The issue is therefore not just under-training or over-training at the scalar
  learning-rate level.

Change-readout only:

- `lambda_change_readout=0.26` weakly reactivated `residual_change_vector`, but
  worsened first-window raw safety.
- `lambda_change_readout=0.34` reduced raw positive-window count and worsened
  sign counterfactual behavior.

Delayed-change only:

- `lambda_delayed_change=0.24` weakened aggregate raw/sign and still failed
  first-window safety.
- `lambda_delayed_change=0.32` strengthened aggregate raw/sign but worsened raw
  first-window MSE to `-1.4773854050120248`.

Conclusion from sweeps:

- Current V29 is not failing because one obvious scalar is set incorrectly.
- Blindly changing more weights one at a time will probably keep generating
  aggregate gains with hidden first-window failure.
- The next phase must tune under explicit hard constraints: first-window
  non-harm, counterfactual margin, primary target pass, and cross-dataset
  retention.

## Correctness Of The Current Correction Direction

The direction is correct at the representation-design level:

- It avoids treating direct vector subtraction as a guaranteed meaningful drift
  object.
- It keeps most representations shared, making alignment and regularization
  easier than fully independent expert streams.
- It places environment inference inside each expert, which is a more plausible
  place to learn expert-local invariance than a single global environment head.
- It exposes a served state package rather than a naked latent difference,
  which is more compatible with Stage III teacher/generator diagnostics.

The direction is not yet proven correct at the empirical-deployment level:

- A correction that improves aggregate validation but harms the first online
  window is not safe.
- A branch that loses to shuffled/sign-flipped/zero counterfactuals is not
  causally persuasive.
- A primary target that fails the gate cannot unlock generator training.
- A single weather seed result cannot support a manuscript accuracy claim.

Therefore the precise answer is:

> The model correction direction is mathematically and implementation-wise more
> defensible than direct `phi_v - phi_i`, but the current trained correction is
> not yet effective enough to claim real Stage III improvement.

## Why More Model Changes Should Wait

Changing the model now would make the evidence harder to interpret.

Current unresolved questions are tuning and validation questions:

- Can the existing V29 structure produce a non-harmful first validation window?
- Can `raw_residual_vector` beat zero and shuffled counterfactuals window by
  window?
- Can `residual_change_vector` be made non-collapsed without destroying raw
  safety?
- Can the sign branch help the primary target instead of remaining a secondary
  diagnostic?
- Can the anchor retain weather aggregate gains without weakening panama?

None of these questions requires another architecture rewrite first. They
require disciplined hyperparameter search and gate-aware diagnostics.

## Minimum-Risk Tuning Specification

Freeze for the next tuning block:

- Keep `models/factor_mechanism_v29.py` unchanged.
- Keep `train/offline_factor_v29.py` unchanged.
- Keep the V29 runner contract unchanged.
- Keep `pooled_servable_change` as the native Stage III drift source.
- Keep `stability_release_deconstrained` as the baseline anchor.

Hard promotion criteria before any generator run:

- `raw_residual_vector` primary gate passes.
- `residual_change_vector` is positive in at least `4/5` windows or is
  explicitly demoted from the primary claim.
- First validation window raw MSE gain is non-negative or within a very small
  documented tolerance.
- Real signal beats `zero`, shuffled, conditional-shuffled, sign-flipped, and
  amplified controls on rolling validation.
- Panama does not regress below the current tuned anchor.
- The same tuning choice is checked on at least two seeds before any manuscript
  claim.

Recommended next tuning block:

1. Do a small two-factor grid around the most informative current result, not a
   broad random search.
2. Use `lambda_release_first_increment` near `0.40` to `0.43` because this axis
   showed the mildest first-window raw harm.
3. Pair it with conservative counterfactual/readout pressure rather than
   changing either one alone.
4. Score candidates by strict ordered priority: first-window raw non-harm,
   raw counterfactual margin, raw positive-window count, residual-sign
   counterfactual margin, then aggregate validation gain.
5. Stop a candidate immediately if first-window raw MSE is below the current
   anchor or if the best counterfactual is `zero` with a strongly negative
   margin.

Example search discipline:

- Do not change more than two loss weights per config.
- Keep `lr=0.00038` unless a later matrix specifically studies optimizer
  sensitivity.
- Keep `lambda_delayed_change=0.28` for now; both `0.24` and `0.32` were
  gate-negative.
- Do not continue readout-only or delayed-change-only sweeps until the
  first-window failure is better understood.
- Record every run in this audit ledger, including negative results.
- For every bounded Stage III tuning run, set `--max-train-batches`,
  `--max-generator-batches`, and `--max-eval-batches` together. The
  `first042_null0055` timeout showed that `--max-train-batches` alone does not
  bound Stage III train split tensor collection.

Latest bounded candidate result:

- `first042_null0055` changed only `lambda_release_first_increment=0.42` and
  `lambda_release_null=0.055`.
- The completed formal weather seed42 b60 screen
  `v29first042null0055_weather_s42_b60_02` is gate-negative.
- It should not replace `stability_release_deconstrained` as the current anchor.
- The useful lesson is that a one-epoch diagnostic can show strong raw signal,
  but the full 5-epoch matched-budget screen can still shift the correction
  behind `history_only` and worsen counterfactual safety.

Latest epoch-sensitivity diagnostic:

- The diagnostic returned to
  `configs/stage3_v29_expert_local_invariant_release_probe_tune_stability_release_deconstrained.yaml`
  and changed only the offline epoch count.
- Both runs used `max_train_batches=60`, `max_generator_batches=60`,
  `max_eval_batches=60`, and `validation_windows=5`.
- `epochs=1` output:
  `logs/s3v29expert_tuning_diag/v29stable_weather_s42_b60g60e1_epochdiag01/weather/seed_42`.
- `epochs=1` keeps raw window `0` non-harmful (`0.47259663917691164`) and has
  near-zero raw counterfactual failure (`-0.011714003971613818`), but it still
  fails the gate with only `2/5` positive partial windows and negative aggregate
  partial MAE (`-0.023886260849879992`).
- `epochs=2` output:
  `logs/s3v29expert_tuning_diag/v29stable_weather_s42_b60g60e2_epochdiag01/weather/seed_42`.
- `epochs=2` is worse for the current diagnostic purpose: aggregate raw partial
  MSE becomes negative (`-0.011986833559262`), aggregate raw partial MAE is
  negative (`-0.0833125710728263`), and raw counterfactual margin worsens to
  `-0.31489352634826306`.
- The formal 5-epoch anchor has stronger aggregate raw partial MSE
  (`0.34745194792605627`) and `4/5` positive partial windows, but it fails raw
  window `0` (`-0.7246091209150117`) and raw counterfactual margin
  (`-0.7246091209150117`).
- Conclusion: epoch count is a real tuning dimension, but "more epochs" is not
  a safe default. Future epoch tuning must use a predeclared gate-aware
  selection rule instead of selecting by aggregate validation gain alone.

Gate-aware selection tool:

- Added `train/analyze_stage3_v29_tuning_selection.py`.
- Added `tests/test_stage3_v29_tuning_selection.py`.
- Test command: `pytest tests\test_stage3_v29_tuning_selection.py -q`.
- Test result: `3 passed`.
- Current output:
  `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection.md`.
- The selector ranks completed V29 runs by teacher gate pass, generator unlock,
  raw window `0` non-harm, counterfactual margin, positive partial-window count,
  min partial MSE/MAE, and aggregate partial/raw gains.
- Current ranked input set:
  `v29stable_weather_s42_b60g60e1_epochdiag01`,
  `v29stable_weather_s42_b60g60e2_epochdiag01`,
  `v29first042only_weather_s42_b60g60e1_diag01`,
  `v29first042null0055_weather_s42_b60_02`, and
  `v29stable_weather_s42_b60_01`.
- Current rank `1` is
  `v29stable_weather_s42_b60g60e1_epochdiag01` with label
  `gate_near_diagnostic`.
- The rank `1` run is still not promotable: target gate pass is `False`,
  generator unlocked is `False`, and positive partial windows are only `2/4`.
- This formalizes the next tuning policy: use selector-ranked safety evidence
  before spending budget on larger screens, and do not choose a candidate solely
  because aggregate validation partial MSE is larger.

Post-selector single-factor diagnostic:

- Run:
  `v29first042only_weather_s42_b60g60e1_diag01`.
- Config:
  `configs/stage3_v29_expert_local_invariant_release_probe_tune_release_first_042_only.yaml`.
- Budget: `epochs=1`, `max_train_batches=60`, `max_generator_batches=60`,
  `max_eval_batches=60`, `validation_windows=5`.
- Result: still gate-negative and generator-locked.
- Raw primary positive partial windows remain `2/4`.
- Aggregate raw partial MSE improves to `0.0860759770012467`, but raw
  counterfactual margin worsens to `-0.01632314949103475` versus the one-epoch
  stability anchor's `-0.011714003971613818`.
- Raw window `0` remains non-harmful (`0.46851635401983394`), but its
  counterfactual margin is negative (`-0.008011704781347995`).
- Worst raw partial window is now window `4` at `-0.3471599666919558`.
- Updated selector ranks this run second, below
  `v29stable_weather_s42_b60g60e1_epochdiag01`.
- Conclusion: increasing `lambda_release_first_increment` to `0.42` under the
  one-epoch schedule improves aggregate partial MSE but does not fix the strict
  gate; it should not be promoted or escalated to larger-budget experiments
  without a separate late-window partial-failure control.

Window-level selector extension:

- Current output:
  `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection_windows.md`.
- The artifact contains `25` rows: five validation windows for each of five
  ranked V29 weather runs.
- For the rank `1` one-epoch stability anchor, windows `0`, `1`, and `4` fail
  because `history_only` outperforms `fluctuation_plus_history`; windows `2` and
  `3` have positive partial MSE but negative counterfactual margins.
- For the rank `2` `release_first_042_only` diagnostic, the failure shape is
  nearly unchanged: windows `0`, `1`, and `4` remain partial-negative, and
  windows `2` and `3` remain counterfactual-negative.
- This means the immediate blocker is not simply insufficient amplitude.
  Increasing release-first pressure mostly strengthens already-positive middle
  windows and leaves history-dominated windows unresolved.
- Next tuning should target conditional feature-mode use or history-dominated
  window protection. Do not continue blind single-axis amplitude sweeps unless a
  candidate explicitly protects windows `0`, `1`, and `4`.

Conditional feature-mode policy oracle:

- Current output:
  `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection_feature_mode_policy.md`.
- For the rank `1` one-epoch stability anchor, `history_protected_oracle` and
  `best_available_feature_oracle` both remain `2/4` positive partial windows,
  so FH/history switching alone does not solve the strict gate.
- For rank `2` `release_first_042_only`, the same result holds: FH/history
  protection remains `2/4`, and counterfactual margin is weaker than rank `1`.
- The formal 5-epoch anchor exposes an upper-bound direction:
  `cf_safe_best_feature_oracle` is gate-like passing with `4/4` positive
  windows, min partial MSE `0`, and min counterfactual margin `0`.
- That oracle selects
  `0:fluctuation_only;1:level_offset_plus_history;2:level_offset_plus_history;3:level_offset_plus_history;4:history_only`.
- This is not a deployable result because it uses validation-window outcomes to
  select modes. It only says the next useful implementation work should be an
  online-available mode gate or abstention policy over existing V29 feature
  modes, not another single scalar sweep.

Mode-gate label artifact:

- Current output:
  `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection_mode_gate_labels.md`.
- It contains `25` per-window oracle labels for the five currently ranked V29
  weather runs.
- The artifact records `best_available_feature_label`,
  `cf_safe_feature_label`, `history_protected_label`, and whether the oracle
  requires a non-`fluctuation_plus_history` mode.
- Every row explicitly records
  `label_source=validation_window_outcome_oracle`.
- Every row also records `online_label_available=False` and
  `online_gate_feature_status=not_defined_in_current_v29_artifact`.
- Therefore, the next step is not to report the oracle as a result; it is to
  define serving-time gate features and evaluate whether a causal gate can
  approximate the oracle without future-window outcomes.

Online gate feature matrix:

- Current output:
  `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection_online_gate_features.md`.
- It contains `25` rows aligned with the same five ranked V29 weather runs and
  five validation windows per run.
- Columns with the `feature_` prefix are limited to pre-outcome metadata:
  validation-window schedule, available feature-mode set, train-fit signal
  contract diagnostics, train-fit history contract diagnostics, and resolved
  static config fields.
- Oracle labels are separated under `oracle_label_`; they are not online
  inputs.
- A leakage check found no `feature_` columns containing `gain`,
  `counterfactual`, `label`, `partial`, `teacher_real`, `outcome`, or `best`.
- Every row still records
  `feature_dynamic_signal_status=missing_current_serving_window_statistics`.
- The code now has a forward-dynamic collection hook:
  `train/stage3_v29_online_gate_features.py::build_online_gate_dynamic_feature_row`
  and
  `train/run_stage3_v29_expert_local_invariant_release_probe.py::collect_stage3_v29_online_gate_dynamic_feature_rows`.
- The V29 wrapper now installs
  `write_stage3_v29_online_gate_dynamic_features` into the V18-compatible
  probe hook, so new bounded V29 probes write
  `stage3_v29_online_gate_dynamic_features.csv` before teacher/validation
  outcome evaluation.
- The V29 output renamer also records that CSV in `run_manifest_v29.json` under
  `stage3_v29_online_gate_dynamic_features_csv`, with path, sha256, row count,
  pre-outcome scope, and the no-ground-truth/no-oracle leakage contract.
- That hook summarizes model-forward state only, including router entropy,
  release confidence, local-environment entropy, servable-change norm, release
  state norm, and correction statistics. It does not accept ground truth,
  gains, counterfactual outcomes, partial-vs-history results, or oracle labels.
- Therefore the current historical tuning artifact remains a safe input
  specification and leakage guard, not a trained online gate. New bounded or
  formal tuning probes must use this code and have their manifests checked
  before an online selector can be evaluated as a deployment policy.

Bounded dynamic-smoke validation:

- Completed run:
  `logs/s3v29expert_dynamic_smoke/v29dynamic_manifest_smoke_s42_b1w2_20260612_03/weather/seed_42`.
- Scope: `epochs=1`, `max_train_batches=1`, `max_generator_batches=1`,
  `max_eval_batches=1`, `validation_windows=2`, target
  `raw_residual_vector`, feature mode `fluctuation_plus_history`, CPU.
- The run writes `stage3_v29_online_gate_dynamic_features.csv` and records it
  in `run_manifest_v29.json.stage3_v29_online_gate_dynamic_features_csv` with
  path, sha256, `row_count=1`, pre-outcome scope, and leakage contract.
- Smoke selector artifact:
  `artifacts/stage3_v29_dynamic_smoke_validation/v29dynamic_manifest_smoke_w2_selection_online_gate_features.md`.
- Smoke selector result: `Feature rows: 1` and
  `feature_dynamic_signal_status=available_from_model_forward`.
- This is not performance evidence: `teacher_only_v29_target_gate.csv` reports
  `primary_gate_pass=False`, and the generator remains locked.

Online mode-gate v1 implementation and threshold tuning:

- Added `train/stage3_v29_online_mode_gate.py`.
- Added `tests/test_stage3_v29_online_mode_gate.py`.
- Gate design: standardized nearest-centroid classifier over selector
  `feature_` columns only. Columns containing outcome-like tokens are removed
  with the same banned-token policy used by the selector. `oracle_label_`
  columns are used only as supervision/evaluation labels.
- Split design: global validation-window order, with windows `0-2` used for
  training and windows `3-4` held out when
  `--heldout-tail-windows 2` is used on the current `25`-row V29 weather
  selector matrix.
- Main input:
  `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection_online_gate_features.csv`.
- Dynamic-smoke input:
  `artifacts/stage3_v29_dynamic_smoke_validation/v29dynamic_manifest_smoke_w2_selection_online_gate_features.csv`.
- Dynamic-smoke result:
  `artifacts/stage3_v29_dynamic_smoke_validation/v29dynamic_manifest_smoke_w2_online_mode_gate_summary.json`
  records `status=insufficient_data`, as expected for a one-row smoke.
- Current tuning matrix result at `min_confidence=0.0`:
  `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection_online_mode_gate_summary.json`
  records train accuracy `0.8666666666666667`, held-out accuracy `0.1`, and
  `0` fallback predictions.
- Conservative fallback result at `min_confidence=0.35`:
  `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection_online_mode_gate_conf035_summary.json`
  records train accuracy `0.9333333333333333`, held-out accuracy `1.0`,
  held-out correct `10/10`, and `15` total fallback predictions.
- Serving-path interface:
  `train.online_loop_stage3_v29.select_stage3_v29_serving_mode` now delegates to
  the V29 online mode-gate policy without changing the inherited V26 online
  adaptation loop. This keeps the current no-generator-unlock boundary intact.
- Serving-path smoke with the real `conf035` artifact:
  held-out `v29stable_weather_s42_b60g60e1_epochdiag01/window=4`, with all
  `oracle_label_` fields removed, selects `history_only` with
  `reject_reason=confidence_below_threshold`,
  `online_gate_confidence=0.21649074285327313`, and
  `oracle_label_columns_seen=[]`.
- Rolling outcome replay:
  `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection_online_mode_gate_conf035_outcome_replay_summary.json`
  maps the serving-selected mode back to the corresponding
  `teacher_only_rolling_validation_window_summary_v29.csv` outcome rows for
  the same five V29 weather tuning runs.
- Replay result: `25/25` rows evaluated, selected mean MSE gain
  `0.5506004468595593`, fixed `fluctuation_plus_history` candidate mean MSE
  gain `0.5030690635309452`, fixed `history_only` mean MSE gain
  `0.45593240636333965`, and oracle cf-safe mean MSE gain
  `0.5545271305345248`.
- Harm control: selected-mode harmful windows are `0`, versus `2` for fixed
  `fluctuation_plus_history` and `1` for fixed `history_only`.
- Interpretation: the serving policy is now measurable against rolling teacher
  outcomes and does reduce harmful correction in this replay. It is still an
  offline teacher-outcome replay, not a real online forecasting run or
  generator-unlock result.
- Served policy package:
  `artifacts/stage3_v29_served_mode_gate/v29_conf035_weather_served_mode_gate_manifest.json`.
- Package scope: `mode_selection_only_no_generator_unlock`.
- Admission criteria: at least `25` evaluated rows, selected harmful windows
  `<= 0`, selected mean MSE gain no worse than fixed
  `fluctuation_plus_history`, and selected mean MSE gain no worse than fixed
  `history_only`.
- Admission result: `admission_pass=true`, with manifest-locked sha256 hashes
  for policy summary/model/predictions/report and outcome replay
  summary/rows/report.
- Serving load smoke: loading the manifest succeeds and the same held-out
  `v29stable_weather_s42_b60g60e1_epochdiag01/window=4` row without
  `oracle_label_` columns selects `history_only` with
  `confidence_below_threshold`.
- Threshold sweep summary:
  `0.00 -> 0.1` held-out accuracy, `0` fallbacks;
  `0.15 -> 0.3`, `8` fallbacks;
  `0.25 -> 0.7`, `12` fallbacks;
  `0.35 -> 1.0`, `15` fallbacks;
  `0.45 -> 0.9`, `21` fallbacks.
- Interpretation: current online-available features support a conservative
  abstention/fallback gate more than an aggressive multi-mode correction gate.
  The best held-out result comes mostly from falling back to `history_only`;
  it does not unlock generator training or prove Stage III forecasting
  improvement.

Tuning queue coverage audit:

- Added non-training queue builder:
  `train/plan_stage3_v29_tuning_queue.py`.
- Added tests:
  `tests/test_stage3_v29_tuning_queue.py`.
- Queue artifacts:
  `artifacts/stage3_v29_tuning_queue/current_v29_tuning_queue.csv` and
  `artifacts/stage3_v29_tuning_queue/current_v29_tuning_queue.md`.
- Protocol: `stage3_v29_tuning_queue_v1`.
- Matching rule: candidate YAML files are matched to completed V29 runs by
  direct `run_id` when available, and by resolved-config tuning signature when
  the actual run used CLI `--run-id` overrides.
- Current queue result: `18` V29 tune config rows; `15` are
  `completed_needs_selector_refresh`, and `3` are `covered_by_selector`.
- Practical implication: the current selector/replay artifacts cover only a
  subset of already completed tuning runs. Do not launch another tuning run
  until the selector is refreshed over the full completed queue; otherwise the
  next search risks duplicating already negative axes or ignoring completed
  evidence.
- This queue is a run-control artifact only. It is not performance evidence,
  does not unlock the generator, and does not change the served package scope.

Full-queue selector refresh and threshold sweep:

- Queue matching was tightened after audit: resolved-config signature matching
  now excludes non-tuning roots such as dynamic-smoke runs and requires dataset
  compatibility, so weather tune configs do not absorb panama runs.
- Selector ranking was tightened with `selector_metric_complete`; rows with
  missing/nan validation partial metrics are labeled `incomplete_diagnostic`
  and are demoted below complete gate-near diagnostics.
- Refreshed selector artifacts:
  `artifacts/stage3_v29_tuning_selection_full_queue/v29_queue_refreshed_selection.csv`
  and
  `artifacts/stage3_v29_tuning_selection_full_queue/v29_queue_refreshed_selection.md`.
- Refreshed selector result: `24` ranked weather V29 tune candidates and `1`
  skipped runtime-triage directory. The top complete candidate remains
  `v29stable_weather_s42_b60g60e1_epochdiag01`, still gate-negative and
  generator-locked.
- Incomplete diagnostics are now explicit: `v29first042null0055_weather_s42_b8g8e1_diag02`
  and `v29first042null0055_weather_s42_b60g60e1_diag03` are demoted to
  `incomplete_diagnostic`; `v29first042null0055_weather_s42_e0g1_runtime02`
  is skipped because rolling outcome files are missing.
- Full-queue mode-gate sweep artifacts:
  `artifacts/stage3_v29_tuning_selection_full_queue/v29_queue_refreshed_mode_gate_threshold_sweep.csv`
  and
  `artifacts/stage3_v29_tuning_selection_full_queue/v29_queue_refreshed_mode_gate_threshold_sweep.md`.
- Full-queue replay uses `117` online-gate feature rows, `112` labeled rows,
  `68` train rows, and `44` held-out rows. Outcome replay evaluates `110`
  rows and marks `7` missing outcome rows.
- Harm-first threshold result: `min_confidence=0.15` gives selected mean MSE
  gain `0.2270443942984707`, fixed `fluctuation_plus_history`
  `0.22015735047664628`, fixed `history_only` `0.16830160575541872`,
  oracle cf-safe `0.3217049720629355`, and selected harmful windows `0`
  versus `18` for fixed `fluctuation_plus_history` and `6` for fixed
  `history_only`.
- Promotion boundary: this full-queue gate is not served-package promotable
  because replay status is `incomplete_outcome_rows`; the existing
  hash-locked served package remains the earlier `conf035` mode-selection-only
  artifact.

Complete-only selector and served package:

- Added selector option `--require-complete-selector-metrics`; it moves
  `selector_metric_complete=False` rows into skipped artifacts before writing
  window, policy, label, and online-gate feature tables.
- Complete selector artifacts:
  `artifacts/stage3_v29_tuning_selection_full_queue/v29_queue_complete_selection.csv`
  and
  `artifacts/stage3_v29_tuning_selection_full_queue/v29_queue_complete_selection.md`.
- Complete selector result: `22` complete weather tuning candidates and `3`
  skipped rows/directories. The top candidate remains
  `v29stable_weather_s42_b60g60e1_epochdiag01`, still gate-negative and
  generator-locked.
- Complete threshold sweep artifacts:
  `artifacts/stage3_v29_tuning_selection_full_queue/v29_queue_complete_mode_gate_threshold_sweep.csv`
  and
  `artifacts/stage3_v29_tuning_selection_full_queue/v29_queue_complete_mode_gate_threshold_sweep.md`.
- Complete replay result: `110/110` rows evaluated and `0` missing outcome
  rows. `min_confidence=0.15` is harm-first best: selected mean MSE gain
  `0.25546113326914255`, fixed `fluctuation_plus_history`
  `0.2018502975357777`, fixed `history_only` `0.16830160575541872`, oracle
  cf-safe `0.32755415337317073`, and selected harmful windows `0` versus `18`
  for fixed `fluctuation_plus_history` and `6` for fixed `history_only`.
- New served package:
  `artifacts/stage3_v29_served_mode_gate/v29_complete_conf015_weather_served_mode_gate_manifest.json`.
- Admission result: `admission_pass=true`, but package scope remains
  `mode_selection_only_no_generator_unlock`; it is a conservative mode-gate
  governance artifact, not generator-unlocked online forecasting evidence.
- Serving load smoke: loading the new manifest succeeds and held-out
  `v29stable_weather_s42_b60g60e1_epochdiag01/window=4` without
  `oracle_label_` columns selects `history_only` with
  `confidence_below_threshold`.

Generator-unlock preflight:

- Added a read-only preflight tool:
  `train/stage3_v29_generator_unlock_preflight.py`.
- Test coverage:
  `tests/test_stage3_v29_generator_unlock_preflight.py`.
- Real preflight artifacts:
  `artifacts/stage3_v29_generator_unlock_preflight/v29_complete_conf015_generator_unlock_preflight.json`
  and
  `artifacts/stage3_v29_generator_unlock_preflight/v29_complete_conf015_generator_unlock_preflight.md`.
- Result: the complete-only served mode gate passes as a prerequisite
  (`mode_gate_prerequisite_pass=True`), but generator unlock still fails.
- Audited top candidate:
  `v29stable_weather_s42_b60g60e1_epochdiag01`.
- Preflight blockers:
  `target_gate_not_passed`, `no_primary_gate_passed`,
  `teacher_abort_written`, `generator_not_unlocked`,
  `stop_reason:do_not_train_generator`,
  `stop_reason:no_primary_target_passed`,
  `positive_partial_windows_below_required`,
  `min_partial_mse_below_floor`, `min_partial_mae_below_floor`, and
  `counterfactual_margin_negative`.
- Decision: `do_not_train_generator`. The next experiment must improve the
  teacher gate before any generator-unlock run.

Teacher-gate tuning matrix:

- Added a non-training tuning planner:
  `train/plan_stage3_v29_teacher_gate_tuning_matrix.py`.
- Test coverage:
  `tests/test_stage3_v29_teacher_gate_tuning_matrix.py`.
- Real matrix artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_teacher_gate_tuning_matrix.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_teacher_gate_tuning_matrix.md`.
- New frozen-architecture config directory:
  `configs/stage3_v29_teacher_gate_tuning_matrix/`.
- Candidate configs:
  `stage3_v29_v29tg_conf015_01_history_guarded_release_floor.yaml`,
  `stage3_v29_v29tg_conf015_02_counterfactual_margin_micro_repair.yaml`,
  `stage3_v29_v29tg_conf015_03_late_window_partial_floor_guard.yaml`,
  `stage3_v29_v29tg_conf015_04_delayed_change_tempered_history.yaml`, and
  `stage3_v29_v29tg_conf015_05_release_null_coupled_guard.yaml`.
- Scope: these are coupled scalar tuning candidates around the
  `stability_release_deconstrained` anchor. They keep V29 topology frozen and
  avoid repeating single-axis sweeps that already failed.
- Execution rule: run at most one bounded candidate at a time, then refresh the
  complete-only selector and rerun generator-unlock preflight. Do not start
  generator training unless a future preflight returns `preflight_pass=true`.

Safe execution plan:

- Added a non-executing plan tool:
  `train/plan_stage3_v29_tuning_execution.py`.
- Test coverage:
  `tests/test_stage3_v29_tuning_execution_plan.py`.
- Real execution artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_tuning_execution_plan.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_tuning_execution_plan.md`.
- Planned log root: `logs/s3v29expert_teacher_gate_matrix`.
- Planned checkpoint root: `ckpt/s3v29expert_teacher_gate_matrix`.
- Status: all `5` candidates are `ready_to_run`; no planned command contains
  `--overwrite`.
- Execution-plan bug fix: `weather` is a config-native dataset path
  (`dataset/weather.csv`), not a supported public-dataset override. The plan now
  omits `--dataset weather` while keeping `weather` in the output path label.
- Boundary: this plan still does not execute training. It only prevents path
  conflicts and records the exact bounded commands for future experiments.

Teacher-gate candidate `01` bounded result:

- Command source:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_tuning_execution_plan.csv`.
- Run:
  `logs/s3v29expert_teacher_gate_matrix/v29tg_conf015_01_history_guarded_release_floor/weather/seed_42`.
- Config:
  `configs/stage3_v29_teacher_gate_tuning_matrix/stage3_v29_v29tg_conf015_01_history_guarded_release_floor.yaml`.
- Scope: `epochs=1`, `max_train_batches=60`, `max_generator_batches=60`,
  `max_eval_batches=60`, and `validation_windows=5`.
- Selector refresh artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate01_selection.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate01_selection.md`.
- Generator preflight artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate01_generator_unlock_preflight.json`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate01_generator_unlock_preflight.md`.
- Result: candidate `01` ranks `3/23`, not first. It remains gate-negative and
  generator-locked.
- Raw primary gate metrics: positive partial windows `2/4`, min partial MSE
  `-0.45713857481802944`, min partial MAE `-0.5075939740964291`, and
  counterfactual margin `-0.016909116092234222`.
- Preflight decision: `do_not_train_generator`; candidate `01` still has
  `target_gate_not_passed`, `no_primary_gate_passed`, `teacher_abort_written`,
  `generator_not_unlocked`, `do_not_train_generator`,
  `no_primary_target_passed`, partial-window insufficiency, partial MSE/MAE
  floor failures, and negative counterfactual margin.
- Interpretation: `history_guarded_release_floor` did not repair the current
  teacher gate and worsened the partial floors relative to the previous top
  anchor. Do not promote this candidate or use it as performance evidence.

Teacher-gate candidate `02` bounded result:

- Command source:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_tuning_execution_plan.csv`.
- Run:
  `logs/s3v29expert_teacher_gate_matrix/v29tg_conf015_02_counterfactual_margin_micro_repair/weather/seed_42`.
- Config:
  `configs/stage3_v29_teacher_gate_tuning_matrix/stage3_v29_v29tg_conf015_02_counterfactual_margin_micro_repair.yaml`.
- Scope: `epochs=1`, `max_train_batches=60`, `max_generator_batches=60`,
  `max_eval_batches=60`, and `validation_windows=5`.
- Selector refresh artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate02_selection.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate02_selection.md`.
- Generator preflight artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate02_generator_unlock_preflight.json`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate02_generator_unlock_preflight.md`.
- Result: candidate `02` ranks `5/24`, not first. It remains gate-negative and
  generator-locked.
- Raw primary gate metrics: positive partial windows `2/4`, min partial MSE
  `-0.46744894881173493`, min partial MAE `-0.5296254452103905`, and
  counterfactual margin `-0.038892373912445266`.
- Preflight decision: `do_not_train_generator`; candidate `02` still has the
  same teacher-gate and generator-unlock blockers.
- Interpretation: `counterfactual_margin_micro_repair` did not repair the
  counterfactual margin; it worsened both partial floors and counterfactual
  margin relative to candidate `01` and the prior top anchor. Do not continue
  this exact micro-shuffle direction as positive evidence.

Teacher-gate candidate `03` bounded result:

- Command source:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_tuning_execution_plan.csv`.
- Run:
  `logs/s3v29expert_teacher_gate_matrix/v29tg_conf015_03_late_window_partial_floor_guard/weather/seed_42`.
- Config:
  `configs/stage3_v29_teacher_gate_tuning_matrix/stage3_v29_v29tg_conf015_03_late_window_partial_floor_guard.yaml`.
- Scope: `epochs=1`, `max_train_batches=60`, `max_generator_batches=60`,
  `max_eval_batches=60`, and `validation_windows=5`.
- Selector refresh artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate03_selection.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate03_selection.md`.
- Generator preflight artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate03_generator_unlock_preflight.json`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate03_generator_unlock_preflight.md`.
- Result: candidate `03` ranks `4/25`, not first. It remains gate-negative and
  generator-locked.
- Raw primary gate metrics: positive partial windows `2/4`, min partial MSE
  `-0.4671504578747007`, min partial MAE `-0.49524774880538697`, and
  counterfactual margin `-0.02420296088538021`.
- Preflight decision: `do_not_train_generator`; candidate `03` still has the
  same teacher-gate and generator-unlock blockers.
- Interpretation: `late_window_partial_floor_guard` did not repair the partial
  floor. It slightly improves MAE versus candidate `02` but remains worse than
  the prior top anchor and keeps counterfactual margin negative.

Teacher-gate candidate `04` bounded result:

- Command source:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_tuning_execution_plan.csv`.
- Run:
  `logs/s3v29expert_teacher_gate_matrix/v29tg_conf015_04_delayed_change_tempered_history/weather/seed_42`.
- Config:
  `configs/stage3_v29_teacher_gate_tuning_matrix/stage3_v29_v29tg_conf015_04_delayed_change_tempered_history.yaml`.
- Scope: `epochs=1`, `max_train_batches=60`, `max_generator_batches=60`,
  `max_eval_batches=60`, and `validation_windows=5`.
- Selector refresh artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate04_selection.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate04_selection.md`.
- Generator preflight artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate04_generator_unlock_preflight.json`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate04_generator_unlock_preflight.md`.
- Result: candidate `04` ranks `7/26`, not first. It remains gate-negative and
  generator-locked.
- Raw primary gate metrics: positive partial windows `2/4`, min partial MSE
  `-0.422131539589501`, min partial MAE `-0.500774945598131`, and
  counterfactual margin `-0.043278617034422484`.
- Preflight decision: `do_not_train_generator`; candidate `04` still has the
  same teacher-gate and generator-unlock blockers.
- Interpretation: `delayed_change_tempered_history` improved window `0` raw
  gain but did not repair the partial-floor requirement, and its
  counterfactual margin is worse than the prior top anchor. It is not
  generator-unlock evidence.

Teacher-gate candidate `05` bounded result:

- Command source:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_tuning_execution_plan.csv`.
- Run:
  `logs/s3v29expert_teacher_gate_matrix/v29tg_conf015_05_release_null_coupled_guard/weather/seed_42`.
- Config:
  `configs/stage3_v29_teacher_gate_tuning_matrix/stage3_v29_v29tg_conf015_05_release_null_coupled_guard.yaml`.
- Scope: `epochs=1`, `max_train_batches=60`, `max_generator_batches=60`,
  `max_eval_batches=60`, and `validation_windows=5`.
- Selector refresh artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate05_selection.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate05_selection.md`.
- Generator preflight artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate05_generator_unlock_preflight.json`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate05_generator_unlock_preflight.md`.
- Result: candidate `05` ranks `8/27`, not first. It remains gate-negative and
  generator-locked.
- Raw primary gate metrics: positive partial windows `2/4`, min partial MSE
  `-0.4669403209264789`, min partial MAE `-0.4851098794798296`, and
  counterfactual margin `-0.043559999700157115`.
- Preflight decision: `do_not_train_generator`; candidate `05` still has
  `target_gate_not_passed`, `no_primary_gate_passed`, `teacher_abort_written`,
  `generator_not_unlocked`, `do_not_train_generator`,
  `no_primary_target_passed`, partial-window insufficiency, partial MSE/MAE
  floor failures, and negative counterfactual margin.
- Interpretation: `release_null_coupled_guard` did not repair the teacher
  gate. It is weaker than the current one-epoch stability anchor and provides
  no generator-unlock evidence.

Teacher-gate matrix exhaustion:

- All five predeclared frozen-architecture matrix candidates completed under
  the same bounded weather seed42 probe budget.
- None passed the primary teacher gate.
- None changed the preflight decision from `do_not_train_generator`.
- The complete selector still ranks the prior one-epoch stability diagnostic,
  `v29stable_weather_s42_b60g60e1_epochdiag01`, first.
- The repeated blockers are now stable: positive partial windows remain below
  requirement, partial MSE/MAE floors remain violated, and counterfactual
  margin remains negative.
- Decision: stop this local matrix. Do not continue minor scalar tuning around
  these five axes as if it were a positive direction. The next step should be a
  higher-level diagnostic of why validation-window partial gains and
  counterfactual ordering conflict, not another blind scalar candidate.

Gate-failure decomposition:

- Added a read-only decomposition tool:
  `train/analyze_stage3_v29_gate_failure_decomposition.py`.
- Test coverage:
  `tests/test_stage3_v29_gate_failure_decomposition.py`.
- Real decomposition artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_matrix_gate_failure_decomposition.json`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_matrix_gate_failure_decomposition.csv`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_matrix_gate_failure_decomposition_windows.csv`,
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_matrix_gate_failure_decomposition.md`.
- Scope: the decomposition covers the current complete selector population of
  `27` V29 weather candidates, not only the last five matrix candidates.
- Result: `candidate_count=27`, `gate_pass_count=0`,
  `generator_unlocked_count=0`, `preflight_decision=do_not_train_generator`,
  and `matrix_decision=stop_local_scalar_matrix`.
- Dominant failure axes: `no_primary_teacher_gate_pass`,
  `positive_partial_window_shortfall`, `partial_floor_violation`,
  `counterfactual_ordering_failure`, `window_level_conflict`, and
  `generator_unlock_blocked`.
- Window `0`: partial MSE floor failures `27/27`, partial MAE floor failures
  `27/27`, counterfactual-negative rows `25/27`, and real-best rows `2/27`.
- Window `1`: partial MSE/MAE floor failures `10/27`,
  counterfactual-negative rows `23/27`, and real-best rows `4/27`.
- Window `2`: partial MSE/MAE floor failures `0/27`, but
  counterfactual-negative rows `25/27`, showing that counterfactual ordering
  can fail even where partial floor is not the immediate blocker.
- Window `3`: partial MSE/MAE floor failures `2/27`,
  counterfactual-negative rows `17/27`, and real-best rows `10/27`.
- Window `4`: partial MSE/MAE floor failures `9/27`,
  counterfactual-negative rows `22/27`, and real-best rows `5/27`.
- Interpretation: the current V29 failure is not a single scalar-weight
  failure. It is a coupled window-level conflict: early/late partial floors,
  positive-window shortfall, and counterfactual ordering remain jointly
  unresolved. The next experiment should first redesign the teacher selection
  or diagnostic objective around these joint constraints, then predeclare a
  new matrix. Do not start generator training from this state.

Decomposition-driven repair matrix:

- Added a non-training repair-matrix planner:
  `train/plan_stage3_v29_decomposition_repair_matrix.py`.
- Test coverage:
  `tests/test_stage3_v29_decomposition_repair_matrix.py`.
- Implementation plan:
  `docs/superpowers/plans/2026-06-13-stage3-v29-decomposition-driven-repair-matrix.md`.
- Generated config directory:
  `configs/stage3_v29_decomposition_repair_matrix/`.
- Generated repair-matrix artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_matrix.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_matrix.md`.
- Generated safe execution plan:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_execution_plan.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_execution_plan.md`.
- Candidate axes: `window0_partial_floor_rebalance`,
  `counterfactual_ordering_without_floor_pressure`,
  `dual_axis_floor_and_cf_coupled`, and
  `mode_abstention_teacher_selection_probe`.
- Status: all four execution-plan rows are `ready_to_run`, and commands omit
  `--overwrite`; `weather` remains config-native and is not passed as
  `--dataset weather`.
- Scope: this is a predeclared next-run queue only. It does not execute
  training, does not change V29 topology, and does not unlock generator
  training.
- Clarification: `teacher_selection_overrides` inside the generated YAML files
  are diagnostic metadata for the repair matrix. The current V29 training code
  does not yet consume them as active teacher-selection logic; the active
  training changes are the recorded invariant loss-weight overrides.
- Interpretation: the new queue is more defensible than another scalar sweep
  because each candidate is tied to the decomposition axes. It is still only a
  tuning plan; proof requires bounded execution, selector refresh,
  gate-failure decomposition refresh, and generator-unlock preflight.

Decomposition-repair candidate `01` bounded result:

- Candidate axis: `window0_partial_floor_rebalance`.
- Run:
  `logs/s3v29expert_decomposition_repair_matrix/v29repair_conf015_01_window0_partial_floor_rebalance/weather/seed_42`.
- Config:
  `configs/stage3_v29_decomposition_repair_matrix/stage3_v29_v29repair_conf015_01_window0_partial_floor_rebalance.yaml`.
- Scope: `epochs=1`, `max_train_batches=60`, `max_generator_batches=60`,
  `max_eval_batches=60`, and `validation_windows=5`.
- Refreshed selector artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair01_selection.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair01_selection.md`.
- Refreshed generator preflight artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair01_generator_unlock_preflight.json`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair01_generator_unlock_preflight.md`.
- Refreshed gate-failure decomposition artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair01_gate_failure_decomposition.json`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair01_gate_failure_decomposition.csv`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair01_gate_failure_decomposition_windows.csv`,
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair01_gate_failure_decomposition.md`.
- Refreshed execution plan:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_execution_plan_after_repair01.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_execution_plan_after_repair01.md`.
- Selector result: candidate `01` ranks `9/28`, not first. The top candidate
  remains `v29stable_weather_s42_b60g60e1_epochdiag01`, still gate-negative and
  generator-locked.
- Candidate `01` raw primary gate metrics: positive partial windows `2/4`, min
  partial MSE `-0.5455704944904651`, min partial MAE
  `-0.5699537539405484`, counterfactual margin `-0.04551155267896673`, and
  window0 FH MSE gain `0.2653257120709129`.
- Preflight decision: `do_not_train_generator`; blockers remain
  `target_gate_not_passed`, `no_primary_gate_passed`,
  `teacher_abort_written`, `generator_not_unlocked`,
  `do_not_train_generator`, `no_primary_target_passed`,
  `positive_partial_windows_below_required`, partial MSE/MAE floor failures,
  and negative counterfactual margin.
- Decomposition result: `candidate_count=28`, `gate_pass_count=0`,
  `generator_unlocked_count=0`, and
  `matrix_decision=stop_local_scalar_matrix`.
- Window-level update: window `0` partial MSE/MAE floor failures increased to
  `28/28`, counterfactual-negative rows increased to `26/28`, and real-best
  rows remain `2/28`. Window `2` still has `0/28` partial-floor failures but
  `26/28` counterfactual-negative rows.
- Updated execution-plan status: candidate `01` is now
  `blocked_existing_output`; candidates `02` through `04` remain
  `ready_to_run`.
- Interpretation: `window0_partial_floor_rebalance` did not repair the window
  `0` partial-floor blocker and worsened the candidate's min partial floors
  relative to the prior one-epoch stability anchor. Do not rerun candidate
  `01`; any further run should use the refreshed after-repair01 execution plan
  and still stop before generator training unless preflight passes.

Decomposition-repair candidate `02` original long-path failure:

- Candidate axis:
  `counterfactual_ordering_without_floor_pressure`.
- Original long-run-id attempt:
  `logs/s3v29expert_decomposition_repair_matrix/v29repair_conf015_02_counterfactual_ordering_without_floor_pressure/weather/seed_42`.
- Failure point: PyTorch checkpoint save for
  `teacher_v18_raw_residual_vector_fluctuationonly.pt`.
- Root cause: the absolute checkpoint file path reached `263` characters on
  Windows. A shorter sibling checkpoint at length `259` succeeded, and a dummy
  `torch.save` to the exact failing path reproduced the failure. The failure is
  therefore a runtime path-length boundary, not a model-quality result.
- Evidence boundary: this partial run is invalid as performance evidence. It
  must not be ranked, compared, or used for generator-unlock decisions.
- Code hardening: `train/plan_stage3_v29_decomposition_repair_matrix.py` now
  emits short run slugs, and `train/plan_stage3_v29_tuning_execution.py` now
  computes the maximum expected checkpoint path and blocks unsafe rows with
  `checkpoint_file_path_too_long`.

Path-safe decomposition-repair matrix:

- Generated path-safe config directory:
  `configs/stage3_v29_decomposition_repair_matrix_pathsafe/`.
- Generated path-safe matrix artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_matrix_pathsafe.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_matrix_pathsafe.md`.
- Generated path-safe execution plan:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_execution_plan_pathsafe.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_execution_plan_pathsafe.md`.
- Path-safe run ids: `v29rps_conf015_01_w0floor`,
  `v29rps_conf015_02_cforder`, `v29rps_conf015_03_floorcf`, and
  `v29rps_conf015_04_abstain`.
- Execution-plan safety: all four rows are `ready_to_run`; the maximum expected
  checkpoint file path length is `200`, below the enforced `259`-character
  guard.
- Supersession rule: future bounded repair runs should use the path-safe queue,
  not the original long-run-id queue. The original candidate `01` already
  completed negatively, so the path-safe equivalent `01` should not be rerun.

Path-safe decomposition-repair candidate `02` bounded result:

- Candidate axis:
  `counterfactual_ordering_without_floor_pressure`.
- Run:
  `logs/s3v29rps/v29rps_conf015_02_cforder/weather/seed_42`.
- Config:
  `configs/stage3_v29_decomposition_repair_matrix_pathsafe/stage3_v29_v29rps_conf015_02_cforder.yaml`.
- Scope: `epochs=1`, `max_train_batches=60`, `max_generator_batches=60`,
  `max_eval_batches=60`, and `validation_windows=5`.
- Refreshed selector artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair02_selection.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair02_selection.md`.
- Refreshed generator preflight artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair02_generator_unlock_preflight.json`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair02_generator_unlock_preflight.md`.
- Refreshed gate-failure decomposition artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair02_gate_failure_decomposition.json`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair02_gate_failure_decomposition.csv`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair02_gate_failure_decomposition_windows.csv`,
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair02_gate_failure_decomposition.md`.
- Refreshed path-safe execution plan:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_execution_plan_pathsafe_after_repair02.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_execution_plan_pathsafe_after_repair02.md`.
- Selector result: candidate `02` ranks `7/29`, not first. The top candidate
  remains `v29stable_weather_s42_b60g60e1_epochdiag01`, still gate-negative and
  generator-locked.
- Candidate `02` raw primary gate metrics: positive partial windows `2/4`, min
  partial MSE `-0.4204164211957788`, min partial MAE
  `-0.500599967584385`, counterfactual margin `-0.04310910574869553`, and
  window0 FH MSE gain `0.3989539074568899`.
- Preflight decision: `do_not_train_generator`; blockers remain
  `target_gate_not_passed`, `no_primary_gate_passed`,
  `teacher_abort_written`, `generator_not_unlocked`,
  `positive_partial_windows_below_required`, partial MSE/MAE floor failures,
  and negative counterfactual margin.
- Decomposition result: `candidate_count=29`, `gate_pass_count=0`,
  `generator_unlocked_count=0`, and
  `matrix_decision=stop_local_scalar_matrix`.
- Window-level update: window `0` partial MSE/MAE floor failures remain
  `29/29`, counterfactual-negative rows are `27/29`, and real-best rows are
  `2/29`. Window `2` still has `0/29` partial-floor failures but `27/29`
  counterfactual-negative rows.
- Updated execution-plan status: candidate `02` is
  `blocked_existing_output`; candidates `01`, `03`, and `04` are technically
  `ready_to_run`, but candidate `01` is path-safe-equivalent to the already
  completed negative candidate `01` and should still not be rerun.
- Interpretation: `counterfactual_ordering_without_floor_pressure` did not
  repair the primary teacher gate. It improves some scalar diagnostics relative
  to candidate `01`, but the minimum partial floors, positive-window count, and
  counterfactual margin remain unacceptable. Do not train the generator from
  candidate `02`.

Path-safe decomposition-repair candidate `03` bounded result:

- Candidate axis: `dual_axis_floor_and_cf_coupled`.
- Run:
  `logs/s3v29rps/v29rps_conf015_03_floorcf/weather/seed_42`.
- Config:
  `configs/stage3_v29_decomposition_repair_matrix_pathsafe/stage3_v29_v29rps_conf015_03_floorcf.yaml`.
- Scope: `epochs=1`, `max_train_batches=60`, `max_generator_batches=60`,
  `max_eval_batches=60`, and `validation_windows=5`.
- Refreshed selector artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair03_selection.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair03_selection.md`.
- Refreshed generator preflight artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair03_generator_unlock_preflight.json`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair03_generator_unlock_preflight.md`.
- Refreshed gate-failure decomposition artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair03_gate_failure_decomposition.json`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair03_gate_failure_decomposition.csv`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair03_gate_failure_decomposition_windows.csv`,
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair03_gate_failure_decomposition.md`.
- Refreshed path-safe execution plan:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_execution_plan_pathsafe_after_repair03.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_execution_plan_pathsafe_after_repair03.md`.
- Selector result: candidate `03` ranks `4/31`, not first. The top candidate
  remains `v29stable_weather_s42_b60g60e1_epochdiag01`, still gate-negative and
  generator-locked.
- Candidate `03` raw primary gate metrics: positive partial windows `2/4`, min
  partial MSE `-0.5299312143911022`, min partial MAE
  `-0.545154931341216`, counterfactual margin `-0.024159312022691004`, and
  window0 FH MSE gain `0.2764126358705603`.
- Preflight decision: `do_not_train_generator`; blockers remain
  `target_gate_not_passed`, `no_primary_gate_passed`,
  `teacher_abort_written`, `generator_not_unlocked`,
  `positive_partial_windows_below_required`, partial MSE/MAE floor failures,
  and negative counterfactual margin.
- Decomposition result: `candidate_count=31`, `gate_pass_count=0`,
  `generator_unlocked_count=0`, and
  `matrix_decision=stop_local_scalar_matrix`.
- Window-level update: window `0` partial MSE floor failures are `31/31`,
  partial MAE floor failures are `30/31`, counterfactual-negative rows are
  `29/31`, and real-best rows remain `2/31`. Window `2` still has `0/31`
  partial-floor failures but `29/31` counterfactual-negative rows.
- Updated execution-plan status: candidate `03` is
  `blocked_existing_output`; candidate `04` is the only meaningful remaining
  path-safe row. Candidate `01` is still listed as technically
  `ready_to_run`, but its long-run equivalent already completed negatively and
  it should not be rerun.
- Interpretation: the coupled floor-plus-counterfactual candidate did not
  solve the joint conflict. It ranks higher than candidate `02` by selector
  ordering, but it worsens the minimum partial floors and remains far from
  generator unlock. Do not train the generator from candidate `03`.

Path-safe decomposition-repair candidate `04` bounded result:

- Candidate axis: `mode_abstention_teacher_selection_probe`.
- Run:
  `logs/s3v29rps/v29rps_conf015_04_abstain/weather/seed_42`.
- Config:
  `configs/stage3_v29_decomposition_repair_matrix_pathsafe/stage3_v29_v29rps_conf015_04_abstain.yaml`.
- Scope: `epochs=1`, `max_train_batches=60`, `max_generator_batches=60`,
  `max_eval_batches=60`, and `validation_windows=5`.
- Refreshed selector artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair04_selection.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair04_selection.md`.
- Refreshed generator preflight artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair04_generator_unlock_preflight.json`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair04_generator_unlock_preflight.md`.
- Refreshed gate-failure decomposition artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair04_gate_failure_decomposition.json`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair04_gate_failure_decomposition.csv`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair04_gate_failure_decomposition_windows.csv`,
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair04_gate_failure_decomposition.md`.
- Final path-safe execution plan:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_execution_plan_pathsafe_after_repair04.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_execution_plan_pathsafe_after_repair04.md`.
- Selector result: candidate `04` ranks `6/32`, not first. The top candidate
  remains `v29stable_weather_s42_b60g60e1_epochdiag01`, still gate-negative and
  generator-locked.
- Candidate `04` raw primary gate metrics: positive partial windows `2/4`, min
  partial MSE `-0.46285146776087893`, min partial MAE
  `-0.5138718209477093`, counterfactual margin `-0.03642005087344469`, and
  window0 FH MSE gain `0.35833191100094924`.
- Preflight decision: `do_not_train_generator`; blockers remain
  `target_gate_not_passed`, `no_primary_gate_passed`,
  `teacher_abort_written`, `generator_not_unlocked`,
  `positive_partial_windows_below_required`, partial MSE/MAE floor failures,
  and negative counterfactual margin.
- Decomposition result: `candidate_count=32`, `gate_pass_count=0`,
  `generator_unlocked_count=0`, and
  `matrix_decision=stop_local_scalar_matrix`.
- Window-level update: window `0` partial MSE floor failures are `32/32`,
  partial MAE floor failures are `31/32`, counterfactual-negative rows are
  `30/32`, and real-best rows remain `2/32`. Window `2` still has `0/32`
  partial-floor failures but `30/32` counterfactual-negative rows.
- Final execution-plan status: path-safe candidates `02`, `03`, and `04` are
  blocked by existing outputs; path-safe candidate `01` is technically
  `ready_to_run` only because its equivalent completed in the original long-run
  queue. It should not be rerun.
- Interpretation: the abstention/selection probe did not repair the teacher
  gate. The repair matrix is now exhausted in the meaningful sense. Do not
  launch generator training, do not promote V29 as an accuracy improvement, and
  do not add another blind scalar candidate. The next step must be a new
  diagnosis or a predeclared matrix that changes the failure mechanism being
  tested.

History-gate recovery matrix:

- Added a post-repair non-training planner:
  `train/plan_stage3_v29_history_gate_recovery_matrix.py`.
- Test coverage:
  `tests/test_stage3_v29_history_gate_recovery_matrix.py`.
- Implementation plan:
  `docs/superpowers/plans/2026-06-13-stage3-v29-history-gate-recovery-matrix.md`.
- Generated config directory:
  `configs/stage3_v29_history_gate_recovery_matrix/`.
- Generated matrix artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_history_gate_recovery_matrix.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_history_gate_recovery_matrix.md`.
- Generated safe execution plan:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_history_gate_recovery_execution_plan.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_history_gate_recovery_execution_plan.md`.
- Candidate axes: `window0_safe_release_ramp`,
  `late_history_dominance_bridge`,
  `counterfactual_tie_breaker_floor_safe`, and
  `history_cap_release_balance`.
- Status before execution: all four rows were `ready_to_run`; maximum expected
  checkpoint file paths were below the `259`-character guard.
- Mechanism shift: this is not another copy of the exhausted repair matrix. It
  targets the observed tradeoff where the top one-epoch rows are window-0
  non-harmful but lose to `history_only` in windows `0`, `1`, and `4`, while
  stronger release settings increase partial-window count but make window `0`
  harmful.
- Boundary: this matrix does not change V29 topology, does not train the
  generator, and does not support any manuscript performance claim unless a
  candidate later passes the predeclared selector/decomposition/preflight gates.

History-gate recovery candidate `01` bounded result:

- Candidate axis: `window0_safe_release_ramp`.
- Run:
  `logs/s3v29hgr/v29hgr_conf015_01_w0ramp/weather/seed_42`.
- Config:
  `configs/stage3_v29_history_gate_recovery_matrix/stage3_v29_v29hgr_conf015_01_w0ramp.yaml`.
- Scope: `epochs=1`, `max_train_batches=60`, `max_generator_batches=60`,
  `max_eval_batches=60`, and `validation_windows=5`.
- Refreshed selector artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr01_selection.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr01_selection.md`.
- Refreshed generator preflight artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr01_generator_unlock_preflight.json`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr01_generator_unlock_preflight.md`.
- Refreshed gate-failure decomposition artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr01_gate_failure_decomposition.json`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr01_gate_failure_decomposition.csv`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr01_gate_failure_decomposition_windows.csv`,
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr01_gate_failure_decomposition.md`.
- Refreshed history-gate execution plan:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_history_gate_recovery_execution_plan_after_hgr01.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_history_gate_recovery_execution_plan_after_hgr01.md`.
- Selector result: candidate `01` ranks `2/33`, not first. The top candidate
  remains `v29stable_weather_s42_b60g60e1_epochdiag01`, still gate-negative and
  generator-locked.
- Candidate `01` raw primary gate metrics: positive partial windows `2/4`, min
  partial MSE `-0.43209738075898796`, min partial MAE
  `-0.4597064241028177`, counterfactual margin `-0.014453849719397449`, and
  window0 FH MSE gain `0.36675840890386036`.
- Preflight decision: `do_not_train_generator`; blockers remain
  `target_gate_not_passed`, `no_primary_gate_passed`,
  `teacher_abort_written`, `generator_not_unlocked`,
  `positive_partial_windows_below_required`, partial MSE/MAE floor failures,
  and negative counterfactual margin.
- Decomposition result: `candidate_count=33`, `gate_pass_count=0`,
  `generator_unlocked_count=0`, and
  `matrix_decision=stop_local_scalar_matrix`.
- Updated execution-plan status: candidate `01` is
  `blocked_existing_output`; candidates `02`, `03`, and `04` remain
  `ready_to_run`.
- Interpretation: candidate `01` is not an improvement over the one-epoch
  stability anchor. It is less harmful than several repair candidates and ranks
  near the top, but it still worsens the minimum partial floor and counterfactual
  margin relative to rank `1`. Do not train the generator from candidate `01`.

History-gate recovery candidate `02` bounded result:

- Candidate axis: `late_history_dominance_bridge`.
- Run:
  `logs/s3v29hgr/v29hgr_conf015_02_latehist/weather/seed_42`.
- Config:
  `configs/stage3_v29_history_gate_recovery_matrix/stage3_v29_v29hgr_conf015_02_latehist.yaml`.
- Scope: `epochs=1`, `max_train_batches=60`, `max_generator_batches=60`,
  `max_eval_batches=60`, and `validation_windows=5`.
- Refreshed selector artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr02_selection.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr02_selection.md`.
- Refreshed generator preflight artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr02_generator_unlock_preflight.json`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr02_generator_unlock_preflight.md`.
- Refreshed gate-failure decomposition artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr02_gate_failure_decomposition.json`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr02_gate_failure_decomposition.csv`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr02_gate_failure_decomposition_windows.csv`,
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr02_gate_failure_decomposition.md`.
- Refreshed history-gate execution plan:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_history_gate_recovery_execution_plan_after_hgr02.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_history_gate_recovery_execution_plan_after_hgr02.md`.
- Selector result: candidate `02` ranks `1/34`, ahead of the one-epoch
  stability anchor, but still remains gate-negative and generator-locked.
- Candidate `02` raw primary gate metrics: positive partial windows `2/4`, min
  partial MSE `-0.5290475768525864`, min partial MAE
  `-0.5365663873681754`, counterfactual margin `-0.009411902842471953`, and
  window0 FH MSE gain `0.2906420799259992`.
- Preflight decision: `do_not_train_generator`; blockers remain
  `target_gate_not_passed`, `no_primary_gate_passed`,
  `teacher_abort_written`, `generator_not_unlocked`,
  `positive_partial_windows_below_required`, partial MSE/MAE floor failures,
  and negative counterfactual margin.
- Decomposition result: `candidate_count=34`, `gate_pass_count=0`,
  `generator_unlocked_count=0`, and
  `matrix_decision=stop_local_scalar_matrix`. Dominant failure axes remain
  `no_primary_teacher_gate_pass`, `positive_partial_window_shortfall`,
  `partial_floor_violation`, `counterfactual_ordering_failure`,
  `window_level_conflict`, and `generator_unlock_blocked`.
- Updated execution-plan status: candidates `01` and `02` are
  `blocked_existing_output`; candidates `03` and `04` remain `ready_to_run`.
- Interpretation: candidate `02` improves the selector ordering mainly by
  making the counterfactual margin slightly less negative than the prior anchor,
  but it worsens the minimum partial floors and still has only `2/4` positive
  partial windows. This is not teacher-gate evidence, not generator-unlock
  evidence, and not a manuscript performance result.

History-gate recovery candidate `03` bounded result:

- Candidate axis: `counterfactual_tie_breaker_floor_safe`.
- Run:
  `logs/s3v29hgr/v29hgr_conf015_03_cftie/weather/seed_42`.
- Config:
  `configs/stage3_v29_history_gate_recovery_matrix/stage3_v29_v29hgr_conf015_03_cftie.yaml`.
- Scope: `epochs=1`, `max_train_batches=60`, `max_generator_batches=60`,
  `max_eval_batches=60`, and `validation_windows=5`.
- Refreshed selector artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr03_selection.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr03_selection.md`.
- Refreshed generator preflight artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr03_generator_unlock_preflight.json`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr03_generator_unlock_preflight.md`.
- Refreshed gate-failure decomposition artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr03_gate_failure_decomposition.json`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr03_gate_failure_decomposition.csv`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr03_gate_failure_decomposition_windows.csv`,
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr03_gate_failure_decomposition.md`.
- Refreshed history-gate execution plan:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_history_gate_recovery_execution_plan_after_hgr03.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_history_gate_recovery_execution_plan_after_hgr03.md`.
- Selector result: candidate `03` ranks `1/35`, ahead of candidate `02` and
  the one-epoch stability anchor, but still remains gate-negative and
  generator-locked.
- Candidate `03` raw primary gate metrics: positive partial windows `2/4`, min
  partial MSE `-0.41351019804173306`, min partial MAE
  `-0.466581375695994`, counterfactual margin
  `-0.006009365114644338`, and window0 FH MSE gain
  `0.3719791529173322`.
- Preflight decision: `do_not_train_generator`; blockers remain
  `target_gate_not_passed`, `no_primary_gate_passed`,
  `teacher_abort_written`, `generator_not_unlocked`,
  `positive_partial_windows_below_required`, partial MSE/MAE floor failures,
  and negative counterfactual margin.
- Decomposition result: `candidate_count=35`, `gate_pass_count=0`,
  `generator_unlocked_count=0`, and
  `matrix_decision=stop_local_scalar_matrix`. Dominant failure axes remain
  `no_primary_teacher_gate_pass`, `positive_partial_window_shortfall`,
  `partial_floor_violation`, `counterfactual_ordering_failure`,
  `window_level_conflict`, and `generator_unlock_blocked`.
- Updated execution-plan status: candidates `01`, `02`, and `03` are
  `blocked_existing_output`; candidate `04` remains `ready_to_run`.
- Interpretation: candidate `03` is the best selector-ranked HGR row so far
  because it weakens the negative counterfactual margin and improves partial
  floors relative to candidate `02`. It still does not pass the primary gate:
  the positive-window count is unchanged at `2/4`, partial floors remain
  negative, and counterfactual ordering is still not positive. It therefore
  cannot unlock generator training or support a manuscript performance claim.

History-gate recovery candidate `04` bounded result and matrix decision:

- Candidate axis: `history_cap_release_balance`.
- Run:
  `logs/s3v29hgr/v29hgr_conf015_04_histcap/weather/seed_42`.
- Config:
  `configs/stage3_v29_history_gate_recovery_matrix/stage3_v29_v29hgr_conf015_04_histcap.yaml`.
- Scope: `epochs=1`, `max_train_batches=60`, `max_generator_batches=60`,
  `max_eval_batches=60`, and `validation_windows=5`.
- Refreshed selector artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr04_selection.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr04_selection.md`.
- Refreshed generator preflight artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr04_generator_unlock_preflight.json`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr04_generator_unlock_preflight.md`.
- Refreshed gate-failure decomposition artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr04_gate_failure_decomposition.json`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr04_gate_failure_decomposition.csv`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr04_gate_failure_decomposition_windows.csv`,
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr04_gate_failure_decomposition.md`.
- Final history-gate execution plan:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_history_gate_recovery_execution_plan_after_hgr04.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_history_gate_recovery_execution_plan_after_hgr04.md`.
- Selector result: candidate `04` ranks `9/36`, below candidate `03`, candidate
  `02`, and the one-epoch stability anchor. It remains gate-negative and
  generator-locked.
- Candidate `04` raw primary gate metrics: positive partial windows `2/4`, min
  partial MSE `-0.4066591740244906`, min partial MAE
  `-0.4354709147954243`, counterfactual margin
  `-0.030275156997368824`, and window0 FH MSE gain
  `0.3889744979968982`.
- Preflight decision: `do_not_train_generator`; blockers remain
  `target_gate_not_passed`, `no_primary_gate_passed`,
  `teacher_abort_written`, `generator_not_unlocked`,
  `positive_partial_windows_below_required`, partial MSE/MAE floor failures,
  and negative counterfactual margin.
- Final decomposition result: `candidate_count=36`, `gate_pass_count=0`,
  `generator_unlocked_count=0`, and
  `matrix_decision=stop_local_scalar_matrix`.
- Final execution-plan status: all four history-gate recovery candidates are
  `blocked_existing_output`.
- Interpretation: the history-gate recovery matrix is now exhausted in the
  meaningful sense. Candidate `03` remains the best selector-ranked HGR row, but
  every HGR row fails the same promotion blockers: no primary teacher gate,
  only `2/4` positive partial windows, negative partial floors, negative
  counterfactual margin, and generator unlock blocked. Do not train the
  generator from this matrix and do not promote it as performance evidence.

## What Would Count As Real Improvement

Weak improvement:

- Aggregate validation raw/sign gains improve, but strict gate still fails.
- This is useful for algorithm development only.

Meaningful teacher improvement:

- A primary target passes strict teacher gate.
- First-window harm is controlled.
- Real signal beats counterfactuals on rolling validation.

Meaningful online improvement:

- A primary teacher target passes.
- Generator training is allowed.
- Online validation and test summaries exist.
- Online prediction shift limits and risk gates remain safe.

Paper-grade improvement:

- The above holds across datasets/seeds.
- Baselines and protocols are fair.
- The effect is not confined to one favorable weather seed.
- The manuscript claim is phrased no stronger than the evidence.

Current status:

- V29 has weak development improvement evidence.
- V29 does not yet have meaningful teacher improvement evidence.
- V29 has no generator-unlocked online improvement evidence.
- V29 has no paper-grade accuracy improvement evidence.

## Recommended Decision

Keep the current model correction direction, but do not claim it works yet.

The next engineering move should be rigorous tuning under frozen architecture,
not another structural update. If multiple constrained tuning blocks still
cannot remove first-window harm and counterfactual failure, then the next model
change should be narrowly targeted at serving-time abstention, history
reliability, or target selection rather than another broad representation
redesign.
