# Stage III V19 Conditional-Release Implementation Record

## Scope

This record covers the first implementation pass of a new official `V19` line.

`V19` extends `V18` without overwriting any previous files. Its purpose is to
push the user's expert-local invariant idea one step further:

- keep the expert-local anchor / suppression / release decomposition;
- keep environment inference causal and online-available;
- stop treating “cleaner `phi_i / phi_v` geometry” as the main target;
- explicitly supervise a history-like proxy branch and a release-increment
  branch inside the base model.

## Why V19 exists

The two Chinese reports and the `V17/V18` records jointly pointed to the same
problem:

- structure was improving,
- but the model was still not directly trained to make the release branch beat
  a history-like alternative.

`V19` changes the base prediction path from:

`expert_y_hat = expert_y_inv + expert_y_var`

to:

`expert_y_hat = expert_y_inv + expert_y_hist + expert_y_var`

where:

- `expert_y_inv` is invariant forecast,
- `expert_y_hist` is a history-like residual proxy from stable context,
- `expert_y_var` is the release increment from release state, confidence, and
  environment shift.

This is the main new mechanism in `V19`.

## New files

Core code:

- `models/factor_mechanism_v19.py`
- `train/offline_factor_v19.py`
- `train/run_stage3_v19_conditional_release_probe.py`
- `train/stage3_generator_training_v19.py`
- `train/stage3_v19_artifacts.py`
- `train/online_loop_stage3_v19.py`
- `configs/stage3_v19_conditional_release_probe.yaml`

Tests:

- `tests/test_factor_mechanism_v19.py`
- `tests/test_offline_factor_v19.py`
- `tests/test_stage3_v19_conditional_release.py`
- `tests/test_stage3_generator_training_v19.py`

Design records:

- `docs/superpowers/specs/2026-06-10-stage3-v19-conditional-release-design.md`
- `docs/superpowers/plans/2026-06-10-stage3-v19-conditional-release.md`

## What V19 adds beyond V18

### 1. Explicit history-proxy branch

`V19` adds `history_output_heads` per expert. They read stable context from:

- `pooled_anchor`
- `pooled_shared`

and produce `expert_y_hist`.

This branch is meant to absorb the “easy” correction that does not require the
release state.

### 2. Release branch is trained as incremental correction

`variant_output_heads` now read:

- `pooled_raw_release`
- `pooled_release_confidence`
- `pooled_env_shift`

instead of only a cleaner `delta_phi_v` projection.

The intent is to make the release branch explicitly conditional and
service-oriented.

### 3. New loss family

`V19` adds:

- `loss_local_history_proxy`
- `loss_local_release_increment`
- `loss_conditional_advantage`
- `loss_release_null`
- `loss_counterfactual_shuffle`

These are designed to approximate the reports' recommendation:

> do not only make the representation cleaner; make the release state carry
> information that is useful beyond easier alternatives.

### 4. Formal versioned probe entry

`train/run_stage3_v19_conditional_release_probe.py` was added as a dedicated
experiment entry point. It reuses the stable `V18` probe skeleton, but swaps in
the `V19` model, loss builder, generator wrapper, online wrapper, and artifact
loader, then rewrites output filenames to `v19`.

This means `V19` is not just a model file. It is now a full runnable version
line at the code level.

## Deterministic validation

The following checks passed:

- `python -m py_compile models\factor_mechanism_v19.py train\offline_factor_v19.py train\stage3_generator_training_v19.py train\stage3_v19_artifacts.py train\online_loop_stage3_v19.py train\run_stage3_v19_conditional_release_probe.py tests\test_factor_mechanism_v19.py tests\test_offline_factor_v19.py tests\test_stage3_v19_conditional_release.py tests\test_stage3_generator_training_v19.py`
- targeted `V19` tests:
  - `pytest tests\test_factor_mechanism_v19.py tests\test_offline_factor_v19.py tests\test_stage3_v19_conditional_release.py tests\test_stage3_generator_training_v19.py -q`
  - result: `12 passed`
- cross-version regression:
  - `pytest tests\test_factor_mechanism_v18.py tests\test_offline_factor_v18.py tests\test_stage3_v18_structured_env_release.py tests\test_stage3_generator_training_v18.py tests\test_factor_mechanism_v19.py tests\test_offline_factor_v19.py tests\test_stage3_v19_conditional_release.py tests\test_stage3_generator_training_v19.py -q`
  - result: `25 passed`

## V19 smoke run

Command:

```bash
C:\Users\admin\anaconda3\python.exe train\run_stage3_v19_conditional_release_probe.py --config configs\stage3_v19_conditional_release_probe.yaml --device cpu --epochs 1 --max-train-batches 1 --max-generator-batches 1 --max-eval-batches 1 --validation-windows 2 --run-id v19_smoke_01 --out-root logs\s3v19expert_smoke --checkpoint-root ckpt\s3v19expert_smoke --overwrite
```

Output directory:

- `logs/s3v19expert_smoke/v19_smoke_01/weather/seed_42`

Verified outputs:

- `offline_metrics_v19.csv`
- `teacher_only_v19_target_gate.csv`
- `teacher_only_abort.json`
- `run_manifest_v19.json`

Abort file:

```json
{
  "stop_reasons": [
    "do_not_train_generator",
    "no_primary_target_passed"
  ],
  "protocol_version": "v19_conditional_release_probe"
}
```

Smoke conclusion:

- the `V19` base path runs;
- the `V19` probe wrapper writes version-correct artifacts;
- the teacher gate is active and still blocks unsupported generator training.

## Weather seed42 b240 screening

Command:

```bash
C:\Users\admin\anaconda3\python.exe train\run_stage3_v19_conditional_release_probe.py --config configs\stage3_v19_conditional_release_probe.yaml --epochs 5 --max-train-batches 240 --max-generator-batches 240 --max-eval-batches 240 --validation-windows 5 --run-id v19_weather_s42_b240_01 --out-root logs\s3v19expert_screen --checkpoint-root ckpt\s3v19expert_screen --overwrite
```

Output directory:

- `logs/s3v19expert_screen/v19_weather_s42_b240_01/weather/seed_42`

Primary gate file:

- `teacher_only_v19_target_gate.csv`

Validation partial-gain file:

- `teacher_only_validation_partial_gain_vs_history_v19.csv`

Abort file:

- `teacher_only_abort.json`

### Gate summary

- `raw_residual_vector`
  - `positive_partial_window_count = 1 / 5`
  - `min_partial_mse_gain_vs_history = -0.12313987664415116`
  - `min_partial_mae_gain_vs_history = -0.12210348898906087`
  - `counterfactual_margin = -0.06464396052506349`
  - gate failed
- `residual_change_vector`
  - `positive_partial_window_count = 0 / 5`
  - `min_partial_mse_gain_vs_history = -0.06923471557813798`
  - `min_partial_mae_gain_vs_history = -0.043232416718186464`
  - gate failed
- `residual_sign`
  - `positive_partial_window_count = 2 / 5`
  - gate failed

### Validation partial-gain summary

- `raw_residual_vector`
  - `partial_mse_gain_vs_history = 0.010815781079246078`
  - `partial_mae_gain_vs_history = 0.006465193648783396`
- `residual_change_vector`
  - `partial_mse_gain_vs_history = -0.022770197429976276`
  - `partial_mae_gain_vs_history = -0.027662769110066737`
- `residual_sign`
  - `partial_mse_gain_vs_history = 0.03241688013076782`
  - `partial_mae_gain_vs_history = 0.026070594787597656`

## V19 narrow loss sweep on `weather seed42 b240`

To test whether the first `V19` result was merely under-weighting the release
branch, two narrow loss variants were added:

- `configs/stage3_v19_conditional_release_probe_moderate_release.yaml`
- `configs/stage3_v19_conditional_release_probe_release_focus.yaml`

The intended hypothesis was narrow and concrete:

> if the history-proxy branch is absorbing too much of the correction budget,
> then weakening `lambda_local_history_proxy` while strengthening
> `lambda_local_release_increment` and `lambda_conditional_advantage` should
> improve the primary target signal.

### `moderate_release` run

Command:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v19_conditional_release_probe.py --config configs\stage3_v19_conditional_release_probe_moderate_release.yaml --epochs 5 --max-train-batches 240 --max-generator-batches 240 --max-eval-batches 240 --validation-windows 5 --run-id v19mod_weather_s42_b240_01 --out-root logs\s3v19expert_screen --checkpoint-root ckpt\s3v19expert_screen --overwrite
```

Output directory:

- `logs/s3v19expert_screen/v19mod_weather_s42_b240_01/weather/seed_42`

Validation partial-gain summary:

- `raw_residual_vector`
  - `partial_mse_gain_vs_history = -0.012165016259213624`
  - `partial_mae_gain_vs_history = -0.014381632718737547`
- `residual_change_vector`
  - `partial_mse_gain_vs_history = -0.13205818681519335`
  - `partial_mae_gain_vs_history = -0.1245595428033277`
- `residual_sign`
  - `partial_mse_gain_vs_history = -0.01053619384765625`
  - `partial_mae_gain_vs_history = -0.009964168071746826`

Gate summary:

- `raw_residual_vector`
  - `positive_partial_window_count = 1 / 5`
  - `min_partial_mse_gain_vs_history = -0.1765395018921223`
  - `min_partial_mae_gain_vs_history = -0.1460732370303538`
  - `counterfactual_margin = -0.12253834561868741`
  - gate failed
- `residual_change_vector`
  - `positive_partial_window_count = 0 / 5`
  - `min_partial_mse_gain_vs_history = -0.29516461253576926`
  - `min_partial_mae_gain_vs_history = -0.2001920153794958`
  - gate failed

Interpretation:

- `moderate_release` is strictly worse than base `V19` on all three validation
  targets.
- Simply turning up release-related losses while slightly turning down the
  history-proxy term did not unlock a usable incremental signal.

### `release_focus` run

Command:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v19_conditional_release_probe.py --config configs\stage3_v19_conditional_release_probe_release_focus.yaml --epochs 5 --max-train-batches 240 --max-generator-batches 240 --max-eval-batches 240 --validation-windows 5 --run-id v19rel_weather_s42_b240_01 --out-root logs\s3v19expert_screen --checkpoint-root ckpt\s3v19expert_screen --overwrite
```

Output directory:

- `logs/s3v19expert_screen/v19rel_weather_s42_b240_01/weather/seed_42`

Validation partial-gain summary:

- `raw_residual_vector`
  - `partial_mse_gain_vs_history = -0.010856369442358266`
  - `partial_mae_gain_vs_history = -0.007818962114359285`
- `residual_change_vector`
  - `partial_mse_gain_vs_history = -0.09527017865213978`
  - `partial_mae_gain_vs_history = -0.09105475171088687`
- `residual_sign`
  - `partial_mse_gain_vs_history = -0.020805716514587402`
  - `partial_mae_gain_vs_history = -0.02339911460876465`

Gate summary:

- `raw_residual_vector`
  - `positive_partial_window_count = 1 / 5`
  - `min_partial_mse_gain_vs_history = -0.0996965475244126`
  - `min_partial_mae_gain_vs_history = -0.07048625470290681`
  - `counterfactual_margin = -0.0548439345531051`
  - gate failed
- `residual_change_vector`
  - `positive_partial_window_count = 0 / 5`
  - `min_partial_mse_gain_vs_history = -0.2466695501425584`
  - `min_partial_mae_gain_vs_history = -0.15062362128819592`
  - gate failed

Interpretation:

- `release_focus` is also worse than base `V19`.
- Relative to `moderate_release`, it is slightly less harmful on
  `raw_residual_vector`, but still removes the only weak positive validation
  aggregate signal seen in the base configuration.

### Narrow-sweep comparison

Current ranking on `weather seed42 b240` is:

1. Base `V19`
2. `release_focus`
3. `moderate_release`

This is a negative but useful result. It means the next bottleneck is not
simply "the release branch needs a bigger scalar weight." The problem is more
likely structural: branch coupling, router interaction, or how the local
history-proxy branch partitions correction budget away from the release state.

## Position against existing version lines

Using already recorded `weather` evidence:

- `V17 weather seed42 b480`
  - `raw_residual_vector`
    - `partial_mse_gain_vs_history = 0.013656406744541993`
    - `partial_mae_gain_vs_history = 0.023935744651397806`
  - `residual_change_vector`
    - `partial_mse_gain_vs_history = 0.015024360492013708`
    - `partial_mae_gain_vs_history = 0.01568583073432975`
  - gate status:
    - `residual_change_vector` had `positive_partial_window_count = 5 / 5`
    - but `counterfactual_margin = -0.0016897684783019227`, so generator still
      stayed locked

- `V18 balanced weather seed42 b240`
  - `raw_residual_vector`
    - `partial_mse_gain_vs_history = 0.0070728888592978476`
    - `partial_mae_gain_vs_history = 0.00308686616356736`
  - `residual_change_vector`
    - `partial_mse_gain_vs_history = 0.0`
    - `partial_mae_gain_vs_history = 0.0`
  - gate status:
    - `raw_residual_vector` had `positive_partial_window_count = 2 / 5`
    - `counterfactual_margin = -0.007588815800448869`

- Base `V19 weather seed42 b240`
  - `raw_residual_vector`
    - `partial_mse_gain_vs_history = 0.010815781079246078`
    - `partial_mae_gain_vs_history = 0.006465193648783396`
  - `residual_change_vector`
    - `partial_mse_gain_vs_history = -0.022770197429976276`
    - `partial_mae_gain_vs_history = -0.027662769110066737`
  - gate status:
    - `raw_residual_vector` had `positive_partial_window_count = 1 / 5`
    - `counterfactual_margin = -0.06464396052506349`

Current comparative reading:

- Against `V18 balanced b240`, base `V19 b240` is slightly stronger on
  aggregate `raw_residual_vector`, but clearly worse on counterfactual margin.
- Against `V17 weather b480`, `V19` is not competitive yet. `V17` remains the
  strongest empirical weather line because it is the only version that produced
  a cross-window positive `residual_change_vector` signal, even though it still
  failed the final strict gate.
- Therefore the most accurate status is:
  - `V19` is the more principled objective redesign line
  - `V17` is still the current empirical weather benchmark to beat

## Current interpretation

Current defensible statement:

> `V19` is not a null rewrite. It changes the prediction semantics in the
> intended direction: the release branch is now trained against a history-like
> proxy instead of only living inside a cleaner representation geometry. On
> `weather seed42 b240`, the base configuration produces a nonzero aggregate
> `raw_residual_vector` signal, but two follow-up loss sweeps show that this
> signal is fragile and can be destroyed by simply pushing more weight into the
> release branch. Therefore `V19` is a meaningful new version line, but the
> current bottleneck is already shifting from loss-weight tuning to structural
> branch allocation.

What this means in practice:

- `V19` is code-complete enough to run formal probes.
- `V19` is not stuck at exact-zero teacher outputs.
- Base `V19` currently improves aggregate validation behavior on
  `raw_residual_vector`, but the signal is not stable enough across windows.
- `residual_change_vector` degraded in the first `b240` run and remained weak
  under both narrow loss sweeps.
- The first narrow sweep result is informative: stronger release-oriented loss
  weights do not rescue the primary target and instead make the model worse.

## Current limitations

1. No formal `V19` run has passed the strict teacher gate.
2. Base `V19` on `weather seed42 b240` shows only a weak aggregate
   `raw_residual_vector` signal and poor cross-window stability.
3. `residual_change_vector` did not improve in the base run and stayed weak in
   both `moderate_release` and `release_focus`.
4. Two narrow loss sweeps indicate that the next failure mode is not trivially
   fixable by scalar reweighting inside the current objective family.
5. `V19` has not yet been compared against `V17` or `V18` under matched formal
   budgets beyond these `b240` screens.
6. Generator and online paths exist at the code level, but remain scientifically
   inactive until the teacher gate passes.

## Recommended next actions

1. Keep `V19` as the new official line for continued work, because it upgrades
   the training objective rather than only the geometry.
2. Stop broad scalar tuning inside the current `V19` loss family. The three
   `b240` screens are enough to show that the current bottleneck is no longer
   just loss-weight selection.
3. Compare base `V19` directly against:
   - `V18 balanced b240`
   - `V17 weather seed42 b480`
4. Make the next change structural rather than scalar. The most likely bounded
   targets are:
   - reduce how much `expert_y_hist` can absorb before `expert_y_var`
   - change router coupling between release confidence and expert allocation
   - audit whether the local history-proxy branch is learning a shortcut that
     suppresses release-state usefulness

## Superpowers and subagent note

This work followed the superpowers workflow:

- design before edits
- version-safe implementation
- deterministic tests before experiment claims
- experiment evidence written back to audit docs

The repository instructions also requested subagents when possible. In this
session there is still no executable child-agent dispatch tool available, so
the work used subagent-style decomposition but not real runnable subagent
execution.
