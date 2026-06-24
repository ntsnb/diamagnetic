# Stage III V21 Release-First Budgeted Implementation Record

## Scope

This record tracks the first implementation and first formal experiment slice
for `V21`.

`V21` exists because `V20` identified a likely structural bottleneck:

- the correction-operator idea itself was not null;
- but the budget allocator appeared too conservative;
- the history branch may still have been absorbing too much before the release
  branch could establish a primary signal.

So `V21` keeps the `V20` invariant correction operator but changes the branch
order:

- release claims residual first
- history only models the post-release remainder

## What V21 changes relative to V20

`V21` keeps:

- `phi_i = base + invariant_adapter - env_correction`
- environment inference as correction/suppression rather than direct invariant
  definition
- compatibility with the existing Stage III served-path contract:
  - `pooled_delta_phi_v = pooled_budgeted_release`
  - `pooled_phi_v = pooled_phi_i + pooled_budgeted_release`

But it changes the sequence:

1. compute release proposal and release budget first
2. build budgeted release state
3. update history branch on the release-adjusted remainder
4. let history model only the leftover stable residue

At the training level, `V21` also changes supervision order:

- release branch is trained against the residual after the invariant path
- history branch is trained against the residual after invariant plus release
- a release-priority advantage loss explicitly compares `inv + var` against
  `inv + hist`

## Files added

### Design and planning

- `docs/superpowers/specs/2026-06-10-stage3-v21-release-first-budgeted-design.md`
- `docs/superpowers/plans/2026-06-10-stage3-v21-release-first-budgeted.md`

### Core model and training

- `models/factor_mechanism_v21.py`
- `train/offline_factor_v21.py`
- `train/run_stage3_v21_release_first_budgeted_probe.py`
- `configs/stage3_v21_release_first_budgeted_probe.yaml`

### Generator / artifact / online path

- `train/stage3_generator_training_v21.py`
- `train/stage3_v21_artifacts.py`
- `train/online_loop_stage3_v21.py`

### Tests

- `tests/test_factor_mechanism_v21.py`
- `tests/test_offline_factor_v21.py`
- `tests/test_stage3_v21_release_first_budgeted.py`
- `tests/test_stage3_generator_training_v21.py`

## Deterministic validation

Completed checks:

- `python -m py_compile models\factor_mechanism_v21.py train\offline_factor_v21.py train\stage3_generator_training_v21.py train\stage3_v21_artifacts.py train\online_loop_stage3_v21.py train\run_stage3_v21_release_first_budgeted_probe.py tests\test_factor_mechanism_v21.py tests\test_offline_factor_v21.py tests\test_stage3_v21_release_first_budgeted.py tests\test_stage3_generator_training_v21.py`
- `pytest tests\test_factor_mechanism_v21.py tests\test_offline_factor_v21.py tests\test_stage3_v21_release_first_budgeted.py tests\test_stage3_generator_training_v21.py -q`
  - result: `11 passed`
- `pytest tests\test_factor_mechanism_v20.py tests\test_offline_factor_v20.py tests\test_stage3_v20_corrective_budgeted.py tests\test_stage3_generator_training_v20.py tests\test_factor_mechanism_v21.py tests\test_offline_factor_v21.py tests\test_stage3_v21_release_first_budgeted.py tests\test_stage3_generator_training_v21.py -q`
  - result: `22 passed`

The tests confirm:

- `phi_i = base + invariant_adapter - env_correction`
- `pooled_delta_phi_v = pooled_budgeted_release`
- `expert_y_hat = expert_y_inv + expert_y_var + expert_y_hist`
- `history_budget + release_budget = 1`
- generator targeting remains limited to:
  - `history_output_heads`
  - `variant_output_heads`

## Smoke run

Command:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v21_release_first_budgeted_probe.py --config configs\stage3_v21_release_first_budgeted_probe.yaml --device cpu --epochs 1 --max-train-batches 1 --max-generator-batches 1 --max-eval-batches 1 --validation-windows 2 --run-id v21_smoke_01 --out-root logs\s3v21expert_smoke --checkpoint-root ckpt\s3v21expert_smoke --overwrite
```

Output directory:

- `logs/s3v21expert_smoke/v21_smoke_01/weather/seed_42`

Smoke outcome:

- `run_manifest_v21.json` exists
- `teacher_only_v21_target_gate.csv` exists
- `teacher_only_abort.json` exists
- `protocol_version = v21_release_first_budgeted_probe`
- stop reasons:
  - `do_not_train_generator`
  - `no_primary_target_passed`

This is acceptable for smoke. The important result is that the full `V21`
pipeline contract is intact.

## First meaningful run: `weather seed42 b240`

Command:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v21_release_first_budgeted_probe.py --config configs\stage3_v21_release_first_budgeted_probe.yaml --epochs 5 --max-train-batches 240 --max-generator-batches 240 --max-eval-batches 240 --validation-windows 5 --run-id v21_weather_s42_b240_01 --out-root logs\s3v21expert_screen --checkpoint-root ckpt\s3v21expert_screen --overwrite
```

Output directory:

- `logs/s3v21expert_screen/v21_weather_s42_b240_01/weather/seed_42`

Gate summary:

- `raw_residual_vector`
  - `positive_partial_window_count = 2 / 5`
  - `min_partial_mse_gain_vs_history = -0.09581947461416859`
  - `min_partial_mae_gain_vs_history = -0.05792836157404739`
  - `counterfactual_margin = -0.0492276391028037`
  - gate failed
- `residual_change_vector`
  - `positive_partial_window_count = 0 / 5`
  - `min_partial_mse_gain_vs_history = 0.0`
  - `min_partial_mae_gain_vs_history = 0.0`
  - gate failed
- `residual_sign`
  - `positive_partial_window_count = 2 / 5`
  - gate failed

Validation partial-gain summary:

- `raw_residual_vector`
  - `partial_mse_gain_vs_history = -0.00536056546860289`
  - `partial_mae_gain_vs_history = 0.06256070653505785`
- `residual_change_vector`
  - `partial_mse_gain_vs_history = 0.0`
  - `partial_mae_gain_vs_history = 0.0`
- `residual_sign`
  - `partial_mse_gain_vs_history = 0.014377236366271973`
  - `partial_mae_gain_vs_history = 0.009754598140716553`

Abort outcome:

- `do_not_train_generator`
- `no_primary_target_passed`

## Interpretation of the first V21 result

The most important result is comparative:

- `V21` still does **not** unlock the strongest primary target
  `residual_change_vector`
- but relative to `V20`, it improves the raw target directionally

Concrete comparison on `weather seed42 b240`:

- base `V20`
  - `raw_residual_vector partial_mse_gain_vs_history = -0.01972691923212508`
  - `counterfactual_margin = -0.09156975486390212`
- `V20 release_priority`
  - `raw_residual_vector partial_mse_gain_vs_history = -0.006590608917462576`
  - `counterfactual_margin = -0.09155519219640097`
- `V21`
  - `raw_residual_vector partial_mse_gain_vs_history = -0.00536056546860289`
  - `counterfactual_margin = -0.0492276391028037`

So `V21` is not yet a gate-passing result, but it is a meaningful structural
improvement over `V20`:

- the raw target is less harmed
- the counterfactual attribution penalty is materially smaller

That is the strongest evidence so far that the release-first ordering is more
correct than the `V20` ordering.

## Position against V17 to V20

Current defensible ordering on `weather` is:

1. `V17` remains the empirical benchmark because it is still the only version
   with a cross-window positive `residual_change_vector` signal.
2. `V19` remains the best objective-redesign line on aggregate
   `raw_residual_vector`.
3. `V21` is now the strongest correction-operator line, because it improves on
   both base `V20` and `V20 release_priority`.
4. `V20 release_priority` showed the allocator could be loosened, but did not
   fix the target.
5. base `V20` is the most conservative and weakest of the `V20/V21`
   correction-operator family.

So the most accurate statement is:

- `V21` is not yet the empirical best line overall
- but it is the current best implementation of the user's structural idea
  inside the `V20+` family

## Current interpretation

Current defensible statement:

> `V21` is the first version that makes the release branch structurally first
> class inside the expert-local correction architecture. It preserves the
> invariant correction operator from `V20`, but changes the causal order so
> that history models the remainder after release instead of competing with it
> from the start. On `weather seed42 b240`, this does not yet unlock generator
> training, and `residual_change_vector` remains absent. However, it improves
> the raw target and reduces the counterfactual penalty relative to `V20`,
> which is real evidence that the branch-order fix is directionally correct.
> Therefore `V21` is a serious official next-generation line and a better
> structural basis than `V20`, but still not a validated Stage III replacement.

## Current limitations

1. No formal `V21` run has passed the strict teacher gate.
2. `residual_change_vector` remains missing.
3. `V21` improves `raw_residual_vector` relative to `V20`, but not enough to
   challenge `V17`.
4. Generator and online paths are implemented and tested, but scientifically
   inactive until the teacher gate passes.

## Recommended next actions

1. Keep `V21` as the active structural line for this idea.
2. Do not revert to `V20`; `V21` is a better basis.
3. The next bounded change should stay structural and be narrow:
   - increase release observability of post-invariant residual error
   - let the router see release-priority evidence instead of only release
     budget state
   - consider an explicit delayed-residual target branch for
     `residual_change_vector`
4. Keep `V17` as the immediate empirical benchmark and `V19` as the strongest
   aggregate raw-target comparison point.

## Superpowers and subagent note

This work followed the superpowers workflow:

- design before edits
- version-safe implementation
- deterministic tests before experiment claims
- formal probe execution before scientific interpretation
- evidence written back to audit docs

The repository instructions also requested subagents when possible. In this
session there is still no executable child-agent dispatch tool available, so
the work used subagent-style decomposition but not real runnable subagent
execution.
