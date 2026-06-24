# Stage III V20 Corrective Budgeted Dual-Stream Implementation Record

## Scope

This record tracks the first full implementation and first experiment slice for
`V20`, a new official version line that pushes the user idea further than
`V19`.

`V20` was introduced because:

- the two Chinese audit reports argued that environment inference should act as
  a correction / suppression operator rather than directly defining
  `phi_i`;
- `V19` already moved the objective toward conditional incremental value, but
  its `history` and `release` branches still competed too freely;
- the next bottleneck appeared structural, not merely scalar.

Therefore `V20` changes the expert-local structure itself.

## What V20 adds beyond V19

`V20` keeps the expert-local shared representation idea, but changes the
decomposition to:

- shared expert-local vector
- invariant base
- invariant adapter
- environment correction operator
- `phi_i = base + invariant_adapter - env_correction`
- history proposal branch
- release proposal branch
- explicit residual budget allocator between history and release

This means the native drift source is no longer treated as a free subtraction
artifact. The model-native signal is the budgeted release state:

- `v20_model_native_drift_source = pooled_budgeted_release`

Compatibility with the existing Stage III toolchain is preserved through:

- `pooled_delta_phi_v = pooled_budgeted_release`
- `pooled_phi_v = pooled_phi_i + pooled_budgeted_release`

## Files added

### Design and planning

- `docs/superpowers/specs/2026-06-10-stage3-v20-corrective-budgeted-dual-stream-design.md`
- `docs/superpowers/plans/2026-06-10-stage3-v20-corrective-budgeted-dual-stream.md`

### Core model and training

- `models/factor_mechanism_v20.py`
- `train/offline_factor_v20.py`
- `train/run_stage3_v20_corrective_budgeted_probe.py`
- `configs/stage3_v20_corrective_budgeted_probe.yaml`
- `configs/stage3_v20_corrective_budgeted_probe_release_priority.yaml`

### Generator / artifact / online path

- `train/stage3_generator_training_v20.py`
- `train/stage3_v20_artifacts.py`
- `train/online_loop_stage3_v20.py`

### Tests

- `tests/test_factor_mechanism_v20.py`
- `tests/test_offline_factor_v20.py`
- `tests/test_stage3_v20_corrective_budgeted.py`
- `tests/test_stage3_generator_training_v20.py`

## Deterministic validation

Completed checks:

- `python -m py_compile models\factor_mechanism_v20.py train\offline_factor_v20.py train\stage3_generator_training_v20.py train\stage3_v20_artifacts.py train\online_loop_stage3_v20.py train\run_stage3_v20_corrective_budgeted_probe.py tests\test_factor_mechanism_v20.py tests\test_offline_factor_v20.py tests\test_stage3_v20_corrective_budgeted.py tests\test_stage3_generator_training_v20.py`
- `pytest tests\test_factor_mechanism_v20.py tests\test_offline_factor_v20.py tests\test_stage3_v20_corrective_budgeted.py tests\test_stage3_generator_training_v20.py -q`
  - result: `11 passed`
- `pytest tests\test_factor_mechanism_v19.py tests\test_offline_factor_v19.py tests\test_stage3_v19_conditional_release.py tests\test_stage3_generator_training_v19.py tests\test_factor_mechanism_v20.py tests\test_offline_factor_v20.py tests\test_stage3_v20_corrective_budgeted.py tests\test_stage3_generator_training_v20.py -q`
  - result: `23 passed`

The tests confirm:

- `phi_i = base + invariant_adapter - env_correction`
- `pooled_delta_phi_v = pooled_budgeted_release`
- `expert_y_hat = expert_y_inv + expert_y_hist + expert_y_var`
- `history_budget + release_budget = 1`
- generator targeting remains constrained to:
  - `history_output_heads`
  - `variant_output_heads`

## Smoke run

Command:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v20_corrective_budgeted_probe.py --config configs\stage3_v20_corrective_budgeted_probe.yaml --device cpu --epochs 1 --max-train-batches 1 --max-generator-batches 1 --max-eval-batches 1 --validation-windows 2 --run-id v20_smoke_01 --out-root logs\s3v20expert_smoke --checkpoint-root ckpt\s3v20expert_smoke --overwrite
```

Output directory:

- `logs/s3v20expert_smoke/v20_smoke_01/weather/seed_42`

Key contract checks:

- `run_manifest_v20.json` exists
- `teacher_only_v20_target_gate.csv` exists
- `teacher_only_abort.json` exists
- `protocol_version = v20_corrective_budgeted_probe`
- `v20_model_native_drift_source = pooled_budgeted_release`

Smoke outcome:

- stop reasons:
  - `do_not_train_generator`
  - `no_primary_target_passed`

This is acceptable for smoke. The important result is that the full `V20`
pipeline contract is intact.

## First meaningful run: base V20 on `weather seed42 b240`

Command:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v20_corrective_budgeted_probe.py --config configs\stage3_v20_corrective_budgeted_probe.yaml --epochs 5 --max-train-batches 240 --max-generator-batches 240 --max-eval-batches 240 --validation-windows 5 --run-id v20_weather_s42_b240_01 --out-root logs\s3v20expert_screen --checkpoint-root ckpt\s3v20expert_screen --overwrite
```

Output directory:

- `logs/s3v20expert_screen/v20_weather_s42_b240_01/weather/seed_42`

Gate summary:

- `raw_residual_vector`
  - `positive_partial_window_count = 1 / 5`
  - `min_partial_mse_gain_vs_history = -0.07795187018159`
  - `min_partial_mae_gain_vs_history = -0.07137102289241079`
  - `counterfactual_margin = -0.09156975486390212`
  - gate failed
- `residual_change_vector`
  - `positive_partial_window_count = 0 / 5`
  - `min_partial_mse_gain_vs_history = 0.0`
  - `min_partial_mae_gain_vs_history = 0.0`
  - gate failed
- `residual_sign`
  - `positive_partial_window_count = 4 / 5`
  - gate failed

Validation partial-gain summary:

- `raw_residual_vector`
  - `partial_mse_gain_vs_history = -0.01972691923212508`
  - `partial_mae_gain_vs_history = -0.007688067933258158`
- `residual_change_vector`
  - `partial_mse_gain_vs_history = 0.0`
  - `partial_mae_gain_vs_history = 0.0`
- `residual_sign`
  - `partial_mse_gain_vs_history = 0.046664535999298096`
  - `partial_mae_gain_vs_history = 0.02708524465560913`

Immediate interpretation:

- Base `V20` is structurally coherent but empirically too conservative.
- It suppresses the primary signal too much.
- Relative to base `V19`, this first `V20` run is worse on
  `raw_residual_vector` and collapses `residual_change_vector` to zero.

## Targeted diagnostic: `release_priority` variant

Hypothesis:

> the first `V20` failure may come from an overly conservative residual-budget
> allocator, not from the correction-operator idea itself.

To test that, one focused diagnostic config was added:

- `configs/stage3_v20_corrective_budgeted_probe_release_priority.yaml`

Changes were narrow and hypothesis-driven:

- reduce `lambda_local_history_proxy`
- increase `lambda_local_release_increment`
- increase `lambda_budget_alignment`
- weaken `lambda_history_budget_cap`

Command:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v20_corrective_budgeted_probe.py --config configs\stage3_v20_corrective_budgeted_probe_release_priority.yaml --epochs 5 --max-train-batches 240 --max-generator-batches 240 --max-eval-batches 240 --validation-windows 5 --run-id v20rel_weather_s42_b240_01 --out-root logs\s3v20expert_screen --checkpoint-root ckpt\s3v20expert_screen --overwrite
```

Output directory:

- `logs/s3v20expert_screen/v20rel_weather_s42_b240_01/weather/seed_42`

Gate summary:

- `raw_residual_vector`
  - `positive_partial_window_count = 2 / 5`
  - `min_partial_mse_gain_vs_history = -0.09726963739004513`
  - `min_partial_mae_gain_vs_history = -0.08701397195473684`
  - `counterfactual_margin = -0.09155519219640097`
  - gate failed
- `residual_change_vector`
  - `positive_partial_window_count = 0 / 5`
  - `min_partial_mse_gain_vs_history = 0.0`
  - `min_partial_mae_gain_vs_history = 0.0`
  - gate failed
- `residual_sign`
  - `positive_partial_window_count = 4 / 5`
  - gate failed

Validation partial-gain summary:

- `raw_residual_vector`
  - `partial_mse_gain_vs_history = -0.006590608917462576`
  - `partial_mae_gain_vs_history = 0.011869996814671269`
- `residual_change_vector`
  - `partial_mse_gain_vs_history = 0.0`
  - `partial_mae_gain_vs_history = 0.0`
- `residual_sign`
  - `partial_mse_gain_vs_history = 0.07836782932281494`
  - `partial_mae_gain_vs_history = 0.0626598596572876`

Interpretation:

- This diagnostic partially recovers `raw_residual_vector` relative to base
  `V20`.
- It does **not** recover `residual_change_vector`.
- The teacher gate still blocks generator training.

This means the allocator is not irreparably broken, but the current `V20`
structure still fails to produce a strong primary residual-release signal.

## Position against V17 to V19

Current `weather` reading:

- `V17` remains the strongest empirical weather line because it is still the
  only version that produced a cross-window positive
  `residual_change_vector` signal, even though strict counterfactual gating
  still blocked generator training.
- Base `V19` produced a weak positive aggregate `raw_residual_vector` signal on
  `weather seed42 b240`, but not a stable gate pass.
- Base `V20` is more faithful to the user idea than `V19`, but its first run
  is weaker than base `V19`.
- `V20 release_priority` recovers some `raw_residual_vector` behavior but still
  does not challenge `V17` on the strongest primary target.

So the defensible ordering is:

1. `V17` as the current empirical weather benchmark
2. `V19` as the current objective-redesign line with weak aggregate raw signal
3. `V20 release_priority` as a more principled correction-operator line that
   partially recovers raw signal
4. base `V20` as the over-conservative first structural instantiation

## Current interpretation

Current defensible statement:

> `V20` is the cleanest implementation so far of the user's original idea:
> share expert-local representations first, then use environment-aware
> correction to form `phi_i`, and use a separate release branch with explicit
> residual budget allocation. The code path is complete and the first formal
> experiments show that the structure is active, not null. However, on
> `weather seed42 b240`, the current instantiation is still too conservative on
> primary residual targets. A release-priority diagnostic partially restores
> `raw_residual_vector` behavior, but the strongest target,
> `residual_change_vector`, remains absent. Therefore `V20` is a serious new
> official version line, but not yet a validated Stage III replacement.

## Current limitations

1. No formal `V20` run has passed the strict teacher gate.
2. Base `V20` underperforms base `V19` on `raw_residual_vector`.
3. `residual_change_vector` stayed at zero in both `V20` runs.
4. `V20 release_priority` only partially recovers `raw_residual_vector` and
   does not resolve the main bottleneck.
5. Generator and online paths exist and are tested, but remain scientifically
   inactive until the teacher gate passes.

## Recommended next actions

1. Keep `V20` as a live official line because it is the most faithful
   implementation of the user idea so far.
2. Do not broaden into another large weight sweep. `V20` has already shown that
   its next issue is structural, not purely scalar.
3. The next bounded structural target should be inside the allocator itself:
   - give the release branch direct access to the unbudgeted residual after the
     invariant path but before history compression
   - reduce how much `history` can mask `release` on high env-shift segments
   - consider routing that depends on correction geometry but not residual
     budget state
4. Keep `V17` as the immediate empirical benchmark to beat in subsequent `V20`
   iterations.

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
