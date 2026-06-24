# Stage III V16 Expert-Local Dual-Stream Implementation Record

## Status

V16 has been introduced as a new non-overwriting Stage III line.

This record documents:

- what was implemented,
- what was validated,
- what the first smoke run showed,
- what remains before any paper-grade claim is possible.

## New versioned files

### Specs and plans

- `docs/superpowers/specs/2026-06-10-stage3-v16-expert-local-dual-stream-design.md`
- `docs/superpowers/plans/2026-06-10-stage3-v16-expert-local-dual-stream.md`

### Model and training

- `models/factor_mechanism_v16.py`
- `train/offline_factor_v16.py`
- `train/online_loop_stage3_v16.py`
- `train/stage3_generator_training_v16.py`
- `train/stage3_v16_artifacts.py`
- `train/run_stage3_v16_expert_local_dual_stream_probe.py`
- `configs/stage3_v16_expert_local_dual_stream_probe.yaml`

### Tests

- `tests/test_factor_mechanism_v16.py`
- `tests/test_offline_factor_v16.py`
- `tests/test_stage3_v16_expert_local_dual_stream.py`
- `tests/test_stage3_generator_training_v16.py`

## What V16 changes

V16 moves invariant/variant decomposition into each expert branch.

The base model now emits:

- `pooled_phi_i`
- `pooled_delta_phi_v`
- `pooled_phi_v`
- `y_inv`
- `y_var`
- `y_hat = y_inv + y_var`

This removes the hard dependency on an external aligned representation model for the first V16 diagnostic path.

This pass also adds the missing second-half V16 plumbing:

- model-native V16 generator training,
- V16 artifact bundle loading,
- candidate checkpoint selection on validation replay,
- V16 risk-gate example collection and training,
- V16 online replay using selected generator artifacts.

## Validation completed

### Static validation

Ran:

```bash
python -m py_compile models\factor_mechanism_v16.py train\offline_factor_v16.py train\online_loop_stage3_v16.py train\stage3_generator_training_v16.py train\stage3_v16_artifacts.py train\run_stage3_v16_expert_local_dual_stream_probe.py tests\test_factor_mechanism_v16.py tests\test_offline_factor_v16.py tests\test_stage3_v16_expert_local_dual_stream.py tests\test_stage3_generator_training_v16.py
```

Result:

- passed

### Focused pytest validation

Ran:

```bash
pytest tests\test_factor_mechanism_v16.py tests\test_offline_factor_v16.py tests\test_stage3_v16_expert_local_dual_stream.py tests\test_stage3_generator_training_v16.py -q
```

Result:

- `13 passed`

Covered:

- V16 forward/decomposition behavior
- deterministic eval behavior
- offline training-step execution
- V16 config contract
- V16 native drift extraction
- V16 online causal batch rejection
- V16 generator construction on output-head targets
- V16 generator candidate checkpoint writing
- V16 risk-gate example collection
- V16 artifact-bundle round trip

### Config-level smoke probes

Ran:

```bash
C:\Users\admin\anaconda3\python.exe train\run_stage3_v16_expert_local_dual_stream_probe.py --config configs\stage3_v16_expert_local_dual_stream_probe.yaml --device cpu --epochs 1 --max-train-batches 1 --max-generator-batches 1 --max-eval-batches 1 --validation-windows 2 --run-id v16_smoke_01 --out-root logs\s3v16expert_smoke --checkpoint-root ckpt\s3v16expert_smoke --overwrite
```

Result:

- command completed
- output directory created:
  `logs/s3v16expert_smoke/v16_smoke_01/weather/seed_42`

Artifacts observed:

- `offline_metrics_v16.csv`
- `residual_teacher_v16_diagnostics.csv`
- `stage3_v16_signal_contract_diagnostics.csv`
- `stage3_v16_history_contract_diagnostics.csv`
- `teacher_only_validation_summary_v16.csv`
- `teacher_only_test_summary_v16.csv`
- `teacher_only_rolling_validation_aggregate_v16.csv`
- `teacher_only_v16_target_gate.csv`
- `teacher_only_abort.json`
- `run_manifest_v16.json`

Ran again after adding V16 generator/artifact plumbing:

```bash
C:\Users\admin\anaconda3\python.exe train\run_stage3_v16_expert_local_dual_stream_probe.py --config configs\stage3_v16_expert_local_dual_stream_probe.yaml --device cpu --epochs 1 --max-train-batches 1 --max-generator-batches 1 --max-eval-batches 1 --validation-windows 2 --run-id v16_smoke_02 --out-root logs\s3v16expert_smoke --checkpoint-root ckpt\s3v16expert_smoke --overwrite
```

Result:

- command completed
- output directory created:
  `logs/s3v16expert_smoke/v16_smoke_02/weather/seed_42`
- the run still stopped at the teacher gate, which is the expected scientific behavior under a one-epoch one-batch budget
- `run_manifest_v16.json` now explicitly records:
  - `stage3_stop_reasons`
  - `stage3_generator_enabled`
  - `stage3_generator_candidate_count`
  - `stage3_generator_checkpoint_sha256`
  - `stage3_risk_gate_checkpoint_sha256`
  - `stage3_risk_gate_threshold`
  - `stage3_online_summary_rows`

## Smoke-run interpretation

The smoke run was not a success claim run.

It behaved as a diagnostic canary should:

- base V16 training path executed;
- model-native `pooled_delta_phi_v` was extracted;
- V15-style teacher-only target-predictability diagnostics ran;
- generator training was not unlocked.

Recorded stop file:

```json
{
  "stop_reasons": [
    "do_not_train_generator",
    "no_primary_target_passed"
  ],
  "protocol_version": "v16_expert_local_dual_stream_probe"
}
```

This is scientifically acceptable for a first smoke run. It does **not** mean V16 has failed as an implementation. It means the tiny smoke budget did not produce primary-target evidence, which is expected under one-epoch, one-batch constraints.

An additional bug was found and fixed during this pass:

- `train/stage3_signals.py` originally did not recognize `fluctuation_delta` or `level_offset` even though V13-V16 configs rely on those names.
- That mismatch would have broken V16 generator training and V16 online replay while leaving teacher-only diagnostics apparently healthy.
- The signal dispatcher now accepts these V13+ aliases, restoring training-serving signal compatibility for the V16 line.

## What is verified vs not verified

### Verified

- V16 exists as a separate, non-overwriting code line.
- V16 base model integrates expert-local dual-stream invariant learning.
- V16 offline training loop runs with integrated representation losses.
- V16 online replay path exists and consumes internal model-native representations.
- V16 generator training can consume internal `pooled_phi_i / pooled_phi_v` without an external aligned model.
- V16 artifact loading now has a dedicated version-safe bundle.
- V16 risk-gate example collection and checkpoint loading are wired for the new line.
- V16 probe runner produces versioned artifacts and negative-stop outputs rather than crashing.

### Not yet verified

- full-budget V16 residual-target predictability
- generator training usefulness under V16
- served online gain versus V15 or V2 baselines
- manuscript-grade ablation and cross-dataset stability

## Residual risks

- the current V16 design may still fail the same `history_only` barrier seen in V15;
- environment posterior may collapse without stronger regularization or better schedules;
- current smoke run is too small to support any paper claim.

## Medium-budget formal probe results

After the smoke pass, two medium-budget runs were executed on the `weather`
dataset with the predeclared V16 probe protocol.

Commands:

```bash
C:\Users\admin\anaconda3\python.exe train\run_stage3_v16_expert_local_dual_stream_probe.py --config configs\stage3_v16_expert_local_dual_stream_probe.yaml --epochs 5 --max-train-batches 480 --max-generator-batches 480 --max-eval-batches 480 --validation-windows 5 --run-id v16_weather_s42_b480_01 --out-root logs\s3v16expert_b480 --checkpoint-root ckpt\s3v16expert_b480 --overwrite
```

```bash
C:\Users\admin\anaconda3\python.exe train\run_stage3_v16_expert_local_dual_stream_probe.py --config configs\stage3_v16_expert_local_dual_stream_probe.yaml --seed 43 --epochs 5 --max-train-batches 480 --max-generator-batches 480 --max-eval-batches 480 --validation-windows 5 --run-id v16_weather_s43_b480_01 --out-root logs\s3v16expert_b480 --checkpoint-root ckpt\s3v16expert_b480 --overwrite
```

Output directories:

- `logs/s3v16expert_b480/v16_weather_s42_b480_01/weather/seed_42`
- `logs/s3v16expert_b480/v16_weather_s43_b480_01/weather/seed_43`

### Verified gate outcomes

Both runs completed end-to-end and both stopped at the teacher gate:

```json
{
  "stop_reasons": [
    "do_not_train_generator",
    "no_primary_target_passed"
  ],
  "protocol_version": "v16_expert_local_dual_stream_probe"
}
```

This means the new V16 generator/risk-gate/online branches remain
scientifically blocked, not broken.

### Seed 42 summary

Primary gate file:
`logs/s3v16expert_b480/v16_weather_s42_b480_01/weather/seed_42/teacher_only_v16_target_gate.csv`

- `raw_residual_vector`
  - `positive_partial_window_count = 3 / 5`
  - `min_partial_mse_gain_vs_history = -0.029448871716980518`
  - `min_partial_mae_gain_vs_history = 0.003027508394576217`
  - `counterfactual_margin = -0.0868974031974975`
  - `primary_gate_pass = False`
- `residual_change_vector`
  - `positive_partial_window_count = 5 / 5`
  - `min_partial_mse_gain_vs_history = 0.0020278395764003294`
  - `min_partial_mae_gain_vs_history = 0.002645771118558394`
  - `counterfactual_margin = -0.00037911838841161744`
  - `primary_gate_pass = False`

Interpretation:

- `residual_change_vector` showed a weak but consistent positive
  `fluctuation_plus_history` gain over `history_only` across rolling windows.
- That signal still failed the predeclared counterfactual-attribution gate.
- So seed 42 provides **weak positive evidence for incremental predictability**
  on `residual_change_vector`, but **not enough evidence to unlock generator
  training**.

### Seed 43 summary

Primary gate file:
`logs/s3v16expert_b480/v16_weather_s43_b480_01/weather/seed_43/teacher_only_v16_target_gate.csv`

- `raw_residual_vector`
  - `positive_partial_window_count = 1 / 5`
  - `min_partial_mse_gain_vs_history = -0.17767975321713048`
  - `min_partial_mae_gain_vs_history = -0.14144781282135369`
  - `counterfactual_margin = -0.05910561221222982`
  - `primary_gate_pass = False`
- `residual_change_vector`
  - `positive_partial_window_count = 5 / 5`
  - `min_partial_mse_gain_vs_history = 0.0006038909380198163`
  - `min_partial_mae_gain_vs_history = 0.00045795649448362617`
  - `counterfactual_margin = -0.002646778304405098`
  - `primary_gate_pass = False`

Interpretation:

- `residual_change_vector` again showed a small positive partial gain versus
  `history_only`, which reproduces the direction seen in seed 42.
- The effect size is smaller than seed 42.
- `raw_residual_vector` is unstable and worse than `history_only`.
- The counterfactual margin remains negative, so attribution is still not
  credible enough under the declared protocol.

### Cross-seed conclusion

Across `seed_42` and `seed_43` on `weather` with a `b480` budget:

- V16 does **not** pass the teacher gate on either primary target.
- `raw_residual_vector` is not stable enough to support a deployment claim.
- `residual_change_vector` shows a **repeatable but weak** positive
  `fluctuation_plus_history` signal relative to `history_only`.
- That weak signal is still rejected by the stricter counterfactual-attribution
  criterion.

This is the strongest scientifically defensible current statement:

> V16 has moved from a pure smoke-stage implementation to a reproducible
> medium-budget diagnostic regime where `residual_change_vector` exhibits weak
> cross-seed incremental signal, but the new expert-local dual-stream drift
> state still does not meet the repository's strict evidence threshold for
> generator training or served online correction.

### What these runs do and do not prove

These medium-budget runs **do prove**:

- the V16 line is functional beyond toy smoke settings;
- the V16 teacher-only protocol can surface nonzero target-level structure from
  model-native drift states;
- the new design is not trivially equivalent to zero signal.

These medium-budget runs **do not prove**:

- that V16 beats V15 on paper-grade residual-target predictability;
- that V16 supports safe generator training;
- that V16 improves served online adaptation;
- that the expert-local environment/invariant factorization is already
  mathematically sufficient.

### Current recommendation after b480

Do **not** write manuscript claims that V16 has solved Stage III.

The defensible next step is:

1. treat `residual_change_vector` as the only currently promising V16 primary
   target;
2. keep the strict counterfactual gate unchanged;
3. run one higher-budget confirmation pass or one second dataset pass before
   changing the scientific narrative.

## Public load-benchmark check: Panama b480 seed42

To test whether the weak `weather` signal transfers into a more paper-relevant
load benchmark, one additional medium-budget run was executed on the public
`panama` dataset.

Command:

```bash
C:\Users\admin\anaconda3\python.exe train\run_stage3_v16_expert_local_dual_stream_probe.py --config configs\stage3_v16_expert_local_dual_stream_probe.yaml --dataset panama --seed 42 --epochs 5 --max-train-batches 480 --max-generator-batches 480 --max-eval-batches 480 --validation-windows 5 --run-id v16_panama_s42_b480_01 --out-root logs\s3v16expert_b480 --checkpoint-root ckpt\s3v16expert_b480 --overwrite
```

Output directory:

- `logs/s3v16expert_b480/v16_panama_s42_b480_01/panama/seed_42`

Verified gate outcome:

- the run completed end-to-end;
- `teacher_only_abort.json` again reports:
  - `do_not_train_generator`
  - `no_primary_target_passed`

Primary gate file:
`logs/s3v16expert_b480/v16_panama_s42_b480_01/panama/seed_42/teacher_only_v16_target_gate.csv`

- `raw_residual_vector`
  - `positive_partial_window_count = 0 / 5`
  - `min_partial_mse_gain_vs_history = 0.0`
  - `min_partial_mae_gain_vs_history = 0.0`
  - `counterfactual_margin = 0.0`
  - `primary_gate_pass = False`
- `residual_change_vector`
  - `positive_partial_window_count = 1 / 5`
  - `min_partial_mse_gain_vs_history = -0.0018147809653974202`
  - `min_partial_mae_gain_vs_history = -0.002662111026845451`
  - `counterfactual_margin = -0.00254581827865733`
  - `primary_gate_pass = False`

Validation partial-gain file:
`logs/s3v16expert_b480/v16_panama_s42_b480_01/panama/seed_42/teacher_only_validation_partial_gain_vs_history_v16.csv`

- `residual_change_vector` has:
  - `partial_mse_gain_vs_history = 0.0006738479908016144`
  - `partial_mae_gain_vs_history = -0.0007598407791469062`
  - `partial_sign_agreement_vs_history = -0.00451388955116272`

Interpretation:

- unlike `weather`, `panama` does **not** show a cross-window stable positive
  V16 signal on either primary target;
- `residual_change_vector` retains only a tiny mixed-sign effect;
- the `weather` weak-signal pattern therefore should be treated as
  dataset-conditional rather than as general evidence.

Current cross-dataset statement:

> At medium budget, V16 produces weak repeatable `residual_change_vector`
> signal on `weather`, but that behavior does not currently transfer to the
> `panama` public load benchmark under the same strict gate. The design remains
> scientifically interesting, but it is not yet a validated Stage III
> replacement.

## Superpowers / subagent note

The repository and top-level `AGENTS.md` explicitly prefer superpowers and bounded subagent routing.

During this implementation pass, the session exposed:

- superpowers skill instructions,
- local agent registry files under `C:\Users\admin\.codex\agents`,
- but **not** an executable `spawn_agent`-style tool in the active tool surface.

So this pass used the superpowers design/plan workflow and subagent-style task boundaries, but it did **not** actually dispatch runnable child agents. That limitation should be treated as a session/tooling constraint, not as completed subagent execution.

## Next recommended actions

1. Add a dedicated V16 generator-training path if V16 teacher diagnostics pass at larger budgets.
2. Run medium-budget V16 probes on at least one target dataset with pre-declared seeds and windows.
3. Compare V16 `history_only` vs `fluctuation_plus_history` before any online-serving claim.
4. Only after that evaluate served policy with risk gate and blend on V16-native drift states.
