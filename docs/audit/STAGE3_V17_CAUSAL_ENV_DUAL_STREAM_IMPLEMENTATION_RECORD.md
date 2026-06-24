# Stage III V17 Causal-Env Dual-Stream Implementation Record

Date: 2026-06-10

## Scope

This record covers the first implementation pass of a new `V17` line that
extends `V16` rather than overwriting it.

The user-requested design direction was:

- keep the expert-local dual-stream invariant idea;
- strengthen environment inference so it is more causally defensible;
- avoid a toy sidecar implementation;
- create a new canonical next-generation version without breaking `V16` or
  earlier Stage III lines.

## Primary architectural change

`V16` already moved invariant/variant decomposition inside each expert, but its
environment inference was still a per-step feed-forward router with no explicit
causal prior state, and prediction supervision still happened mainly after
expert fusion.

`V17` changes the ML system boundary to:

`expert state -> causal environment prior/posterior filter -> expert-local invariant/drift decomposition -> expert-local predictions -> alpha fusion -> Stage III teacher-only probe`

New versioned files:

- `models/factor_mechanism_v17.py`
- `train/offline_factor_v17.py`
- `train/online_loop_stage3_v17.py`
- `train/run_stage3_v17_causal_env_dual_stream_probe.py`
- `configs/stage3_v17_causal_env_dual_stream_probe.yaml`
- `tests/test_factor_mechanism_v17.py`
- `tests/test_offline_factor_v17.py`
- `tests/test_stage3_v17_causal_env_dual_stream.py`

Old V16/V15/V2.x files were not removed or renamed.

## What V17 adds beyond V16

### 1. Causal environment filtering inside each expert

Each expert-local dual-stream head now includes a `CausalEnvironmentFilter`
that produces:

- `env_prior_logits_seq`
- `env_prior_posteriors_seq`
- `env_logits_seq`
- `env_posterior_seq`
- `env_shift_seq = posterior - prior`

This is materially different from `V16`:

- `V16` environment inference was per-step and memoryless.
- `V17` makes the environment signal explicitly causal and time-filtered.

### 2. Drift branch uses environment shift, not only environment level

`delta_phi_v` in `V17` is built from:

- a base drift branch;
- an environment-level modulation term;
- an environment-transition term driven by `posterior - prior`.

So the drift state is no longer only "environment-conditioned residual"; it is
also sensitive to *changes* in inferred regime.

### 3. Expert-local prediction supervision is now part of the base model

`V17` produces:

- `expert_y_inv`
- `expert_y_var`
- `expert_y_hat`

and the final output is:

- `y_hat = sum_k alpha_k * expert_y_hat_k`

This is a stronger realization of the user's idea than `V16`, because each
expert branch now carries its own invariant/drift prediction path before final
fusion.

### 4. Routing is conditioned on environment summaries

`V17` adds `env_router_bias`, so expert routing is not only a function of the
global latent `z`; it also depends on pooled expert-local environment shifts.

### 5. New training losses

`train/offline_factor_v17.py` adds:

- `loss_local_inv_forecast`
- `loss_local_residual`
- `loss_env_temporal`
- `loss_router_stability`

These are in addition to the aligned-representation-style losses reused from
earlier lines.

## Validation completed

### TDD sequence

New tests were written first and confirmed failing before implementation
because `V17` modules did not exist.

Observed failing command before implementation:

```bash
pytest tests\test_factor_mechanism_v17.py tests\test_offline_factor_v17.py tests\test_stage3_v17_causal_env_dual_stream.py -q
```

Failure mode:

- `ModuleNotFoundError: No module named 'models.factor_mechanism_v17'`

### Unit/integration verification after implementation

Targeted V17 tests:

```bash
pytest tests\test_factor_mechanism_v17.py tests\test_offline_factor_v17.py tests\test_stage3_v17_causal_env_dual_stream.py -q
```

Result:

- `9 passed`

Cross-version regression check:

```bash
pytest tests\test_factor_mechanism_v16.py tests\test_offline_factor_v16.py tests\test_stage3_v16_expert_local_dual_stream.py tests\test_factor_mechanism_v17.py tests\test_offline_factor_v17.py tests\test_stage3_v17_causal_env_dual_stream.py -q
```

Result:

- `18 passed`

Syntax verification:

```bash
python -m py_compile models\factor_mechanism_v17.py train\offline_factor_v17.py train\online_loop_stage3_v17.py train\run_stage3_v17_causal_env_dual_stream_probe.py
```

Result:

- passed

## Smoke run

Command:

```bash
C:\Users\admin\anaconda3\python.exe train\run_stage3_v17_causal_env_dual_stream_probe.py --config configs\stage3_v17_causal_env_dual_stream_probe.yaml --device cpu --epochs 1 --max-train-batches 1 --max-generator-batches 1 --max-eval-batches 1 --validation-windows 2 --run-id v17_smoke_01 --out-root logs\s3v17expert_smoke --checkpoint-root ckpt\s3v17expert_smoke --overwrite
```

Output directory:

- `logs/s3v17expert_smoke/v17_smoke_01/weather/seed_42`

Observed artifacts:

- `offline_metrics_v17.csv`
- `residual_teacher_v17_diagnostics.csv`
- `stage3_v17_signal_contract_diagnostics.csv`
- `stage3_v17_history_contract_diagnostics.csv`
- `teacher_only_validation_summary_v17.csv`
- `teacher_only_test_summary_v17.csv`
- `teacher_only_rolling_validation_aggregate_v17.csv`
- `teacher_only_v17_target_gate.csv`
- `teacher_only_abort.json`
- `run_manifest_v17.json`

Stop file:

```json
{
  "stop_reasons": [
    "do_not_train_generator",
    "no_primary_target_passed"
  ],
  "protocol_version": "v17_causal_env_dual_stream_probe"
}
```

Interpretation:

- the `V17` training and teacher-only diagnostic path is operational;
- the new version is protocol-safe at smoke budget;
- generator training was intentionally not unlocked by the evidence gate.

## Medium-budget formal experiments

### Weather seed42 b480

Command:

```bash
C:\Users\admin\anaconda3\python.exe train\run_stage3_v17_causal_env_dual_stream_probe.py --config configs\stage3_v17_causal_env_dual_stream_probe.yaml --epochs 5 --max-train-batches 480 --max-generator-batches 480 --max-eval-batches 480 --validation-windows 5 --run-id v17_weather_s42_b480_01 --out-root logs\s3v17expert_b480 --checkpoint-root ckpt\s3v17expert_b480 --overwrite
```

Output directory:

- `logs/s3v17expert_b480/v17_weather_s42_b480_01/weather/seed_42`

Primary gate file:

- `logs/s3v17expert_b480/v17_weather_s42_b480_01/weather/seed_42/teacher_only_v17_target_gate.csv`

Key results:

- `raw_residual_vector`
  - `positive_partial_window_count = 2 / 5`
  - `min_partial_mse_gain_vs_history = -0.06894322598520891`
  - `min_partial_mae_gain_vs_history = -0.0535088519343383`
  - `counterfactual_margin = -0.10546347647316584`
  - gate failed
- `residual_change_vector`
  - `positive_partial_window_count = 5 / 5`
  - `min_partial_mse_gain_vs_history = 0.00365798641761761`
  - `min_partial_mae_gain_vs_history = 0.002521005271483251`
  - `counterfactual_margin = -0.0016897684783019227`
  - gate failed

Validation partial-gain summary:

- `residual_change_vector`
  - `partial_mse_gain_vs_history = 0.015024360492013708`
  - `partial_mae_gain_vs_history = 0.01568583073432975`

Interpretation:

- compared with `V16 weather seed42 b480`, `V17` amplifies the
  `residual_change_vector` signal substantially;
- however the stricter counterfactual-attribution gate is still not passed.

### Weather seed43 b480

Command:

```bash
C:\Users\admin\anaconda3\python.exe train\run_stage3_v17_causal_env_dual_stream_probe.py --config configs\stage3_v17_causal_env_dual_stream_probe.yaml --seed 43 --epochs 5 --max-train-batches 480 --max-generator-batches 480 --max-eval-batches 480 --validation-windows 5 --run-id v17_weather_s43_b480_01 --out-root logs\s3v17expert_b480 --checkpoint-root ckpt\s3v17expert_b480 --overwrite
```

Output directory:

- `logs/s3v17expert_b480/v17_weather_s43_b480_01/weather/seed_43`

Primary gate file:

- `logs/s3v17expert_b480/v17_weather_s43_b480_01/weather/seed_43/teacher_only_v17_target_gate.csv`

Key results:

- `raw_residual_vector`
  - `positive_partial_window_count = 3 / 5`
  - `min_partial_mse_gain_vs_history = -0.0062340355006325066`
  - `min_partial_mae_gain_vs_history = -0.008938198234835992`
  - `counterfactual_margin = -0.07904187093184228`
  - gate failed
- `residual_change_vector`
  - `positive_partial_window_count = 0 / 5`
  - all partial-gain gate statistics effectively zero
  - gate failed

Validation partial-gain summary:

- `raw_residual_vector`
  - `partial_mse_gain_vs_history = 0.09422509044846933`
  - `partial_mae_gain_vs_history = 0.07946953186022049`
  - `partial_sign_agreement_vs_history = 0.04218745231628418`
- `residual_change_vector`
  - all zeros

Interpretation:

- `V17` changes the signal topology relative to `V16`.
- In seed42, the main visible signal is `residual_change_vector`.
- In seed43, the visible gain shifts toward `raw_residual_vector` aggregate
  partial gain, but the rolling gate and counterfactual tests still reject it.

## Comparative interpretation: V16 vs V17

Current defensible statement:

> `V17` is not yet a validated Stage III replacement, but it is not a null
> rewrite either. Relative to `V16`, it changes the residual-target behavior in
> a way that is consistent with the intended mechanism: stronger causal
> environment modeling and expert-local supervision can amplify target-level
> incremental signal. However, the amplified signal is still unstable across
> seeds and still fails the repository's strict counterfactual-attribution gate.

What appears improved:

- larger `residual_change_vector` partial gains on `weather seed42`
- environment inference is now causally structured instead of memoryless
- expert-local supervision is explicit in the base model rather than only after
  fusion

What remains unproven:

- cross-seed stability
- cross-dataset transfer
- counterfactual specificity
- safe generator training
- served online improvement

## V17 generator/risk/artifact path status

After the first V17 teacher-only pass, the version line was extended with a
dedicated back-half path instead of reusing V16 assumptions:

- `train/stage3_generator_training_v17.py`
- `train/stage3_v17_artifacts.py`
- `train/online_loop_stage3_v17.py`
- generator/risk-gate/online integration inside
  `train/run_stage3_v17_causal_env_dual_stream_probe.py`

Verification completed:

- targeted V17 generator-path tests passed:
  - `tests/test_stage3_generator_training_v17.py`
- combined V16+V17 regression passed:
  - `18 passed`
- a second smoke run with the integrated path also completed safely:
  - `logs/s3v17expert_smoke/v17_smoke_02/weather/seed_42`

Interpretation:

- the V17 version line is now a full protocol line at the code level:
  `base -> teacher -> generator/risk/artifact -> online wrapper`;
- however the evidence gate still prevented generator training from being
  unlocked in smoke or formal runs.

## Public load benchmark transfer: Panama seed42 b480

To check whether the V17 mechanism transfers beyond `weather`, one additional
medium-budget run was executed on the public `panama` dataset.

Command:

```bash
C:\Users\admin\anaconda3\python.exe train\run_stage3_v17_causal_env_dual_stream_probe.py --config configs\stage3_v17_causal_env_dual_stream_probe.yaml --dataset panama --seed 42 --epochs 5 --max-train-batches 480 --max-generator-batches 480 --max-eval-batches 480 --validation-windows 5 --run-id v17_panama_s42_b480_01 --out-root logs\s3v17expert_b480 --checkpoint-root ckpt\s3v17expert_b480 --overwrite
```

Output directory:

- `logs/s3v17expert_b480/v17_panama_s42_b480_01/panama/seed_42`

Verified gate outcome:

- the run completed end-to-end;
- `teacher_only_abort.json` again reports:
  - `do_not_train_generator`
  - `no_primary_target_passed`

Primary gate file:
`logs/s3v17expert_b480/v17_panama_s42_b480_01/panama/seed_42/teacher_only_v17_target_gate.csv`

- `raw_residual_vector`
  - `positive_partial_window_count = 5 / 5`
  - `min_partial_mse_gain_vs_history = 0.0024726218897780684`
  - `min_partial_mae_gain_vs_history = 0.0066659659057101625`
  - `counterfactual_margin = 0.0012869920323301948`
  - `primary_gate_pass = False`
- `residual_change_vector`
  - `positive_partial_window_count = 0 / 5`
  - `min_partial_mse_gain_vs_history = -0.04907615070421861`
  - `min_partial_mae_gain_vs_history = -0.028343973028232543`
  - `counterfactual_margin = 0.0018944491984944096`
  - `primary_gate_pass = False`

Validation partial-gain file:
`logs/s3v17expert_b480/v17_panama_s42_b480_01/panama/seed_42/teacher_only_validation_partial_gain_vs_history_v17.csv`

- `raw_residual_vector` has:
  - `partial_mse_gain_vs_history = 0.005422230047755178`
  - `partial_mae_gain_vs_history = 0.008928217976699633`
  - `partial_sign_agreement_vs_history = 0.003125004470348358`
- `residual_change_vector` has:
  - `partial_mse_gain_vs_history = -0.042183220661439466`
  - `partial_mae_gain_vs_history = -0.024635680756115212`
  - `partial_sign_agreement_vs_history = -0.0039930567145347595`

Interpretation:

- relative to `V16 panama seed42 b480`, V17 does not stay at zero on
  `raw_residual_vector`;
- V17 pushes `raw_residual_vector` to a weak but cross-window positive signal;
- however the counterfactual margin is still below the required `0.002`, so the
  strict gate still correctly rejects generator training;
- the signal pattern also differs from `weather`, where the main visible gain on
  one seed was `residual_change_vector`.

Current cross-dataset statement:

> V17 improves over V16 on `panama` in the narrow sense that it turns the
> previously null `raw_residual_vector` signal into a weak positive
> cross-window signal. But the improvement is still below the repository's
> counterfactual-attribution threshold, and the strongest target is not stable
> across datasets or seeds. V17 is therefore scientifically more promising than
> V16, but still not a validated Stage III replacement.

## Current limitations

1. All formal V17 runs still stopped at `teacher_only_abort.json`; the code path
   exists, but the evidence gate has not unlocked generator training.
2. The strongest target changes across runs:
   - `weather seed42`: `residual_change_vector`
   - `weather seed43`: `raw_residual_vector` aggregate partial gain
   - `panama seed42`: weak cross-window `raw_residual_vector`
3. The signal can be stronger than V16 without becoming decision-safe under the
   declared counterfactual protocol.
4. Served online improvement remains unverified because no formal run has yet
   passed the teacher gate.

## Recommended next actions

1. Run one more public load benchmark such as `tetouan` or
   `spain_energy_weather` under the same `b480` budget to test whether the weak
   `raw_residual_vector` improvement on `panama` transfers.
2. Increase the diagnostic focus on target instability itself:
   record which primary target is strongest by seed and dataset instead of
   assuming `residual_change_vector` remains dominant.
3. If a later run crosses the counterfactual margin on any primary target, use
   the existing V17 generator/risk/artifact path as-is for the next canary
   stage rather than redesigning the back-half again.

## Superpowers and subagent note

The implementation followed the superpowers workflow:

- brainstorming/design first,
- TDD for new files,
- version-safe non-overwriting changes,
- verification before success claims.

The repository-level `AGENTS.md` also requested subagent use when possible.
This session still did not expose an executable subagent-dispatch tool, so the
work used subagent-style decomposition but not actual runnable child-agent
execution.
