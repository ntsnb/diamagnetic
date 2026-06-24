# STAGE3 V24 TAIL OBSERVABLE CHANGE IMPLEMENTATION RECORD

## Scope

This record documents the `V24` official version line added on `2026-06-11`.
`V24` does not overwrite `V16`-`V23`. It adds a new base-model and Stage III
probe line under a dedicated `s3v24expert` namespace.

`V24` was created after the first `V23` screens showed a very specific
pattern:

- `V23 weather seed42 b240` recovered a weak non-zero
  `residual_change_vector` signal.
- `V23 weather seed43 b240` did not preserve that gain.

The working diagnosis was that `V23` had improved observability, but its
observable signal was still too sequence-mean-like and could be washing out
late-window change structure.

`V24` therefore focuses on **tail-weighted observable change** rather than
another rewrite of the internal invariant/variant geometry.

## ML System Boundary

The changed ML boundary for this version is:

`V24 expert-local release state -> tail-weighted observable change -> Stage III teacher gate -> generator/risk-gate/online wrappers`

This is a training-serving consistency change at the drift-interface level.
The main question is not whether `phi_i / phi_v` can be parameterized more
elegantly, but whether the teacher-facing signal is more observable for
`residual_change_vector`.

## Core V24 Mechanism

`V24` keeps the `V23` expert-local release-state backbone:

- shared expert-local representation
- invariant anchor
- environment-conditioned correction
- release increment
- accumulated release state
- history branch after release state

The actual changes are:

1. Add an explicit sample-internal observable change sequence:
   - `observable_change_seq`
   - constructed as the time-difference of `budgeted_release_seq`
2. Keep the geometric release state:
   - `delta_phi_v = release_state_seq`
3. Define the teacher/serve-facing signal as:
   - `tail_observable_change_seq = concat(delta_phi_v, budgeted_release_seq, observable_change_seq, release_confidence_seq)`
4. Replace simple mean pooling with tail pooling for drift-relevant expert
   summaries, including:
   - `pooled_budgeted_release`
   - `pooled_release_state`
   - `pooled_delta_phi_v`
   - `pooled_phi_v`
   - `pooled_observable_change`
   - `pooled_tail_observable_change`
5. Let the router observe change magnitude explicitly by augmenting
   `router_features` with `pooled_observable_change_norm`
6. Expand the local variant readout so it sees:
   - release state
   - current budgeted release
   - observable change
   - release confidence
   - release budget
   - environment shift
7. Add an offline auxiliary objective:
   - `lambda_observable_change_alignment`
   - aligns expert observable-change magnitude with internal env-shift change
     magnitude without introducing cross-sample leakage

The authoritative model-native drift source for `V24` is:

- `pooled_tail_observable_change`

This is the signal consumed by:

- split-tensor collection
- teacher-only gate
- generator training wrapper
- online adaptation wrapper

## File Boundary

Files added for `V24`:

- `models/factor_mechanism_v24.py`
- `train/offline_factor_v24.py`
- `train/stage3_generator_training_v24.py`
- `train/stage3_v24_artifacts.py`
- `train/online_loop_stage3_v24.py`
- `train/run_stage3_v24_tail_observable_change_probe.py`
- `configs/stage3_v24_tail_observable_change_probe.yaml`
- `tests/test_factor_mechanism_v24.py`
- `tests/test_offline_factor_v24.py`
- `tests/test_stage3_v24_tail_observable_change.py`
- `tests/test_stage3_generator_training_v24.py`
- `docs/superpowers/specs/2026-06-11-stage3-v24-tail-observable-change-design.md`
- `docs/superpowers/plans/2026-06-11-stage3-v24-tail-observable-change.md`

Key implementation locations:

- `observable_change_seq` and `tail_observable_change_seq`
  - [models/factor_mechanism_v24.py](/C:/Users/admin/Desktop/最近/1.28/IEEE-Transactions-LaTeX2e-templates-and-instructions/models/factor_mechanism_v24.py:279)
- tail pooling helper
  - [models/factor_mechanism_v24.py](/C:/Users/admin/Desktop/最近/1.28/IEEE-Transactions-LaTeX2e-templates-and-instructions/models/factor_mechanism_v24.py:188)
- router features augmented with observable-change norm
  - [models/factor_mechanism_v24.py](/C:/Users/admin/Desktop/最近/1.28/IEEE-Transactions-LaTeX2e-templates-and-instructions/models/factor_mechanism_v24.py:514)
- observable-change alignment loss
  - [train/offline_factor_v24.py](/C:/Users/admin/Desktop/最近/1.28/IEEE-Transactions-LaTeX2e-templates-and-instructions/train/offline_factor_v24.py:42)
- generator wrapper drift extraction
  - [train/stage3_generator_training_v24.py](/C:/Users/admin/Desktop/最近/1.28/IEEE-Transactions-LaTeX2e-templates-and-instructions/train/stage3_generator_training_v24.py:28)
- probe protocol and manifest drift-source recording
  - [train/run_stage3_v24_tail_observable_change_probe.py](/C:/Users/admin/Desktop/最近/1.28/IEEE-Transactions-LaTeX2e-templates-and-instructions/train/run_stage3_v24_tail_observable_change_probe.py:197)

## Deterministic Validation

Completed checks:

- `python -m py_compile models\factor_mechanism_v24.py train\offline_factor_v24.py train\stage3_generator_training_v24.py train\stage3_v24_artifacts.py train\online_loop_stage3_v24.py train\run_stage3_v24_tail_observable_change_probe.py tests\test_factor_mechanism_v24.py tests\test_offline_factor_v24.py tests\test_stage3_v24_tail_observable_change.py tests\test_stage3_generator_training_v24.py`
- `pytest tests\test_factor_mechanism_v24.py tests\test_offline_factor_v24.py tests\test_stage3_v24_tail_observable_change.py tests\test_stage3_generator_training_v24.py -q`
  - result: `11 passed`
- `pytest tests\test_factor_mechanism_v23.py tests\test_offline_factor_v23.py tests\test_stage3_v23_observable_drift.py tests\test_stage3_generator_training_v23.py tests\test_factor_mechanism_v24.py tests\test_offline_factor_v24.py tests\test_stage3_v24_tail_observable_change.py tests\test_stage3_generator_training_v24.py -q`
  - result: `22 passed`

These checks verify:

- schema compatibility of the new drift tensors
- dynamic drift-dimension handling through the generator wrapper
- offline trainer logging of the new alignment term
- no immediate regression against the `V23` deterministic path

## Smoke Run

Smoke command completed:

- run id: `v24_smoke_01`
- output:
  [v24_smoke_01](/C:/Users/admin/Desktop/最近/1.28/IEEE-Transactions-LaTeX2e-templates-and-instructions/logs/s3v24expert_smoke/v24_smoke_01/weather/seed_42)

The smoke run completed the protocol and still stopped at:

- `do_not_train_generator`
- `no_primary_target_passed`

This was acceptable for smoke because the purpose was end-to-end protocol
validation for the new drift interface.

## Formal Screen 1: Weather seed42 b240

Formal screen output:

- run id: `v24_weather_s42_b240_01`
- directory:
  [v24_weather_s42_b240_01](/C:/Users/admin/Desktop/最近/1.28/IEEE-Transactions-LaTeX2e-templates-and-instructions/logs/s3v24expert_screen/v24_weather_s42_b240_01/weather/seed_42)

Key manifest facts:

- `protocol_version = v24_tail_observable_change_probe`
- `v24_model_native_drift_source = pooled_tail_observable_change`

Teacher gate summary:

- `raw_residual_vector`
  - `positive_partial_window_count = 1 / 5`
  - `min_partial_mse_gain_vs_history = -0.20619244340213305`
  - `min_partial_mae_gain_vs_history = -0.15181480985018714`
  - `counterfactual_margin = -0.04975623733051171`
  - gate: fail
- `residual_change_vector`
  - `positive_partial_window_count = 0 / 5`
  - `min_partial_mse_gain_vs_history = 0.0`
  - `min_partial_mae_gain_vs_history = 0.0`
  - `counterfactual_margin = 0.0`
  - gate: fail
- `residual_sign`
  - gate: fail

Validation partial gain vs history:

- `raw_residual_vector`
  - `history_only_real_mse_gain = 0.23065883109966354`
  - `fluctuation_plus_history_real_mse_gain = 0.2077850587394868`
  - `partial_mse_gain_vs_history = -0.022873772360176736`
  - `history_only_real_mae_gain = 0.2955441778671441`
  - `fluctuation_plus_history_real_mae_gain = 0.27522518554269987`
  - `partial_mae_gain_vs_history = -0.02031899232444423`
- `residual_change_vector`
  - `history_only_real_mse_gain = 0.0`
  - `fluctuation_plus_history_real_mse_gain = 0.0`
  - `partial_mse_gain_vs_history = 0.0`
  - `history_only_real_mae_gain = 0.0`
  - `fluctuation_plus_history_real_mae_gain = 0.0`
  - `partial_mae_gain_vs_history = 0.0`

Abort file:

- `do_not_train_generator`
- `no_primary_target_passed`

## Interpretation of the Formal Screen

`V24` is a negative result.

The new explicit observable-change channel did **not** strengthen the weak
`V23 seed42` change-target recovery. Instead:

- `raw_residual_vector` became slightly worse than `V23 seed42`
- `residual_change_vector` collapsed back to zero

This matters because the entire point of `V24` was to stabilize and amplify the
change-target observability hint seen in `V23`. It did not do that.

## Comparison Against V23, V22, and V21

Reference points on `weather seed42 b240`:

- `V24`
  - `raw partial_mse_gain_vs_history = -0.022873772360176736`
  - `raw counterfactual_margin = -0.04975623733051171`
  - `residual_change_vector partial_mse_gain_vs_history = 0.0`
- `V23`
  - `raw partial_mse_gain_vs_history = -0.022176874611375547`
  - `raw counterfactual_margin = -0.07100568366234472`
  - `residual_change_vector partial_mse_gain_vs_history = 0.0004171525744068808`
- `V22`
  - `raw partial_mse_gain_vs_history = -0.03921956662575807`
  - `raw counterfactual_margin = -0.20234545087930134`
  - `residual_change_vector partial_mse_gain_vs_history = 0.0`
- `V21`
  - `raw counterfactual_margin = -0.0492276391028037`
  - `residual_change_vector partial_mse_gain_vs_history = 0.0`

Precise position of `V24`:

- better than `V22` on raw-target harm
- comparable to `V21` on raw counterfactual margin
- worse than `V23` on the only encouraging signal, namely weak
  `residual_change_vector` recovery

So `V24` does not become the new lead line inside the `V20+` family.

## Current Interpretation

The strongest defensible reading is:

1. `V23` correctly identified observability as a bottleneck.
2. But replacing mean-like observable packaging with a tail-weighted
   explicit-change channel was not sufficient.
3. The bottleneck is now more likely in **how residual change is read out and
   supervised**, not merely in how it is aggregated.

In other words:

> `V24` falsifies the simpler hypothesis that a stronger tail-focused
> observable-change representation alone would stabilize the `V23`
> `residual_change_vector` hint.

## Current Limitations

1. `V24` has only been screened on `weather seed42 b240`.
2. Generator training remains blocked.
3. The line does not justify a second formal seed run at this stage because the
   first formal screen already regressed the key target.
4. No paper-grade positive claim is justified.

## Recommended Next Actions

1. Do not promote `V24` as the new mainline.
2. Keep `V23` as the stronger observability-oriented reference inside the
   `V20+` family.
3. If continuing to `V25`, focus on structure rather than more `V24` weight
   tuning:
   - dedicated `residual_change_vector` readout
   - delayed-change auxiliary target
   - router features that measure observable-drift quality, not only magnitude
4. Keep `V17` as the empirical weather reference until a newer line beats it
   more convincingly.

## Tooling Note

This work continued to use the superpowers workflow style: isolated version
lines, deterministic validation, smoke before formal screen, and evidence-first
writeback.

The current tool surface still does not expose a real child-agent dispatch
capability, so no actual executable subagent was spawned. That should not be
misreported.
