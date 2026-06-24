# Stage III V29 Expert-Local Invariant Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `V29` as the official next-generation implementation of expert-local invariant correction plus directly learned residual-release state, without overwriting `V16`-`V28`.

**Architecture:** `V29` reuses the mature `V26` correction backbone for train/serve compatibility, then adds a per-expert invariant-release corrector that starts from each expert's shared anchor state, performs environment inference, corrects the invariant representation, and learns a residual-release state consumed directly by Stage III. Native Stage III drift must be `pooled_servable_change`, not a naked `phi_v - phi_i` subtraction.

**Tech Stack:** PyTorch model modules under `models/`, Stage III probe runners under `train/`, YAML configs under `configs/`, deterministic tests under `tests/`, audit records under `docs/audit/`.

---

## File Structure

- Create `models/factor_mechanism_v29.py` for the official V29 model, config, output wrapper, and builder.
- Create `train/offline_factor_v29.py` for V29 loss config, V29-specific invariant/release losses, offline train loop, and evaluation.
- Create `train/run_stage3_v29_expert_local_invariant_release_probe.py` for config validation, model/loss builders, split tensor collection, and V18 runner monkey-patch boundary.
- Create `train/stage3_generator_training_v29.py` to normalize V29 generator names into existing V26 generator implementations while preserving V29 protocol names.
- Create `train/stage3_v29_artifacts.py` and `train/online_loop_stage3_v29.py` as versioned wrappers around the compatible V26 artifact/online path.
- Create `configs/stage3_v29_expert_local_invariant_release_probe.yaml` with `s3v29expert` roots and V29-specific model/loss keys.
- Create tests:
  - `tests/test_factor_mechanism_v29.py`
  - `tests/test_offline_factor_v29.py`
  - `tests/test_stage3_v29_expert_local_invariant_release.py`
  - `tests/test_stage3_generator_training_v29.py`
- Create `docs/audit/STAGE3_V29_EXPERT_LOCAL_INVARIANT_RELEASE_IMPLEMENTATION_RECORD.md` for the implementation and experiment ledger.

## Task 1: V29 Model

**Files:**

- Create: `models/factor_mechanism_v29.py`
- Test: `tests/test_factor_mechanism_v29.py`

- [ ] **Step 1: Define V29 output contract**

Expose at least these V29-native tensors:

```python
expert_shared_anchor
expert_local_env_logits
expert_local_env_posteriors
expert_invariant_correction
expert_invariant_correction_gate
expert_phi_i
expert_residual_release_state
expert_release_support_score
expert_release_suppression_score
expert_release_delta_y
expert_release_suppression_y
expert_servable_change
pooled_phi_i
pooled_residual_release_state
pooled_servable_change
```

- [ ] **Step 2: Build `ExpertLocalInvariantReleaseCorrector`**

For each expert, construct:

```text
shared_anchor = base.expert_base + base.expert_invariant_adapter
env_input = [shared_anchor, base.expert_env_correction, base.expert_release_budget,
             base.expert_release_confidence, base.expert_raw_release,
             base.expert_change_memory, base.expert_observable_change]
local_env_posterior = softmax(env_head(env_input))
phi_i = shared_anchor + invariant_residual - correction_gate * invariant_correction
residual_release_state = release_state_head([phi_i, env_input, local_env_posterior])
expert_y_var = base.expert_y_change_raw
             + support_score * release_delta_y
             - suppression_score * release_suppression_y
```

- [ ] **Step 3: Preserve compatibility aliases**

Expose:

```python
expert_delta_phi_v = expert_residual_release_state
pooled_delta_phi_v = pooled_residual_release_state
expert_phi_v = expert_phi_i + expert_residual_release_state
pooled_phi_v = pooled_phi_i + pooled_residual_release_state
expert_support_score = expert_release_support_score
expert_suppression_score = expert_release_suppression_score
expert_supported_delta = expert_release_delta_y
expert_suppression_vector = expert_release_suppression_y
```

These aliases exist for legacy losses and generator compatibility; the V29 audit must state that the native served signal is the residual-release state package, not the semantic claim that `phi_v - phi_i` is true drift.

- [ ] **Step 4: Write model tests**

Run:

```powershell
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_factor_mechanism_v29.py -q
```

Expected: all V29 shape, score-range, deterministic mean-mode, and formula tests pass.

## Task 2: V29 Offline Losses

**Files:**

- Create: `train/offline_factor_v29.py`
- Test: `tests/test_offline_factor_v29.py`

- [ ] **Step 1: Reuse V28/V26 verified losses**

Import V28 support/suppression losses and V26 release/history losses where compatible.

- [ ] **Step 2: Add V29-specific losses**

Add:

```python
loss_local_env_consistency
loss_invariant_correction_budget
loss_residual_release_alignment
loss_residual_release_advantage
loss_expert_release_orthogonal
loss_residual_state_noncollapse
```

- [ ] **Step 3: Log all V29 losses**

Each training row must include the six new loss keys and inherited V28 keys.

- [ ] **Step 4: Write offline training test**

Run:

```powershell
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_offline_factor_v29.py -q
```

Expected: a tiny training batch runs, returns rows, and logs all V29-specific losses.

## Task 3: Stage III V29 Probe Boundary

**Files:**

- Create: `train/run_stage3_v29_expert_local_invariant_release_probe.py`
- Create: `train/stage3_generator_training_v29.py`
- Create: `train/stage3_v29_artifacts.py`
- Create: `train/online_loop_stage3_v29.py`
- Create: `configs/stage3_v29_expert_local_invariant_release_probe.yaml`
- Test: `tests/test_stage3_v29_expert_local_invariant_release.py`
- Test: `tests/test_stage3_generator_training_v29.py`

- [ ] **Step 1: Define protocol**

Use:

```python
PROTOCOL_VERSION = "v29_expert_local_invariant_release_probe"
```

Require:

```text
stage3_v2.protocol_version = v29_expert_local_invariant_release_probe
online.stage3.version = v29_expert_local_invariant_release_probe
online.stage3.signal_mode = fluctuation_delta
stage3 roots contain s3v29expert
```

- [ ] **Step 2: Build V29 model/loss from YAML**

Read `model_v29` and `invariant` sections. Keep default dimensions close to V28 to make matched-budget comparisons valid.

- [ ] **Step 3: Collect V29 split tensors**

Use:

```python
raw = out.pooled_servable_change
```

Add manifest field:

```json
"v29_model_native_drift_source": "pooled_servable_change"
```

- [ ] **Step 4: Versioned generator and online wrappers**

Map `layer_aware_hyperadapter_v29` to the compatible V26 generator internally, while keeping V29 names in config and logs.

- [ ] **Step 5: Write Stage III tests**

Run:

```powershell
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_stage3_v29_expert_local_invariant_release.py tests\test_stage3_generator_training_v29.py -q
```

Expected: config contract, split tensor source, causal-online guard, and generator cfg normalization tests pass.

## Task 4: Deterministic Validation

**Files:**

- No new source files unless a validation failure exposes a real defect.

- [ ] **Step 1: Compile V29 files**

Run:

```powershell
C:\Users\admin\anaconda3\python.exe -m py_compile models\factor_mechanism_v29.py train\offline_factor_v29.py train\run_stage3_v29_expert_local_invariant_release_probe.py train\stage3_generator_training_v29.py train\stage3_v29_artifacts.py train\online_loop_stage3_v29.py tests\test_factor_mechanism_v29.py tests\test_offline_factor_v29.py tests\test_stage3_v29_expert_local_invariant_release.py tests\test_stage3_generator_training_v29.py
```

- [ ] **Step 2: Run V29 tests**

Run:

```powershell
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_factor_mechanism_v29.py tests\test_offline_factor_v29.py tests\test_stage3_v29_expert_local_invariant_release.py tests\test_stage3_generator_training_v29.py -q
```

- [ ] **Step 3: Run regression tests against V28**

Run:

```powershell
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_factor_mechanism_v28.py tests\test_offline_factor_v28.py tests\test_stage3_v28_support_suppression_release.py tests\test_stage3_generator_training_v28.py tests\test_factor_mechanism_v29.py tests\test_offline_factor_v29.py tests\test_stage3_v29_expert_local_invariant_release.py tests\test_stage3_generator_training_v29.py -q
```

Expected: V28 remains intact and V29 passes.

## Task 5: Experiments and Audit Record

**Files:**

- Create: `docs/audit/STAGE3_V29_EXPERT_LOCAL_INVARIANT_RELEASE_IMPLEMENTATION_RECORD.md`
- Modify only if V29 evidence exists: `docs/audit/EXPERIMENT_EVIDENCE_LEDGER.md`

- [ ] **Step 1: Run smoke**

Run a small weather smoke only after deterministic tests pass:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v29_expert_local_invariant_release_probe.py --config configs\stage3_v29_expert_local_invariant_release_probe.yaml --seed 42 --epochs 1 --max-train-batches 2 --max-generator-batches 2 --max-eval-batches 2 --validation-windows 2 --run-id v29_smoke_01 --out-root logs\s3v29expert_smoke --checkpoint-root ckpt\s3v29expert_smoke --overwrite
```

- [ ] **Step 2: Run matched b60 screens if smoke path is valid**

Run weather and panama matched-budget screens:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v29_expert_local_invariant_release_probe.py --config configs\stage3_v29_expert_local_invariant_release_probe.yaml --seed 42 --epochs 4 --max-train-batches 60 --max-generator-batches 60 --max-eval-batches 60 --validation-windows 5 --run-id v29_weather_s42_b60_01 --out-root logs\s3v29expert_screen --checkpoint-root ckpt\s3v29expert_screen --overwrite
```

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v29_expert_local_invariant_release_probe.py --config configs\stage3_v29_expert_local_invariant_release_probe.yaml --dataset panama --seed 42 --epochs 4 --max-train-batches 60 --max-generator-batches 60 --max-eval-batches 60 --validation-windows 5 --run-id v29_panama_s42_b60_01 --out-root logs\s3v29expert_screen --checkpoint-root ckpt\s3v29expert_screen --overwrite
```

- [ ] **Step 3: Write paper-level interpretation**

The audit record must distinguish:

```text
verified implementation facts
deterministic validation results
smoke/formal experiment results
gate-passing or gate-failing status
whether generator unlocked
whether V29 is empirically better than V28
residual risks and next tuning direction
```

Do not claim improvement unless logs pass the relevant teacher/generator gates.

## Self-Review

Spec coverage:

- Non-overwriting official version: covered by new V29 filenames and `s3v29expert` roots.
- User's expert-local invariant idea: covered by expert-local shared anchor, local env inference, invariant correction, and residual-release state.
- Avoid toy implementation: covered by model, losses, runner, generator, online wrapper, config, tests, and audit record.
- Avoid path confusion: covered by versioned filenames and root checks.
- Paper-grade evidence: covered by deterministic tests, smoke, matched b60 screens, and conservative audit wording.

Known execution constraint:

- This Codex session does not expose a real subagent-spawn tool. The plan is written in subagent-ready task form, but execution in this session must be coordinated inline unless a future session provides a subagent dispatcher.

## Execution Status: 2026-06-12

Completed:

- V29 model, offline training, runner, generator wrapper, artifact wrapper,
  online wrapper, config, tests, and audit record were added.
- Deterministic compile check passed.
- V29 unit/integration tests passed: `11 passed`.
- V28+V29 regression passed: `22 passed`.
- Smoke run completed:
  - `logs/s3v29expert_smoke/v29_smoke_01/weather/seed_42`
- Matched b60 screens completed:
  - `logs/s3v29expert_screen/v29_weather_s42_b60_01/weather/seed_42`
  - `logs/s3v29expert_screen/v29_panama_s42_b60_01/panama/seed_42`
- First V29 tuning config completed:
  - `configs/stage3_v29_expert_local_invariant_release_probe_tune_stability_release_deconstrained.yaml`
- First tuned b60 screens completed:
  - `logs/s3v29expert_tuning_screen/v29stable_weather_s42_b60_01/weather/seed_42`
  - `logs/s3v29expert_tuning_screen/v29stable_panama_s42_b60_01/panama/seed_42`
- V29 runtime contract was tightened so final resolved configs record
  `online.stage3.drift_source = pooled_servable_change`.
- V28+V29 regression after the V29 contract fix: `23 passed`.
- Second V29 tuning config completed:
  - `configs/stage3_v29_expert_local_invariant_release_probe_tune_counterfactual_guarded.yaml`
- Second tuned weather b60 screen completed:
  - `logs/s3v29expert_tuning_screen/v29guard_weather_s42_b60_01/weather/seed_42`
- User requested pausing structural changes and focusing on rigorous tuning.
- Third V29 tuning config completed:
  - `configs/stage3_v29_expert_local_invariant_release_probe_tune_early_window_stability.yaml`
- Third tuned weather b60 screen completed:
  - `logs/s3v29expert_tuning_screen/v29earlystab_weather_s42_b60_01/weather/seed_42`
- Fourth V29 tuning config completed:
  - `configs/stage3_v29_expert_local_invariant_release_probe_tune_stability_micro_cf.yaml`
- Fourth tuned weather b60 screen completed:
  - `logs/s3v29expert_tuning_screen/v29microcf_weather_s42_b60_01/weather/seed_42`
- Single-factor counterfactual-shuffle low-side config completed:
  - `configs/stage3_v29_expert_local_invariant_release_probe_tune_cf_shuffle_007_only.yaml`
- Single-factor counterfactual-shuffle low-side weather b60 screen completed:
  - `logs/s3v29expert_tuning_screen/v29cf007_weather_s42_b60_01/weather/seed_42`
- Single-factor counterfactual-shuffle high-side config completed:
  - `configs/stage3_v29_expert_local_invariant_release_probe_tune_cf_shuffle_009_only.yaml`
- Single-factor counterfactual-shuffle high-side weather b60 screen completed:
  - `logs/s3v29expert_tuning_screen/v29cf009_weather_s42_b60_01/weather/seed_42`
- Single-factor release-null low-side config completed:
  - `configs/stage3_v29_expert_local_invariant_release_probe_tune_release_null_0045_only.yaml`
- Single-factor release-null low-side weather b60 screen completed:
  - `logs/s3v29expert_tuning_screen/v29null0045_weather_s42_b60_01/weather/seed_42`
- Single-factor release-null high-side config completed:
  - `configs/stage3_v29_expert_local_invariant_release_probe_tune_release_null_0055_only.yaml`
- Single-factor release-null high-side weather b60 screen completed:
  - `logs/s3v29expert_tuning_screen/v29null0055_weather_s42_b60_01/weather/seed_42`
- Single-factor release-first-increment low-side config completed:
  - `configs/stage3_v29_expert_local_invariant_release_probe_tune_release_first_038_only.yaml`
- Single-factor release-first-increment low-side weather b60 screen completed:
  - `logs/s3v29expert_tuning_screen/v29first038_weather_s42_b60_01/weather/seed_42`
- Single-factor release-first-increment high-side config completed:
  - `configs/stage3_v29_expert_local_invariant_release_probe_tune_release_first_042_only.yaml`
- Single-factor release-first-increment high-side weather b60 screen completed:
  - `logs/s3v29expert_tuning_screen/v29first042_weather_s42_b60_01/weather/seed_42`
- Single-factor release-first-increment midpoint config completed:
  - `configs/stage3_v29_expert_local_invariant_release_probe_tune_release_first_041_only.yaml`
- Single-factor release-first-increment midpoint weather b60 screen completed:
  - `logs/s3v29expert_tuning_screen/v29first041_weather_s42_b60_01/weather/seed_42`
- Single-factor learning-rate low-side config completed:
  - `configs/stage3_v29_expert_local_invariant_release_probe_tune_lr_00034_only.yaml`
- Single-factor learning-rate low-side weather b60 screen completed:
  - `logs/s3v29expert_tuning_screen/v29lr00034_weather_s42_b60_01/weather/seed_42`
- Single-factor learning-rate high-side config completed:
  - `configs/stage3_v29_expert_local_invariant_release_probe_tune_lr_00042_only.yaml`
- Single-factor learning-rate high-side weather b60 screen completed:
  - `logs/s3v29expert_tuning_screen/v29lr00042_weather_s42_b60_01/weather/seed_42`
- Single-factor change-readout low-side config completed:
  - `configs/stage3_v29_expert_local_invariant_release_probe_tune_change_readout_026_only.yaml`
- Single-factor change-readout low-side weather b60 screen completed:
  - `logs/s3v29expert_tuning_screen/v29cr026_weather_s42_b60_01/weather/seed_42`
- Single-factor change-readout high-side config completed:
  - `configs/stage3_v29_expert_local_invariant_release_probe_tune_change_readout_034_only.yaml`
- Single-factor change-readout high-side weather b60 screen completed:
  - `logs/s3v29expert_tuning_screen/v29cr034_weather_s42_b60_01/weather/seed_42`
- Single-factor delayed-change low-side config completed:
  - `configs/stage3_v29_expert_local_invariant_release_probe_tune_delayed_change_024_only.yaml`
- Single-factor delayed-change low-side weather b60 screen completed:
  - `logs/s3v29expert_tuning_screen/v29dc024_weather_s42_b60_01/weather/seed_42`
- Single-factor delayed-change high-side config completed:
  - `configs/stage3_v29_expert_local_invariant_release_probe_tune_delayed_change_032_only.yaml`
- Single-factor delayed-change high-side weather b60 screen completed:
  - `logs/s3v29expert_tuning_screen/v29dc032_weather_s42_b60_01/weather/seed_42`

Current result:

- V29 is implemented as a complete official version line.
- First matched-budget base evidence is negative.
- First tuning pass recovers strong weather aggregate raw/sign signal, but it
  still fails counterfactual gates.
- First tuning pass stabilizes panama change from negative to zero, but does
  not produce positive primary change evidence.
- Second guarded tuning pass is negative: it preserves aggregate weather raw
  gain but worsens worst-window counterfactual margin.
- Third early-window-stability tuning pass is negative: it raises aggregate
  weather raw MSE gain to `0.6487557143040037`, but catastrophically worsens
  window `0` partial MSE to `-3.2306852535381294` and counterfactual margin to
  `-5.456856716067559`.
- Fourth micro-counterfactual tuning pass is negative relative to the current
  anchor: aggregate raw MSE is only `0.2107932743377172`, positive raw windows
  fall to `3/5`, and counterfactual margin remains negative at
  `-0.5858395508423936`.
- The first true single-factor sweep around
  `lambda_counterfactual_shuffle = 0.08` is also negative for the primary raw
  target:
  - `0.07` gives raw aggregate MSE `0.3656415117996853`, but window `0` raw
    partial MSE is `-3.093504781268863` and counterfactual margin is
    `-3.4914044381611973`.
  - `0.09` gives raw aggregate MSE `0.5549855579529909`, but window `0` raw
    partial MSE and counterfactual margin are both `-2.304469675805639`.
- The single-factor sweep around `lambda_release_null = 0.05` is also not a
  gate improvement:
  - `0.045` gives raw aggregate MSE `0.28996881937757046`, but window `0` raw
    partial MSE and counterfactual margin are both `-3.5556763517029277`.
  - `0.055` gives raw aggregate MSE `0.5222828881075632`, but raw min partial
    worsens to `-0.44838369064419603`, counterfactual worsens to
    `-0.823874883701122`, and sign weakens relative to the anchor.
- The single-factor sweep around `lambda_release_first_increment = 0.40` is
  mixed:
  - `0.38` gives raw aggregate MSE `0.46718376446062737`, but window `0` raw
    partial MSE and counterfactual margin are both `-1.805349288341766`.
  - `0.41` gives raw aggregate MSE `0.566033730128696`, but window `0` raw
    partial MSE remains harmful at `-0.39785273932262527`, raw counterfactual
    margin worsens to `-0.8293233156592138`, and `residual_change_vector`
    remains `0/5`.
  - `0.42` gives raw aggregate MSE `0.2843195154550431`; it improves raw
    counterfactual margin to `-0.42083107598996183` and sign counterfactual to
    `-0.0507509708404541`, but still fails the raw harmful-window threshold.
- The single-factor sweep around `train.lr = 0.00038` is not a gate
  improvement:
  - `0.00034` gives raw aggregate MSE `0.43066202825842365`, but window `0`
    raw partial MSE is `-3.7506495003938225` and counterfactual margin is
    `-4.664620134611712`.
  - `0.00042` gives raw aggregate MSE `0.5935039309515732` and sign aggregate
    MSE `0.7864210605621338`, but raw window `0` remains harmful at
    `-0.7459058197140563`, counterfactual is still `-0.7459058197140563`, and
    `residual_change_vector` remains `0/5`.
- The single-factor sweep around `lambda_change_readout = 0.30` is also not a
  gate improvement:
  - `0.26` gives raw aggregate MSE `0.36031556562316` and weakly reactivates
    `residual_change_vector` to `1/5`, but raw window `0` and counterfactual
    both worsen to `-1.9118969332201075`.
  - `0.34` gives raw aggregate MSE only `0.2034717094000943`, raw positive
    windows drop to `3/5`, and sign counterfactual worsens to
    `-0.5392091274261475`.
- The single-factor sweep around `lambda_delayed_change = 0.28` is also not a
  gate improvement:
  - `0.24` gives raw aggregate MSE `0.2778015450214807`, raw window `0`
    `-0.998103525805619`, and sign counterfactual `-0.32214784622192383`.
  - `0.32` gives raw aggregate MSE `0.48845339421202494` and sign aggregate
    MSE `0.6566083133220673`, but raw window `0` and raw counterfactual worsen
    to `-1.4773854050120248`, while `residual_change_vector` remains `0/5`.
- The current best V29 tuning anchor remains
  `stability_release_deconstrained`, not any later guarded/stability/micro
  config.
- No V29 run passed the strict teacher gate.
- Generator training remains locked.
- A dedicated tuning-priority correction report was added at
  `docs/audit/STAGE3_V29_TUNING_PRIORITY_CORRECTION_REPORT.md`.
- Next work should remain V29 tuning under frozen structure, not another model
  structure change.

Next tuning target:

- preserve tuned weather raw/sign aggregate gains
- improve counterfactual margin
- recover positive panama `residual_change_vector`
- keep matched b60 budgets until a setting is gate-near
- avoid continuing the current `counterfactual_guarded` direction unless the
  early-window instability mechanism is addressed
- avoid bundled regularization changes like `early_window_stability`
- use single-factor or very small-grid sweeps around
  `stability_release_deconstrained`
- stop sweeping `lambda_counterfactual_shuffle` for now; both `0.07` and
  `0.09` break raw window `0` more severely than the `0.08` anchor
- stop sweeping `lambda_release_null` for now; `0.045` collapses window `0`,
  while `0.055` improves aggregate raw but worsens the strict gate terms
- stop using `lambda_release_first_increment` as the only lever; `0.41` was the
  disciplined midpoint test and still failed the strict teacher gate
- stop broad learning-rate sweeps for now; the low side is catastrophic and the
  high side improves aggregate raw/sign without fixing the strict gate
- stop readout-only sweeps for now; both `0.26` and `0.34` are worse than the
  `0.30` anchor under strict gate criteria
- stop delayed-change-only sweeps for now; both `0.24` and `0.32` are worse than
  the `0.28` anchor under strict gate criteria
- next inspect per-window release diagnostics or run a targeted small matrix
  explicitly optimized for window `0` raw non-harm and counterfactual safety
- test at most one or two loss weights per run; do not bundle another broad
  structure-like regularization change into the tuning sweep
- check weather window `0` and raw counterfactual margin before running panama
  transfer screens
- first constrained two-factor candidate was added:
  `configs/stage3_v29_expert_local_invariant_release_probe_tune_first042_null0055.yaml`
- attempted matched-budget run
  `v29first042null0055_weather_s42_b60_01` timed out before teacher-only gate
  files were written; this is queued execution work, not performance evidence
- next execution should either finish the same matched-budget candidate in a
  longer/background run window or run a separately named short diagnostic budget
  without treating it as comparable b60 evidence
- timeout root cause was identified: Stage III train split tensor collection is
  controlled by `--max-generator-batches`, not by `--max-train-batches`
- runtime triage `v29first042null0055_weather_s42_e0g1_runtime02` completed and
  verified the V29 teacher/gate output chain when `--max-generator-batches` was
  set
- short diagnostic `v29first042null0055_weather_s42_b60g60e1_diag04_hist`
  showed strong one-epoch raw signal but still failed partial-vs-history and
  counterfactual criteria
- completed formal screen `v29first042null0055_weather_s42_b60_02` is negative:
  raw primary gate fails (`1/5` positive partial windows, min partial MSE
  `-0.34518940793525116`, counterfactual `-0.48101357731444827`), history-only
  beats fluctuation-plus-history in aggregate, and generator remains locked
- do not promote or continue the `first042_null0055` pair as the next anchor;
  return to `stability_release_deconstrained` and target the longer-training
  shift that makes raw correction fall behind `history_only`
- epoch-sensitivity diagnostics around `stability_release_deconstrained` were
  completed with matched b60/g60/e60 caps:
  `v29stable_weather_s42_b60g60e1_epochdiag01` and
  `v29stable_weather_s42_b60g60e2_epochdiag01`
- `epochs=1` is not gate-passing, but it is safer than the formal 5-epoch
  anchor on raw window `0` (`0.47259663917691164`) and raw counterfactual
  margin (`-0.011714003971613818`)
- `epochs=2` is worse for the current diagnostic purpose because aggregate raw
  partial MSE turns negative (`-0.011986833559262`) and raw counterfactual
  margin worsens to `-0.31489352634826306`
- the formal 5-epoch anchor still has stronger aggregate raw partial MSE
  (`0.34745194792605627`) and `4/5` positive raw partial windows, but its raw
  window `0` and raw counterfactual margin both fail at
  `-0.7246091209150117`
- current decision: treat epoch count and early-stop selection as a controlled
  tuning axis; do not run an unconstrained `epochs=3` escalation and do not
  change model structure before a gate-aware epoch/signal matrix is exhausted

## Current Execution Subplan: V29 Gate-Aware Tuning Selector

**Goal:** build a read-only selector that ranks completed V29 tuning runs by the
strict teacher-gate priorities already documented, so the next tuning step is
chosen from evidence rather than manual cherry-picking.

**Files:**

- Create: `train/analyze_stage3_v29_tuning_selection.py`
- Create: `tests/test_stage3_v29_tuning_selection.py`
- Generate:
  `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection.csv`
- Generate:
  `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection.md`
- Update:
  `docs/audit/STAGE3_V29_EXPERT_LOCAL_INVARIANT_RELEASE_IMPLEMENTATION_RECORD.md`
- Update:
  `docs/audit/STAGE3_V29_TUNING_PRIORITY_CORRECTION_REPORT.md`
- Update:
  `docs/audit/EXPERIMENT_EVIDENCE_LEDGER.md`

### Task A: Implement Selector

- [x] Add CSV/JSON readers that accept explicit `--run-dir` values and optional
  `--root` discovery directories.
- [x] Require V29 teacher files before summarizing a run:
  `teacher_only_v29_target_gate.csv`,
  `teacher_only_validation_diagnostic_summary_v29.csv`,
  `teacher_only_validation_partial_gain_vs_history_v29.csv`,
  `teacher_only_rolling_validation_window_summary_v29.csv`, and
  `teacher_only_rolling_partial_gain_vs_history_v29.csv`.
- [x] Extract target-level fields for each V29 run:
  primary gate pass, positive partial-window count, min partial MSE/MAE,
  counterfactual margin, validation raw/history/partial gains, window `0`
  raw/history/partial gains, worst-window counterfactual margin, and generator
  lock state.
- [x] Sort candidates lexicographically by strict safety priorities:
  gate pass, generator unlock, non-harmful raw window `0`, counterfactual
  margin, positive partial windows, min partial MSE, min partial MAE, aggregate
  partial MSE, aggregate partial MAE, aggregate raw MSE.
- [x] Write CSV and Markdown outputs without silently overwriting unless
  `--overwrite` is supplied.

### Task B: Test Selector

- [x] Add synthetic run-directory fixtures covering an `epochs=1`-like candidate,
  an `epochs=2`-like candidate, and a formal 5-epoch-like candidate.
- [x] Verify the selector ranks the safer `epochs=1`-like candidate above the
  `epochs=2`-like candidate when no run passes the strict gate.
- [x] Verify incomplete run directories are skipped with a clear missing-file
  diagnostic instead of being ranked as negative evidence.
- [x] Verify direct script invocation writes both CSV and Markdown artifacts.

### Task C: Apply To Current Evidence

- [x] Run:
  `C:\Users\admin\anaconda3\python.exe train\analyze_stage3_v29_tuning_selection.py --run-dir logs\s3v29expert_tuning_diag\v29stable_weather_s42_b60g60e1_epochdiag01\weather\seed_42 --run-dir logs\s3v29expert_tuning_diag\v29stable_weather_s42_b60g60e2_epochdiag01\weather\seed_42 --run-dir logs\s3v29expert_tuning_screen\v29stable_weather_s42_b60_01\weather\seed_42 --run-dir logs\s3v29expert_tuning_screen\v29first042null0055_weather_s42_b60_02\weather\seed_42 --out-dir artifacts\stage3_v29_tuning_selection --prefix current_v29_tuning_selection --overwrite`
- [x] Expected output files:
  `current_v29_tuning_selection.csv` and
  `current_v29_tuning_selection.md`.
- [x] Record the selector result in V29 audit docs as tuning-control evidence,
  not as a performance claim.

Current selector result:

- artifact directory: `artifacts/stage3_v29_tuning_selection`
- ranked runs: `5`
- skipped run directories: `0`
- rank `1`: `v29stable_weather_s42_b60g60e1_epochdiag01`
- rank `1` label: `gate_near_diagnostic`
- rank `1` is still not promotable: target gate pass `False`, generator
  unlocked `False`, positive partial windows `2/4`
- rank `1` safety reason: raw window `0` MSE `0.472596639177` and
  counterfactual margin `-0.0117140039716`, better than `epochs=2` and the
  formal 5-epoch anchor under the strict safety-first ordering
- post-selector bounded run:
  `v29first042only_weather_s42_b60g60e1_diag01`
- config:
  `configs/stage3_v29_expert_local_invariant_release_probe_tune_release_first_042_only.yaml`
- budget:
  `epochs=1`, `max_train_batches=60`, `max_generator_batches=60`,
  `max_eval_batches=60`, `validation_windows=5`
- result: ranked `2/5`; still gate-negative and generator-locked
- diagnostic lesson: `lambda_release_first_increment=0.42` improves aggregate
  partial MSE (`0.0860759770012467`) versus the e1 stability anchor, but does
  not improve positive partial-window count (`2/4`) and worsens counterfactual
  margin (`-0.01632314949103475`)
- next tuning boundary: do not escalate `release_first_042_only` to larger
  budget; next candidate must target the specific late-window partial failure
  without sacrificing the e1 anchor's window `0` safety
- selector now also writes window-level artifacts:
  `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection_windows.csv`
  and
  `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection_windows.md`
- window-level result: rank `1` and rank `2` share the same structural failure
  pattern; windows `0`, `1`, and `4` are partial-negative because
  `history_only` is stronger than `fluctuation_plus_history`, while windows `2`
  and `3` are mainly counterfactual-negative
- current decision: do not run another blind single-axis e1 candidate this turn;
  the next useful candidate must explicitly protect history-dominated windows
  or implement a conditional feature-mode selection/tuning rule without changing
  the V29 model structure
- selector now also writes conditional feature-mode policy oracle artifacts:
  `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection_feature_mode_policy.csv`
  and
  `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection_feature_mode_policy.md`
- policy diagnostic result: FH/history switching alone remains insufficient for
  rank `1` and rank `2` (`2/4` positive windows), but the formal 5-epoch anchor
  has a gate-like passing `cf_safe_best_feature_oracle`
- passing oracle mode sequence:
  `0:fluctuation_only;1:level_offset_plus_history;2:level_offset_plus_history;3:level_offset_plus_history;4:history_only`
- interpretation boundary: this is validation-oracle evidence only; the next
  implementation work must define an online-available gating signal before any
  run can be treated as deployable or paper-facing
- selector now also writes per-window mode-gate label artifacts:
  `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection_mode_gate_labels.csv`
  and
  `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection_mode_gate_labels.md`
- mode-gate label result: the artifact currently has `25` rows across the
  explicit V29 tuning set and records oracle labels such as
  `best_available_feature_label`, `cf_safe_feature_label`,
  `history_protected_label`, and `needs_non_fh_mode`
- online boundary: every current row is marked
  `label_source=validation_window_outcome_oracle`,
  `online_label_available=False`, and
  `online_gate_feature_status=not_defined_in_current_v29_artifact`; therefore
  it is a supervision-design artifact, not an online policy or manuscript
  performance result
- next safe tuning direction: define pre-outcome, serving-time gate features
  over the existing V29 feature modes before running another candidate or
  claiming a deployable improvement
- selector now also writes online-gate feature matrix artifacts:
  `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection_online_gate_features.csv`
  and
  `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection_online_gate_features.md`
- online feature matrix result: `25` rows, with `feature_` columns restricted
  to pre-outcome static metadata, validation-window schedule, train-fit signal
  contract diagnostics, train-fit history contract diagnostics, available
  feature modes, and resolved config fields
- leakage boundary: oracle labels are separated under `oracle_label_`, and a
  direct column audit found no `feature_` columns containing `gain`,
  `counterfactual`, `label`, `partial`, `teacher_real`, `outcome`, or `best`
- remaining blocker: every row records
  `feature_dynamic_signal_status=missing_current_serving_window_statistics`;
  a deployable mode gate still needs runtime pre-outcome dynamic features before
  another candidate run or performance claim is justified
- dynamic feature hook now exists for future V29 probes:
  `train/stage3_v29_online_gate_features.py::build_online_gate_dynamic_feature_row`
  and
  `train/run_stage3_v29_expert_local_invariant_release_probe.py::collect_stage3_v29_online_gate_dynamic_feature_rows`
- V29 wrapper now installs
  `write_stage3_v29_online_gate_dynamic_features` into the V18-compatible
  probe hook; new bounded V29 probes write
  `stage3_v29_online_gate_dynamic_features.csv` before teacher/validation
  outcome evaluation
- V29 output renaming now records
  `stage3_v29_online_gate_dynamic_features_csv` in `run_manifest_v29.json`,
  with path, sha256, row count, pre-outcome scope, and leakage contract
- dynamic hook scope: model-forward state only, including router entropy,
  top-1/margin, local-environment entropy, release confidence/support,
  servable-change norm, residual-release-state norm, invariant-correction
  norm, correction-gate mean, and prediction norm
- selector merge scope: if a future run writes
  `stage3_v29_online_gate_dynamic_features.csv`, the selector merges only safe
  `feature_dynamic_` columns and keeps oracle labels under `oracle_label_`
- current boundary: the five historical ranked V29 weather runs still do not
  include dynamic CSVs, so current artifacts remain design/leakage-guard
  evidence, not a trained online gate or performance result
- validation before online mode-gate v1: targeted V29 tests passed with
  `18 passed`

2026-06-12 continuation dynamic-smoke evidence:

- `_01` attempted `--dataset weather`; this failed before training because
  `--dataset` is a public-dataset override key and `weather` is not registered
  there. Correct usage is to omit `--dataset` and use the config path
  `dataset/weather.csv`.
- `_01` was then retried with `--modes zero`; it wrote partial files but failed
  because the inherited V14/V18 teacher gate requires the full counterfactual
  mode set. Treat `_01` as a failure diagnostic, not evidence.
- `_02` completed with `validation_windows=1` and proved
  `stage3_v29_online_gate_dynamic_features.csv` plus
  `run_manifest_v29.json.stage3_v29_online_gate_dynamic_features_csv`, but the
  selector had `Feature rows: 0` because no rolling selector rows were produced.
- `_03` completed with `validation_windows=2`:
  `logs/s3v29expert_dynamic_smoke/v29dynamic_manifest_smoke_s42_b1w2_20260612_03/weather/seed_42`.
- `_03` scope: `epochs=1`, `max_train_batches=1`,
  `max_generator_batches=1`, `max_eval_batches=1`, target
  `raw_residual_vector`, feature mode `fluctuation_plus_history`, CPU.
- `_03` selector artifact:
  `artifacts/stage3_v29_dynamic_smoke_validation/v29dynamic_manifest_smoke_w2_selection_online_gate_features.md`.
- `_03` selector result: `Feature rows: 1` and
  `feature_dynamic_signal_status=available_from_model_forward`.
- `_03` remains performance-negative/non-evidence:
  `teacher_only_v29_target_gate.csv` reports `primary_gate_pass=False` and the
  generator remains locked.

2026-06-12 continuation online mode-gate v1:

- Added `train/stage3_v29_online_mode_gate.py`.
- Added `tests/test_stage3_v29_online_mode_gate.py`.
- Gate consumes selector `feature_` columns only, drops outcome-like feature
  names via the selector banned-token policy, and uses
  `oracle_label_cf_safe_feature` only as supervision/evaluation.
- Model is a standardized nearest-centroid mode classifier with configurable
  confidence threshold and `history_only` fallback.
- Current 25-row V29 tuning matrix split:
  train windows `0-2`, held-out windows `3-4`.
- Threshold sweep on
  `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection_online_gate_features.csv`:
  `0.00 -> 1/10`, `0.15 -> 3/10`, `0.25 -> 7/10`,
  `0.35 -> 10/10`, `0.45 -> 9/10` held-out correct.
- Best conservative artifact:
  `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection_online_mode_gate_conf035_summary.json`.
- Interpretation: the current evidence supports conservative fallback/abstention
  more than active multi-mode correction. This still does not unlock generator
  training and is not paper-grade forecasting evidence.
- validation after online mode-gate v1:
  `tests/test_stage3_v29_expert_local_invariant_release.py`,
  `tests/test_stage3_v29_online_gate_features.py`,
  `tests/test_stage3_v29_tuning_selection.py`, and
  `tests/test_stage3_v29_online_mode_gate.py` pass with `21 passed`.

2026-06-12 continuation serving helper:

- Added serving policy functions:
  `load_online_mode_gate_policy`, `apply_online_mode_gate_policy`, and
  `select_online_mode_for_row` in `train/stage3_v29_online_mode_gate.py`.
- Added V29 online-loop helper:
  `train.online_loop_stage3_v29.select_stage3_v29_serving_mode`.
- The helper preserves the inherited V26 online adaptation path and does not
  force generator usage while V29 remains generator-locked.
- Real-artifact serving smoke:
  loaded `current_v29_tuning_selection_online_mode_gate_conf035`, removed all
  `oracle_label_` columns from held-out
  `v29stable_weather_s42_b60g60e1_epochdiag01/window=4`, and selected
  `history_only` with `reject_reason=confidence_below_threshold`,
  `online_gate_confidence=0.21649074285327313`, and
  `oracle_label_columns_seen=[]`.
- validation after serving helper:
  the same targeted V29 test set passes with `23 passed`.

2026-06-12 continuation outcome replay:

- Added serving-policy outcome replay in `train/stage3_v29_online_mode_gate.py`.
- CLI path: `--evaluate-outcomes` with `--policy-dir`, `--policy-prefix`, and
  one or more `--run-dir` values.
- Synthetic tests now cover that selected serving modes are mapped to rolling
  teacher outcomes after stripping oracle labels before policy prediction.
- Real replay artifact:
  `artifacts/stage3_v29_tuning_selection/current_v29_tuning_selection_online_mode_gate_conf035_outcome_replay_summary.json`.
- Real replay scope: current five V29 weather tuning runs, `25/25` rows
  evaluated against `teacher_only_rolling_validation_window_summary_v29.csv`.
- Result: selected mean MSE gain `0.5506004468595593`; fixed
  `fluctuation_plus_history` `0.5030690635309452`; fixed `history_only`
  `0.45593240636333965`; oracle cf-safe `0.5545271305345248`.
- Harm control: selected harmful windows `0`; fixed
  `fluctuation_plus_history` harmful windows `2`; fixed `history_only`
  harmful windows `1`.
- Boundary: this is offline teacher-outcome replay, not a generator-unlocked
  online forecasting result.
- validation after outcome replay: targeted V29 tests pass with `25 passed`.

2026-06-12 continuation served policy package:

- Added served package promotion/loading helpers in
  `train/stage3_v29_online_mode_gate.py`:
  `promote_online_mode_gate_serving_package` and
  `load_served_online_mode_gate_policy`.
- Generated manifest:
  `artifacts/stage3_v29_served_mode_gate/v29_conf035_weather_served_mode_gate_manifest.json`.
- Manifest protocol: `v29_served_online_mode_gate_manifest_v1`.
- Package scope: `mode_selection_only_no_generator_unlock`.
- Admission pass: `true`.
- Admission criteria: at least `25` evaluated rows, selected harmful windows
  `<= 0`, selected mean MSE gain no worse than fixed
  `fluctuation_plus_history`, and selected mean MSE gain no worse than fixed
  `history_only`.
- Manifest locks sha256 hashes for policy summary/model/predictions/report and
  outcome replay summary/rows/report.
- Fail-closed loader test: tampered model artifact returns `None`.
- Real load smoke: manifest loads and selects `history_only` on held-out
  `v29stable_weather_s42_b60g60e1_epochdiag01/window=4` without oracle labels.
- validation after served package: targeted V29 tests pass with `27 passed`.

2026-06-12 continuation tuning queue coverage:

- Added non-training queue builder:
  `train/plan_stage3_v29_tuning_queue.py`.
- Added tests:
  `tests/test_stage3_v29_tuning_queue.py`.
- Generated artifacts:
  `artifacts/stage3_v29_tuning_queue/current_v29_tuning_queue.csv` and
  `artifacts/stage3_v29_tuning_queue/current_v29_tuning_queue.md`.
- Protocol: `stage3_v29_tuning_queue_v1`.
- Queue matching uses direct `run_id` plus resolved-config tuning signatures,
  so YAML candidates still match completed logs when CLI `--run-id` overrides
  were used.
- Current result: `18` V29 tune config rows; `15` are
  `completed_needs_selector_refresh`, and `3` are `covered_by_selector`.
- Decision: do not start another V29 tuning run until selector coverage is
  refreshed over the completed queue. The queue is run-control evidence only,
  not a performance result.
- validation after tuning queue: targeted V29 tests pass with `30 passed`;
  `py_compile` passes for the queue, selector, mode-gate, online loop, and
  related V29 probe files.

2026-06-12 continuation full-queue selector refresh:

- Tightened tuning queue matching after finding that resolved-config signature
  matching was too broad:
  dynamic-smoke roots are excluded from tuning signature matches, and candidate
  dataset must match completed-run dataset.
- Tightened selector ranking after finding that incomplete/nan diagnostic rows
  could outrank complete gate-near diagnostics:
  `selector_metric_complete` now demotes rows labeled
  `incomplete_diagnostic`.
- Refreshed selector artifacts:
  `artifacts/stage3_v29_tuning_selection_full_queue/v29_queue_refreshed_selection.csv`
  and
  `artifacts/stage3_v29_tuning_selection_full_queue/v29_queue_refreshed_selection.md`.
- Refreshed selector result: `24` ranked weather tuning candidates, `1` skipped
  incomplete runtime-triage directory, and top candidate
  `v29stable_weather_s42_b60g60e1_epochdiag01`; it remains gate-negative and
  generator-locked.
- Full-queue mode-gate threshold sweep artifacts:
  `artifacts/stage3_v29_tuning_selection_full_queue/v29_queue_refreshed_mode_gate_threshold_sweep.csv`
  and
  `artifacts/stage3_v29_tuning_selection_full_queue/v29_queue_refreshed_mode_gate_threshold_sweep.md`.
- Harm-first threshold: `min_confidence=0.15`, selected mean MSE
  `0.2270443942984707`, fixed `fluctuation_plus_history`
  `0.22015735047664628`, fixed `history_only` `0.16830160575541872`,
  selected harmful windows `0`, fixed FH harmful windows `18`, fixed history
  harmful windows `6`.
- Boundary: replay status is `incomplete_outcome_rows` with `110` evaluated
  rows and `7` missing outcome rows, so this does not replace the served
  package and does not unlock generator training.
- validation after full-queue selector refresh: targeted V29 tests pass with
  `33 passed`; `py_compile` passes for the queue, selector, mode-gate, online
  loop, and related V29 probe files.

2026-06-12 continuation complete-only mode-gate package:

- Added selector option `--require-complete-selector-metrics`.
- Complete selector artifacts:
  `artifacts/stage3_v29_tuning_selection_full_queue/v29_queue_complete_selection.csv`
  and
  `artifacts/stage3_v29_tuning_selection_full_queue/v29_queue_complete_selection.md`.
- Complete selector result: `22` complete weather tuning candidates and `3`
  skipped rows/directories. Top remains
  `v29stable_weather_s42_b60g60e1_epochdiag01`, gate-negative and
  generator-locked.
- Complete threshold sweep artifacts:
  `artifacts/stage3_v29_tuning_selection_full_queue/v29_queue_complete_mode_gate_threshold_sweep.csv`
  and
  `artifacts/stage3_v29_tuning_selection_full_queue/v29_queue_complete_mode_gate_threshold_sweep.md`.
- Complete replay result: `110/110` evaluated, `0` missing outcomes.
  `min_confidence=0.15` has selected mean MSE `0.25546113326914255`, fixed
  `fluctuation_plus_history` `0.2018502975357777`, fixed `history_only`
  `0.16830160575541872`, selected harmful windows `0`, fixed FH harmful
  windows `18`, fixed history harmful windows `6`.
- New served manifest:
  `artifacts/stage3_v29_served_mode_gate/v29_complete_conf015_weather_served_mode_gate_manifest.json`.
- Admission pass: `true`; scope remains
  `mode_selection_only_no_generator_unlock`.
- Real load smoke: manifest loads and selects `history_only` with
  `confidence_below_threshold` on held-out
  `v29stable_weather_s42_b60g60e1_epochdiag01/window=4` after removing
  `oracle_label_` columns.
- Boundary: this is a conservative mode-selection package only, not a generator
  unlock and not paper-grade online forecasting evidence.
- validation after complete-only selector path: targeted V29 tests pass with
  `34 passed`; `py_compile` passes for the queue, selector, mode-gate, online
  loop, and related V29 probe files.

2026-06-12 continuation generator-unlock preflight:

- Added read-only preflight module:
  `train/stage3_v29_generator_unlock_preflight.py`.
- Added tests:
  `tests/test_stage3_v29_generator_unlock_preflight.py`.
- Real preflight artifacts:
  `artifacts/stage3_v29_generator_unlock_preflight/v29_complete_conf015_generator_unlock_preflight.json`
  and
  `artifacts/stage3_v29_generator_unlock_preflight/v29_complete_conf015_generator_unlock_preflight.md`.
- Result: `mode_gate_prerequisite_pass=True`, but `preflight_pass=False`.
- Top candidate audited:
  `v29stable_weather_s42_b60g60e1_epochdiag01`.
- Blockers: target gate did not pass, no primary gate passed, teacher abort is
  written, generator remains locked, stop reasons include
  `do_not_train_generator` and `no_primary_target_passed`, positive partial
  windows are below requirement, min partial MSE/MAE are below the floor, and
  counterfactual margin is negative.
- Decision: `do_not_train_generator`. The next step remains teacher-gate tuning,
  not generator training.
- validation after preflight: targeted V29 tests pass with `38 passed`.
  `py_compile` passes for the new preflight module and related V29 queue,
  selector, mode-gate, online loop, and probe files.

2026-06-12 continuation teacher-gate tuning matrix:

- Added non-training planner:
  `train/plan_stage3_v29_teacher_gate_tuning_matrix.py`.
- Added tests:
  `tests/test_stage3_v29_teacher_gate_tuning_matrix.py`.
- Generated artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_teacher_gate_tuning_matrix.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_teacher_gate_tuning_matrix.md`.
- Generated config directory:
  `configs/stage3_v29_teacher_gate_tuning_matrix/`.
- Candidate axes:
  `history_guarded_release_floor`,
  `counterfactual_margin_micro_repair`,
  `late_window_partial_floor_guard`,
  `delayed_change_tempered_history`, and
  `release_null_coupled_guard`.
- Boundary: this is the next teacher-gate tuning queue around the
  `stability_release_deconstrained` anchor. It does not modify model topology,
  does not launch training, and does not unlock generator training.
- validation after teacher-gate tuning matrix: targeted V29 tests pass with
  `42 passed`. `py_compile` passes for the new planner, preflight module,
  queue, selector, mode-gate, online loop, and probe files.

2026-06-13 continuation safe tuning execution plan:

- Added non-executing planner:
  `train/plan_stage3_v29_tuning_execution.py`.
- Added tests:
  `tests/test_stage3_v29_tuning_execution_plan.py`.
- Generated artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_tuning_execution_plan.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_tuning_execution_plan.md`.
- Planned output roots:
  `logs/s3v29expert_teacher_gate_matrix` and
  `ckpt/s3v29expert_teacher_gate_matrix`.
- Execution status: all `5` candidates are `ready_to_run`; all planned commands
  omit `--overwrite`.
- Boundary: this still does not run training. It only makes the next bounded
  experiments reproducible and non-overwriting.
- validation after safe execution plan: targeted V29 tests pass with
  `48 passed`. `py_compile` passes for the execution planner, matrix planner,
  preflight module, queue, selector, mode-gate, online loop, and probe files.

2026-06-13 continuation candidate `01` bounded result:

- Execution-plan fix: omit `--dataset weather` because weather is config-native
  (`dataset/weather.csv`) rather than a supported public-dataset override.
- Completed bounded run:
  `logs/s3v29expert_teacher_gate_matrix/v29tg_conf015_01_history_guarded_release_floor/weather/seed_42`.
- Refreshed selector artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate01_selection.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate01_selection.md`.
- Refreshed preflight artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate01_generator_unlock_preflight.json`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate01_generator_unlock_preflight.md`.
- Result: candidate `01` ranks `3/23`, remains gate-negative and
  generator-locked.
- Raw primary metrics: positive partial windows `2/4`, min partial MSE
  `-0.45713857481802944`, min partial MAE `-0.5075939740964291`, and
  counterfactual margin `-0.016909116092234222`.
- Decision: `do_not_train_generator`; do not promote this candidate or continue
  this exact direction as positive evidence.

2026-06-13 continuation candidate `02` bounded result:

- Completed bounded run:
  `logs/s3v29expert_teacher_gate_matrix/v29tg_conf015_02_counterfactual_margin_micro_repair/weather/seed_42`.
- Refreshed selector artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate02_selection.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate02_selection.md`.
- Refreshed preflight artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate02_generator_unlock_preflight.json`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate02_generator_unlock_preflight.md`.
- Result: candidate `02` ranks `5/24`, remains gate-negative and
  generator-locked.
- Raw primary metrics: positive partial windows `2/4`, min partial MSE
  `-0.46744894881173493`, min partial MAE `-0.5296254452103905`, and
  counterfactual margin `-0.038892373912445266`.
- Decision: `do_not_train_generator`; this exact
  `counterfactual_margin_micro_repair` direction worsened the counterfactual
  margin and partial floors.

2026-06-13 continuation candidate `03` bounded result:

- Completed bounded run:
  `logs/s3v29expert_teacher_gate_matrix/v29tg_conf015_03_late_window_partial_floor_guard/weather/seed_42`.
- Refreshed selector artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate03_selection.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate03_selection.md`.
- Refreshed preflight artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate03_generator_unlock_preflight.json`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate03_generator_unlock_preflight.md`.
- Result: candidate `03` ranks `4/25`, remains gate-negative and
  generator-locked.
- Raw primary metrics: positive partial windows `2/4`, min partial MSE
  `-0.4671504578747007`, min partial MAE `-0.49524774880538697`, and
  counterfactual margin `-0.02420296088538021`.
- Decision: `do_not_train_generator`; `late_window_partial_floor_guard` did not
  repair the partial floor or counterfactual blocker.

2026-06-13 continuation candidate `04` bounded result:

- Completed bounded run:
  `logs/s3v29expert_teacher_gate_matrix/v29tg_conf015_04_delayed_change_tempered_history/weather/seed_42`.
- Refreshed selector artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate04_selection.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate04_selection.md`.
- Refreshed preflight artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate04_generator_unlock_preflight.json`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate04_generator_unlock_preflight.md`.
- Result: candidate `04` ranks `7/26`, remains gate-negative and
  generator-locked.
- Raw primary metrics: positive partial windows `2/4`, min partial MSE
  `-0.422131539589501`, min partial MAE `-0.500774945598131`, and
  counterfactual margin `-0.043278617034422484`.
- Decision: `do_not_train_generator`; `delayed_change_tempered_history` did not
  repair the partial-floor or counterfactual blockers.

2026-06-13 continuation candidate `05` bounded result:

- Completed bounded run:
  `logs/s3v29expert_teacher_gate_matrix/v29tg_conf015_05_release_null_coupled_guard/weather/seed_42`.
- Refreshed selector artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate05_selection.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate05_selection.md`.
- Refreshed preflight artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate05_generator_unlock_preflight.json`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_candidate05_generator_unlock_preflight.md`.
- Result: candidate `05` ranks `8/27`, remains gate-negative and
  generator-locked.
- Raw primary metrics: positive partial windows `2/4`, min partial MSE
  `-0.4669403209264789`, min partial MAE `-0.4851098794798296`, and
  counterfactual margin `-0.043559999700157115`.
- Decision: `do_not_train_generator`; `release_null_coupled_guard` did not
  repair the partial-floor or counterfactual blockers.

2026-06-13 continuation teacher-gate matrix exhaustion:

- Completed all five predeclared frozen-architecture candidates under the
  bounded weather seed42 budget.
- None passed the primary teacher gate.
- None changed generator-unlock preflight from `do_not_train_generator`.
- The complete selector still ranks
  `v29stable_weather_s42_b60g60e1_epochdiag01` first.
- Next step: stop this local scalar matrix and diagnose the persistent conflict
  between validation-window partial gains and counterfactual ordering before
  launching another tuning block.

2026-06-13 continuation gate-failure decomposition:

- Added read-only diagnostic tool:
  `train/analyze_stage3_v29_gate_failure_decomposition.py`.
- Added tests:
  `tests/test_stage3_v29_gate_failure_decomposition.py`.
- Generated artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_matrix_gate_failure_decomposition.json`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_matrix_gate_failure_decomposition.csv`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_matrix_gate_failure_decomposition_windows.csv`,
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_matrix_gate_failure_decomposition.md`.
- Result across `27` complete weather candidates: `gate_pass_count=0`,
  `generator_unlocked_count=0`, and
  `matrix_decision=stop_local_scalar_matrix`.
- Dominant axes: no primary teacher gate pass, positive partial-window
  shortfall, partial-floor violation, counterfactual-ordering failure,
  window-level conflict, and generator unlock blocked.
- Window diagnosis: window `0` has partial floor failures in `27/27` rows and
  counterfactual-negative rows in `25/27`; window `2` has `0/27` partial-floor
  failures but still has `25/27` counterfactual-negative rows.
- Boundary: do not start generator training and do not add another scalar-only
  tuning candidate until a new plan explicitly targets both partial-floor
  stability and counterfactual ordering.

2026-06-13 continuation decomposition-driven repair matrix:

- Added non-training planner:
  `train/plan_stage3_v29_decomposition_repair_matrix.py`.
- Added tests:
  `tests/test_stage3_v29_decomposition_repair_matrix.py`.
- Added implementation plan:
  `docs/superpowers/plans/2026-06-13-stage3-v29-decomposition-driven-repair-matrix.md`.
- Generated config directory:
  `configs/stage3_v29_decomposition_repair_matrix/`.
- Generated matrix artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_matrix.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_matrix.md`.
- Generated safe execution plan:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_execution_plan.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_execution_plan.md`.
- Candidate axes:
  `window0_partial_floor_rebalance`,
  `counterfactual_ordering_without_floor_pressure`,
  `dual_axis_floor_and_cf_coupled`, and
  `mode_abstention_teacher_selection_probe`.
- Execution plan status: `4/4` rows are `ready_to_run`; commands omit
  `--overwrite` and do not pass `--dataset weather`.
- Boundary: this is the next predeclared queue only. It does not run training,
  does not change topology, and does not unlock generator training. The
  generated `teacher_selection_overrides` are diagnostic metadata unless a
  later code change explicitly wires them into the teacher-selection logic.

2026-06-13 continuation repair candidate `01` bounded result:

- Completed bounded run:
  `logs/s3v29expert_decomposition_repair_matrix/v29repair_conf015_01_window0_partial_floor_rebalance/weather/seed_42`.
- Refreshed selector artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair01_selection.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair01_selection.md`.
- Refreshed preflight artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair01_generator_unlock_preflight.json`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair01_generator_unlock_preflight.md`.
- Refreshed decomposition artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair01_gate_failure_decomposition.json`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair01_gate_failure_decomposition.md`.
- Result: candidate `01` ranks `9/28`, remains gate-negative and
  generator-locked.
- Raw primary metrics: positive partial windows `2/4`, min partial MSE
  `-0.5455704944904651`, min partial MAE `-0.5699537539405484`,
  counterfactual margin `-0.04551155267896673`, and window0 FH MSE gain
  `0.2653257120709129`.
- Decomposition result: `candidate_count=28`, `gate_pass_count=0`,
  `generator_unlocked_count=0`, and
  `matrix_decision=stop_local_scalar_matrix`.
- Updated execution plan:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_execution_plan_after_repair01.csv`
  marks candidate `01` as `blocked_existing_output` and candidates `02`-`04`
  as `ready_to_run`.
- Decision: `do_not_train_generator`; candidate `01` did not repair the window
  `0` partial-floor blocker and must not be rerun.

2026-06-13 continuation repair candidate `02` long-path failure:

- Attempted original long-run-id candidate:
  `v29repair_conf015_02_counterfactual_ordering_without_floor_pressure`.
- Failure point:
  `teacher_v18_raw_residual_vector_fluctuationonly.pt` checkpoint save under
  `ckpt/s3v29expert_decomposition_repair_matrix/weather/v29repair_conf015_02_counterfactual_ordering_without_floor_pressure/seed_42/`.
- Diagnosis: the absolute checkpoint path reached `263` characters on Windows.
  A shorter sibling checkpoint at `259` characters saved successfully, and a
  dummy `torch.save` to the exact failing path reproduced the error.
- Evidence boundary: this partial original candidate `02` run is invalid as
  model-quality evidence and must not be used for ranking or generator unlock.
- Fix: shorten repair run slugs in
  `train/plan_stage3_v29_decomposition_repair_matrix.py`, and add checkpoint
  path-length prediction plus `checkpoint_file_path_too_long` blocking in
  `train/plan_stage3_v29_tuning_execution.py`.

2026-06-13 continuation path-safe decomposition-repair queue:

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
- Execution safety: all four rows are `ready_to_run`; the maximum expected
  checkpoint path length is `200`, below the enforced `259`-character guard.
- Boundary: this queue supersedes the original long-run-id queue for future
  bounded repair runs. It still does not unlock generator training.

2026-06-13 continuation path-safe repair candidate `02` bounded result:

- Completed bounded run:
  `logs/s3v29rps/v29rps_conf015_02_cforder/weather/seed_42`.
- Refreshed selector artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair02_selection.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair02_selection.md`.
- Refreshed preflight artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair02_generator_unlock_preflight.json`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair02_generator_unlock_preflight.md`.
- Refreshed decomposition artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair02_gate_failure_decomposition.json`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair02_gate_failure_decomposition.csv`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair02_gate_failure_decomposition_windows.csv`,
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair02_gate_failure_decomposition.md`.
- Updated path-safe execution plan:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_execution_plan_pathsafe_after_repair02.csv`
  marks candidate `02` as `blocked_existing_output`.
- Result: candidate `02` ranks `7/29`, remains gate-negative and
  generator-locked.
- Raw primary metrics: positive partial windows `2/4`, min partial MSE
  `-0.4204164211957788`, min partial MAE `-0.500599967584385`,
  counterfactual margin `-0.04310910574869553`, and window0 FH MSE gain
  `0.3989539074568899`.
- Decomposition result: `candidate_count=29`, `gate_pass_count=0`,
  `generator_unlocked_count=0`, and
  `matrix_decision=stop_local_scalar_matrix`.
- Decision: `do_not_train_generator`; candidate `02` did not repair the primary
  teacher gate. Do not rerun path-safe candidate `01` because its long-run
  equivalent already completed negatively; do not rerun candidate `02` because
  this path-safe run is complete and negative. If continuing, use only one of
  the meaningful remaining path-safe rows `03` or `04`, then refresh selector,
  decomposition, and generator-unlock preflight before any further run.

2026-06-13 continuation path-safe repair candidate `03` bounded result:

- Completed bounded run:
  `logs/s3v29rps/v29rps_conf015_03_floorcf/weather/seed_42`.
- Refreshed selector artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair03_selection.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair03_selection.md`.
- Refreshed preflight artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair03_generator_unlock_preflight.json`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair03_generator_unlock_preflight.md`.
- Refreshed decomposition artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair03_gate_failure_decomposition.json`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair03_gate_failure_decomposition.csv`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair03_gate_failure_decomposition_windows.csv`,
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair03_gate_failure_decomposition.md`.
- Updated path-safe execution plan:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_execution_plan_pathsafe_after_repair03.csv`
  marks candidate `03` as `blocked_existing_output`.
- Result: candidate `03` ranks `4/31`, remains gate-negative and
  generator-locked.
- Raw primary metrics: positive partial windows `2/4`, min partial MSE
  `-0.5299312143911022`, min partial MAE `-0.545154931341216`,
  counterfactual margin `-0.024159312022691004`, and window0 FH MSE gain
  `0.2764126358705603`.
- Decomposition result: `candidate_count=31`, `gate_pass_count=0`,
  `generator_unlocked_count=0`, and
  `matrix_decision=stop_local_scalar_matrix`.
- Decision: `do_not_train_generator`; candidate `03` did not repair the coupled
  floor-plus-counterfactual failure. Do not rerun path-safe candidate `01`,
  `02`, or `03`. Candidate `04` is the only meaningful remaining path-safe row;
  if it is run, refresh selector, decomposition, and generator-unlock preflight
  before designing any new matrix.

2026-06-13 continuation path-safe repair candidate `04` bounded result:

- Completed bounded run:
  `logs/s3v29rps/v29rps_conf015_04_abstain/weather/seed_42`.
- Refreshed selector artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair04_selection.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair04_selection.md`.
- Refreshed preflight artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair04_generator_unlock_preflight.json`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair04_generator_unlock_preflight.md`.
- Refreshed decomposition artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair04_gate_failure_decomposition.json`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair04_gate_failure_decomposition.csv`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair04_gate_failure_decomposition_windows.csv`,
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_repair04_gate_failure_decomposition.md`.
- Final path-safe execution plan:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_decomposition_repair_execution_plan_pathsafe_after_repair04.csv`.
- Result: candidate `04` ranks `6/32`, remains gate-negative and
  generator-locked.
- Raw primary metrics: positive partial windows `2/4`, min partial MSE
  `-0.46285146776087893`, min partial MAE `-0.5138718209477093`,
  counterfactual margin `-0.03642005087344469`, and window0 FH MSE gain
  `0.35833191100094924`.
- Decomposition result: `candidate_count=32`, `gate_pass_count=0`,
  `generator_unlocked_count=0`, and
  `matrix_decision=stop_local_scalar_matrix`.
- Decision: `do_not_train_generator`; candidate `04` did not repair the
  teacher gate. The path-safe repair matrix is now exhausted in the meaningful
  sense. Do not rerun candidates `01`-`04`; any next run must be a new
  predeclared diagnosis or matrix rather than a blind continuation of this
  repair queue.

2026-06-13 continuation history-gate recovery matrix:

- Added post-repair planner:
  `train/plan_stage3_v29_history_gate_recovery_matrix.py`.
- Added tests:
  `tests/test_stage3_v29_history_gate_recovery_matrix.py`.
- Added implementation plan:
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
- Candidate axes:
  `window0_safe_release_ramp`, `late_history_dominance_bridge`,
  `counterfactual_tie_breaker_floor_safe`, and
  `history_cap_release_balance`.
- Execution-plan status before running: `4/4` rows are `ready_to_run`.
- Boundary: this is a new mechanism matrix targeting
  history-dominance/partial-floor conflict and near-tie counterfactual margins.
  It does not change topology, does not train the generator, and does not
  provide manuscript performance evidence.

2026-06-13 continuation history-gate recovery candidate `01` bounded result:

- Completed bounded run:
  `logs/s3v29hgr/v29hgr_conf015_01_w0ramp/weather/seed_42`.
- Refreshed selector artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr01_selection.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr01_selection.md`.
- Refreshed preflight artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr01_generator_unlock_preflight.json`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr01_generator_unlock_preflight.md`.
- Refreshed decomposition artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr01_gate_failure_decomposition.json`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr01_gate_failure_decomposition.csv`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr01_gate_failure_decomposition_windows.csv`,
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr01_gate_failure_decomposition.md`.
- Updated execution plan:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_history_gate_recovery_execution_plan_after_hgr01.csv`
  marks candidate `01` as `blocked_existing_output`.
- Result: candidate `01` ranks `2/33`, remains gate-negative and
  generator-locked.
- Raw primary metrics: positive partial windows `2/4`, min partial MSE
  `-0.43209738075898796`, min partial MAE `-0.4597064241028177`,
  counterfactual margin `-0.014453849719397449`, and window0 FH MSE gain
  `0.36675840890386036`.
- Decomposition result: `candidate_count=33`, `gate_pass_count=0`,
  `generator_unlocked_count=0`, and
  `matrix_decision=stop_local_scalar_matrix`.
- Decision: `do_not_train_generator`; candidate `01` is near the top but is not
  better than the one-epoch stability anchor and must not unlock generator
  training. If continuing this matrix, use only one of candidates `02`-`04`,
  then refresh selector, decomposition, and preflight again.

2026-06-13 continuation history-gate recovery candidate `02` bounded result:

- Completed bounded run:
  `logs/s3v29hgr/v29hgr_conf015_02_latehist/weather/seed_42`.
- Refreshed selector artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr02_selection.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr02_selection.md`.
- Refreshed preflight artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr02_generator_unlock_preflight.json`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr02_generator_unlock_preflight.md`.
- Refreshed decomposition artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr02_gate_failure_decomposition.json`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr02_gate_failure_decomposition.csv`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr02_gate_failure_decomposition_windows.csv`,
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr02_gate_failure_decomposition.md`.
- Updated execution plan:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_history_gate_recovery_execution_plan_after_hgr02.csv`
  marks candidates `01` and `02` as `blocked_existing_output`; candidates
  `03` and `04` remain `ready_to_run`.
- Result: candidate `02` ranks `1/34`, but remains gate-negative and
  generator-locked.
- Raw primary metrics: positive partial windows `2/4`, min partial MSE
  `-0.5290475768525864`, min partial MAE `-0.5365663873681754`,
  counterfactual margin `-0.009411902842471953`, and window0 FH MSE gain
  `0.2906420799259992`.
- Decomposition result: `candidate_count=34`, `gate_pass_count=0`,
  `generator_unlocked_count=0`, and
  `matrix_decision=stop_local_scalar_matrix`.
- Decision: `do_not_train_generator`; candidate `02` is a negative bounded
  tuning result despite ranking first. If continuing this matrix, use only one
  of candidates `03` or `04`, then refresh selector, decomposition, and
  preflight again.

2026-06-13 continuation history-gate recovery candidate `03` bounded result:

- Completed bounded run:
  `logs/s3v29hgr/v29hgr_conf015_03_cftie/weather/seed_42`.
- Refreshed selector artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr03_selection.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr03_selection.md`.
- Refreshed preflight artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr03_generator_unlock_preflight.json`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr03_generator_unlock_preflight.md`.
- Refreshed decomposition artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr03_gate_failure_decomposition.json`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr03_gate_failure_decomposition.csv`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr03_gate_failure_decomposition_windows.csv`,
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr03_gate_failure_decomposition.md`.
- Updated execution plan:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_history_gate_recovery_execution_plan_after_hgr03.csv`
  marks candidates `01`, `02`, and `03` as `blocked_existing_output`;
  candidate `04` remains `ready_to_run`.
- Result: candidate `03` ranks `1/35`, but remains gate-negative and
  generator-locked.
- Raw primary metrics: positive partial windows `2/4`, min partial MSE
  `-0.41351019804173306`, min partial MAE `-0.466581375695994`,
  counterfactual margin `-0.006009365114644338`, and window0 FH MSE gain
  `0.3719791529173322`.
- Decomposition result: `candidate_count=35`, `gate_pass_count=0`,
  `generator_unlocked_count=0`, and
  `matrix_decision=stop_local_scalar_matrix`.
- Decision: `do_not_train_generator`; candidate `03` is the best selector row
  so far, but it is still a negative bounded tuning result. If continuing this
  matrix, use only candidate `04`, then refresh selector, decomposition, and
  preflight again.

2026-06-13 continuation history-gate recovery candidate `04` bounded result:

- Completed bounded run:
  `logs/s3v29hgr/v29hgr_conf015_04_histcap/weather/seed_42`.
- Refreshed selector artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr04_selection.csv`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr04_selection.md`.
- Refreshed preflight artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr04_generator_unlock_preflight.json`
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr04_generator_unlock_preflight.md`.
- Refreshed decomposition artifacts:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr04_gate_failure_decomposition.json`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr04_gate_failure_decomposition.csv`,
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr04_gate_failure_decomposition_windows.csv`,
  and
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr04_gate_failure_decomposition.md`.
- Final execution plan:
  `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_history_gate_recovery_execution_plan_after_hgr04.csv`
  marks all four HGR candidates as `blocked_existing_output`.
- Result: candidate `04` ranks `9/36`, remains gate-negative and
  generator-locked.
- Raw primary metrics: positive partial windows `2/4`, min partial MSE
  `-0.4066591740244906`, min partial MAE `-0.4354709147954243`,
  counterfactual margin `-0.030275156997368824`, and window0 FH MSE gain
  `0.3889744979968982`.
- Final decomposition result: `candidate_count=36`, `gate_pass_count=0`,
  `generator_unlocked_count=0`, and
  `matrix_decision=stop_local_scalar_matrix`.
- Decision: `do_not_train_generator`; the HGR matrix is exhausted in the
  meaningful sense. Do not rerun candidates `01`-`04`. Any next tuning block
  must be a new predeclared diagnosis or matrix that changes the tested
  mechanism.
