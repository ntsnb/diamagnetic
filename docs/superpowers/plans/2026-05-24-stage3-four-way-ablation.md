# Stage III Four-Way Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a non-overwriting, paper-grade four-way Stage III ablation comparing scalar and vector concept-drift signals under the existing online load-forecasting protocol.

**Architecture:** Keep original training/model files intact. Add a configurable copied online loop that supports `level_scalar`, `delta_scalar`, `level_vector`, and `delta_vector`, then add a benchmark harness that tunes Stage III hyperparameters on the validation stream and evaluates the selected setting once on the test stream. Results are written to new timestamped CSV/JSON artifacts with MSE, MAE, gains, tuning metadata, and drift diagnostics.

**Tech Stack:** Python, PyTorch, existing `train/run_experiment.py` offline artifact path, existing dataset/config definitions, pytest smoke tests.

---

### Task 1: Implement Stage III Signal Utilities

**Files:**
- Create: `train/stage3_signals.py`
- Test: `tests/test_stage3_signals.py`

- [ ] **Step 1: Add signal helper API**

Create `Stage3SignalState`, `Stage3SignalResult`, and `build_stage3_drift_vector()` with this contract:

```python
result = build_stage3_drift_vector(
    drift_raw=phi_v - phi_i,
    mode="delta_vector",
    bottleneck_dim=16,
    state=state,
    update_state=True,
)
assert result.drift_vec.ndim == 1
```

`level_scalar` must reproduce the original implementation: global mean divided by global absolute mean, repeated to `bottleneck_dim`.

`delta_scalar` must use the same scalarization after replacing `drift_raw` by the sample-wise temporal difference.

`level_vector` must average over batch only and preserve embedding dimensions.

`delta_vector` must preserve embedding dimensions after sample-wise temporal differencing.

- [ ] **Step 2: Add deterministic tests**

Add tests proving:

```python
assert level_vector.shape == (embedding_dim,)
assert not torch.allclose(level_vector, level_vector.mean().repeat(embedding_dim))
assert delta_vector equals mean([(row_t - row_t_minus_1) ...], dim=0)
assert scalar_vec.numel() == bottleneck_dim
assert scalar_vec.unique().numel() == 1
```

- [ ] **Step 3: Run tests**

Run:

```powershell
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_stage3_signals.py -q
```

Expected: all tests pass.

### Task 2: Implement Non-Overwriting Ablation Online Loop

**Files:**
- Create: `train/online_loop_stage3_ablation.py`
- Reuse: `train/online_loop.py`
- Test: `tests/test_stage3_signals.py`

- [ ] **Step 1: Copy the original loop**

Copy behavior from `train/online_loop.py`, but read `online.stage3.signal_mode`.

- [ ] **Step 2: Use separate Stage III generator when vector mode is active**

Stage II keeps the original `lrp_bottleneck` generator. Stage III uses:

```python
stage3_gen = AdaptationGenerator(
    drift_dim=stage3_drift_dim,
    rank=online_cfg["stage2"]["lrp_bottleneck"],
    max_dim=512,
)
```

`stage3_drift_dim` must come from the actual vector signal dimension, not from scalar repeat.

- [ ] **Step 3: Log diagnostics**

Add trace columns:

```text
stage3_signal_mode, stage3_drift_dim, stage3_level_abs_mean,
stage3_signal_abs_mean, stage3_signal_norm, stage3_pred_shift, stage3_blend
```

- [ ] **Step 4: Run syntax check**

Run:

```powershell
C:\Users\admin\anaconda3\python.exe -m py_compile train\stage3_signals.py train\online_loop_stage3_ablation.py
```

Expected: exit code 0.

### Task 3: Implement Validation-Tuned Benchmark Harness

**Files:**
- Create: `train/run_stage3_ablation_benchmark.py`

- [ ] **Step 1: Train offline artifacts once per dataset/seed**

Call `run_single_experiment(..., return_artifacts=True)` with `online_run.enable=False` so all four Stage III modes share the same offline model, invariant encoders, train/validation/test split, and seed.

- [ ] **Step 2: Tune on validation stream**

For each dataset, seed, variant, and signal mode, sweep only Stage III knobs:

```text
lr_scale: 0.00025, 0.0005, 0.001
blend_alpha: 0.20, 0.28, 0.34
max_pred_shift: 0.10, 0.15
```

Select by lowest validation `online_stage3_mse`, breaking ties by validation MAE.

- [ ] **Step 3: Evaluate once on test stream**

Use the selected Stage III settings, reset RNG, deep-copy the offline model, and evaluate on `test_ds`.

- [ ] **Step 4: Write result artifacts**

Write:

```text
logs/load_benchmarks/stage3_ablation/<run_id>_tuning.csv
logs/load_benchmarks/stage3_ablation/<run_id>_test.csv
logs/load_benchmarks/stage3_ablation/<run_id>_summary.json
```

Rows include MSE, MAE, gains, seed, dataset, variant, mode, selected hyperparameters, offline metrics, drift diagnostics, trace path, and split metadata.

### Task 4: Run Checks and Launch Formal Experiments

**Files:**
- Use: `tests/test_stage3_signals.py`
- Use: `train/run_stage3_ablation_benchmark.py`

- [ ] **Step 1: Run unit and syntax checks**

Run:

```powershell
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_stage3_signals.py tests\test_shapes.py -q
C:\Users\admin\anaconda3\python.exe -m py_compile train\stage3_signals.py train\online_loop_stage3_ablation.py train\run_stage3_ablation_benchmark.py
```

- [ ] **Step 2: Run a formal-interface smoke**

Run one dataset, one seed, reduced epochs only to verify plumbing:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_ablation_benchmark.py --datasets panama --seeds 42 --epochs 1 --invariant-epochs 1 --variants stage3 --tune-grid smoke --run-id codex_stage3_ablation_smoke
```

- [ ] **Step 3: Launch larger run**

Run all three public load datasets, three seeds, paper-level epochs, and full tuning grid:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_ablation_benchmark.py --datasets panama tetouan spain_energy_weather --seeds 42 43 44 --epochs 5 --invariant-epochs 5 --variants stage3 stage1_stage2_stage3 --tune-grid full --run-id codex_stage3_ablation_full
```

This is expected to be expensive; keep outputs in new `stage3_ablation` paths only.

### Task 5: Summarize Evidence

**Files:**
- Create: `docs/audit/STAGE3_ABLATION_REPORT.md`

- [ ] **Step 1: Load final CSVs**

Compute per-dataset and mean MSE/MAE gains for all four modes.

- [ ] **Step 2: Mark paper eligibility**

Classify each claim as:

```text
supported, weakly supported, unsupported, contradicted
```

- [ ] **Step 3: Recommend manuscript action**

Only recommend paper updates if test evidence shows a real improvement under the validation-selected protocol.
