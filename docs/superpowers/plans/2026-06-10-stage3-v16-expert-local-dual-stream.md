# Stage III V16 Expert-Local Dual-Stream Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a version-safe V16 Stage III line that integrates expert-local dual-stream invariant learning into the base model, reuses chronological teacher diagnostics, and adds a new online replay path without overwriting V2-V15.

**Architecture:** V16 introduces a new `FactorMechanism` generation where each expert late-branches into invariant correction and drift residual heads. The base model directly emits pooled `phi_i`, `delta_phi_v`, and `phi_v`, so Stage III diagnostics and serving can consume model-native drift states instead of an external aligned representation model.

**Tech Stack:** PyTorch, existing train/eval utilities, Stage3SignalBuilderV13, V14 teacher diagnostics, repository YAML config system, pytest.

---

### Task 1: Add V16 base model

**Files:**
- Create: `models/factor_mechanism_v16.py`
- Test: `tests/test_factor_mechanism_v16.py`

- [ ] Define V16 dataclasses for outputs and configs.
- [ ] Implement expert-local dual-stream head with environment posterior, invariant correction, and drift residual branches.
- [ ] Implement V16 `FactorMechanism` using the existing temporal encoder / VAE / gate skeleton.
- [ ] Expose pooled `phi_i`, `pooled_delta_phi_v`, `pooled_phi_v`, `y_inv`, `y_var`, and `y_hat`.
- [ ] Add deterministic eval behavior equivalent to V2.
- [ ] Write forward-shape and deterministic-eval tests.

### Task 2: Add V16 offline training loop

**Files:**
- Create: `train/offline_factor_v16.py`
- Modify: `models/factor_mechanism_v16.py`
- Test: `tests/test_offline_factor_v16.py`

- [ ] Define a V16 loss config that combines forecast, residual decomposition, anti-collapse, and optional VAE losses.
- [ ] Reuse aligned-representation penalties on pooled V16 states.
- [ ] Add load-balance, environment-balance, and expert-diversity penalties.
- [ ] Implement `train_offline_factor_v16` and `evaluate_factor_v16`.
- [ ] Write a smoke test proving one optimizer step runs and losses are finite.

### Task 3: Add V16 diagnostic runner

**Files:**
- Create: `train/run_stage3_v16_expert_local_dual_stream_probe.py`
- Create: `configs/stage3_v16_expert_local_dual_stream_probe.yaml`
- Test: `tests/test_stage3_v16_expert_local_dual_stream.py`

- [ ] Build V16 model construction from config.
- [ ] Reuse repository data loading and split-safe setup.
- [ ] Replace external aligned-model drift collection with V16 model-native `pooled_delta_phi_v`.
- [ ] Reuse V13 signal builder and V14 teacher machinery for residual-target diagnostics.
- [ ] Enforce V16-only output roots and non-overwrite behavior.
- [ ] Add tests for config contract and split tensor collection.

### Task 4: Add V16 online replay path

**Files:**
- Create: `train/online_loop_stage3_v16.py`
- Test: `tests/test_stage3_v16_expert_local_dual_stream.py`

- [ ] Add a served path that computes `phi_i / phi_v` internally from the V16 model.
- [ ] Preserve risk gate, shift gate, blend, and batch-size-1 causal behavior.
- [ ] Keep old V2 online loop untouched.
- [ ] Add a test that V16 online replay rejects unsupported batch settings when causal mode is enabled.

### Task 5: Add implementation record

**Files:**
- Create: `docs/audit/STAGE3_V16_EXPERT_LOCAL_DUAL_STREAM_IMPLEMENTATION_RECORD.md`

- [ ] Document version boundaries, new files, output roots, and what V16 changes relative to V15.
- [ ] Record validation commands and remaining experiment gaps.

### Task 6: Verification

**Files:**
- Run-only verification against new V16 files and tests

- [ ] Run `python -m py_compile` on all new V16 Python files.
- [ ] Run focused pytest for V16 model/training/diagnostic tests.
- [ ] If feasible, run one config-level smoke invocation with tiny batch limits and document whether it finishes or where it stops.
- [ ] Capture any residual risks that prevent a paper-grade claim.
