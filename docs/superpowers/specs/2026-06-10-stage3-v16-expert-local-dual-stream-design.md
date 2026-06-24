# Stage III V16 Expert-Local Dual-Stream Design

## Goal

V16 is the canonical next-generation Stage III line for this repository. It promotes expert-local invariant learning from an external side model into the base forecasting model itself.

V16 is designed to test the user's new idea in a non-overwriting, version-safe way:

- each expert owns an internal dual-stream invariant/variant decomposition;
- most expert layers share one representation before late branching;
- `phi_i` is produced by an expert-local correction path rather than by subtracting unrelated embeddings;
- Stage III drift diagnostics and online serving consume representations emitted directly from the base model;
- V2-V15 remain intact as historical evidence and comparison baselines.

## Core Design Decision

V16 is allowed to redefine and retrain the base `FactorMechanism` line.

This is deliberate. The goal is not to bolt a new sidecar onto V2, but to make expert-local dual-stream invariant learning part of the main architecture. Old checkpoints are not required to load into V16. V16 must instead:

- use new versioned filenames,
- use new log/checkpoint/result roots,
- keep old code paths untouched,
- document incompatible artifact boundaries explicitly.

## Architectural Summary

V16 keeps the high-level forecasting skeleton:

`temporal encoder -> latent gate -> expert mixture -> forecast`

but replaces the expert payload with a dual-stream decomposition:

`shared expert state -> invariant correction branch + drift branch -> pooled phi_i / delta_phi_v / phi_v`

### Per-expert structure

For expert `k` at time `t`:

```text
h_k,t = VariableAxisExpert_k(x_t)
z_k,t = shared_post_k(h_k,t)
u_pre_k,t = invariant_anchor_k(z_k,t)
e_k,t = q_env_k(z_k,t)
c_env_k,t = env_basis_k(e_k,t)
u_k,t = u_pre_k,t - g_inv_k(z_k,t) * c_env_k,t
d_pre_k,t = drift_head_k(z_k,t)
d_env_k,t = drift_env_head_k(z_k,t, e_k,t)
d_k,t = d_pre_k,t + d_env_k,t
phi_i_k,t = u_k,t
phi_v_k,t = phi_i_k,t + d_k,t
```

This keeps invariant and variant representations in the same expert-local coordinate system.

### Global aggregation

The gate `alpha` still mixes experts. V16 aggregates three expert-local objects:

- pooled invariant state
- pooled drift residual state
- pooled full variant state

These are exposed as:

- `pooled_phi_i`
- `pooled_delta_phi_v`
- `pooled_phi_v`

V16 also predicts:

- `y_inv` from `pooled_phi_i`
- `y_var` from `pooled_delta_phi_v`
- `y_hat = y_inv + y_var`

This allows the base model and representation losses to live in one graph.

## Why this version is different from V2-V15

V2-V15 use:

- a standard MoE base predictor, and
- a separate aligned representation model for `phi_i / phi_v`.

V16 removes that split source of truth. Stage III representation now comes from the same internal states that serve forecasting.

This addresses three concrete problems:

1. representation-source mismatch between forecasting and drift analysis;
2. post-hoc alignment dependence;
3. awkward interpretation of `phi_v - phi_i` across loosely coupled representations.

## What V16 does not assume

V16 does not assume that the new geometry automatically solves residual-target predictability.

The V13-V15 negative evidence still matters. Therefore V16 must preserve:

- chronological validation,
- train-only signal fitting,
- history-only comparison,
- real-versus-counterfactual drift checks,
- online batch size 1 constraints where required.

V16 is a structural redesign, not an excuse to drop the old gates.

## Environment inference policy

V16 does not reuse old `MTEI` or label-dependent `EnvInference` in the online path.

Instead, V16 uses a causal expert-local environment posterior:

```text
q_env_k = softmax(env_router_k(z_k,t))
```

and learned environment bases:

```text
env_context_k = q_env_k @ env_basis_k
```

This means:

- training does not require current-step labels to form `phi_i`,
- serving can compute `phi_i` and `phi_v` from `x` alone,
- environment inference becomes a latent causal modulation variable, not an offline oracle.

## Loss design

V16 base training combines forecasting and representation objectives in one optimization.

### Base representation losses

Reuse the proven aligned-representation loss family on pooled states:

- forecast reconstruction through `y_hat`
- invariant-only forecast `y_inv`
- residual forecast `y_var`
- orthogonality penalty
- smoothness penalty on drift
- HSIC / Barlow penalties
- temporal ranking and deployed temporal ranking
- temporal contrastive loss
- VICReg variance/covariance penalties

### New V16-only structural losses

Add:

- expert load-balance loss on `alpha`
- expert diversity penalty on expert-local pooled drift states
- environment marginal balance penalty to reduce trivial one-environment collapse

These new losses are not meant to prove semantic disentanglement. They are structural anti-collapse controls.

## Train/serve contract

V16 defines a new contract:

- the base model emits `phi_i`, `delta_phi_v`, and `phi_v`;
- Stage III teacher diagnostics consume these directly;
- online serving may compute drift internally from base-model outputs rather than from an external aligned model;
- old V2 online replay remains untouched.

## Versioned file boundaries

V16 introduces new versioned files only.

### New model files

- `models/factor_mechanism_v16.py`
- optional future siblings such as `models/stage3_residual_teacher_v16.py` only if V14 reuse becomes too awkward

### New training / online files

- `train/offline_factor_v16.py`
- `train/online_loop_stage3_v16.py`
- `train/run_stage3_v16_expert_local_dual_stream_probe.py`

### New configs

- `configs/stage3_v16_expert_local_dual_stream_probe.yaml`

### New tests

- `tests/test_factor_mechanism_v16.py`
- `tests/test_offline_factor_v16.py`
- `tests/test_stage3_v16_expert_local_dual_stream.py`

### New evidence docs

- `docs/audit/STAGE3_V16_EXPERT_LOCAL_DUAL_STREAM_IMPLEMENTATION_RECORD.md`

## Output root policy

V16 must use independent roots:

- `logs/s3v16expert`
- `ckpt/s3v16expert`
- `results/s3v16expert`

No V16 command may write to V15 roots.

## Experiment protocol

The first V16 experiment is not a manuscript claim run. It is a structured canary:

1. train V16 base model offline;
2. collect direct model-emitted `delta_phi_v`;
3. run V15-style residual-target predictability diagnostics;
4. compare `history_only` vs `fluctuation_plus_history`;
5. keep counterfactual checks;
6. stop before generator claims if the primary gate fails.

Only after that should V16 generator training and served online replay become claim-bearing artifacts.

## Success conditions for the first implementation phase

The first implementation phase is successful if all of the following are true:

- V16 model builds and runs forward deterministically in eval mode.
- V16 exposes direct pooled `phi_i / delta_phi_v / phi_v`.
- V16 offline training loop executes with integrated representation losses.
- V16 diagnostic runner collects raw drift from the base model rather than an external aligned model.
- V16 online replay can consume internal V16 representations.
- Tests cover offline path, online path, and config/version safety.

## Known risks

- Better geometry may still fail residual-target predictability.
- Environment posterior may collapse without careful regularization.
- Expert drift branches may become redundant without diversity control.
- V16 may increase training instability due to multi-objective coupling.
- Existing generator logic may still be too tied to V2 assumptions and may require a second V16 pass.

## Implementation posture

V16 should not be built as a toy branch.

It should be built as a new canonical line with:

- dedicated filenames,
- explicit contracts,
- anti-collapse controls,
- reusable diagnostics,
- evidence-first experiment flow.
