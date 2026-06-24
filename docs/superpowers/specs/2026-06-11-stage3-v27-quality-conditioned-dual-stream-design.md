# Stage III V27 Quality-Conditioned Dual-Stream Design

## Objective

Build a new official version line after the `V26` tuning plateau without
overwriting any old version. The design must stay faithful to the two Chinese
algorithm audit reports:

- keep the expert-local shared-representation plus late correction structure
- keep environment inference as a correction operator rather than as the
  definition of `phi_i`
- keep a teacher-facing observable drift package
- avoid treating "better orthogonality" as sufficient success

## Why V27 Exists

The most important verified findings before `V27` were:

- `V17` remained the strongest empirical weather reference
- `V23` was the first `V20+` line to recover a weak non-zero
  `residual_change_vector` hint
- `V26` tuning proved that local hyperparameter changes can move the raw-side
  signal, but the family still saturates and does not reliably recover the
  primary change target

So the next coherent move was not another small tuning branch on `V26`. It was
to keep the correction backbone but change the way the variant branch is
packaged and trusted.

## Core Hypothesis

`V27` tests the following hypothesis:

> The expert-local correction backbone may already be rich enough, but the
> system still lacks an explicit quality-conditioned aggregation layer that
> decides when delayed expert-local change should be trusted and routed into the
> served drift vector.

## Core Design

`V27` is built on top of the `V26` expert-local correction backbone and adds a
new top-level quality-conditioning layer.

### 1. Keep the expert-local correction backbone

`V26` already contains:

- shared expert-local representation
- invariant anchor plus environment correction
- release-state construction
- history branch
- change-memory branch

`V27` keeps that backbone trainable and reuses its internal signals.

### 2. Add a separate quality adapter per expert

For each expert, `V27` consumes:

- observable change
- change memory
- raw release
- pooled environment shift
- release budget
- release confidence
- raw change readout
- delayed change readout

and maps them into:

- `quality_state`
- `quality_score`

This makes the "should this delayed change be trusted?" decision explicit
instead of reusing `V26`'s multiscale score semantics.

### 3. Replace the variant aggregation rule

`V27` defines:

`expert_y_var = expert_y_change_raw + quality_score * expert_change_delta`

So the delayed change branch is not always injected. It is conditioned by a
separate quality estimate.

### 4. Replace the served drift package

The native Stage III source is now a new quality-conditioned served vector:

- `quality_state`
- observable change
- change memory
- quality-weighted change memory
- quality score
- release confidence
- release budget
- quality-weighted output change
- raw change readout
- delayed change readout

This is exported as `pooled_servable_change`.

### 5. Add a router-level quality bias

`V27` does not keep `V26` routing untouched. It adds a second router bias using:

- pooled environment shift
- observable-change norm
- change-memory norm
- quality-state norm
- quality score
- release budget
- release confidence

and reweights `alpha_logits` before final fusion.

## Why This Is Not A Toy Edit

`V27` changes all of these boundaries:

- base-model top-level outputs
- expert-local variant aggregation semantics
- router bias logic
- served drift packaging
- generator target coverage
- offline training losses
- probe protocol namespace

It is therefore a full version line, not a sidecar patch.

## Expected Success Condition

The intended success condition was:

- preserve the expert-local correction idea from the audit reports
- outperform the null `V26`-style local saturation pattern
- recover at least weak non-zero teacher evidence on primary targets

The current implementation status is:

- code path complete
- deterministic validation complete
- first completed screen still null

So the design is now ready for fuller experimental confirmation, but it has not
yet earned promotion over earlier empirical references.
