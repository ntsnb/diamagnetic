# Stage III V19 Conditional-Release Design

## Scope

`V19` is the next official non-overwriting Stage III line after `V18`.

It keeps the expert-local invariant dual-stream structure, but changes the
training objective and downstream interpretation so that the model is no longer
optimized only for a cleaner `phi_i / phi_v` geometry. Instead, `V19` treats
the expert-local release state as the primary object to supervise.

This spec is derived from:

- `docs/audit/2026-06-09_EXPERT_INVARIANT_DUAL_STREAM_REVIEW_CN.md`
- `docs/audit/2026-06-10_EXPERT_LOCAL_INVARIANT_DUAL_STREAM_ALGORITHM_CORRECTION_REPORT_CN.md`
- `docs/audit/STAGE3_V17_CAUSAL_ENV_DUAL_STREAM_IMPLEMENTATION_RECORD.md`
- `docs/audit/STAGE3_V18_STRUCTURED_ENV_RELEASE_IMPLEMENTATION_RECORD.md`

## Why a new line is needed

The reports and experiments agree on two points:

1. Moving invariant / variant structure inside each expert was the correct
   structural direction.
2. The deepest failure is not just the geometry of `phi_v - phi_i`, but the
   lack of stable conditional incremental value beyond `history_only`.

`V18` improved the structure by separating:

- invariant anchor
- environment-conditioned suppression
- release state
- release confidence

But `V18` still trained the downstream prediction path too indirectly. It did
not make the release branch explicitly compete against a history-like proxy
inside the base model.

## Core V19 idea

Per expert `k`, the representation remains:

- stable shared expert state
- invariant anchor
- environment suppression
- release state
- confidence

But the prediction path is refined into three pieces:

1. `expert_y_inv`
   - invariant forecast from `phi_i`
2. `expert_y_hist`
   - history-like / easy residual proxy from stable context
3. `expert_y_var`
   - release increment from release state, confidence, and environment shift

The full expert forecast becomes:

`expert_y_hat = expert_y_inv + expert_y_hist + expert_y_var`

This changes the training objective from “make `phi_i` and `phi_v` cleaner” to:

- let the stable path absorb easy predictable structure;
- let the release path focus on what remains after the stable proxy;
- penalize the release path if it does not beat the history-like proxy.

## Architecture change

`V19` keeps the `V18` local head semantics:

- `phi_i = anchor - suppression`
- `delta_phi_v = raw_release`
- `phi_v = phi_i + delta_phi_v`

But the global readout changes:

- add `history_output_heads`
- keep `invariant_output_heads`
- keep `variant_output_heads`
- route over the same expert-local structured context

The intended semantics are:

- `delta_phi_v` is now the model-native release state by construction
- `phi_v - phi_i` is no longer an arbitrary post-hoc difference between two
  separately learned spaces
- the release state is directly supervised as an incremental correction branch

## New losses

`V19` adds five new supervision terms on top of the `V18` loss family:

1. `local_history_proxy`
   - supervise `expert_y_hist` against the residual left after `expert_y_inv`
2. `local_release_increment`
   - supervise `expert_y_var` against the residual left after
     `expert_y_inv + expert_y_hist`
3. `conditional_advantage`
   - require the full expert forecast to beat the history-proxy path by margin
4. `release_null`
   - when environment shift is small and confidence is low, the release
     increment should stay small
5. `counterfactual_shuffle`
   - shuffled release increments should be less predictive than the real ones

These do not prove conditional incremental value at the paper claim level.
That still requires the existing teacher-only and target-gate protocol. But the
base model is now trained with a closer approximation to the desired behavior.

## Intended Stage III semantics

For `V19`, the official interpretation is:

- `pooled_raw_release`
  - model-native release state
- `pooled_delta_phi_v`
  - equal to `pooled_raw_release` by construction

The old Stage III machinery can still consume `delta_phi_v`, but for `V19`
this field should be interpreted as release state, not as a generic “variant
minus invariant” artifact.

## Non-goals

`V19` does not yet claim:

- stable teacher-gate success
- generator unlock
- served online improvement
- superiority over `V17` on formal weather or public-load benchmarks

Those require explicit experimental proof after implementation.

## Tooling note

The project-level instructions asked to use superpowers and subagents when
possible. In this environment, the superpowers workflow is available as a
methodology, but no executable child-agent dispatch tool is exposed. Therefore
the implementation follows:

- design-first planning
- version-safe file ownership
- TDD / regression checks
- evidence-first experiment writeback

without claiming real spawned subagent execution.
