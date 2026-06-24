# Stage III V24 Tail Observable Change Design

## Objective

Build a new official version line after `V23` that tests a narrower hypothesis:

- `V23` may have recovered a weak `residual_change_vector` signal because it
  improved teacher-facing observability.
- But that signal may still be too diffuse because the observable packaging is
  not tail-sensitive enough.

`V24` therefore keeps the `V23` expert-local correction-operator structure and
changes the drift interface seen by Stage III.

## Hypothesis

The specific hypothesis is:

> If the useful part of the release signal is concentrated in later timesteps
> inside each input window, then a tail-focused observable-change summary should
> preserve more of the causal change information than mean-like pooling.

This is intentionally narrower than redesigning invariant learning again.

## Core Design

`V24` introduces three linked changes:

1. Explicit within-window observable change:
   - `observable_change_seq[t] = budgeted_release_seq[t] - budgeted_release_seq[t-1]`
2. Tail-focused aggregation:
   - a new `_tail_pool(...)` summary replaces simple mean pooling for the main
     release-related expert summaries
3. Expanded teacher-facing drift:
   - `pooled_tail_observable_change`
   - combines:
     - release state
     - current budgeted release
     - observable change
     - release confidence

The geometric interpretation remains:

- `phi_i` is still the invariant-side representation
- `delta_phi_v` still corresponds to the internal release state

But the operational Stage III drift source becomes:

- `pooled_tail_observable_change`

## Why This Was Worth Testing

Relative to `V23`, this is a small but coherent systems change:

- it preserves version hygiene
- it keeps train/serve parity at the drift interface
- it adds no cross-sample leakage
- it makes the teacher-facing signal more explicitly change-aware

If it had worked, it would have supported a simpler story than another large
architectural jump.

## Expected Outcome

Success criteria were modest:

- maintain or improve `raw_residual_vector` validation partial gain vs history
- keep counterfactual harm from getting worse
- most importantly, amplify the weak `residual_change_vector` signal already
  seen in `V23 weather seed42 b240`

## Result Summary

The actual result did not support the hypothesis:

- `V24 weather seed42 b240` kept generator training blocked
- `raw_residual_vector` was slightly worse than `V23 seed42`
- `residual_change_vector` dropped back to zero

So the design conclusion is:

> tail-focused observable packaging alone is not enough; the next version must
> target readout/supervision of residual change, not just observable-drift
> aggregation.
