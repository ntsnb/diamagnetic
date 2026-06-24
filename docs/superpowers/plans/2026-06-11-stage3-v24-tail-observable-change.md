# Stage III V24 Tail Observable Change Plan

## Goal

Evaluate whether a tail-focused observable-change interface can stabilize the
weak `V23` `residual_change_vector` hint without another large architectural
rewrite.

## Planned Work

1. Copy the `V23` version line into a dedicated `V24` namespace.
2. Keep the `V23` expert-local correction-operator backbone unchanged.
3. Add an explicit `observable_change_seq` from within-window release
   differences.
4. Replace mean-style pooling with tail pooling for the main release-related
   expert summaries.
5. Make `pooled_tail_observable_change` the authoritative drift source for:
   - split collection
   - teacher-only gate
   - generator wrapper
   - online wrapper
6. Add a lightweight offline observable-change alignment loss.
7. Run deterministic tests and `V23+V24` regression.
8. Run one smoke and one formal `weather seed42 b240` screen.
9. Only run `seed43` if `seed42` shows a clear improvement over `V23`.

## Outcome

Completed:

- isolated `V24` implementation
- deterministic validation
- smoke run
- formal `weather seed42 b240` screen
- evidence-first audit writeback

Not continued:

- `seed43` formal screen

Reason:

- the first formal screen did not improve over `V23`
- `residual_change_vector` collapsed back to zero
- continuing with another seed would add cost without changing the current
  ranking of the version lines

## Final Decision

`V24` is retained as a documented negative result, not promoted as the new
mainline.

The next version, if pursued, should target explicit residual-change readout or
delayed-change supervision rather than further `V24` weight tuning.
