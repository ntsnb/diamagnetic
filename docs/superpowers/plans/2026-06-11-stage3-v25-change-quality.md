# Stage III V25 Change Quality Plan

## Goal

Evaluate whether an explicit residual-change readout plus delayed-change
supervision plus quality-aware drift packaging can turn the `V23` weak
`residual_change_vector` hint into a stronger official version line.

## Planned Work

1. Copy the `V24` version line into a dedicated `V25` namespace.
2. Keep the expert-local correction backbone and preserve old version paths.
3. Add a change-target readout head.
4. Add a change-quality head and route quality into both the router and the
   served drift signal.
5. Add delayed-change supervision and quality-alignment losses.
6. Expand generator target coverage to include `change_quality_heads`.
7. Run deterministic validation and `V24+V25` regression.
8. Run one smoke and one formal `weather seed42 b240` screen.
9. Only run more seeds if the first formal screen is at least directionally
   positive.

## Outcome

Completed:

- isolated `V25` implementation
- deterministic validation
- cross-version regression
- smoke run
- formal `weather seed42 b240` screen
- evidence-first audit writeback

Not continued:

- second formal seed

Reason:

- the first formal screen was clearly negative
- the key target `residual_change_vector` regressed rather than weakly improved

## Final Decision

`V25` is retained as a documented negative result, not promoted as the new
mainline.

The next version, if pursued, should change the correction-state structure
rather than continuing to stack more target-side readout losses on the same
backbone.
