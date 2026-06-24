# Stage III V26 Change Quality Design

## Objective

Build a new official version line after `V24` that directly tests a sharper
hypothesis:

- maybe the expert-local correction backbone is already rich enough
- but the system still lacks
  - an explicit residual-change readout
  - delayed-change supervision
  - and a notion of drift quality rather than drift magnitude alone

`V26` therefore keeps the `V24` correction backbone and changes the target-side
readout and gating logic.

## Hypothesis

The specific hypothesis is:

> If the weak `V23` `residual_change_vector` hint failed to stabilize because
> Stage III did not get a clean change-target readout, then an explicit
> change-readout head plus delayed-change supervision plus quality-aware served
> drift packaging should improve teacher-gate evidence.

## Core Design

`V26` introduces four linked changes:

1. Dedicated change readout
   - the variant head is explicitly treated as a change-target head
2. Delayed-change auxiliary supervision
   - finite-difference supervision is added on the output vector
3. Per-expert change quality score
   - a new scalar gate estimates whether the local change readout looks useful
4. Quality-aware served drift source
   - `pooled_servable_change`
   - combines base observable change, quality-weighted observable change,
     quality score, raw change readout, and delayed change readout

Router context is also expanded to include:

- change-quality score
- change-delta norm

## Why This Was Worth Testing

This was not a toy edit. It changed:

- the local expert readout semantics
- the supervision family
- the router context
- the teacher/generator/online drift boundary

It was therefore a coherent next-generation test of the user's algorithmic
direction rather than a small compatibility patch.

## Expected Outcome

Success criteria were:

- preserve or improve `raw_residual_vector`
- most importantly, recover and strengthen `residual_change_vector`
- make the `V23` weak seed42 hint less fragile

## Result Summary

The actual result contradicted the hypothesis:

- `V26 weather seed42 b240` remained blocked by the strict teacher gate
- `raw_residual_vector` became clearly worse
- `residual_change_vector` became clearly negative

So the design conclusion is:

> change-readout supervision and quality-aware gating alone are not enough on
> top of the current expert-local correction backbone.

