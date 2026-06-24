# Stage III V21 Release-First Budgeted Design

## Objective

Build a new official version line that keeps the `V20` correction-operator idea
but fixes the suspected ordering problem:

- release should see the residual after the invariant path first;
- history should model the remaining stable residue after release, not compete
  with release from the start.

## Core change relative to V20

`V20` used an explicit residual budget allocator, but the empirical signal
suggested that history and release were still effectively competing in a way
that left the primary release target too weak.

`V21` reorders the expert-local logic:

1. build shared expert-local state
2. build invariant representation with:
   - base
   - invariant adapter
   - environment correction operator
3. compute release proposal and release budget first
4. form budgeted release state
5. compute history proposal on the release-adjusted remainder
6. allocate the remaining budget to history

So the intended semantics are:

- `phi_i` is still the corrected invariant representation
- `delta_phi_v` is still the served drift source
- but history is now explicitly downstream of release

## Expected benefit

This version directly targets the bottleneck identified in `V20`:

- `V20` was structurally correct but too conservative
- `V20 release_priority` partially restored `raw_residual_vector`
- `residual_change_vector` still stayed at zero

`V21` should give the release branch a stronger chance to encode the causal
increment before history compresses it away.

