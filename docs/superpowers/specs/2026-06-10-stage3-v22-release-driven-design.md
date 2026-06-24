# Stage III V22 Release-Driven Design

## Objective

Build a new official version line after `V21` that specifically targets the
remaining failure mode:

- `V21` improved the raw-target side inside the `V20+` correction-operator
  family.
- But `residual_change_vector` still remained absent.

The `V22` hypothesis is that release should not be represented only as a
one-step budgeted correction. Instead, it should be represented as an
**explicit accumulated release state** driven by expert-local release
increments.

## Core Change Relative to V21

`V21` used release-first budgeting, but the served drift source was still the
budgeted release output directly.

`V22` changes the expert-local mechanism to:

1. build shared expert-local state
2. build invariant representation with:
   - base
   - invariant adapter
   - environment-conditioned correction operator
3. predict a budgeted release increment
4. integrate that increment into a release state
5. expose the release state as `delta_phi_v`
6. let history model the remainder after subtracting the accumulated release
   state

So the intended semantics are:

- `phi_i` remains the invariant anchor after correction
- release increment is the immediate adjustment proposal
- release state is the accumulated served drift source
- history is explicitly downstream of the release state

## Expected Benefit

This version directly targets the `V21` bottleneck:

- if `delta_phi_v` is an accumulated state, then
  `fluctuation_delta(delta_phi_v)` should behave more like a meaningful release
  change signal
- this may improve teacher-only evidence on `residual_change_vector`

## Required Version Boundary

`V22` must not overwrite:

- `V16`
- `V17`
- `V18`
- `V19`
- `V20`
- `V21`

All paths must use the dedicated `s3v22expert` namespace.
