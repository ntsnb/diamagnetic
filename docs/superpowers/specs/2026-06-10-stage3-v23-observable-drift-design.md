# Stage III V23 Observable Drift Design

## Objective

Build a new official version line after `V22` that directly targets the
observability bottleneck identified by the `V22` negative result.

The working hypothesis is:

- the model may already contain useful release-related structure,
- but Stage III cannot exploit it if everything is forced through a single
  internal `delta_phi_v` representation.

`V23` therefore separates:

- internal geometric representation
- teacher/serve-facing observable drift representation

## Core Change Relative to V22

`V22` used:

- internal release state
- `pooled_delta_phi_v` as the model-native drift source

`V23` keeps the internal release state but adds:

- `pooled_observable_drift`
- formed from:
  - release state
  - current release increment
  - release confidence

This observable drift becomes the authoritative signal for:

- teacher-only split collection
- Stage III generator training wrapper
- risk-gate example collection
- online adaptation wrapper

## Expected Benefit

This version directly targets the new boundary condition:

- better training-serving parity for drift semantics
- richer teacher-facing signal without destroying the `phi_i / phi_v`
  geometric meaning
- a chance to recover `residual_change_vector` evidence without regressing to
  mathematically awkward direct subtraction logic

## Required Version Boundary

`V23` must not overwrite:

- `V16`
- `V17`
- `V18`
- `V19`
- `V20`
- `V21`
- `V22`

All paths must use the dedicated `s3v23expert` namespace.
