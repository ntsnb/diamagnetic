# Stage III V27 Quality-Conditioned Dual-Stream Plan

## Goal

Turn the user's expert-local invariant/correction idea into a new official
version line that:

- does not overwrite old versions
- keeps repository paths clean
- includes base model, probe path, generator path, artifact loading, online
  wrapper, config, and tests

## Planned Work

1. Re-audit the current strongest and weakest evidence lines:
   - `V17`
   - `V23`
   - `V26`
2. Keep the `V26` correction backbone instead of creating another toy branch.
3. Add a dedicated quality-conditioning layer above the backbone.
4. Rebuild the served drift vector around quality-conditioned change packaging.
5. Add a new protocol namespace:
   - `model_v27`
   - `s3v27expert`
   - `v27_quality_conditioned_dual_stream_probe`
6. Add deterministic tests.
7. Run smoke and one initial formal screen.
8. Write back the implementation record with explicit limits.

## Outcome

Completed:

- new `V27` model file
- new offline training file
- new probe runner
- new generator/artifact/online wrappers
- new config
- `V27` tests
- `V26+V27` regression
- smoke run
- one completed `weather seed42 b20` initial screen
- completed `weather seed42 b60`
- completed `weather seed43 b60`
- completed `panama seed42 b60`
- completed `panama seed43 b60`
- completed narrow `panama seed43 b60` change-priority tuning pass

Not yet completed:

- larger `b120` completed formal screen
- paper-grade positive evidence

## Current Decision

`V27` is retained as a full official version line.

It is still not a validated new mainline, but the current evidence is now
stronger than before:

- `weather seed42 b60` and `seed43 b60` both show positive raw-side teacher
  signal
- `panama seed42 b60` shows positive `residual_change_vector` direction
- `panama seed43 b60` does not replicate that change-target direction and falls
  back to raw-side gain
- the first narrow change-priority retune on `panama seed43 b60` is negative:
  - it weakens the raw-side gain
  - it still does not recover the primary change target
- the primary change target still fails
- the strict gate still blocks generator training
