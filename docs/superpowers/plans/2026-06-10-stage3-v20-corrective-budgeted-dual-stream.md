# Stage III V20 Corrective Budgeted Dual-Stream Plan

1. Add a new expert-local head that turns the user idea into:
   - shared vector
   - invariant correction operator
   - causal environment suppression
   - history proposal
   - release proposal
   - residual budget allocator

2. Build a full `V20` version line:
   - model
   - offline trainer
   - Stage III probe runner
   - generator wrapper
   - artifact loader
   - online wrapper

3. Keep old version lines intact:
   - do not edit `V16` to `V19` behavior
   - isolate all new outputs under `s3v20expert`

4. Add deterministic tests for:
   - model invariants
   - trainer metrics
   - probe contract
   - generator/artifact path

5. Run:
   - `py_compile`
   - `pytest`
   - one smoke probe
   - one meaningful `weather` screen

6. Write back an audit record with:
   - exact files added
   - validation status
   - first scientific interpretation
   - position versus `V17` to `V19`

