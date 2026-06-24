# Stage III V21 Release-First Budgeted Plan

1. Add a `V21` expert-local head where release is computed before history.
2. Keep the invariant correction operator from `V20`.
3. Build a full isolated version line:
   - model
   - offline trainer
   - probe runner
   - generator wrapper
   - artifact loader
   - online wrapper
   - config
   - tests
4. Run deterministic validation.
5. Run smoke plus one `weather seed42 b240` formal slice.
6. Write an audit record positioning `V21` against `V17` to `V20`.

