# Stage III V23 Observable Drift Plan

1. Copy the `V22` version line into a dedicated `V23` namespace.
2. Preserve `phi_i / phi_v` geometric semantics.
3. Add a model-native `observable drift` channel for teacher and serving use.
4. Align generator/risk-gate/online wrappers to consume that same drift source.
5. Run deterministic tests and cross-version regression.
6. Run one smoke and two `weather b240` formal screens for `seed42` and
   `seed43`.
7. Write an evidence-first audit record positioning `V23` against `V17`-`V22`.
