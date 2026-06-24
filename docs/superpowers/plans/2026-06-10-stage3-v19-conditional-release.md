# Stage III V19 Conditional-Release Plan

## Objective

Add a new official `V19` line that strengthens the user's expert-local
invariant idea by explicitly supervising:

- history-like residual proxy
- release increment
- release-vs-history advantage

without overwriting `V16-V18`.

## Workstreams

1. New base model
   - add `history_output_heads`
   - keep `anchor / suppression / release / confidence`
   - expose `expert_y_hist` and `y_hist`

2. New offline loss
   - add history-proxy, release-increment, conditional-advantage,
     release-null, and counterfactual-shuffle losses

3. Version-safe config and wrappers
   - add `V19` config roots
   - keep old lines untouched
   - preserve downstream compatibility where possible

4. Deterministic validation
   - new targeted tests
   - regression against existing lines where relevant

5. Experiment entry
   - start with smoke or bounded validation after code path is stable
   - do not claim success before the teacher gate evidence exists

## Version boundary

New files must be version-safe and use the `v19` naming family:

- `models/factor_mechanism_v19.py`
- `train/offline_factor_v19.py`
- `train/stage3_generator_training_v19.py`
- `train/online_loop_stage3_v19.py`
- `train/stage3_v19_artifacts.py`
- `configs/stage3_v19_*.yaml`
- `tests/test_*v19*.py`

`V16-V18` remain authoritative historical baselines.
