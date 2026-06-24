# Stage III V20 Corrective Budgeted Dual-Stream Design

## Objective

Create a new official version line that follows the expert-local idea more
directly than `V19`:

- each expert first builds a shared representation;
- environment inference acts as a correction / suppression operator instead of
  directly defining the invariant representation;
- invariant and variant semantics stay expert-local;
- the model explicitly controls how much residual budget is allocated to a
  history-like branch versus a release branch.

## Core design

For expert `k` and time `t`:

- shared post-expert state: `s_{k,t}`
- causal environment posterior: `q(e_{k,t} | s_{k,<=t})`
- invariant base: `b_{k,t}`
- invariant adapter: `a_{k,t}`
- environment correction: `c_{k,t}`
- invariant representation:
  - `phi_i^{(k,t)} = b_{k,t} + a_{k,t} - c_{k,t}`

The drift / release path is not defined as a free subtraction target. Instead
it becomes a budgeted release increment:

- history proposal: `h_prop^{(k,t)}`
- release proposal: `r_prop^{(k,t)}`
- release confidence: `gamma_{k,t}`
- residual budgets:
  - `beta_hist^{(k,t)}`
  - `beta_rel^{(k,t)}`
  - with `beta_hist + beta_rel = 1`

Then:

- budgeted history state: `h^{(k,t)} = beta_hist * h_prop`
- budgeted release state: `r^{(k,t)} = beta_rel * gamma * r_prop`
- compatibility form:
  - `phi_v^{(k,t)} = phi_i^{(k,t)} + r^{(k,t)}`

The key point is that the native drift source is the budgeted release state,
not the abstract subtraction itself.

## Why V20 exists

`V19` showed that:

- moving to a history-proxy plus release-increment objective is directionally
  meaningful;
- but scalar reweighting does not fix the core issue;
- the next bottleneck is structural branch allocation.

Therefore `V20` changes the model structure itself:

- environment inference only explains / suppresses nuisance directions;
- invariant correction is an operator on a shared vector;
- history and release no longer freely compete without an explicit allocator.

## Planned code artifacts

- `models/factor_mechanism_v20.py`
- `train/offline_factor_v20.py`
- `train/stage3_generator_training_v20.py`
- `train/stage3_v20_artifacts.py`
- `train/online_loop_stage3_v20.py`
- `train/run_stage3_v20_corrective_budgeted_probe.py`
- `configs/stage3_v20_corrective_budgeted_probe.yaml`
- `tests/test_factor_mechanism_v20.py`
- `tests/test_offline_factor_v20.py`
- `tests/test_stage3_v20_corrective_budgeted.py`
- `tests/test_stage3_generator_training_v20.py`

