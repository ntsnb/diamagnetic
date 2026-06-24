# Stage III V28 Support-Suppression Release Design

## Goal

Create a new official version line that is more faithful to the two Chinese
algorithm reports than `V27`, without overwriting older versions.

## Core Design Decision

`V28` keeps the `V26` expert-local correction backbone, but replaces the single
quality-conditioning mechanism of `V27` with two explicit conditionors:

- `suppression`: estimates the environment-explainable, harmful raw-change part
- `support`: estimates the delayed-change part that should be released

## Mechanism

For each expert:

1. keep the `V26` anchor and causal env filter
2. infer a suppressible raw-change component
3. infer a supported delayed-change component
4. form the served variant as:

`corrected_change = raw_change - suppression_score * suppression_vector + supported_delta`

This follows the audit conclusion that environment inference should act more as
a suppression/routing prior than as the direct definition of `phi_i`.

## Version Boundary

`V28` adds:

- `models/factor_mechanism_v28.py`
- `train/offline_factor_v28.py`
- `train/run_stage3_v28_support_suppression_release_probe.py`
- `train/stage3_generator_training_v28.py`
- `train/stage3_v28_artifacts.py`
- `train/online_loop_stage3_v28.py`
- `configs/stage3_v28_support_suppression_release_probe.yaml`
- `tests/test_*_v28.py`

All older lines remain untouched.
