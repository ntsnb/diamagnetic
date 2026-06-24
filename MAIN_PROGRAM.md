# Main Project Autoresearch Program

This project is the primary OOD time-series forecasting system with:
- offline invariant learning
- offline factor-mechanism forecasting
- online adaptation via Stage I / Stage II / Stage III

The goal is not just a lower offline validation error. The goal is:
- `baseline` online error should be measurable and reproducible
- `stage1`, `stage2`, `stage3`, and their combinations should be directly comparable
- the full combination `stage1+stage2+stage3` should outperform `baseline`

## Scope

Modify code in this repository as needed, but keep the evaluation protocol fixed:
- use the same dataset split
- use the same normalization
- compare online variants under the same `eval()` setting
- log every run to `main_results.tsv`

## Canonical variants

Every experiment should compare these variants:
- `baseline`
- `stage1`
- `stage2`
- `stage3`
- `stage1_stage2`
- `stage1_stage3`
- `stage2_stage3`
- `stage1_stage2_stage3`

## Primary metric

Primary metric:
- online `stage*_mse` versus `baseline`

Secondary metrics:
- offline `val_mse`
- offline `test_mse`
- online drift diagnostics
- stability trigger rate

## Experiment loop

1. Start from `configs/base.yaml`.
2. Run the canonical ablation suite.
3. Log all results in `main_results.tsv`.
4. Change only a small set of hyperparameters at once:
   - offline LR schedule
   - stage1 LR / decay
   - stage2 scale / decay / threshold / interval
   - stage3 scale / decay / clamp
   - stability gamma / window size
5. Keep changes only if they improve the online comparison, especially `stage1_stage2_stage3`.

## Simplicity rule

Prefer smaller changes that improve online performance consistently.
Do not keep complexity that only improves a single noisy run.
