# Stage III V7 Residualized RRCCA Alignment Record

Date: 2026-05-27

## 1. Purpose

V7 was added to test a stronger answer to the `phi_v - phi_i` alignment
problem. V6 used train-only whitening/CORAL plus Procrustes alignment before
temporal residual CCA. The V7 hypothesis is different:

1. keep the residual meaning of `phi_v - phi_i`;
2. remove the part of `phi_v - phi_i` that is linearly explainable by
   `phi_i` on the train split;
3. fit CCA between residual-drift transitions and forecast-residual
   transitions on an inner chronological train-fit split;
4. keep only active CCA dimensions that remain stable on an inner
   chronological train-calibration split;
5. validation/test only use the frozen transform.

This is a model repair hypothesis, not a manuscript claim.

## 2. Code Changes

New and modified files:

- `models/stage3_alignment_v7.py`
  - adds `TrainSetChronologicalResidualizedReducedRankCCAAligner`.
  - supports `fit`, `transform_phi_pair`, `transform_delta`, `state_dict`,
    `load_state_dict`, and `diagnostic_summary`.
- `train/run_stage3_v2_sensitivity_audit.py`
  - opt-in support for `canonicalizer_method: residualized_rrcca_v7`.
  - adds V7 diagnostic summary fields to `alignment_diagnostics.csv` without
    changing V5/V6 defaults.
- `train/stage3_v2_artifacts.py`
  - loads V7 canonicalizer states by class name.
- `configs/stage3_v7_residualized_rrcca_served_policy_gate.yaml`
  - V7 served-policy config with the same strict claim gate as V5/V6.
- `tests/test_stage3_alignment_v7.py`
  - unit tests for residualization, active-rank truncation, fail-closed
    behavior, and serialization.
- Existing protocol/artifact tests were extended for V7 dispatch and loading.

Verification before the V7 canary:

- `C:\Users\admin\anaconda3\python.exe -m pytest tests -q`
- Result: `120 passed in 4.76s`

## 3. V7 Canary Run

Command:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v2_sensitivity_audit.py --config configs\stage3_v7_residualized_rrcca_served_policy_gate.yaml --dataset panama --run-id v7ps42g5b480r1 --seed 42 --device auto --modes real zero shuffled sign_flipped amplified_2x amplified_5x random_matched_norm --validation-windows 5 --drift-source delta_phi_v --drift-polarity 1.0 --signal-mode delta_vector --normalize-vector --epochs 5 --invariant-epochs 5 --generator-epochs 5 --max-generator-batches 480 --max-eval-batches 480 --out-root logs\s3v7p --checkpoint-root ckpt\s3v7p
```

Output:

- logs: `logs/s3v7p/v7ps42g5b480r1/panama/seed_42`
- checkpoints: `ckpt/s3v7p/panama/v7ps42g5b480r1/seed_42`
- final served artifact: not saved
- reason: `served-policy generator selection abstained (no_candidate_passed_claim_gate)`

## 4. Served-policy Selection Evidence

| Epoch | Pass | Window pass rate | Failure reasons | Worst MSE gain | Worst MAE gain | Worst real-vs-CF margin | Max reject |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |
| 0 | `False` | `0.0` | `window_pass_rate;real_mse_gain;real_mae_gain;counterfactual` | `-0.002010` | `-0.001224` | `-0.002010` | `0.145833` |
| 1 | `False` | `0.0` | `window_pass_rate;real_mse_gain;real_mae_gain;counterfactual` | `-0.000990` | `-0.001009` | `-0.000990` | `0.129167` |
| 2 | `False` | `0.0` | `window_pass_rate;real_mse_gain;real_mae_gain;counterfactual` | `-0.003985` | `-0.002308` | `-0.003985` | `0.070833` |
| 3 | `False` | `0.0` | `window_pass_rate;real_mse_gain;real_mae_gain;counterfactual` | `-0.000482` | `-0.000979` | `-0.000482` | `0.083333` |
| 4 | `False` | `0.0` | `window_pass_rate;real_mse_gain;real_mae_gain;counterfactual` | `0.000490` | `0.000534` | `0.000267` | `0.045833` |

Best candidate by score was epoch `4`, but it still passed `0/5` rolling
validation windows. It is far below the pre-declared gates:

- MSE gain required: `0.005`; observed worst-window best: `0.000490`.
- MAE gain required: `0.002`; observed worst-window best: `0.000534`.
- real-vs-counterfactual margin required: `0.002`; observed best:
  `0.000267`.
- rolling pass required: `5/5`; observed best: `0/5`.

## 5. Alignment Diagnostics

V7 alignment was active and passed its internal alignment gate:

- `alignment_method`: `residualized_rrcca_v7`
- `alignment_gate_pass`: `True`
- train-only inner fit/calibration: `360 / 120`
- active rank: `8`
- inactive dimensions zeroed: `56`
- active dim fraction: `0.125`
- residualizer R2 mean: `0.931512`
- residualizer R2 max: `0.970671`
- calibration top correlation: `0.312118`
- calibration mean active correlation: `0.209289`
- calibration transition R2: `0.097418`
- bootstrap sign stability min: `0.703125`
- bootstrap sign stability mean: `0.931641`
- projection condition number: `1.242098`

However, the post-canonical drift became lower-rank and less correlated with
residual transitions on held-out splits:

| Split | Top corr pre | Top corr post | Mean-best corr pre | Mean-best corr post | Norm corr pre | Norm corr post | Collapse fraction post |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | `0.438055` | `0.243765` | `0.345399` | `0.189180` | `0.675577` | `0.471651` | `0.875` |
| validation | `0.280047` | `0.270101` | `0.242872` | `0.198292` | `0.337630` | `0.255140` | `0.875` |
| test | `0.344814` | `0.230035` | `0.261135` | `0.148047` | `0.469346` | `0.217500` | `0.875` |

Interpretation:

- V7 successfully enforced a reduced active subspace.
- The high residualizer R2 means much of `delta_phi_v` was explainable by
  `phi_i`, confirming invariant leakage risk.
- But after removing that leakage and truncating unstable dimensions, the
  remaining signal was too weak to support served-policy adaptation.

## 6. Comparison With V5/V6 Canaries

| Run | Best rolling pass | Best worst MSE gain | Best worst MAE gain | Best margin | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| V5 g5_b480 | `4/5` | `0.004620` | `0.004778` | `0.004575` | near miss, abstained |
| V5 g8_b960 | `3/5` | `0.003612` | `-0.002499` | `0.003612` | abstained |
| V6 g5_b480 | `1/5` | `-0.000633` | `0.000473` | `-0.000775` | abstained |
| V7 g5_b480 | `0/5` | `0.000490` | `0.000534` | `0.000267` | abstained |

V7 is methodologically cleaner for the alignment question, but it is not a
performance improvement in this canary.

## 7. Scientific Conclusion

Supported:

- The V7 code path is opt-in, tested, train-only, and auditable.
- `phi_v - phi_i` contains substantial invariant-explainable structure in this
  run; this validates the concern that raw subtraction is not a clean drift
  signal.
- Reduced-rank residualized alignment can identify a stable train-calibrated
  subspace.

Not supported:

- Stage III V7 improves Panama seed42 forecasting under the strict served-policy
  gate.
- V7 is better than V5 or V6 for final predictive performance.
- Stage III can be promoted as a verified concept-drift learning contribution.

Manuscript implication:

- Do not update the paper to claim Stage III validity from V7.
- If Stage III remains in the manuscript, it should be framed as a rigorous
  diagnostic pathway showing the difficulty of converting representation shift
  into reliable online parameter adaptation.

## 8. 2026-05-27/28 Non-downgraded Diagnostic Rerun

Purpose:

- The first V7 canary abstained before normal `step_metrics.csv` generation.
- The runner was updated to preserve candidate/window/mode evidence before the
  strict abstain error is raised.
- This change does not relax the claim gate and does not save a failed served
  artifact.

Additional code/test verification:

- `train/run_stage3_v2_sensitivity_audit.py`
  - Added `served_policy_candidate_window_breakdown.csv`.
  - Added `served_policy_selected_candidate_window_breakdown.csv`.
  - These files are written before `_raise_if_final_generator_selection_abstained`.
- `tests/test_stage3_v2_protocol_safety.py`
  - Added coverage for candidate-window diagnostics and selected-candidate
    artifact preservation.
- `C:\Users\admin\anaconda3\python.exe -m pytest tests -q`
  - Result before rerun: `121 passed in 3.99s`.
- `C:\Users\admin\anaconda3\python.exe -m py_compile models\stage3_alignment_v7.py train\run_stage3_v2_sensitivity_audit.py train\stage3_v2_artifacts.py`
  - Result: pass.

Command:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v2_sensitivity_audit.py --config configs\stage3_v7_residualized_rrcca_served_policy_gate.yaml --dataset panama --run-id v7ps42g5b480diag1 --seed 42 --device auto --modes real zero shuffled sign_flipped amplified_2x amplified_5x random_matched_norm --validation-windows 5 --drift-source delta_phi_v --drift-polarity 1.0 --signal-mode delta_vector --normalize-vector --epochs 5 --invariant-epochs 5 --generator-epochs 5 --max-generator-batches 480 --max-eval-batches 480 --out-root logs\s3v7pdiag --checkpoint-root ckpt\s3v7pdiag
```

Output:

- logs: `logs/s3v7pdiag/v7ps42g5b480diag1/panama/seed_42`
- checkpoints: `ckpt/s3v7pdiag/panama/v7ps42g5b480diag1/seed_42`
- final served artifact: not saved
- reason: `served-policy generator selection abstained (no_candidate_passed_claim_gate)`
- new all-candidate diagnostic file:
  `served_policy_candidate_window_breakdown.csv`
- new selected-candidate diagnostic file:
  `served_policy_selected_candidate_window_breakdown.csv`

All-candidate diagnostic coverage:

- rows: `125`
- candidates: epochs `0, 1, 2, 3, 4`
- windows per candidate: `5`
- selection modes in served-policy validation:
  `real`, `zero`, `sign_flipped`, `amplified_2x`, `amplified_5x`
- offline diagnostic modes such as `shuffled` and `random_matched_norm` remained
  excluded from served-policy selection.

Selected candidate: epoch `4`.

| Window | Gate pass | Failure reasons | Real MSE gain | Real MAE gain | Best CF | Best CF MSE gain | Margin | Real reject |
| ---: | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| 0 | `False` | `real_mse_gain;real_mae_gain;counterfactual` | `0.000561` | `0.000534` | `amplified_2x` | `0.000005` | `0.000556` | `0.033333` |
| 1 | `False` | `real_mse_gain;real_mae_gain` | `0.002325` | `0.001505` | `zero` | `0.000000` | `0.002325` | `0.031250` |
| 2 | `False` | `real_mse_gain;real_mae_gain;counterfactual` | `0.001524` | `0.001651` | `amplified_2x` | `0.000624` | `0.000900` | `0.037500` |
| 3 | `False` | `real_mse_gain;real_mae_gain;counterfactual` | `0.000490` | `0.000634` | `amplified_2x` | `0.000224` | `0.000267` | `0.037500` |
| 4 | `False` | `real_mse_gain;real_mae_gain;counterfactual` | `0.001710` | `0.000719` | `amplified_2x` | `0.000026` | `0.001684` | `0.045833` |

Pre-declared claim gate:

- `min_real_mse_gain`: `0.005`
- `min_real_mae_gain`: `0.002`
- `counterfactual_margin`: `0.002`
- `max_real_reject_rate`: `0.2`
- `rolling_min_window_pass_rate`: `1.0`

Diagnostic interpretation:

- The best V7 served candidate uses real drift frequently
  (`online_stage3_use_rate` about `0.954` to `0.969`) and keeps reject rate
  below the configured maximum.
- `sign_flipped` consistently hurts, with MSE gains around `-0.0052` to
  `-0.0073`; this means the repaired path is not completely sign-insensitive.
- `zero` and amplified modes are mostly rejected by the risk gate, which is
  safer than the legacy behavior.
- However, real-drift gains are far below the manuscript-grade threshold.
  The best real MSE gain across selected-candidate windows is only `0.002325`,
  still less than the required `0.005`; the best MAE gain is `0.001651`, still
  less than the required `0.002`.
- Therefore the failure is not just a missing diagnostic artifact. The current
  V7 path remains scientifically insufficient for a Stage III performance or
  concept-drift-learning claim.
