# Stage III Four-Way Ablation: Expanded Evidence Log

Date: 2026-05-24

## Objective

Audit whether the proposed temporal concept-drift definition improves Stage III online adaptation, without overwriting the original model or training code.

Four controlled Stage III signal definitions were evaluated:

| Mode | Meaning | Signal shape before generator |
| --- | --- | --- |
| `level_scalar` | Original current-time `phi_v(t) - phi_i(t)` compressed to scalar and repeated | 16 |
| `delta_scalar` | Temporal change: `[phi_v(t)-phi_i(t)] - [phi_v(t-1)-phi_i(t-1)]`, compressed to scalar and repeated | 16 |
| `level_vector` | Current-time vector drift signal | 64, train-PCA projected to 16 |
| `delta_vector` | Temporal vector drift signal | 64, train-PCA projected to 16 |

The vector modes preserve vector information until a train-only PCA projection is applied to match the existing Stage III generator bottleneck. This avoids changing the original generator interface while preventing scalar collapse.

## Files Added Or Modified

- `train/stage3_signals.py`: isolated Stage III signal definitions.
- `train/online_loop_stage3_ablation.py`: copy of the online loop for ablation only; original `train/online_loop.py` was not modified.
- `train/run_stage3_ablation_benchmark.py`: paired offline-once / replay-many benchmark harness.
- `tests/test_stage3_signals.py`: signal-level regression tests.
- `tests/test_stage3_ablation_benchmark.py`: result-row, CSV append, and short trace-path tests.
- `docs/superpowers/plans/2026-05-24-stage3-four-way-ablation.md`: execution plan.

## Verification

Command:

```powershell
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_stage3_signals.py tests\test_stage3_ablation_benchmark.py tests\test_shapes.py -q
```

Result:

```text
9 passed in 1.88s
```

## Expanded Runs

### Run A: Formal-Config-Aligned Run

Run id: `s3abl_pub3_s42_e5_inv1_v1`

Scope:

- Datasets: `panama`, `tetouan`, `spain_energy_weather`
- Seed: `42`
- Offline training epochs: `5`
- Invariant epochs: `1` (matches `configs/best_load_online.yaml`)
- Variant: `stage3`
- Tuning grid: full public grid, 27 settings per signal mode
- Tuning rows: `324`
- Test rows: `12`

Artifacts:

- `logs/load_benchmarks/stage3_ablation/s3abl_pub3_s42_e5_inv1_v1_tuning.csv`
- `logs/load_benchmarks/stage3_ablation/s3abl_pub3_s42_e5_inv1_v1_test.csv`
- `logs/load_benchmarks/stage3_ablation/s3abl_pub3_s42_e5_inv1_v1_summary.json`

Test results:

| Dataset | Signal | Selected val MSE/MAE | Test MSE | Test MAE | Base MSE | Base MAE | Gain MSE | Gain MAE | Selected `(lr,max_delta,blend)` |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| panama | `level_scalar` | 0.130755 / 0.251774 | 0.180288 | 0.296508 | 0.183736 | 0.300767 | 0.003447 | 0.004260 | 0.0008, 0.015, 0.34 |
| panama | `delta_scalar` | 0.130755 / 0.251774 | 0.180288 | 0.296508 | 0.183736 | 0.300767 | 0.003447 | 0.004260 | 0.0008, 0.015, 0.34 |
| panama | `level_vector` | 0.130754 / 0.251774 | 0.180288 | 0.296507 | 0.183736 | 0.300767 | 0.003447 | 0.004260 | 0.0008, 0.015, 0.34 |
| panama | `delta_vector` | 0.130755 / 0.251774 | 0.180288 | 0.296508 | 0.183736 | 0.300767 | 0.003447 | 0.004260 | 0.0008, 0.015, 0.34 |
| tetouan | `level_scalar` | 0.108071 / 0.224543 | 0.083327 | 0.193289 | 0.083851 | 0.194866 | 0.000523 | 0.001577 | 0.0003, 0.015, 0.34 |
| tetouan | `delta_scalar` | 0.108071 / 0.224543 | 0.083327 | 0.193289 | 0.083851 | 0.194866 | 0.000523 | 0.001577 | 0.0003, 0.015, 0.34 |
| tetouan | `level_vector` | 0.108071 / 0.224544 | 0.083327 | 0.193290 | 0.083851 | 0.194866 | 0.000523 | 0.001577 | 0.0003, 0.015, 0.34 |
| tetouan | `delta_vector` | 0.108071 / 0.224543 | 0.083327 | 0.193289 | 0.083851 | 0.194866 | 0.000523 | 0.001577 | 0.0003, 0.015, 0.34 |
| spain_energy_weather | `level_scalar` | 0.208720 / 0.329683 | 0.205232 | 0.324566 | 0.210550 | 0.330485 | 0.005318 | 0.005919 | 0.0003, 0.015, 0.34 |
| spain_energy_weather | `delta_scalar` | 0.208720 / 0.329683 | 0.205232 | 0.324566 | 0.210550 | 0.330485 | 0.005318 | 0.005919 | 0.0003, 0.015, 0.34 |
| spain_energy_weather | `level_vector` | 0.208720 / 0.329683 | 0.205232 | 0.324566 | 0.210550 | 0.330485 | 0.005318 | 0.005919 | 0.0003, 0.015, 0.34 |
| spain_energy_weather | `delta_vector` | 0.208720 / 0.329683 | 0.205232 | 0.324566 | 0.210550 | 0.330485 | 0.005318 | 0.005919 | 0.0003, 0.015, 0.34 |

Trace-level maximum differences versus `level_scalar`:

| Dataset | `delta_scalar` max pred diff | `level_vector` max pred diff | `delta_vector` max pred diff |
| --- | ---: | ---: | ---: |
| panama | 1.46e-6 | 1.00e-5 | 1.94e-6 |
| tetouan | 1.79e-7 | 4.95e-6 | 2.62e-6 |
| spain_energy_weather | 2.38e-7 | 2.74e-6 | 5.96e-7 |

### Run B: Stronger Invariant Training Stress Run

Run id: `s3abl_pub3_s42_e5_v2`

Scope:

- Same as Run A, except invariant epochs were increased to `5`.
- Tuning rows: `324`
- Test rows: `12`

Artifacts:

- `logs/load_benchmarks/stage3_ablation/s3abl_pub3_s42_e5_v2_tuning.csv`
- `logs/load_benchmarks/stage3_ablation/s3abl_pub3_s42_e5_v2_test.csv`
- `logs/load_benchmarks/stage3_ablation/s3abl_pub3_s42_e5_v2_summary.json`

Test results:

| Dataset | Signal | Test MSE | Test MAE | Base MSE | Base MAE | Gain MSE | Gain MAE | Selected `(lr,max_delta,blend)` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| panama | all four modes | 0.180838 | 0.298057 | 0.184265 | 0.301879 | 0.003427 | 0.003822 | 0.0003, 0.015, 0.34 |
| tetouan | all four modes | 0.084506 | 0.187505 | 0.085752 | 0.190184 | 0.001245 | 0.002679 | 0.0003, 0.015, 0.34 |
| spain_energy_weather | all four modes | 0.206604 | 0.333366 | 0.211027 | 0.337768 | 0.004423 | 0.004403 | 0.0003, 0.015, 0.34 |

Trace-level maximum differences versus `level_scalar`:

| Dataset | `delta_scalar` max pred diff | `level_vector` max pred diff | `delta_vector` max pred diff |
| --- | ---: | ---: | ---: |
| panama | 1.79e-7 | 5.25e-6 | 5.51e-7 |
| tetouan | 2.38e-7 | 3.04e-6 | 2.44e-6 |
| spain_energy_weather | 2.38e-7 | 3.67e-6 | 3.58e-7 |

## Evidence-Based Interpretation

1. The proposed temporal-delta concept-drift definition is conceptually more defensible than the current-level definition for isolating short-term fluctuations.
2. However, the expanded experiments do not show empirical improvement from `delta_scalar` or `delta_vector`.
3. Scalar and vector signal modes produce nearly identical predictions. The largest trace-level prediction difference is about `1e-5`, far below any meaningful forecasting-performance change.
4. Stage III gives a small gain over the frozen online base on these runs, but that gain is not attributable to the choice of drift signal.
5. The current evidence suggests that the Stage III generator is effectively insensitive to the drift input. The observed improvement is more likely driven by generic prediction blending / bounded shift behavior than by a discriminative concept-drift representation.

## Paper Implication

Do not update the manuscript to claim that the temporal-delta drift definition improves results. Current evidence does not support that claim.

A defensible statement, if the paper needs to discuss this audit, would be:

> We found that Stage III adaptation yields small online gains under the current benchmark setting, but the present generator is not sufficiently sensitive to alternative drift-signal definitions. This motivates a future revision of the generator objective or conditioning pathway before making claims about vector or temporal-delta drift modeling.

## Highest-Risk Finding

The highest-risk technical finding is not the scalar-vs-vector implementation detail by itself. The stronger risk is that changing the drift signal barely changes the prediction trajectory. This weakens any claim that Stage III's improvement is caused by a specifically designed concept-drift signal.

## Minimum Safe Next Step

Before changing the manuscript, run a direct generator-sensitivity audit:

1. Freeze the offline model and invariant encoders.
2. Feed controlled drift signals: zero, real, shuffled, amplified, and sign-flipped.
3. Log generated LRP vector norms, prediction-shift norms, and final MSE/MAE.
4. If zero or shuffled drift performs the same as real drift, revise the generator architecture or training objective before rerunning the paper experiments.

