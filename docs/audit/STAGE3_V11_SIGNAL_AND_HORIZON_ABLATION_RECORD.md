# Stage III V11 Signal And Horizon Ablation Record

Date: 2026-06-01

This record follows `docs/audit/STAGE3_V11_TEACHER_MATRIX_RECORD.md`. It tests
two narrow hypotheses without training a generator:

1. The V11 failure might be caused by using `signal_mode: delta_vector` while
   the aligner scores level-like post-canonical drift.
2. The V11 failure might hide useful horizon-specific corrections that are
   suppressed by a single block-scalar alpha.

No original logs, checkpoints, tables, or figures were overwritten.

## A1: `level_vector` Teacher-Only Matrix

Command:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v11_teacher_audit_matrix.py --datasets panama tetouan spain_energy_weather --seeds 42 43 44 --epochs 5 --invariant-epochs 5 --generator-epochs 5 --max-generator-batches 480 --max-eval-batches 480 --validation-windows 5 --signal-mode level_vector --out-root logs\s3v11lv --checkpoint-root ckpt\s3v11lv --run-prefix v11lv
```

The first Tetouan seed42 run exposed a numerical robustness issue in the V11
teacher solver:

```text
torch.linalg.solve: The solver failed because the input matrix is singular.
```

The V11 copy was patched to add finite-input checks, jittered linear-solve
fallback, and pseudoinverse fallback in `models/stage3_residual_teacher_v11.py`.
This does not change the teacher gate logic. Verification after the patch:

```text
py_compile models\stage3_residual_teacher_v11.py: passed
pytest tests\test_stage3_v11_teacher_protocol.py -q: 5 passed
git diff --check: passed
```

Tetouan seed42 was then rerun without overwriting the failed record:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v11_teacher_audit_matrix.py --datasets tetouan --seeds 42 --epochs 5 --invariant-epochs 5 --generator-epochs 5 --max-generator-batches 480 --max-eval-batches 480 --validation-windows 5 --signal-mode level_vector --out-root logs\s3v11lvfix --checkpoint-root ckpt\s3v11lvfix --run-prefix v11lvfix
```

Effective `level_vector` result:

| Signal Mode | Runs | Teacher Gates | Rolling Windows | Nonzero Alpha |
| --- | ---: | ---: | ---: | ---: |
| `delta_vector` | 9 | `0/9` | `0/45` | `0/9` |
| `level_vector` | 9 | `0/9` | `0/45` | `0/9` |

Compact CSV:

- `docs/audit/STAGE3_V11_SIGNAL_MODE_ABLATION_SUMMARY.csv`

Interpretation:

`level_vector` does not rescue the V11 teacher. The failure is not explained by
`delta_vector` temporal differencing alone. Several `level_vector` runs show a
small positive inner-calibration `best_gain`, but the worst chronological block
and sign gates still force `alpha=0`.

Representative `level_vector` inner-calibration findings:

| Dataset | Seed | Best Gain | Best Alpha | Worst MSE Gain | Worst MAE Gain | Worst Sign | Block Pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Panama | 43 | `0.005020` | `0.025` | `-0.000950` | `-0.000792` | `0.461806` | `3/5` |
| Panama | 44 | `0.014344` | `0.025` | `-0.001583` | `-0.001869` | `0.413194` | `2/5` |
| Tetouan | 42 | `0.014679` | `0.025` | `-0.000618` | `-0.000703` | `0.354167` | `4/5` |
| Spain | 43 | `0.001950` | `0.025` | `-0.001056` | `-0.000111` | `0.527778` | `4/5` |

## A2: `block_horizon` Teacher-Only Matrix

Config:

- `configs/stage3_v11_block_horizon_teacher_gate.yaml`

Command:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v11_teacher_audit_matrix.py --config configs\stage3_v11_block_horizon_teacher_gate.yaml --datasets panama tetouan spain_energy_weather --seeds 42 43 44 --epochs 5 --invariant-epochs 5 --generator-epochs 5 --max-generator-batches 480 --max-eval-batches 480 --validation-windows 5 --signal-mode delta_vector --out-root logs\s3v11bh --checkpoint-root ckpt\s3v11bh --run-prefix v11bh
```

Compact CSV:

- `docs/audit/STAGE3_V11_BLOCK_HORIZON_TEACHER_SUMMARY.csv`

Result:

| Variant | Runs | Teacher Gates | Rolling Windows | Nonzero Alpha | Mean Worst MSE Gain | Mean Worst MAE Gain | Mean Test MSE Gain | Mean Test MAE Gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `block_horizon` | 9 | `0/9` | `0/45` | `4/9` | `-0.000762` | `-0.000592` | `-0.000101` | `-0.000217` |

Per-run pattern:

- Panama seeds `42/43/44`: still `alpha=0`.
- Tetouan seeds `42/43`: nonzero mean alpha (`0.0104` and `0.0333`) but rolling validation is harmful; worst sign agreement is about `0.045` to `0.048`.
- Spain seeds `42/43`: nonzero mean alpha (`0.0417`) but rolling validation is harmful; worst sign agreement is about `0.024` to `0.028`.
- Tetouan seed44 and Spain seed44 still abstain with `alpha=0`.

No `generator*.pt` artifact was found under `ckpt/s3v11bh`.

## Scientific Conclusion

These ablations strengthen the negative conclusion:

```text
The V11 teacher failure is not fixed by using level-vector drift instead of
delta-vector drift, and horizon-wise nonzero corrections do not pass rolling
validation. Generator training remains scientifically blocked.
```

The evidence now distinguishes three facts:

1. `delta_vector` is not the sole cause of V11 abstention.
2. Some horizon-specific teacher corrections can become nonzero, but they are
   not stable or sign-correct enough to pass validation.
3. The correct next target is the residual teacher/alignment objective itself,
   not generator capacity.

## Next Safe Diagnostic

The next useful experiment should still be teacher-only:

1. export per-horizon residual sign and candidate-alpha details;
2. compare against simple residual baselines, such as previous residual,
   seasonal residual, and train residual mean;
3. evaluate whether the residual target is predictable from any causal train
   feature before attempting another generator.

Do not relax the teacher gate, and do not train a V11 generator from these
ablation results.
