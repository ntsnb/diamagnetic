# Stage III V10 Teacher-Only Matrix Record

Date: 2026-05-29

This record extends `docs/audit/STAGE3_V10_CROSSFIT_TEACHER_RECORD.md` from a
single Panama seed42 canary to a public-3, three-seed teacher-only matrix. It
does not train or report a V10 generator. The purpose is to answer one question
before any generator expansion:

```text
Can the cross-fitted, zero-preserving V10 drift teacher itself pass the
pre-declared rolling validation gate across datasets and seeds?
```

## Implementation Added For This Matrix

- `train/run_stage3_v10_sensitivity_audit.py`
  - added `--teacher-only`;
  - after teacher validation/test diagnostics, the runner returns before
    generator training even if a teacher passes;
  - it writes `teacher_only_complete.json` for explicit teacher-only stops.
- `train/run_stage3_v10_teacher_audit_matrix.py`
  - non-overwriting batch runner;
  - creates a matrix manifest, per-run command logs, and summary CSV;
  - defaults now use short Windows-safe paths: `logs/s3v10m` and `ckpt/s3v10m`.
- `docs/audit/STAGE3_V10_TEACHER_MATRIX_SUMMARY.csv`
  - compact audited table of all nine effective runs.

Verification before running the matrix:

```powershell
C:\Users\admin\anaconda3\python.exe -m py_compile train\run_stage3_v10_sensitivity_audit.py train\run_stage3_v10_teacher_audit_matrix.py
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_stage3_v10_teacher_protocol.py tests\test_stage3_v9_teacher_protocol.py -q
```

Result:

```text
13 passed in 1.50s
```

## Matrix Commands

First dry-run:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v10_teacher_audit_matrix.py --dry-run --datasets panama tetouan spain_energy_weather --seeds 42 43 44 --epochs 5 --invariant-epochs 5 --generator-epochs 5 --max-generator-batches 480 --max-eval-batches 480 --validation-windows 5 --out-root logs\s3v10teacher_matrix_dryrun --checkpoint-root ckpt\s3v10teacher_matrix_dryrun
```

Main matrix:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v10_teacher_audit_matrix.py --datasets panama tetouan spain_energy_weather --seeds 42 43 44 --epochs 5 --invariant-epochs 5 --generator-epochs 5 --max-generator-batches 480 --max-eval-batches 480 --validation-windows 5 --out-root logs\s3v10teacher_matrix --checkpoint-root ckpt\s3v10teacher_matrix
```

The first Spain runs hit a Windows path-length failure while writing the long
`teacher_only_validation_diagnostic_summary.csv` filename. This was a filesystem
path issue, not a model result. Spain was rerun with short paths:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v10_teacher_audit_matrix.py --datasets spain_energy_weather --seeds 42 43 44 --epochs 5 --invariant-epochs 5 --generator-epochs 5 --max-generator-batches 480 --max-eval-batches 480 --validation-windows 5 --out-root logs\s3v10m --checkpoint-root ckpt\s3v10m --run-prefix v10m
```

Manifests:

- `logs/s3v10teacher_matrix/_manifests/v10teacher_matrix_20260529_112849_e5_ie5_b480_ew480`
- `logs/s3v10m/_manifests/v10m_20260529_121906_e5_ie5_b480_ew480`

## Result Summary

| Dataset | Seeds | Teacher Gate | Rolling Windows Passed | Mean Worst MSE Gain | Mean Worst MAE Gain | Mean Test MSE Gain | Mean Test MAE Gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Panama | 3 | `0/3` | `7/15` | `+0.007158` | `+0.000809` | `+0.026752` | `+0.015343` |
| Tetouan | 3 | `0/3` | `1/15` | `-0.144104` | `-0.096893` | `-0.026226` | `-0.058438` |
| Spain | 3 | `0/3` | `0/15` | `-0.025551` | `-0.018493` | `+0.001271` | `-0.001937` |
| All | 9 | `0/9` | `8/45` | `-0.054166` | `-0.038192` | `+0.000599` | `-0.015011` |

Per-run rolling validation:

| Dataset | Seed | Gate | Window Pass | Failure Reasons | Worst MSE | Worst MAE | Margin | Worst Sign |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |
| Panama | 42 | fail | `3/5` | sign, MAE | `+0.005937` | `-0.000710` | `+0.005937` | `0.529688` |
| Panama | 43 | fail | `2/5` | MSE, counterfactual, MAE | `-0.016564` | `+0.001770` | `-0.016564` | `0.554514` |
| Panama | 44 | fail | `2/5` | MAE, sign | `+0.032100` | `+0.001367` | `+0.032100` | `0.539236` |
| Tetouan | 42 | fail | `1/5` | MSE, MAE, counterfactual, sign | `-0.165450` | `-0.093404` | `-0.165450` | `0.510417` |
| Tetouan | 43 | fail | `0/5` | MAE, sign, MSE, counterfactual | `-0.176170` | `-0.127094` | `-0.176170` | `0.421354` |
| Tetouan | 44 | fail | `0/5` | MSE, MAE, counterfactual, sign | `-0.090691` | `-0.070182` | `-0.090691` | `0.515799` |
| Spain | 42 | fail | `0/5` | MSE, MAE, counterfactual, sign | `-0.031804` | `-0.029031` | `-0.031804` | `0.502257` |
| Spain | 43 | fail | `0/5` | MSE, MAE, counterfactual, sign | `-0.000310` | `-0.000539` | `-0.001878` | `0.497743` |
| Spain | 44 | fail | `0/5` | MSE, counterfactual, sign, MAE | `-0.044540` | `-0.025909` | `-0.044540` | `0.486458` |

No `generator_served_policy.pt` was found under the V10 matrix checkpoint roots.
This is expected and correct: the matrix was run in teacher-only mode, and the
teacher gate failed in every effective run.

## Interpretation

The single Panama seed42 V10 canary was genuinely better than V9 teacher-only,
but the broader matrix does not support a V10 performance claim:

1. Panama has a positive signal, but it is not stable enough. No seed reaches
   `5/5` rolling windows. The failure alternates between sign agreement, MAE,
   and counterfactual separation.
2. Tetouan is clearly negative under the same protocol. The teacher has
   nontrivial train R2 (`0.13` to `0.17`) but transfers poorly to rolling
   validation and test.
3. Spain is mostly inert or harmful. Seed43 is close to zero because shrinkage
   collapsed to `0.05`, but it still fails every validation window.
4. The V10 alignment gate passing is not sufficient evidence. All nine runs
   passed the internal alignment gate, yet all nine failed the teacher claim
   gate. Cleaner `phi_v - phi_i` geometry alone does not guarantee useful
   drift-conditioned correction.

Scientific conclusion:

```text
V10 is a strong auditability repair and a partial Panama mechanism signal.
It is not validated drift-conditioned online adaptation.
The generator must remain blocked until teacher-only rolling validation passes
prospectively across at least a single dataset/seed, and preferably across
multiple seeds before public-3 expansion.
```

## Next Repair Direction

The next model revision should not simply increase generator capacity. The
failure is upstream of generator training.

Recommended V11 direction:

1. Replace aggregate shrinkage calibration with worst-block chronological
   calibration, so a teacher cannot pass inner calibration by improving only
   easy early segments.
2. Add horizon-wise nonnegative MAE-aware shrinkage, but constrain it by
   worst-block MSE/MAE and sign agreement rather than per-output aggregate fit.
3. Add selective correction gates trained only on inner calibration features,
   such as correction norm, drift norm, calibration residual uncertainty, and
   local sign reliability. Zero-drift identity must remain exact.
4. Move beyond post-hoc `phi_v - phi_i` cleanup by training or calibrating the
   representation against residual-correction transfer, not only invariant
   leakage removal.
5. Continue to stop before generator training unless the teacher-only gate
   passes. If V11 only reduces harm by abstaining, report it as safety evidence,
   not as an accuracy contribution.
