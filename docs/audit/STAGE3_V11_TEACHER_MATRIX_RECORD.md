# Stage III V11 Teacher-Only Matrix Record

Date: 2026-06-01

This record extends the V10 public-3 teacher-only audit. V11 was created as an
opt-in copy and did not overwrite legacy Stage III, V9, or V10 code paths. The
question for this run was deliberately narrow:

```text
Can a block-calibrated, residual-transfer V11 teacher produce nonzero,
validation-stable drift-conditioned residual corrections before any generator
training is allowed?
```

## Implementation Added For This Matrix

- `models/stage3_residual_teacher_v11.py`
  - adds `TrainOnlyRobustBlockResidualTeacherV11`;
  - keeps exact zero-drift identity through a bias-free feature map;
  - calibrates shrinkage by chronological calibration blocks rather than only
    aggregate validation loss;
  - supports a reliability gate, but the public-3 matrix never reached a
    nonzero reliable teacher.
- `models/stage3_alignment_v11.py`
  - adds `TrainSetCrossFittedResidualTransferPLSAlignerV11`;
  - keeps V10 zero-preserving invariant-leakage projection;
  - replaces transition CCA with direct residual-transfer PLS selection.
- `train/run_stage3_v11_sensitivity_audit.py`
  - uses explicit V9/V10/V11 teacher dispatch;
  - writes `residual_teacher_v11_diagnostics.csv`;
  - uses `load_stage3_v11_artifact_bundle`;
  - treats `--teacher-only` as a hard guard before generator training.
- `train/run_stage3_v11_teacher_audit_matrix.py`
  - non-overwriting public-3, multi-seed matrix runner;
  - always calls the V11 runner with `--teacher-only`;
  - defaults to short Windows-safe paths: `logs/s3v11m` and `ckpt/s3v11m`.
- `configs/stage3_v11_block_calibrated_teacher_gate.yaml`
  - configures `residual_teacher.version: v11`;
  - uses `cross_fitted_residual_transfer_pls_v11`;
  - retains the same teacher claim gate thresholds used for V10.
- `tests/test_stage3_v11_teacher_protocol.py`
  - covers nonzero helpful teacher behavior, harmful-block abstention,
    train-only update rejection, checkpoint round-trip, V11 aligner
    serialization, and V11 artifact loading.

## Verification Before Matrix

```powershell
C:\Users\admin\anaconda3\python.exe -m py_compile models\stage3_residual_teacher_v11.py models\stage3_alignment_v11.py train\stage3_v11_artifacts.py train\run_stage3_v11_sensitivity_audit.py train\run_stage3_v11_teacher_audit_matrix.py tests\test_stage3_v11_teacher_protocol.py
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_stage3_v11_teacher_protocol.py tests\test_stage3_v10_teacher_protocol.py tests\test_stage3_v9_teacher_protocol.py -q
C:\Users\admin\anaconda3\python.exe -m pytest tests -q
git diff --check
```

Results:

```text
18 passed in 2.58s
145 passed in 5.49s
git diff --check: passed
```

The repository-level `pytest -q` was also attempted, but it is not a meaningful
project check because the repository previously contained external `CLI-Anything/` and
LibreOffice `core/` test trees at the root; they are now archived under `archive/external_sources/` and still require separate packages/UNO runtime and
one inaccessible temporary directory. The project-local suite under `tests/`
passed.

## Smoke Run

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v11_sensitivity_audit.py --config configs\stage3_v11_block_calibrated_teacher_gate.yaml --dataset panama --run-id v11_smoke_s42_b16_20260601_01 --seed 42 --device auto --modes real zero shuffled sign_flipped amplified_2x amplified_5x random_matched_norm --validation-windows 2 --drift-source delta_phi_v --drift-polarity 1.0 --signal-mode delta_vector --normalize-vector --epochs 1 --invariant-epochs 1 --generator-epochs 1 --max-generator-batches 16 --max-eval-batches 16 --teacher-only --out-root logs\s3v11_smoke --checkpoint-root ckpt\s3v11_smoke
```

Smoke output:

- `logs/s3v11_smoke/v11_smoke_s42_b16_20260601_01/panama/seed_42`
- `ckpt/s3v11_smoke/panama/v11_smoke_s42_b16_20260601_01/seed_42`

The smoke completed and wrote only base/aligned/canonicalizer/V11-teacher
artifacts. No generator artifact was created. Because the smoke used only `16`
teacher samples, the teacher abstained with `teacher_alpha_source =
fallback_block_gate`, which confirmed the safety guard rather than performance.

## Matrix Command

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v11_teacher_audit_matrix.py --datasets panama tetouan spain_energy_weather --seeds 42 43 44 --epochs 5 --invariant-epochs 5 --generator-epochs 5 --max-generator-batches 480 --max-eval-batches 480 --validation-windows 5 --out-root logs\s3v11m --checkpoint-root ckpt\s3v11m --run-prefix v11tm
```

Manifest:

- `logs/s3v11m/_manifests/v11tm_20260601_100708_e5_ie5_b480_ew480`

Compact summary:

- `docs/audit/STAGE3_V11_TEACHER_MATRIX_SUMMARY.csv`

## Result Summary

| Dataset | Seeds | Teacher Gate | Rolling Windows Passed | Mean Worst MSE Gain | Mean Worst MAE Gain | Mean Test MSE Gain | Mean Test MAE Gain | Mean Train R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Panama | 3 | `0/3` | `0/15` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `-0.012011` |
| Tetouan | 3 | `0/3` | `0/15` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `-0.027989` |
| Spain | 3 | `0/3` | `0/15` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `-0.008380` |
| All | 9 | `0/9` | `0/45` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `-0.016127` |

Per-run summary:

| Dataset | Seed | Gate | Window Pass | Alpha Source | Alpha | Train R2 | Alignment Gate | Active Rank | Leakage Rank |
| --- | ---: | --- | ---: | --- | ---: | ---: | --- | ---: | ---: |
| Panama | 42 | fail | `0/5` | `fallback_block_gate` | `0.0` | `-0.006421` | pass | 4 | 3 |
| Panama | 43 | fail | `0/5` | `fallback_block_gate` | `0.0` | `-0.023882` | pass | 4 | 1 |
| Panama | 44 | fail | `0/5` | `fallback_block_gate` | `0.0` | `-0.005729` | pass | 5 | 2 |
| Tetouan | 42 | fail | `0/5` | `fallback_block_gate` | `0.0` | `-0.029592` | pass | 3 | 12 |
| Tetouan | 43 | fail | `0/5` | `fallback_block_gate` | `0.0` | `-0.026911` | pass | 3 | 16 |
| Tetouan | 44 | fail | `0/5` | `fallback_block_gate` | `0.0` | `-0.027465` | pass | 3 | 10 |
| Spain | 42 | fail | `0/5` | `fallback_block_gate` | `0.0` | `-0.016275` | pass | 4 | 9 |
| Spain | 43 | fail | `0/5` | `fallback_block_gate` | `0.0` | `-0.003163` | pass | 3 | 16 |
| Spain | 44 | fail | `0/5` | `fallback_block_gate` | `0.0` | `-0.005702` | pass | 2 | 13 |

No `generator*.pt` artifact was found under `ckpt/s3v11m`. This is expected:
every run wrote `teacher_only_abort.json` before generator training.

## Interpretation

V11 is not a performance repair. It is a safety-abstention repair.

The key scientific signal is the contradiction between the alignment gate and
the teacher gate:

```text
alignment_gate_pass = True in all 9 runs
teacher_gate_pass = False in all 9 runs
teacher_alpha_source = fallback_block_gate in all 9 runs
teacher_shrinkage_alpha = 0.0 in all 9 runs
```

This means that a cleaner internal drift subspace is not enough to establish a
useful drift-conditioned residual correction. The final served feature can pass
an alignment diagnostic while still failing the actual forecasting correction
test.

The negative `teacher_train_r2` values should not be interpreted as a nonzero
harmful served correction. Because V11 abstained with `alpha=0`, the served
correction is exactly zero. The R2 diagnostic is computed against the train
residual mean baseline; if base residuals have nonzero mean, an abstaining
zero-correction teacher can have negative R2. The decision-relevant fields are
therefore `teacher_alpha_source`, `teacher_shrinkage_alpha`, rolling gate
status, and test/rolling gain fields.

Scientific conclusion:

```text
V11 confirms that the current Stage III teacher path should abstain under the
pre-declared public-3 protocol. It does not validate concept-drift learning,
online adaptation, or a deployable Stage III generator. Generator training
must remain blocked until a nonzero teacher first passes prospective rolling
teacher-only validation.
```

## Next Repair Direction

The next useful step is not larger generator training. The evidence localizes
the failure upstream of generator capacity.

Recommended next protocol:

1. Keep `--teacher-only` as a hard guard.
2. Instrument the V11 teacher calibration with per-block candidate-alpha rows:
   block MSE gain, block MAE gain, sign agreement, and chosen alpha before
   fallback.
3. Add horizon-wise diagnostics to determine whether every horizon is harmful
   or whether useful short-horizon correction is being suppressed by long
   horizons.
4. Compare residual targets against a simple chronological residual baseline:
   previous residual, seasonal residual, and train-window residual mean.
5. Only if a nonzero teacher reaches `5/5` rolling windows on a prospective
   canary should a generator canary be allowed.

Manuscript constraint:

```text
Do not write V11 as a successful Stage III contribution. If mentioned at all,
describe it as an audit result showing that strict teacher gating prevents an
unsupported drift correction from entering the served model.
```
