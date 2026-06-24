# Stage III V12 Outcome-Aligned Transport Record

Date: 2026-06-01

This record follows the V11 negative teacher-only matrix and the V11
`level_vector` / `block_horizon` ablations. V12 is implemented as a new opt-in
path. It does not overwrite V7/V10/V11 files, logs, checkpoints, tables, or
figures.

## Purpose

V11 showed that passing an alignment diagnostic does not imply a useful
forecast residual correction. V12 therefore targets the upstream
`phi_v - phi_i` alignment problem more directly:

```text
phi_i and phi_v are first transported into a train-only shared latent anchor;
variant-invariant delta is cleaned in that anchored tangent space;
active dimensions are retained only when chronological residual-transfer
calibration supports them.
```

## Implemented Opt-In Files

- `models/stage3_alignment_v12.py`
  - `TrainOnlyOutcomeAlignedTransportAlignerV12`
  - train-only shared latent anchor
  - orthogonal invariant leakage removal
  - chronological residual-transfer active-dimension calibration
  - zero-drift identity
  - fail-closed `active_rank=0` behavior
- `models/stage3_residual_teacher_v12.py`
  - `TrainOnlyOutcomeAlignedBlockResidualTeacherV12`
  - V12 protocol wrapper around the strict V11 block/reliability teacher gate
  - V12-specific checkpoint save/load metadata
- `train/stage3_v12_artifacts.py`
  - V12 artifact bundle loader for V12 aligner and V12 teacher.
- `train/run_stage3_v12_sensitivity_audit.py`
  - isolated V12 runner copied from V11 and wired to V12 alignment/teacher
    dispatch.
- `train/run_stage3_v12_teacher_audit_matrix.py`
  - non-overwriting V12 matrix runner with `--teacher-only`.
- `configs/stage3_v12_outcome_aligned_teacher_gate.yaml`
  - V12 opt-in config using `canonicalizer_method:
    outcome_aligned_transport_v12` and `residual_teacher.version: v12`.
- `tests/test_stage3_v12_alignment_protocol.py`
- `tests/test_stage3_v12_teacher_protocol.py`
- `tests/test_stage3_v12_artifacts.py`

## Verification

Code-level verification:

```text
py_compile V12 modules and runners: passed
pytest tests/test_stage3_v12_alignment_protocol.py \
       tests/test_stage3_v12_teacher_protocol.py \
       tests/test_stage3_v12_artifacts.py -q: 6 passed
git diff --check: passed
```

Smoke runner:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v12_sensitivity_audit.py --config configs\stage3_v12_outcome_aligned_teacher_gate.yaml --dataset panama --run-id smoke_v12_panama_s42_20260601_03 --seed 42 --device cpu --epochs 1 --invariant-epochs 1 --generator-epochs 1 --max-train-batches 2 --max-generator-batches 8 --max-eval-batches 8 --validation-windows 2 --teacher-only --out-root logs\s3v12_smoke --checkpoint-root ckpt\s3v12_smoke --modes real zero sign_flipped amplified_2x
```

Smoke result:

- Output: `logs/s3v12_smoke/smoke_v12_panama_s42_20260601_03/panama/seed_42`.
- V12 aligner method: `outcome_aligned_transport_v12`.
- V12 aligner fail-closed: `alignment_gate_pass=False`, `active_rank=0`.
- V12 teacher: `teacher_method=v12_outcome_aligned_block_residual_teacher`.
- Teacher alpha: `0.0`, `teacher_alpha_source=fallback_block_gate`.
- Rolling teacher-only gate: `0/2`.
- `teacher_only_abort.json` was written.
- No generator artifact was produced.

Smoke matrix runner:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v12_teacher_audit_matrix.py --config configs\stage3_v12_outcome_aligned_teacher_gate.yaml --datasets panama --seeds 42 --epochs 1 --invariant-epochs 1 --generator-epochs 1 --max-train-batches 2 --max-generator-batches 8 --max-eval-batches 8 --validation-windows 2 --device cpu --out-root logs\s3v12_smoke_matrix2 --checkpoint-root ckpt\s3v12_smoke_matrix2 --run-prefix v12sm2
```

Smoke matrix result:

- Manifest:
  `logs/s3v12_smoke_matrix2/_manifests/v12sm2_20260601_134230_e1_ie1_b8_ew8`.
- Runs: `1/1` completed.
- Teacher gate: `0/1`.
- Rolling windows: `0/2`.
- Teacher alpha: `0.0`, `fallback_block_gate`.
- Alignment gate: `False`, `active_rank=0`.
- No `generator*.pt` artifact exists under `ckpt/s3v12_smoke_matrix2`.

## Public-3 Teacher-Only Matrix Result

The V12 public-3 teacher-only matrix has now been executed. This is the first
larger-scale V12 result and supersedes the earlier smoke-only status for
scientific interpretation.

Command:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v12_teacher_audit_matrix.py --config configs\stage3_v12_outcome_aligned_teacher_gate.yaml --datasets panama tetouan spain_energy_weather --seeds 42 43 44 --epochs 5 --invariant-epochs 5 --generator-epochs 5 --max-generator-batches 480 --max-eval-batches 480 --validation-windows 5 --device auto --out-root logs\s3v12m --checkpoint-root ckpt\s3v12m --run-prefix v12tm
```

Primary evidence:

- Manifest:
  `logs/s3v12m/_manifests/v12tm_20260601_134623_e5_ie5_b480_ew480/matrix_manifest.json`.
- Matrix summary:
  `logs/s3v12m/_manifests/v12tm_20260601_134623_e5_ie5_b480_ew480/matrix_summary.csv`.
- Per-run teacher diagnostics:
  `logs/s3v12m/v12tm_20260601_134623_e5_ie5_b480_ew480_*/*/seed_*/residual_teacher_v12_diagnostics.csv`.
- Per-run alignment diagnostics:
  `logs/s3v12m/v12tm_20260601_134623_e5_ie5_b480_ew480_*/*/seed_*/alignment_diagnostics.csv`.

Summary:

| Item | Result |
| --- | ---: |
| Completed runs | `9/9` |
| Datasets | `panama`, `tetouan`, `spain_energy_weather` |
| Seeds | `42`, `43`, `44` |
| Rolling validation windows | `45` total |
| Alignment gate passed | `9/9` |
| Teacher gate passed | `0/9` |
| Rolling windows passed | `0/45` |
| Nonzero teacher alpha | `0/9` |
| Teacher alpha source | all `fallback_block_gate` |
| Teacher-only abort files | `9/9` |
| Generator artifacts under `ckpt/s3v12m` | `0` |

Per-run gate summary:

| Dataset | Seed | Alignment gate | Active rank | OOF transfer MSE gain | Teacher gate | Window pass | Alpha | Failure reasons |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | --- |
| panama | 42 | pass | 3 | 0.640201 | fail | 0/5 | 0.0 | MSE; MAE; counterfactual; sign agreement |
| panama | 43 | pass | 3 | 0.039301 | fail | 0/5 | 0.0 | MSE; MAE; counterfactual; sign agreement |
| panama | 44 | pass | 3 | 0.533407 | fail | 0/5 | 0.0 | MSE; MAE; counterfactual; sign agreement |
| tetouan | 42 | pass | 2 | 0.274735 | fail | 0/5 | 0.0 | MSE; MAE; counterfactual; sign agreement |
| tetouan | 43 | pass | 3 | 0.476276 | fail | 0/5 | 0.0 | MSE; MAE; counterfactual; sign agreement |
| tetouan | 44 | pass | 3 | 0.303220 | fail | 0/5 | 0.0 | MSE; MAE; counterfactual; sign agreement |
| spain_energy_weather | 42 | pass | 2 | 0.031668 | fail | 0/5 | 0.0 | MSE; MAE; counterfactual; sign agreement |
| spain_energy_weather | 43 | pass | 2 | 0.337808 | fail | 0/5 | 0.0 | MSE; MAE; counterfactual; sign agreement |
| spain_energy_weather | 44 | pass | 2 | 0.394015 | fail | 0/5 | 0.0 | MSE; MAE; counterfactual; sign agreement |

## Chinese Experiment Analysis

这轮结果要分成两层理解：

1. V12 的 `phi_v - phi_i` 对齐方向确实比 smoke 更有信号。大矩阵中
   `alignment_gate_pass=9/9`，`active_rank` 在 `2-3` 之间，说明
   train-only shared anchor 和 outcome-aligned transport 能保留一些和
   residual transfer 相关的低维 drift 信号。
2. 这个对齐信号没有转化成可以服务的预测修正。所有 run 的
   `teacher_shrinkage_alpha=0.0`，`teacher_alpha_source=fallback_block_gate`，
   teacher gate 为 `0/9`，rolling validation window 为 `0/45`。这意味着
   V12 teacher 认为非零修正不够稳定，最终选择不改预测。

因此，V12 当前最准确的科学结论是：

```text
V12 improves the auditability and coordinate handling of the drift
representation, but it still does not validate a deployable
drift-conditioned residual correction or Stage III generator.
```

对论文的影响：

- 可以写：V12 是一个更严格的 train-only alignment / teacher-only 审计路径。
- 可以写：V12 证明 alignment gate 本身不足以证明 forecasting correction。
- 不能写：V12 显著提升预测精度。
- 不能写：V12 证明 Stage III 学到了 concept drift。
- 不能写：V12 允许 generator training 进入论文主结果。

下一步不应该直接训练 generator，而应该先重构 V12b teacher。核心原因是：
当前 V12 teacher 仍然是围绕 V11 strict block/reliability gate 的协议包装；
alignment 层找到了候选信号，但 teacher 层没有形成稳定的
out-of-fold residual correction pathway。

## Interpretation

V12 currently proves only code-path feasibility and safety behavior:

```text
The V12 opt-in path can run, save diagnostics, improve alignment auditability,
fail closed, and block generator training when teacher-only evidence is
missing.
```

It does not prove forecasting improvement, concept-drift learning, or
generator validity. The public-3 matrix is stronger evidence than the smoke
run, and it is negative for the teacher/generator claim.

## Go / No-Go

Current status: **No-Go for generator training as manuscript evidence**.

The next valid experiment is a V12b teacher-only redesign, not generator
training:

- public-3 datasets;
- pre-registered seeds;
- no `--overwrite`;
- strict V11/V12 gate thresholds retained;
- nonzero teacher alpha required;
- real drift must beat counterfactual drift modes.

Only if a nonzero V12b teacher passes prospective rolling teacher-only gates
should a generator canary be considered.
