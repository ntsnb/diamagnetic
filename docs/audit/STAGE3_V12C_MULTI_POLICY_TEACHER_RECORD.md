# Stage III V12c Multi-Policy Teacher Record

Date: 2026-06-01

本记录汇总 V12b 失败后的 V12c 诊断实现、单元验证、Panama 8 组合
teacher-only probe，以及 best-delta public-3 大矩阵。本文不是论文正面
结果；它是 Stage III 科研完整性证据链的一部分。

## 1. 为什么进入 V12c

V12 public-3 的结论是：alignment gate `9/9` 通过，但 teacher gate
`0/9`，rolling validation window `0/45`。V12b 随后把 V12 的 wrapper
teacher 替换成 train-only chronological OOF residual teacher，但 public-3
仍然是 `teacher_gate_pass=0/9`、`0/45` rolling windows，并且主矩阵中
alpha 被 `fallback_oof_gate` 清零。

V12b loose-alpha probe 进一步排除了“只是 gate 太严”这个单一解释：

- 路径：
  `logs/s3v12bprobe/probe_v12b_loose_panama_s42_20260601_01/panama/seed_42`
- 内部 alpha gate 放松后，teacher 产生非零 correction。
- 但 rolling validation 仍为 `0/5`。
- worst real MSE gain `-0.000148732`。
- worst real MAE gain `-0.000123566`。
- worst real-vs-best-counterfactual margin `-0.002547824`。
- worst sign agreement `0.126736`。
- 5/5 validation windows 的 best counterfactual 都是 `sign_flipped`。

因此，V12c 的目标不是调低最终 claim gate，而是把诊断拆开：

```text
ungated: 先看 OOF 上理论最优 alpha 是否有信号；
relaxed: 只要求足够多 chronological blocks 通过，允许探索性非零 alpha；
strict: 保留 worst-block 安全条件，作为论文/部署级 gate。
```

这样可以避免 V12b 一旦 strict gate 清零，就完全看不到 teacher 是否有
弱信号。

## 2. 新增 opt-in 文件

V12c 没有覆盖 V12/V12b 文件，新增或复制出独立路径：

- `models/stage3_residual_teacher_v12c.py`
  - `TrainOnlyChronologicalOOFResidualTeacherV12c`
  - 三策略 alpha：`ungated`、`relaxed`、`strict`
  - 保存 `alpha_by_policy` 和 policy-level diagnostics
  - checkpoint metadata 保存 `stage3_signal_builder_v12c_state`
- `train/stage3_v12c_signal_contract.py`
  - V12c train-only signal contract
  - 支持 `level_vector` / `delta_vector`
  - train split 拟合 scale，eval/test 复用
- `train/run_stage3_v12c_sensitivity_audit.py`
  - V12c teacher-only audit runner
  - CLI `--signal-mode` 同步写入 `online.stage3` 和
    `stage3_v2.residual_teacher`
  - final diagnostic gate 默认只用 online-causal counterfactuals；
    offline modes 可作为 report-only
- `train/run_stage3_v12c_teacher_audit_matrix.py`
  - non-overwriting matrix runner
- `configs/stage3_v12c_multi_policy_teacher_probe.yaml`
  - signed alpha probe config
- `configs/stage3_v12c_multi_policy_teacher_probe_nonnegative.yaml`
  - nonnegative alpha probe config
- `tests/test_stage3_v12c_teacher_policies.py`
  - relaxed vs strict policy test
  - policy diagnostics test
  - checkpoint round-trip test
  - signal contract metadata preservation test

## 3. Verification

Fresh checks:

```text
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_stage3_v12c_teacher_policies.py -q
4 passed

C:\Users\admin\anaconda3\python.exe -m pytest \
  tests\test_stage3_v12c_teacher_policies.py \
  tests\test_stage3_v12b_signal_contract.py \
  tests\test_stage3_v12b_teacher_protocol.py \
  tests\test_stage3_v12b_oof_no_leakage.py \
  tests\test_stage3_v12b_gate_diagnostics.py -q
10 passed

C:\Users\admin\anaconda3\python.exe -m py_compile \
  models\stage3_residual_teacher_v12c.py \
  train\stage3_v12c_signal_contract.py \
  train\stage3_v12c_artifacts.py \
  train\run_stage3_v12c_sensitivity_audit.py \
  train\run_stage3_v12c_teacher_audit_matrix.py
passed
```

Note: default `python -m pytest ...` used the local `.venv`, which has no
`pytest` installed. All valid test evidence above uses the Anaconda Python
that the experiments also used.

## 4. V12c Smoke

Command:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v12c_sensitivity_audit.py `
  --config configs\stage3_v12c_multi_policy_teacher_probe.yaml `
  --dataset panama --run-id smoke_v12c_panama_s42_20260601_01 `
  --seed 42 --device auto --epochs 1 --invariant-epochs 1 `
  --generator-epochs 1 --max-train-batches 2 --max-generator-batches 96 `
  --max-eval-batches 32 --validation-windows 2 --teacher-only `
  --out-root logs\s3v12c_smoke --checkpoint-root ckpt\s3v12c_smoke `
  --modes real zero sign_flipped amplified_2x
```

Output:

- `logs/s3v12c_smoke/smoke_v12c_panama_s42_20260601_01/panama/seed_42`

Key finding:

- V12c wrote multi-policy diagnostics.
- Relaxed/ungated OOF correction was strong in this tiny smoke:
  `teacher_oof_mse_gain=0.709636`, `teacher_oof_mae_gain=0.511410`.
- But rolling validation still failed:
  `teacher_gate_pass=False`, `0/2` windows.
- worst rolling real MSE gain `-0.844508`.
- This shows exactly why V12c is useful as a diagnostic: it exposes that
  train/OOF-looking correction can fail badly out of sample.

## 5. Panama Seed42 8-Combination Probe

Purpose:

```text
Check whether V12b/V12c failure is mainly caused by signal mode, drift polarity,
or signed vs nonnegative alpha.
```

Grid:

- `signal_mode`: `level_vector`, `delta_vector`
- `drift_polarity`: `+1`, `-1`
- alpha policy: signed config vs nonnegative config
- dataset/seed: `panama`, `42`
- epochs: `5`
- max teacher/eval batches: `480`
- validation windows: `5`

Manifest root:

- `logs/s3v12c_probe8/_manifests`

All 8 manifests completed:

- `v12c8_signed_level_pos_20260601_181615_e5_ie5_b480_ew480`
- `v12c8_signed_level_neg_20260601_182119_e5_ie5_b480_ew480`
- `v12c8_signed_delta_pos_20260601_182615_e5_ie5_b480_ew480`
- `v12c8_signed_delta_neg_20260601_183111_e5_ie5_b480_ew480`
- `v12c8_nonneg_level_pos_20260601_183607_e5_ie5_b480_ew480`
- `v12c8_nonneg_level_neg_20260601_184104_e5_ie5_b480_ew480`
- `v12c8_nonneg_delta_pos_20260601_184600_e5_ie5_b480_ew480`
- `v12c8_nonneg_delta_neg_20260601_185057_e5_ie5_b480_ew480`

Summary:

| Family | Gate | Windows | Worst MSE | Worst MAE | Worst sign | Test MSE | Test MAE | Test sign |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| level-vector variants | `0/4` | `0/20` | `-0.000156` | `-0.000289` | `0.381250` | `0.000944` | `0.001187` | `0.380729` |
| delta-vector variants | `0/4` | `0/20` | `0.000909` | `0.000707` | `0.466146` | `0.001881` | `0.000834` | `0.453472` |

Interpretation:

1. `delta_vector` is better than `level_vector` on Panama seed42.
2. Polarity `+1` vs `-1` and signed vs nonnegative alpha gave identical
   aggregate results in this probe, suggesting the selected correction is not
   primarily rescued by flipping the drift sign.
3. `delta_vector` real drift beat zero/sign-flipped/amplified online-causal
   controls in the evaluated windows, but the absolute gains were too small
   and sign agreement stayed below `0.55`.
4. No `generator*.pt` artifact was produced under `ckpt/s3v12c_probe8`.

This is a weak positive diagnostic for Panama seed42 only. It is not a
publishable Stage III performance result.

## 6. Best-Delta Public-3 Matrix

Because `delta_vector` was best in the 8-combination probe, it was expanded to
public-3 x 3 seeds using the nonnegative-alpha config.

Command:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v12c_teacher_audit_matrix.py `
  --config configs\stage3_v12c_multi_policy_teacher_probe_nonnegative.yaml `
  --datasets panama tetouan spain_energy_weather `
  --seeds 42 43 44 `
  --epochs 5 --invariant-epochs 5 --generator-epochs 5 `
  --max-generator-batches 480 --max-eval-batches 480 `
  --validation-windows 5 --device auto `
  --signal-mode delta_vector --drift-polarity 1.0 `
  --out-root logs\s3v12cm --checkpoint-root ckpt\s3v12cm `
  --run-prefix v12ctm_delta_nonneg
```

Manifest:

- `logs/s3v12cm/_manifests/v12ctm_delta_nonneg_20260601_190259_e5_ie5_b480_ew480`

Overall:

| Metric | Result |
| --- | ---: |
| Completed runs | `9/9` |
| Teacher gates | `0/9` |
| Rolling validation windows | `0/45` |
| Generator artifacts | `0` |

Per-run summary:

| Dataset | Seed | Gate | Worst MSE | Worst MAE | Worst margin | Worst sign | Test MSE | Test MAE | Test sign | Alpha source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| panama | 42 | fail | `0.000909` | `0.000707` | `0.000909` | `0.466146` | `0.001881` | `0.000834` | `0.453472` | `v12c_multi_policy_oof` |
| panama | 43 | fail | `0.005604` | `0.003026` | `0.002918` | `0.440451` | `0.003033` | `0.001289` | `0.446528` | `v12c_multi_policy_oof` |
| panama | 44 | fail | `-0.000381` | `-0.000222` | `-0.001351` | `0.468924` | `-0.000557` | `-0.000054` | `0.456250` | `v12c_multi_policy_oof` |
| tetouan | 42 | fail | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `fallback_oof_gate` |
| tetouan | 43 | fail | `0.000000071` | `-0.00000137` | `-0.001623` | `0.031944` | `0.000242` | `0.000153` | `0.046875` | `v12c_multi_policy_oof` |
| tetouan | 44 | fail | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `fallback_oof_gate` |
| spain_energy_weather | 42 | fail | `-0.000588` | `-0.000577` | `-0.000588` | `0.469444` | `-0.002120` | `-0.001920` | `0.480208` | `v12c_multi_policy_oof` |
| spain_energy_weather | 43 | fail | `-0.006002` | `-0.005716` | `-0.006002` | `0.480035` | `0.001732` | `0.001304` | `0.494097` | `v12c_multi_policy_oof` |
| spain_energy_weather | 44 | fail | `-0.000187` | `-0.000326` | `-0.003692` | `0.383333` | `0.001144` | `0.000145` | `0.429514` | `v12c_multi_policy_oof` |

Most informative row:

- Panama seed43 is the closest to passing:
  - worst MSE `0.005604` passes the `0.005` threshold;
  - worst MAE `0.003026` passes the `0.002` threshold;
  - worst margin `0.002918` passes the `0.002` threshold;
  - but worst sign `0.440451` fails the `0.55` threshold.

This means V12c can extract a weak useful direction in some Panama runs, but it
does not yet produce a stable, sign-consistent residual correction across
datasets and seeds.

## 7. Scientific Conclusion

V12c is an improvement as a diagnostic protocol, not as a validated forecasting
module.

What V12c proves:

- The previous V12b all-zero result was partly hiding weak teacher signals.
- Separating `ungated` / `relaxed` / `strict` policies is useful for diagnosis.
- `delta_vector` is a better candidate signal than `level_vector` for Panama
  seed42.
- The safety gate correctly prevented generator training when the teacher
  failed public-3 prospective validation.

What V12c does not prove:

- It does not prove Stage III captures concept drift.
- It does not prove drift-conditioned online adaptation improves forecasting.
- It does not justify generator training.
- It does not justify SOTA or significant-improvement language.
- It does not solve cross-dataset robustness.

Current best defensible wording:

```text
The Stage III repair experiments show that stricter train-only alignment and
multi-policy OOF teacher diagnostics can expose weak drift-residual signals,
but the current teacher does not yet satisfy prospective rolling-validation
gates across datasets. Therefore Stage III remains a diagnostic and candidate
mechanism rather than validated performance evidence.
```

## 8. Next Safe Step

Do not train a generator yet. The next scientifically safe step is V12d or an
equivalent teacher redesign that targets the residual mapping itself:

1. Add residual-history/context features, not only drift polynomial features.
2. Test per-horizon correction with explicit sign calibration.
3. Add a base residual autoregressive teacher as a non-drift control.
4. Compare drift-only, residual-history-only, and drift+history teachers.
5. Keep the final public-3 gate unchanged.

Only if a teacher-only public-3 matrix passes the pre-declared rolling gates
should generator training resume.
