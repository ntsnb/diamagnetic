# Stage III V10 Cross-Fitted Teacher Diagnostic Record

日期：2026-05-28

本文记录 V10 的代码修改、实验结果和科学结论。V10 不是论文性能结果，而是针对 V9 失败原因的一轮机制修复与门控实验。

## 1. 这次修了什么

V9 的 teacher-only 诊断已经证明：`y_base + teacher(z)` 在 Panama seed42 rolling validation 上是有害的。因此 V10 没有继续扩大 generator，而是先修 `drift -> residual correction` 证据链。

新增文件：

- `models/stage3_residual_teacher_v10.py`
- `models/stage3_alignment_v10.py`
- `models/adaptation_v10.py`
- `train/stage3_generator_training_v10.py`
- `train/stage3_v10_artifacts.py`
- `train/run_stage3_v10_sensitivity_audit.py`
- `configs/stage3_v10_crossfit_teacher_projection_served_policy_gate.yaml`
- `configs/stage3_v10_crossfit_teacher_projection_signed_alpha_gate.yaml`
- `tests/test_stage3_v10_teacher_protocol.py`

核心变化：

1. `TrainOnlyResidualTeacherV10` 增加 train-only inner calibration shrinkage。teacher 先在 inner train 拟合 residual，再在 inner calibration 上选择 correction 缩放系数。若 calibration 不支持修正，`alpha=0`，即安全 abstain。
2. `TrainSetCrossFittedTeacherResidualProjectionCCAAligner` 用 cross-fitted delta-space leakage projection 替代 V9 的 additive `D - f(phi_i)`。部署变换是线性投影，满足 `delta=0 -> cleaned_delta=0 -> transformed=0`。
3. V10 runner 增加 `abort_if_teacher_diagnostic_fails`。如果 teacher-only rolling gate 没过，不训练 generator，避免把坏 teacher 方向传播到参数扰动。
4. 第二个变体尝试了 per-output signed alpha 和 MAE-aware calibration，用于检验 MAE/符号问题是否来自单一标量 shrinkage 过弱。

未覆盖旧文件、旧日志、旧 checkpoint 或论文原表图。V10 都是 opt-in 新路径。

## 2. 验证

命令：

```powershell
C:\Users\admin\anaconda3\python.exe -m pytest tests -q
```

结果：

```text
140 passed in 4.84s
```

另外：

```powershell
git diff --check
```

结果：通过。

## 3. V10 Smoke

命令：

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v10_sensitivity_audit.py --config configs\stage3_v10_crossfit_teacher_projection_served_policy_gate.yaml --dataset panama --run-id v10_smoke_s42_b8_20260528_01 --seed 42 --device auto --modes real zero shuffled sign_flipped amplified_2x amplified_5x random_matched_norm --validation-windows 2 --drift-source delta_phi_v --drift-polarity 1.0 --signal-mode delta_vector --normalize-vector --epochs 1 --invariant-epochs 1 --generator-epochs 1 --max-generator-batches 8 --max-eval-batches 8 --out-root logs\s3v10_smoke --checkpoint-root ckpt\s3v10_smoke
```

输出目录：

- `logs/s3v10_smoke/v10_smoke_s42_b8_20260528_01/panama/seed_42`

结果：

- teacher shrinkage alpha: `0.0`
- teacher rolling pass: `0/2`
- worst MSE gain: `0.0`
- worst MAE gain: `0.0`
- no generator training
- `teacher_only_abort.json` 已写出

解释：

8 个 train batches 太少，teacher 不能形成可信 correction。V10 安全地退回 `alpha=0`，没有像 V9 那样继续放大错误方向。这个 smoke 只证明 fail-safe 机制有效，不证明性能有效。

## 4. V10 Scalar Shrinkage Same-Budget Canary

命令：

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v10_sensitivity_audit.py --config configs\stage3_v10_crossfit_teacher_projection_served_policy_gate.yaml --dataset panama --run-id v10teacher_s42_g5b480_20260528_01 --seed 42 --device auto --modes real zero shuffled sign_flipped amplified_2x amplified_5x random_matched_norm --validation-windows 5 --drift-source delta_phi_v --drift-polarity 1.0 --signal-mode delta_vector --normalize-vector --epochs 5 --invariant-epochs 5 --generator-epochs 5 --max-generator-batches 480 --max-eval-batches 480 --out-root logs\s3v10teacher_diag --checkpoint-root ckpt\s3v10teacher_diag
```

输出目录：

- `logs/s3v10teacher_diag/v10teacher_s42_g5b480_20260528_01/panama/seed_42`
- `ckpt/s3v10teacher_diag/panama/v10teacher_s42_g5b480_20260528_01/seed_42`

### 4.1 Teacher Fit

`residual_teacher_v10_diagnostics.csv`：

| Metric | Value |
| --- | ---: |
| fit sample count | `480` |
| inner fit / calib | `360 / 120` |
| drift dim | `64` |
| feature dim | `193` |
| output dim | `12` |
| train MSE before | `0.1608959436` |
| train MSE after | `0.1408407837` |
| train residual R2 | `0.1190261841` |
| selected alpha | `0.5` |
| calibration best gain | `0.0367155732` |
| calibration sign agreement | `0.5055555701` |
| calibration cosine | `0.0846847296` |

### 4.2 Alignment Diagnostics

`alignment_diagnostics.csv` train/post-canonical：

| Metric | Value |
| --- | ---: |
| alignment method | `cross_fitted_teacher_projection_rrcca_v10` |
| alignment gate | `True` |
| active rank | `11` |
| selected leakage rank | `3` |
| selected invariant predictability R2 | `0.8688101768` |
| retained variance fraction | `0.9537377357` |
| removed variance fraction | `0.0462622643` |
| teacher residual R2 diagnostic | `0.7431992888` |
| bootstrap sign stability mean | `0.9715909362` |

解释：V10 的 alignment 不再像 V7/V9 那样用 additive residualizer，而是移除了 delta-space 中约 `4.63%` 的 invariant-leakage 子空间，同时保留大部分 drift variance。内部 alignment gate 通过。

### 4.3 Rolling Validation

`teacher_only_rolling_validation_aggregate.csv`：

| Metric | Value |
| --- | ---: |
| teacher gate | `False` |
| failure reasons | `teacher_sign_agreement;teacher_real_mae_gain` |
| window pass | `3/5` |
| worst MSE gain | `0.0059367432` |
| worst MAE gain | `-0.0007096970` |
| worst real-vs-best-CF margin | `0.0059367432` |
| worst sign agreement | `0.5296875238` |

Window-level：

| Window | MSE gain | MAE gain | Sign | Cosine | Best CF | Margin |
| ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 0 | `0.0356502269` | `0.0154853922` | `0.5734375119` | `0.2320549637` | zero | `0.0356502269` |
| 1 | `0.0382004393` | `0.0195561119` | `0.5824652910` | `0.2694369853` | zero | `0.0382004393` |
| 2 | `0.0059367432` | `0.0047289725` | `0.5588541627` | `0.2126939893` | zero | `0.0059367432` |
| 3 | `0.0297367862` | `0.0164831039` | `0.5482639074` | `0.1690579504` | zero | `0.0297367862` |
| 4 | `0.0158427208` | `-0.0007096970` | `0.5296875238` | `0.1197582856` | zero | `0.0158427208` |

解释：

- 与 V9 teacher-only 的 `0/5` 和负收益相比，V10 scalar shrinkage 明显改善：`3/5` 窗口通过，所有窗口 MSE gain 为正，real drift 在每个窗口都优于 counterfactual。
- 但它仍然不能训练 generator：窗口 3 的 sign agreement 低于 `0.55`，窗口 4 的 MAE gain 为负且 sign agreement 更低。
- 因此当前结果是“机制明显改善但未过硬门”，不能写成 Stage III 已验证。

### 4.4 Test

`teacher_only_test_diagnostic_summary.csv`：

| Metric | Value |
| --- | ---: |
| MSE gain | `0.0267751812` |
| MAE gain | `0.0151892459` |
| sign agreement | `0.5416666865` |
| residual-correction cosine | `0.1836607456` |
| best counterfactual | zero |
| real-vs-best-CF margin | `0.0267751812` |

Test 是正的，但 test 不能替代 rolling validation selection gate。

## 5. V10 Signed Per-Output Alpha Canary

命令：

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v10_sensitivity_audit.py --config configs\stage3_v10_crossfit_teacher_projection_signed_alpha_gate.yaml --dataset panama --run-id v10signedteacher_s42_g5b480_20260528_01 --seed 42 --device auto --modes real zero shuffled sign_flipped amplified_2x amplified_5x random_matched_norm --validation-windows 5 --drift-source delta_phi_v --drift-polarity 1.0 --signal-mode delta_vector --normalize-vector --epochs 5 --invariant-epochs 5 --generator-epochs 5 --max-generator-batches 480 --max-eval-batches 480 --out-root logs\s3v10signed_teacher_diag --checkpoint-root ckpt\s3v10signed_teacher_diag
```

输出目录：

- `logs/s3v10signed_teacher_diag/v10signedteacher_s42_g5b480_20260528_01/panama/seed_42`

结果：

| Metric | Value |
| --- | ---: |
| teacher gate | `False` |
| window pass | `2/5` |
| worst MSE gain | `-0.0018595963` |
| worst MAE gain | `-0.0099424396` |
| worst sign agreement | `0.5204861164` |
| test MSE gain | `0.0124717232` |
| test MAE gain | `0.0096130566` |

解释：

per-output signed alpha 没有修复最后窗口，反而引入一个负 MSE gain 窗口。当前不采用这个变体作为后续 generator 训练前置。

## 6. 与 V9 的直接比较

| Run | Teacher rolling pass | Worst MSE gain | Worst MAE gain | Worst margin | Test MSE gain | Test MAE gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V9 teacher-only | `0/5` | `-0.197784` | `-0.118167` | `-0.197784` | `-0.097074` | `-0.052800` |
| V10 scalar shrinkage | `3/5` | `0.005937` | `-0.000710` | `0.005937` | `0.026775` | `0.015189` |
| V10 signed alpha | `2/5` | `-0.001860` | `-0.009942` | `-0.001860` | `0.012472` | `0.009613` |

V10 scalar shrinkage 是目前最好的 Stage III teacher-only 证据：它显著优于 V9，但仍未达到预先声明的硬门。

## 7. 当前科学结论

可以说：

```text
V10 的 cross-fitted zero-preserving alignment 和 train-calibrated shrinkage
显著降低了 V9 teacher-only 的 harmful correction 问题。
在 Panama seed42 same-budget canary 中，teacher-only 从 V9 的 0/5 负收益
改善到 V10 scalar 的 3/5，并且 test aggregate 为正。
```

不能说：

```text
Stage III V10 已经验证了 drift-conditioned online adaptation。
Stage III generator 可以开始作为论文性能贡献。
V10 已经显著超越 SOTA。
```

原因：

- rolling validation gate 仍失败；
- 第 5 个 validation window 的 MAE gain 为负；
- 最差 sign agreement 只有 `0.5297`；
- 还没有任何 V10 generator served-policy artifact；
- 还没有多 seed / 多数据集通过硬门。

## 8. 下一步

最小安全下一步不是写论文，而是继续修 teacher target：

1. 保留 scalar V10 作为目前最强路径。
2. 不训练 generator，直到 teacher-only rolling validation `5/5` 通过。
3. 重点分析窗口 4：为什么 MSE gain 为正但 MAE gain 为负、sign agreement 低。
4. 尝试 horizon-wise robust loss、quantile/Huber residual teacher、或按季节/负荷区间分层 calibration。
5. 只有 teacher-only 先过 Panama seed42 硬门，才扩展到 generator canary。
