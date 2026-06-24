# Stage III V9 Teacher-Only 诊断实验记录

日期：2026-05-28

本记录说明 V9 之后新增的 teacher-only upper-bound / residual-sign 诊断。它不是新的论文性能结果，而是为了回答一个更基础的问题：

```text
如果完全不经过 generator、不做参数扰动，只使用 train-only residual teacher 直接修正预测，
这个 teacher 本身能否在 rolling validation 上产生有益方向？
```

如果 teacher-only 都失败，那么继续扩大 generator 训练通常没有科学意义，因为 generator 只是学习一个本身不可靠的 teacher 方向。

## 1. 背景

上一轮 V9 修复了以下问题：

- `AdaptationGenerator` 未训练；
- zero drift 非零扰动；
- drift 表示被压成标量；
- `phi_v - phi_i` 未严格校准；
- failed candidate 被误发布为最终 served artifact。

但 V9 同预算 Panama seed42 canary 仍失败。尤其是：

- rolling validation `0/5`；
- real drift 在多数窗口伤害 MSE/MAE；
- zero drift 常常是最佳 counterfactual；
- generator train loss 下降，但 served-policy validation 变差。

因此需要拆分失败来源：

1. residual teacher 本身是否能迁移；
2. generator 是否把 teacher 方向学歪；
3. parameter delta 和 residual correction 是否相互干扰；
4. served-policy objective 是否和 train mini-batch objective 错位。

本轮先解决第 1 个问题。

## 2. 实现范围

新增文件：

- `train/stage3_v9_teacher_diagnostics.py`

新增功能：

- `compute_teacher_correction_metrics()`
  - 计算 base/corrected MSE、MAE；
  - 计算 MSE/MAE gain；
  - 计算 residual 与 correction 的 sign agreement；
  - 计算 residual-correction cosine；
  - 计算 correction norm、residual norm、over-correction rate、harmful element rate。
- `summarize_teacher_mode_rows()`
  - 比较 real / zero / shuffled / sign-flipped / amplified / random matched-norm；
  - 输出 real-vs-best-counterfactual margin。

扩展文件：

- `train/run_stage3_v2_sensitivity_audit.py`
  - 新增 `_run_v9_teacher_only_rows()`；
  - 新增 `_run_v9_teacher_rolling_diagnostics()`；
  - 新增 `_teacher_diagnostic_aggregate()`；
  - 在 residual teacher 拟合后、generator 训练前输出 teacher-only 诊断；
  - manifest 增加 residual teacher hash 和 teacher-only 摘要字段。
- `configs/stage3_v9_residual_teacher_rrcca_served_policy_gate.yaml`
  - 增加 `run_teacher_only_diagnostics: true`；
  - 增加 teacher diagnostic gate 阈值。
- `tests/test_stage3_v9_teacher_protocol.py`
  - 增加 toy teacher-only 诊断测试。
- `models/stage3_alignment_v9.py`
  - 修正 `teacher_residual_r2` 的含义：现在是 residualized drift 对 residual 的 ridge predictability R2，而不是 residual 均值相关的伪 R2。

未触碰：

- `train/online_loop.py`
- `models/adaptation.py`
- 原始日志、checkpoint、表格、图像
- legacy Stage III 路径

## 3. 新增 CSV

每个 V9 run 目录新增：

- `teacher_only_validation_summary.csv`
- `teacher_only_validation_diagnostic_summary.csv`
- `teacher_only_validation_step_metrics.csv`
- `teacher_only_rolling_validation_summary.csv`
- `teacher_only_rolling_validation_window_summary.csv`
- `teacher_only_rolling_validation_aggregate.csv`
- `teacher_only_rolling_validation_step_metrics.csv`
- `teacher_only_test_summary.csv`
- `teacher_only_test_diagnostic_summary.csv`
- `teacher_only_test_step_metrics.csv`

这些文件不会复用或覆盖原有 `sensitivity_summary.csv` / `step_metrics.csv` 语义。

## 4. Targeted Test

```text
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_stage3_v9_teacher_protocol.py -q
8 passed
```

新增测试验证：

- real drift teacher correction 能在 toy residual 上明显降低误差；
- zero drift 不改变 base；
- sign-flipped drift 明显劣于 real；
- diagnostic summary 能正确识别 real 优于 counterfactual。

## 5. Smoke Run

命令：

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v2_sensitivity_audit.py --config configs\stage3_v9_residual_teacher_rrcca_served_policy_gate.yaml --dataset panama --run-id v9teacher_smoke_s42_g1b64_01 --seed 42 --device auto --modes real zero shuffled sign_flipped amplified_2x amplified_5x random_matched_norm --validation-windows 5 --drift-source delta_phi_v --drift-polarity 1.0 --signal-mode delta_vector --normalize-vector --epochs 1 --invariant-epochs 1 --generator-epochs 1 --max-generator-batches 64 --max-eval-batches 120 --out-root logs\s3v9teacher_smoke --checkpoint-root ckpt\s3v9teacher_smoke
```

结果：

- exit status: strict abstain，预期内；
- logs: `logs/s3v9teacher_smoke/v9teacher_smoke_s42_g1b64_01/panama/seed_42`
- no `generator_served_policy.pt`

Teacher-only rolling validation aggregate：

| Metric | Value |
| --- | ---: |
| teacher diagnostic gate | fail |
| window pass | `0/5` |
| worst real MSE gain | `-1.116660` |
| worst real MAE gain | `-0.482254` |
| worst real-vs-best-CF margin | `-1.116660` |
| worst sign agreement | `0.477083` |

Smoke 结论：

```text
teacher-only 直接修正比 generator 还危险。
这说明 V9 的 train-only teacher 在小预算下不能迁移到 validation windows。
```

## 6. Same-Budget Diagnostic Canary

命令：

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v2_sensitivity_audit.py --config configs\stage3_v9_residual_teacher_rrcca_served_policy_gate.yaml --dataset panama --run-id v9teacher_s42_g5b480_01 --seed 42 --device auto --modes real zero shuffled sign_flipped amplified_2x amplified_5x random_matched_norm --validation-windows 5 --drift-source delta_phi_v --drift-polarity 1.0 --signal-mode delta_vector --normalize-vector --epochs 5 --invariant-epochs 5 --generator-epochs 5 --max-generator-batches 480 --max-eval-batches 480 --out-root logs\s3v9teacher_diag --checkpoint-root ckpt\s3v9teacher_diag
```

结果：

- exit status: strict abstain，预期内；
- logs: `logs/s3v9teacher_diag/v9teacher_s42_g5b480_01/panama/seed_42`
- checkpoints: `ckpt/s3v9teacher_diag/panama/v9teacher_s42_g5b480_01/seed_42`
- no `generator_served_policy.pt`

### 6.1 Teacher Fit

`residual_teacher_v9_diagnostics.csv`：

| Metric | Value |
| --- | ---: |
| fit split | `train` |
| fit sample count | `480` |
| drift dim | `64` |
| feature dim | `193` |
| output dim | `12` |
| teacher residual MSE before | `0.1608959436` |
| teacher residual MSE after | `0.1450631768` |
| teacher residual R2 | `0.0926146507` |

解释：

- teacher 在训练集上确实有一点 residual explainability；
- 但 R2 只有约 `9.26%`，train 内部都不强；
- 这已经预示 validation 迁移会困难。

### 6.2 Teacher-Only Rolling Validation

`teacher_only_rolling_validation_aggregate.csv`：

| Metric | Value |
| --- | ---: |
| teacher diagnostic gate | fail |
| failure reasons | `teacher_real_mse_gain;teacher_real_mae_gain;teacher_counterfactual` |
| window pass | `0/5` |
| worst real MSE gain | `-0.197784` |
| worst real MAE gain | `-0.118167` |
| worst real-vs-best-CF margin | `-0.197784` |
| worst sign agreement | `0.515625` |

Window-level summary：

| Window | MSE gain | MAE gain | sign agreement | cosine | best CF | margin |
| ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 0 | `-0.197784` | `-0.118167` | `0.536979` | `0.088957` | zero | `-0.197784` |
| 1 | `-0.127959` | `-0.098808` | `0.527778` | `0.099303` | zero | `-0.127959` |
| 2 | `-0.150972` | `-0.086863` | `0.523611` | `0.077501` | zero | `-0.150972` |
| 3 | `-0.128340` | `-0.066789` | `0.518750` | `0.062607` | zero | `-0.128340` |
| 4 | `-0.098364` | `-0.067042` | `0.515625` | `0.053426` | zero | `-0.098364` |

解释：

- 每个 rolling window 中，real teacher correction 都伤害 MSE 和 MAE；
- 每个窗口的 best counterfactual 都是 zero；
- sign agreement 稍高于随机，但幅度和方向不足以改善预测；
- residual-correction cosine 很低，说明 teacher correction 与真实 residual 的向量方向弱相关。

### 6.3 Teacher-Only Test

`teacher_only_test_diagnostic_summary.csv`：

| Metric | Value |
| --- | ---: |
| test MSE gain | `-0.097074` |
| test MAE gain | `-0.052800` |
| sign agreement | `0.513368` |
| residual-correction cosine | `0.057671` |
| best counterfactual | zero |
| margin | `-0.097074` |

Test 也复现同样失败模式。

### 6.4 Generator Served-Policy

`generator_checkpoint_selection_served_policy.csv`：

| Epoch | Rolling pass | Worst MSE gain | Worst MAE gain | Worst margin | Max reject |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `0/5` | `-0.009815` | `-0.008962` | `-0.009815` | `0.0` |
| 1 | `0/5` | `-0.009427` | `-0.008627` | `-0.009427` | `0.0` |
| 2 | `0/5` | `-0.011107` | `-0.004480` | `-0.011107` | `0.0` |
| 3 | `0/5` | `-0.015173` | `-0.011738` | `-0.015173` | `0.0` |
| 4 | `0/5` | `-0.014583` | `-0.012317` | `-0.014583` | `0.0` |

Selected epoch 是 `1`，但仍未通过 gate。

这和 teacher-only 结论一致：generator 没有救回 teacher 方向，真实 drift 修正仍然是负收益。

## 7. 与原 V9 的关系

原 V9 同预算 run：

- `logs/s3v9diag/v9rt_s42_g5b480_01/panama/seed_42`
- selected epoch `0`
- rolling pass `0/5`
- worst MSE gain `-0.007082`
- worst MAE gain `-0.005109`
- no served artifact

新 V9 teacher diagnostic run：

- `logs/s3v9teacher_diag/v9teacher_s42_g5b480_01/panama/seed_42`
- selected epoch `1`
- rolling pass `0/5`
- worst MSE gain `-0.009427`
- worst MAE gain `-0.008627`
- no served artifact
- teacher-only rolling validation `0/5`

两者都不能支持 Stage III claim。新 run 的额外价值是定位失败来源：

```text
V9 teacher 本身在 held-out rolling validation 上就是 harmful correction。
因此下一步不能只继续扩大 generator 训练。
```

## 8. 科学结论

本轮诊断支持以下判断：

1. V9 的工程修复是真实的，但 teacher 方向不可靠。
2. `phi_v - phi_i` 经过 teacher-aware RRCCA 后仍不能稳定预测 residual correction。
3. sign agreement 约 `0.51-0.54`，只略高于随机，且 cosine 很低。
4. zero drift 是所有 validation windows 和 test 的最佳 counterfactual。
5. 继续用当前 teacher 训练 generator，会把错误方向传播到 parameter delta 和 residual head。

因此，当前不能写：

```text
Stage III learns concept drift.
Stage III V9 validates drift-conditioned online adaptation.
Residual teacher solves the Stage III evidence problem.
```

更安全的表述：

```text
The V9 audit shows that train-only residual supervision and zero-drift identity
are necessary but insufficient. Teacher-only diagnostics reveal that the current
drift-to-residual map does not transfer to rolling validation, so Stage III
remains an audited research module rather than a validated performance component.
```

中文：

```text
V9 审计表明，仅基于训练集的残差教师和 zero-drift identity 是必要修复，但仍不充分。
teacher-only 诊断显示，当前 drift-to-residual 映射不能迁移到 rolling validation。
因此 Stage III 目前仍应被定位为经过审计的研究模块，而不是已经验证有效的性能组件。
```

## 9. 下一步

如果继续研究 Stage III，下一步不应扩大当前 V9 到 public-3 性能宣称，而应重构 teacher/alignment：

1. 增加 train-calibrated shrinkage teacher：
   - `y_base + alpha * teacher(z)`；
   - alpha 只能在 train/inner-validation 中校准；
   - 检查是否只是 correction magnitude 过大。
2. 增加 residual-sign weighted loss：
   - 显式惩罚 `correction * true_residual < 0` 的维度；
   - 但必须只用 train inner split。
3. 将 V9 aligner 改为 cross-fitted leakage-subspace projection：
   - 去除与 `phi_i` 相关的泄漏子空间；
   - 避免 `D - f(phi_i)` 的 additive offset 破坏 zero identity；
   - 用 out-of-fold R2 和 sign stability 选 rank。
4. generator 训练改为 rolling/meta-objective：
   - 在 train 内做 chronological inner-train / inner-meta；
   - 候选选择直接优化 served-policy-like robust objective。

在 teacher-only rolling validation 通过之前，不应运行 public-3 大规模性能宣称实验。

