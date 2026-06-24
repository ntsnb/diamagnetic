# Stage III V2.5 Canonical Drift + Generator Checkpoint Selection 实验记录

日期：2026-05-25

本轮 run id：

```text
codex_stage3_v25_canonical_ckptselect_smoke_20260525
codex_stage3_v25_canonical_ckptselect_panama_b60_20260525
codex_stage3_v25_canonical_ckptselect_panama_b120_20260525
```

## 1. 本轮修复目标

V2.3 的问题是：虽然 generator 已经训练并 artifact reload，但 generator checkpoint 仍然按最后 epoch 使用；同时 `phi_v - phi_i` 的坐标系虽然改成 residual 表示，仍没有 train-only 的 residual-transition 规范化坐标。

V2.5 做两类增强：

1. **Generator checkpoint selection on rolling validation**：每个 generator epoch 保存候选 checkpoint，只在 rolling validation 上打分选 checkpoint，test 仍然 report-only。
2. **Train-only residual drift canonicalizer**：只用训练集 ordered stream 中的 `delta_phi_v[t]-delta_phi_v[t-1]` 与 residual transition 拟合 canonical basis，validation/test 冻结使用。

## 2. 代码改动摘要

| 文件 | 改动 | 科研意义 |
| --- | --- | --- |
| `train/stage3_generator_selection_v2.py` | 新增 generator candidate、selection score、tie-breaker | 把 generator selection 从最后 epoch 改成 validation-only evidence |
| `train/run_stage3_v2_sensitivity_audit.py` | 每 epoch 保存候选 generator，rolling validation 选择，manifest 记录 selection | 避免用 test 反向挑 generator |
| `train/stage3_sensitivity_gate_v2.py` | `real_reject_rate` 改为 shift/risk/missing 三类拒绝总和 | 防止“靠 risk gate 大量拒绝”被误判为可靠提升 |
| `models/aligned_representation_v2.py` | 新增 `TrainSetResidualDriftCanonicalizer` | 用 train-only residual transition 固定漂移坐标系 |
| `train/stage3_v2_artifacts.py` | artifact bundle 支持 residual canonicalizer | 保证 canonicalizer 保存后 reload 再评估 |
| `configs/stage3_v2_strong.yaml` | 启用 checkpoint selection 和 canonicalize_delta | V2.5 strong 配置 |

## 3. 验证

新增/目标测试：

```text
tests/test_stage3_generator_selection_v2.py
tests/test_stage3_sensitivity_gate_v2.py
tests/test_aligned_representation_v2.py
tests/test_stage3_v2_artifacts.py
```

关键验证结果：

```text
8 passed in 0.03s
21 passed in 1.43s
```

完整 Stage III V2 回归套件：

```text
82 passed in 3.63s
```

Smoke run 确认：

```text
stage3_residual_drift_canonicalized = true
stage3_artifacts_reloaded_for_eval = true
stage3_generator_checkpoint_selection_enabled = true
stage3_generator_checkpoint_selection_stage = pre_risk_gate
```

## 4. V2.5 Panama seed42 b60/b120 成对实验

协议：

```text
dataset: panama
seed: 42
base epochs: 2
aligned epochs: 2
generator epochs: 4
validation_windows: 3
modes: real, zero, shuffled, sign_flipped, random_matched_norm
test role: report-only
selection role: rolling validation
```

注意：本轮已完成 seed42/43 的 b60/b120 成对实验。它仍只覆盖 Panama 一个数据集，因此不能代表跨数据集统计显著结论。

## 5. 结果

### 5.1 Claim gate

| seed | budget | selected generator epoch | generator selection pass | rolling pass rate | test gate | test failure reasons |
| ---: | --- | ---: | --- | ---: | --- | --- |
| 42 | b60 | 0 | fail | 0.3333 | fail | `real_mse_gain;real_mae_gain;counterfactual` |
| 42 | b120 | 3 | fail | 0.3333 | fail | `real_mae_gain` |
| 43 | b60 | 3 | fail | 0.3333 | fail | `real_mse_gain;real_mae_gain;counterfactual` |
| 43 | b120 | 0 | fail | 0.3333 | fail | `reject_rate` |

### 5.2 Test report-only

| seed | budget | base MSE | Stage III MSE | MSE gain | base MAE | Stage III MAE | MAE gain | best counterfactual | reject rate |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 42 | b60 | 0.328935 | 0.329244 | -0.0938% | 0.485584 | 0.487286 | -0.3505% | random_matched_norm | 0.0833 |
| 42 | b120 | 0.207779 | 0.205150 | +1.2656% | 0.356179 | 0.357926 | -0.4907% | zero | 0.0750 |
| 43 | b60 | 0.317831 | 0.318864 | -0.3250% | 0.476784 | 0.476272 | +0.1074% | random_matched_norm | 0.0500 |
| 43 | b120 | 0.207718 | 0.204320 | +1.6361% | 0.358740 | 0.354609 | +1.1517% | shuffled | 0.3750 |

### 5.3 b120 相对 b60 的配对变化

| seed | delta MSE gain | delta MAE gain | delta rolling pass rate |
| ---: | ---: | ---: | ---: |
| 42 | +1.3594 percentage points | -0.1401 percentage points | 0.0000 |
| 43 | +1.9611 percentage points | +1.0443 percentage points | 0.0000 |
| mean | +1.6602 percentage points | +0.4521 percentage points | 0.0000 |

解释：

1. b120 让两个 seed 的 test MSE 都从负收益变成正收益；
2. b120 在 seed43 上同时改善 MAE，但在 seed42 上 MAE 更差；
3. b120 的 rolling validation 在两个 seed 上仍然只有 1/3；
4. 两个 b120 run 的 generator selection 本身都未通过 rolling gate；
5. seed43 b120 的 test gate 失败原因是 `reject_rate`，说明一部分收益来自过度拒绝；
6. 因此 b120 只能说明“更大 budget 可能改善 report-only test MSE”，不能说明模型整体显著提升。

### 5.4 Rolling validation

| seed | budget | worst-window MSE gain | worst-window MAE gain | best-window counterfactual MSE gain | failure reasons |
| ---: | --- | ---: | ---: | ---: | --- |
| 42 | b60 | -0.6458% | -0.8303% | +0.2621% | `window_pass_rate;real_mse_gain;real_mae_gain;counterfactual` |
| 42 | b120 | +0.1327% | +0.2380% | 0.0000% | `window_pass_rate;real_mse_gain;counterfactual` |
| 43 | b60 | -3.1018% | -1.0657% | +0.8160% | `window_pass_rate;real_mse_gain;real_mae_gain;counterfactual;reject_rate` |
| 43 | b120 | -0.6135% | +0.5226% | +0.3989% | `window_pass_rate;real_mse_gain;counterfactual;reject_rate` |

## 6. 诊断

V2.5 的修复是有效的工程修复，但不是有效的性能证明。

已改善：

1. generator 不再按最后 epoch 隐式选择；
2. manifest 记录了 candidate count、selected epoch、selection score 和 selection failure reasons；
3. `real_reject_rate` 不再漏算 risk-gate rejection；
4. drift 坐标从 raw `delta_phi_v` 增强为 train-only residual canonical coordinates；
5. artifact reload 覆盖 generator、risk gate、delta standardizer 和 residual canonicalizer。

### 6.1 五个原始缺陷的 V2.5 状态

| 原始缺陷 | V2.5 状态 | 证据解释 |
| --- | --- | --- |
| generator 未训练 | V2-only 路径已修复 | V2 runner 训练 `LayerAwareHyperLoRAGenerator`，保存 candidate checkpoints，并在评估前 reload。 |
| zero drift 非零扰动 | generator 输入坐标中已修复 | `models/adaptation_v2.py` 使用 bias-free conditioner 和 identity initialization，zero code 产生 zero delta。注意：经过 standardizer/canonicalizer 后，raw `delta_phi_v=0` 不一定仍是 generator-input zero；论文应说明 zero intervention 是在 generator 输入坐标下定义。 |
| vector drift 被压成 scalar | strong V2.5 路径已修复 | strong 配置使用 `signal_mode: delta_vector`，V2 runner 消费完整 drift 向量；legacy scalar 只应作为消融/旧路径。 |
| eval 阶段 VAE 随机 | V2-only 路径已修复 | `vae_eval_mode: mean` 下 eval 使用 `mu_z`，不再用随机采样污染 MSE/MAE 和 sensitivity。 |
| `phi_v - phi_i` 空间未对齐 | 工程级缓解，不能称为理论完全解决 | 当前采用 `phi_v = phi_i + delta_phi_v` 和 train-only residual drift canonicalizer。它提高了坐标可审计性，但还未证明 `delta_phi_v` 是稳定、因果、可泛化的 concept drift 表示。 |

未改善到可发 claim：

1. b60 和 b120 在两个 seed 上的 rolling validation 都失败；
2. b120 test MSE 在两个 seed 上变好，但 seed42 MAE 仍退化；
3. seed43 b120 的 reject rate 达到 37.5%，超过 `max_real_reject_rate=20%`；
4. test 是 report-only，不能用 b120 test MSE 回头支撑配置选择；
5. 当前仍不能证明 real drift consistently 优于 counterfactual；
6. 当前仍不能证明 Stage III V2.5 是稳定的性能贡献。

## 7. 对论文的影响

不能写：

```text
Stage III V2.5 significantly improves forecasting accuracy.
The canonical drift generator reliably captures concept drift.
The proposed model outperforms SOTA under online drift.
```

可以写在方法/局限中：

```text
We introduce a train-only canonical residual drift calibration and a validation-only generator checkpoint selection protocol to make Stage III online adaptation auditable. In bounded Panama experiments, the repaired protocol improves some MSE outcomes at larger budget, but rolling validation and MAE remain insufficient for a strong performance claim.
```

## 8. 下一步

本轮已经触发停止条件：

1. seed42/43 的 b60/b120 rolling validation 都不是 3/3；
2. seed42 b120 的 MAE gain 为负；
3. 多个 run 中 counterfactual 接近或超过 real；
4. seed43 b120 虽然 test MSE/MAE 为正，但 reject rate 超阈值，且 rolling validation 失败。

因此，Stage III 当前应作为“审计型在线适应机制/负结果诊断”写入论文，而不是作为性能核心创新点。

如果继续研究，不应再优先扩大 test budget，而应先把以下 alignment 诊断作为硬门槛：

1. train/validation 上 `delta_phi_v` transition 对 residual transition 的 explained variance / R2；
2. `phi_i` 与 `delta_phi_v` 的 CKA/CCA/cross-correlation diagonal 与 off-diagonal；
3. raw、standardized、canonicalized drift 三组消融；
4. `risk_gate_off`、`zero_adaptation`、real/zero/shuffled/sign-flipped/random_matched_norm 的逐 rolling-window margin；
5. checkpoint selection 必须要求每个 rolling window 中 real drift 同时优于 counterfactual，而不是只看 test report-only。
