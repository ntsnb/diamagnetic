# Stage III V2.5 Large-Scale Experiment Record

日期：2026-05-25

本文记录用户批准“不考虑成本”后的 Stage III V2.5 大规模审计实验。所有实验继续使用新增 V2/V2.5 路径，不覆盖 legacy Stage III 文件、原始数据、旧日志或旧 checkpoint。

## 1. 大规模协议原则

本轮采用更严格的矩阵原则：

1. `budget` 只限制训练和 generator replay，不限制 final evaluation；
2. test 保持 report-only，不能反向选择配置；
3. generator checkpoint 只通过 rolling validation selection 选择；
4. 每个 run 使用唯一 `run_id`，同一 `run_id + dataset + seed` 不复跑；
5. 若失败或中断，使用新的 `_retry` run id，不覆盖旧目录。

主矩阵目标：

| axis | value |
| --- | --- |
| datasets | `panama`, `tetouan`, `spain_energy_weather` |
| seeds | `42-51` |
| budgets | `b240`, `b480`, `b960`, `b1920` |
| epochs | `--epochs 5 --invariant-epochs 5 --generator-epochs 10` |
| validation windows | `5` |
| sensitivity modes | `real`, `zero`, `shuffled`, `sign_flipped`, `amplified_2x`, `amplified_5x`, `random_matched_norm` |

## 2. 首个 full-eval b240 run

运行：

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v2_sensitivity_audit.py `
  --config configs\stage3_v2_strong.yaml `
  --dataset panama `
  --run-id codex_stage3_v25_matrix_b240_pub3_seed42_fg_20260525 `
  --seed 42 `
  --epochs 5 --invariant-epochs 5 --generator-epochs 10 `
  --device auto `
  --modes real zero shuffled sign_flipped amplified_2x amplified_5x random_matched_norm `
  --max-train-batches 240 --max-generator-batches 240 `
  --validation-windows 5 `
  --drift-source delta_phi_v --drift-polarity 1.0 `
  --signal-mode delta_vector --normalize-vector
```

输出目录：

```text
logs/load_benchmarks/stage3_v2/codex_stage3_v25_matrix_b240_pub3_seed42_fg_20260525/panama/seed_42
checkpoints/stage3_v2/panama/codex_stage3_v25_matrix_b240_pub3_seed42_fg_20260525/seed_42
```

耗时：

```text
约 2 小时 18 分钟，单 GPU RTX 5060，full test sample count = 16783。
```

### 2.1 Claim gate

| dataset | seed | budget | selected epoch | rolling pass rate | test gate | failure reasons |
| --- | ---: | --- | ---: | ---: | --- | --- |
| panama | 42 | b240 | 0 | 0.0000 | fail | `real_mse_gain;real_mae_gain;counterfactual;reject_rate` |

### 2.2 Test report-only metrics

| mode | base MSE | Stage III MSE | base MAE | Stage III MAE | use rate | reject risk count | pred shift | delta norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| real | 0.242081 | 0.241873 | 0.336200 | 0.335824 | 0.1742 | 13859 | 0.042306 | 0.201011 |
| zero | 0.242081 | 0.242081 | 0.336200 | 0.336200 | 0.0000 | 16783 | 0.000000 | 0.000000 |
| shuffled | 0.242081 | 0.242213 | 0.336200 | 0.336359 | 0.1432 | 14380 | 0.038238 | 0.187593 |
| sign_flipped | 0.242081 | 0.242539 | 0.336200 | 0.336944 | 0.1706 | 13920 | 0.042342 | 0.201011 |
| amplified_2x | 0.242081 | 0.241873 | 0.336200 | 0.335824 | 0.1742 | 13859 | 0.042306 | 0.201011 |
| amplified_5x | 0.242081 | 0.241873 | 0.336200 | 0.335824 | 0.1742 | 13859 | 0.042306 | 0.201011 |
| random_matched_norm | 0.242081 | 0.242209 | 0.336200 | 0.336381 | 0.1466 | 14322 | 0.039000 | 0.191552 |

Gate summary:

```text
real_mse_gain = 0.000858
real_mae_gain = 0.001118
best_counterfactual_mode = amplified_5x
real_vs_best_counterfactual_mse_gain_margin = approximately 0
real_reject_rate = 0.8258
```

### 2.3 Rolling validation

| validation windows | pass count | pass rate | worst real MSE gain | worst real MAE gain | best counterfactual MSE gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0 | 0.0000 | 0.000229 | 0.000549 | 0.001711 |

Failure reasons:

```text
window_pass_rate;real_mse_gain;real_mae_gain;counterfactual;reject_rate
```

## 3. 初步诊断

这个 run 不支持 Stage III V2.5 的性能 claim。

原因：

1. rolling validation 5 个窗口全失败；
2. test MSE/MAE 只有极小正收益，低于 claim gate 阈值；
3. `amplified_2x` 和 `amplified_5x` 与 real 几乎完全相同，说明当前 normalize/clamp/generator 路径对幅值不敏感；
4. real reject rate 达到 82.6%，远高于 20% 阈值；
5. risk gate 拒绝了 13859/16783 个 test steps，真实使用率只有 17.4%，不能称为稳定在线适应。

当前结论：

```text
b240/full-eval 没有改善 V2.5 的 claim status；相反，它更强地支持“Stage III V2.5 是审计性诊断模块，而不是已验证性能贡献”。
```

## 4. 第二个 full-eval b240 run

运行：

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v2_sensitivity_audit.py `
  --config configs\stage3_v2_strong.yaml `
  --dataset tetouan `
  --run-id codex_stage3_v25_matrix_b240_pub3_seed42_tetouan_fg2_20260525 `
  --seed 42 `
  --epochs 5 --invariant-epochs 5 --generator-epochs 10 `
  --device auto `
  --modes real zero shuffled sign_flipped amplified_2x amplified_5x random_matched_norm `
  --max-train-batches 240 --max-generator-batches 240 `
  --validation-windows 5 `
  --drift-source delta_phi_v --drift-polarity 1.0 `
  --signal-mode delta_vector --normalize-vector
```

输出目录：

```text
logs/load_benchmarks/stage3_v2/codex_stage3_v25_matrix_b240_pub3_seed42_tetouan_fg2_20260525/tetouan/seed_42
checkpoints/stage3_v2/tetouan/codex_stage3_v25_matrix_b240_pub3_seed42_tetouan_fg2_20260525/seed_42
```

耗时：

```text
约 2 小时 19 分钟，单 GPU RTX 5060，full test sample count = 18312。
```

### 4.1 Claim gate

| dataset | seed | budget | selected epoch | rolling pass rate | test gate | failure reasons |
| --- | ---: | --- | ---: | ---: | --- | --- |
| tetouan | 42 | b240 | 5 | 0.0000 | fail | `real_mse_gain;real_mae_gain;counterfactual;reject_rate` |

### 4.2 Test report-only metrics

| mode | base MSE | Stage III MSE | base MAE | Stage III MAE | use rate | reject risk count | reject shift count | pred shift | delta norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| real | 0.102572 | 0.102549 | 0.206629 | 0.206592 | 0.0009 | 11370 | 6926 | 0.226778 | 1.950653 |
| zero | 0.102572 | 0.102572 | 0.206629 | 0.206629 | 0.0733 | 16970 | 0 | 0.000000 | 0.000000 |
| shuffled | 0.102572 | 0.102569 | 0.206629 | 0.206627 | 0.0009 | 12763 | 5532 | 0.202130 | 1.862374 |
| sign_flipped | 0.102572 | 0.102571 | 0.206629 | 0.206625 | 0.0001 | 10925 | 7385 | 0.233375 | 1.950653 |
| amplified_2x | 0.102572 | 0.102549 | 0.206629 | 0.206592 | 0.0009 | 11370 | 6926 | 0.226778 | 1.950654 |
| amplified_5x | 0.102572 | 0.102549 | 0.206629 | 0.206592 | 0.0009 | 11370 | 6926 | 0.226778 | 1.950654 |
| random_matched_norm | 0.102572 | 0.102562 | 0.206629 | 0.206599 | 0.0007 | 11698 | 6601 | 0.229000 | 1.919602 |

Gate summary:

```text
real_mse_gain = 0.000225
real_mae_gain = 0.000180
best_counterfactual_mode = amplified_5x
real_vs_best_counterfactual_mse_gain_margin = approximately 0
real_reject_rate = 0.9991
```

### 4.3 Rolling validation

| validation windows | pass count | pass rate | worst real MSE gain | worst real MAE gain | best counterfactual MSE gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0 | 0.0000 | 0.000000 | 0.000000 | 0.000000001 |

Failure reasons:

```text
window_pass_rate;real_mse_gain;real_mae_gain;counterfactual;reject_rate
```

### 4.4 Tetouan 诊断

这个 run 比 Panama 更不支持性能 claim。

原因：

1. rolling validation 5 个窗口全失败；
2. test MSE/MAE 收益约为 0.0225%/0.0180%，远低于可支撑性能 claim 的幅度；
3. `amplified_2x` 和 `amplified_5x` 与 real 几乎完全相同，继续说明当前路径对 drift 幅值缺乏有效敏感性；
4. real reject rate 达到 99.91%，Stage III 在几乎全部 test steps 被 risk/shift gate 拒绝；
5. 真实使用率只有 0.087%，不能声称该机制在 Tetouan 上形成了有效在线适应。

当前 public-2 seed42 结论：

```text
Panama 和 Tetouan 的 b240/full-eval 都失败。两者都出现 rolling pass rate = 0、test claim gate fail、real 与 amplified counterfactual 几乎并列，以及 reject rate 过高的问题。
```

## 5. 第三个 full-eval b240 run

运行：

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v2_sensitivity_audit.py `
  --config configs\stage3_v2_strong.yaml `
  --dataset spain_energy_weather `
  --run-id codex_stage3_v25_matrix_b240_pub3_seed42_spain_fg_20260525 `
  --seed 42 `
  --epochs 5 --invariant-epochs 5 --generator-epochs 10 `
  --device auto `
  --modes real zero shuffled sign_flipped amplified_2x amplified_5x random_matched_norm `
  --max-train-batches 240 --max-generator-batches 240 `
  --validation-windows 5 `
  --drift-source delta_phi_v --drift-polarity 1.0 `
  --signal-mode delta_vector --normalize-vector
```

输出目录：

```text
logs/load_benchmarks/stage3_v2/codex_stage3_v25_matrix_b240_pub3_seed42_spain_fg_20260525/spain_energy_weather/seed_42
checkpoints/stage3_v2/spain_energy_weather/codex_stage3_v25_matrix_b240_pub3_seed42_spain_fg_20260525/seed_42
```

耗时：

```text
约 1 小时 33 分钟，单 GPU RTX 5060，full test sample count = 12238。
```

### 5.1 Claim gate

| dataset | seed | budget | selected epoch | rolling pass rate | test gate | failure reasons |
| --- | ---: | --- | ---: | ---: | --- | --- |
| spain_energy_weather | 42 | b240 | 2 | 0.0000 | fail | `real_mse_gain;counterfactual;reject_rate` |

### 5.2 Test report-only metrics

| mode | base MSE | Stage III MSE | base MAE | Stage III MAE | use rate | reject risk count | reject shift count | pred shift | delta norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| real | 0.224890 | 0.224017 | 0.346140 | 0.345010 | 0.6754 | 1603 | 2370 | 0.168283 | 0.801873 |
| zero | 0.224890 | 0.224890 | 0.346140 | 0.346140 | 0.2290 | 9435 | 0 | 0.000000 | 0.000000 |
| shuffled | 0.224890 | 0.226734 | 0.346140 | 0.348075 | 0.6839 | 1860 | 2009 | 0.156426 | 0.780208 |
| sign_flipped | 0.224890 | 0.229551 | 0.346140 | 0.350743 | 0.6370 | 2179 | 2263 | 0.167256 | 0.801873 |
| amplified_2x | 0.224890 | 0.224017 | 0.346140 | 0.345010 | 0.6754 | 1603 | 2370 | 0.168283 | 0.801873 |
| amplified_5x | 0.224890 | 0.224017 | 0.346140 | 0.345010 | 0.6754 | 1603 | 2370 | 0.168283 | 0.801873 |
| random_matched_norm | 0.224890 | 0.226750 | 0.346140 | 0.347775 | 0.6855 | 1778 | 2071 | 0.157781 | 0.787930 |

Gate summary:

```text
real_mse_gain = 0.003882
real_mae_gain = 0.003264
best_counterfactual_mode = amplified_2x
real_vs_best_counterfactual_mse_gain_margin = approximately 0
real_reject_rate = 0.3246
```

### 5.3 Rolling validation

| validation windows | pass count | pass rate | worst real MSE gain | worst real MAE gain | best counterfactual MSE gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0 | 0.0000 | 0.004301 | 0.002761 | 0.010325 |

Failure reasons:

```text
window_pass_rate;real_mse_gain;counterfactual;reject_rate
```

### 5.4 Spain 诊断

Spain 的结果比 Panama/Tetouan 更复杂。

支持信号：

1. `real` 明显优于 `shuffled`、`sign_flipped` 和 `random_matched_norm`；
2. `sign_flipped` 使 MSE/MAE 明显恶化，说明方向信息至少在该数据集上有一定意义；
3. real use rate 达到 67.5%，不像 Tetouan 那样几乎完全被拒绝。

仍然不能支撑性能 claim 的原因：

1. test MSE/MAE 收益只有约 0.388%/0.326%，仍低于强性能 claim 的证据阈值；
2. `amplified_2x`、`amplified_5x` 与 real 几乎完全相同，说明幅值敏感性仍不足；
3. rolling validation 5 个窗口全失败，不能说明机制在时间窗口上稳定；
4. reject rate 约 32.46%，仍高于协议阈值；
5. best counterfactual gain 与 real 几乎并列，claim gate 不能区分“真实 drift 条件适应”和“幅值被归一化/裁剪后的类似扰动”。

当前 public-3 seed42 结论：

| dataset | test gate | rolling pass rate | real MSE gain | real MAE gain | reject rate | strongest warning |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| panama | fail | 0.0000 | 0.000858 | 0.001118 | 0.8258 | reject rate 高，real 与 amplified 并列 |
| tetouan | fail | 0.0000 | 0.000225 | 0.000180 | 0.9991 | 几乎完全不使用 Stage III |
| spain_energy_weather | fail | 0.0000 | 0.003882 | 0.003264 | 0.3246 | 有方向信号，但收益小且 amplified 并列 |

```text
public-3 seed42 的 b240/full-eval 全部未通过 Stage III claim gate。当前证据不支持“Stage III 已经提供稳定、显著、可归因的在线学习性能贡献”。更可辩护的说法是：V2.5 修复后开始具备可诊断的 drift-conditioned pathway，但其有效性高度依赖数据集，并且尚未通过反事实敏感性、rolling validation 稳定性和 reject-rate 协议检验。
```

## 6. 多 seed b240 追加：Panama seed43

运行：

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v2_sensitivity_audit.py `
  --config configs\stage3_v2_strong.yaml `
  --dataset panama `
  --run-id codex_stage3_v25_matrix_b240_pub3_seed43_panama_fg_20260525 `
  --seed 43 `
  --epochs 5 --invariant-epochs 5 --generator-epochs 10 `
  --device auto `
  --modes real zero shuffled sign_flipped amplified_2x amplified_5x random_matched_norm `
  --max-train-batches 240 --max-generator-batches 240 `
  --validation-windows 5 `
  --drift-source delta_phi_v --drift-polarity 1.0 `
  --signal-mode delta_vector --normalize-vector
```

输出目录：

```text
logs/load_benchmarks/stage3_v2/codex_stage3_v25_matrix_b240_pub3_seed43_panama_fg_20260525/panama/seed_43
checkpoints/stage3_v2/panama/codex_stage3_v25_matrix_b240_pub3_seed43_panama_fg_20260525/seed_43
```

耗时：

```text
约 2 小时 19 分钟，单 GPU RTX 5060，full test sample count = 16783。
```

### 6.1 Claim gate

| dataset | seed | budget | selected epoch | rolling pass rate | test gate | failure reasons |
| --- | ---: | --- | ---: | ---: | --- | --- |
| panama | 43 | b240 | 9 | 0.0000 | fail | `counterfactual;reject_rate` |

### 6.2 Test report-only metrics

| mode | base MSE | Stage III MSE | base MAE | Stage III MAE | use rate | reject risk count | reject shift count | pred shift | delta norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| real | 0.219434 | 0.215519 | 0.323105 | 0.320117 | 0.4196 | 4805 | 4936 | 0.207322 | 0.999890 |
| zero | 0.219434 | 0.219434 | 0.323105 | 0.323105 | 0.0166 | 16505 | 0 | 0.000000 | 0.000000 |
| shuffled | 0.219434 | 0.220935 | 0.323105 | 0.325048 | 0.4055 | 4591 | 5387 | 0.207956 | 0.987827 |
| sign_flipped | 0.219434 | 0.226309 | 0.323105 | 0.329186 | 0.3780 | 5496 | 4943 | 0.207684 | 0.999890 |
| amplified_2x | 0.219434 | 0.215519 | 0.323105 | 0.320117 | 0.4196 | 4805 | 4936 | 0.207322 | 0.999890 |
| amplified_5x | 0.219434 | 0.215519 | 0.323105 | 0.320117 | 0.4196 | 4805 | 4936 | 0.207322 | 0.999890 |
| random_matched_norm | 0.219434 | 0.221032 | 0.323105 | 0.324965 | 0.4340 | 4024 | 5475 | 0.209620 | 0.995284 |

Gate summary:

```text
real_mse_gain = 0.017843
real_mae_gain = 0.009248
best_counterfactual_mode = amplified_5x
real_vs_best_counterfactual_mse_gain_margin = approximately 0
real_reject_rate = 0.5804
```

### 6.3 Rolling validation

| validation windows | pass count | pass rate | worst real MSE gain | worst real MAE gain | best counterfactual MSE gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0 | 0.0000 | 0.023176 | 0.008280 | 0.028059 |

Failure reasons:

```text
window_pass_rate;counterfactual;reject_rate
```

### 6.4 Panama seed43 诊断

该 run 说明 seed42 的“极小收益”不是唯一情况：Panama seed43 出现了明显 report-only MSE/MAE 改善，且 `shuffled`、`sign_flipped`、`random_matched_norm` 都明显差于 real。

但它仍然不能支撑性能 claim：

1. rolling validation 5 个窗口全失败；
2. `amplified_2x` 和 `amplified_5x` 与 real 几乎完全一致，幅值敏感性仍未成立；
3. reject rate 为 58.04%，远高于协议阈值；
4. claim gate 失败原因是 `counterfactual;reject_rate`，说明“真实 drift 条件适应”的可归因性和可部署性仍未过关。

当前多 seed 状态：

| dataset | seed | test gate | rolling pass rate | real MSE gain | real MAE gain | reject rate | note |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| panama | 42 | fail | 0.0000 | 0.000858 | 0.001118 | 0.8258 | 小收益，高拒绝 |
| panama | 43 | fail | 0.0000 | 0.017843 | 0.009248 | 0.5804 | report-only 改善明显，但 counterfactual/reject 失败 |
| tetouan | 42 | fail | 0.0000 | 0.000225 | 0.000180 | 0.9991 | 几乎不使用 Stage III |
| spain_energy_weather | 42 | fail | 0.0000 | 0.003882 | 0.003264 | 0.3246 | 有方向信号，但 amplified 并列 |

```text
Panama seed43 提供了“机制可能在某些随机初始化下产生可见 report-only gain”的信号，但没有改变主结论：Stage III V2.5 仍未通过 claim-gated evidence，尤其未通过 rolling validation、counterfactual magnitude sensitivity 和 reject-rate 检验。
```

## 7. 多 seed b240 追加：Tetouan seed43

运行：

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v2_sensitivity_audit.py `
  --config configs\stage3_v2_strong.yaml `
  --dataset tetouan `
  --run-id codex_stage3_v25_matrix_b240_pub3_seed43_tetouan_fg_20260525 `
  --seed 43 `
  --epochs 5 --invariant-epochs 5 --generator-epochs 10 `
  --device auto `
  --modes real zero shuffled sign_flipped amplified_2x amplified_5x random_matched_norm `
  --max-train-batches 240 --max-generator-batches 240 `
  --validation-windows 5 `
  --drift-source delta_phi_v --drift-polarity 1.0 `
  --signal-mode delta_vector --normalize-vector
```

输出目录：

```text
logs/load_benchmarks/stage3_v2/codex_stage3_v25_matrix_b240_pub3_seed43_tetouan_fg_20260525/tetouan/seed_43
checkpoints/stage3_v2/tetouan/codex_stage3_v25_matrix_b240_pub3_seed43_tetouan_fg_20260525/seed_43
```

耗时：

```text
约 2 小时 21 分钟，单 GPU RTX 5060，full test sample count = 18312。
```

### 7.1 Claim gate

| dataset | seed | budget | selected epoch | rolling pass rate | test gate | failure reasons |
| --- | ---: | --- | ---: | ---: | --- | --- |
| tetouan | 43 | b240 | 3 | 0.0000 | fail | `real_mse_gain;real_mae_gain;counterfactual;reject_rate` |

### 7.2 Test report-only metrics

| mode | base MSE | Stage III MSE | base MAE | Stage III MAE | use rate | reject risk count | reject shift count | pred shift | delta norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| real | 0.104233 | 0.104252 | 0.211588 | 0.211639 | 0.0096 | 14787 | 3349 | 0.150431 | 1.043842 |
| zero | 0.104233 | 0.104233 | 0.211588 | 0.211588 | 0.0144 | 18048 | 0 | 0.000000 | 0.000000 |
| shuffled | 0.104233 | 0.104253 | 0.211588 | 0.211610 | 0.0083 | 14247 | 3913 | 0.168818 | 1.119248 |
| sign_flipped | 0.104233 | 0.104257 | 0.211588 | 0.211584 | 0.0130 | 15120 | 2954 | 0.151296 | 1.043842 |
| amplified_2x | 0.104233 | 0.104252 | 0.211588 | 0.211639 | 0.0096 | 14787 | 3349 | 0.150431 | 1.043842 |
| amplified_5x | 0.104233 | 0.104252 | 0.211588 | 0.211639 | 0.0096 | 14787 | 3349 | 0.150431 | 1.043842 |
| random_matched_norm | 0.104233 | 0.104239 | 0.211588 | 0.211595 | 0.0049 | 15418 | 2805 | 0.133636 | 1.011458 |

Gate summary:

```text
real_mse_gain = -0.000181
real_mae_gain = -0.000243
best_counterfactual_mode = zero
real_vs_best_counterfactual_mse_gain_margin = -0.000181
real_reject_rate = 0.9904
```

### 7.3 Rolling validation

| validation windows | pass count | pass rate | worst real MSE gain | worst real MAE gain | best counterfactual MSE gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0 | 0.0000 | -0.000034 | -0.000083 | 0.000083 |

Failure reasons:

```text
window_pass_rate;real_mse_gain;real_mae_gain;counterfactual;reject_rate
```

### 7.4 Tetouan seed43 诊断

Tetouan seed43 复现了 seed42 的核心失败模式。

1. real MSE/MAE 都略微退化；
2. Stage III 实际使用率只有 0.96%；
3. real reject rate 达到 99.04%；
4. rolling validation 5 个窗口全失败；
5. zero drift 是 best counterfactual，说明真实 drift 在该 seed 上没有提供有效条件信息。

当前多 seed 状态：

| dataset | seed | test gate | rolling pass rate | real MSE gain | real MAE gain | reject rate | note |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| panama | 42 | fail | 0.0000 | 0.000858 | 0.001118 | 0.8258 | 小收益，高拒绝 |
| panama | 43 | fail | 0.0000 | 0.017843 | 0.009248 | 0.5804 | report-only 改善明显，但 counterfactual/reject 失败 |
| tetouan | 42 | fail | 0.0000 | 0.000225 | 0.000180 | 0.9991 | 几乎不使用 Stage III |
| tetouan | 43 | fail | 0.0000 | -0.000181 | -0.000243 | 0.9904 | 退化，几乎不使用 Stage III |
| spain_energy_weather | 42 | fail | 0.0000 | 0.003882 | 0.003264 | 0.3246 | 有方向信号，但 amplified 并列 |

```text
Tetouan 的两 seed 都说明：当前 V2.5 risk/shift gating 和 generator 输出无法形成可部署的在线适应路径；即使修复了原始实现问题，该数据集上仍不能声称 Stage III 有有效收益。
```

## 8. 多 seed b240 追加：Spain seed43

运行：

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v2_sensitivity_audit.py `
  --config configs\stage3_v2_strong.yaml `
  --dataset spain_energy_weather `
  --run-id codex_stage3_v25_matrix_b240_pub3_seed43_spain_fg_20260525 `
  --seed 43 `
  --epochs 5 --invariant-epochs 5 --generator-epochs 10 `
  --device auto `
  --modes real zero shuffled sign_flipped amplified_2x amplified_5x random_matched_norm `
  --max-train-batches 240 --max-generator-batches 240 `
  --validation-windows 5 `
  --drift-source delta_phi_v --drift-polarity 1.0 `
  --signal-mode delta_vector --normalize-vector
```

输出目录：

```text
logs/load_benchmarks/stage3_v2/codex_stage3_v25_matrix_b240_pub3_seed43_spain_fg_20260525/spain_energy_weather/seed_43
checkpoints/stage3_v2/spain_energy_weather/codex_stage3_v25_matrix_b240_pub3_seed43_spain_fg_20260525/seed_43
```

耗时：

```text
约 1 小时 36 分钟，单 GPU RTX 5060，full test sample count = 12238。
```

### 8.1 Claim gate

| dataset | seed | budget | selected epoch | rolling pass rate | test gate | failure reasons |
| --- | ---: | --- | ---: | ---: | --- | --- |
| spain_energy_weather | 43 | b240 | 4 | 0.0000 | fail | `real_mse_gain;real_mae_gain;counterfactual;reject_rate` |

### 8.2 Test report-only metrics

| mode | base MSE | Stage III MSE | base MAE | Stage III MAE | use rate | reject risk count | reject shift count | pred shift | delta norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| real | 0.235682 | 0.235699 | 0.350616 | 0.350626 | 0.0013 | 12172 | 50 | 0.081733 | 0.351444 |
| zero | 0.235682 | 0.235682 | 0.350616 | 0.350616 | 1.0000 | 0 | 0 | 0.000000 | 0.000000 |
| shuffled | 0.235682 | 0.235731 | 0.350616 | 0.350655 | 0.0059 | 12145 | 21 | 0.078067 | 0.343116 |
| sign_flipped | 0.235682 | 0.235843 | 0.350616 | 0.350735 | 0.0138 | 12032 | 37 | 0.081359 | 0.351444 |
| amplified_2x | 0.235682 | 0.235699 | 0.350616 | 0.350626 | 0.0013 | 12172 | 50 | 0.081733 | 0.351444 |
| amplified_5x | 0.235682 | 0.235699 | 0.350616 | 0.350626 | 0.0013 | 12172 | 50 | 0.081733 | 0.351444 |
| random_matched_norm | 0.235682 | 0.235741 | 0.350616 | 0.350679 | 0.0069 | 12121 | 33 | 0.080271 | 0.345710 |

Gate summary:

```text
real_mse_gain = -0.000075
real_mae_gain = -0.000029
best_counterfactual_mode = zero
real_vs_best_counterfactual_mse_gain_margin = -0.000075
real_reject_rate = 0.9987
```

### 8.3 Rolling validation

| validation windows | pass count | pass rate | worst real MSE gain | worst real MAE gain | best counterfactual MSE gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0 | 0.0000 | 0.000000 | -0.00000000006 | 0.001189 |

Failure reasons:

```text
window_pass_rate;real_mse_gain;real_mae_gain;counterfactual;reject_rate
```

### 8.4 Spain seed43 诊断

Spain seed43 没有复现 seed42 的正向信号。

1. real MSE/MAE 略微退化；
2. Stage III 实际使用率只有 0.13%；
3. real reject rate 达到 99.87%；
4. zero drift 是 best counterfactual；
5. rolling validation 5 个窗口全失败。

当前 seed42/43 汇总：

| dataset | seed | test gate | rolling pass rate | real MSE gain | real MAE gain | reject rate | note |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| panama | 42 | fail | 0.0000 | 0.000858 | 0.001118 | 0.8258 | 小收益，高拒绝 |
| panama | 43 | fail | 0.0000 | 0.017843 | 0.009248 | 0.5804 | report-only 改善明显，但 counterfactual/reject 失败 |
| tetouan | 42 | fail | 0.0000 | 0.000225 | 0.000180 | 0.9991 | 几乎不使用 Stage III |
| tetouan | 43 | fail | 0.0000 | -0.000181 | -0.000243 | 0.9904 | 退化，几乎不使用 Stage III |
| spain_energy_weather | 42 | fail | 0.0000 | 0.003882 | 0.003264 | 0.3246 | 有方向信号，但 amplified 并列 |
| spain_energy_weather | 43 | fail | 0.0000 | -0.000075 | -0.000029 | 0.9987 | 退化，几乎不使用 Stage III |

```text
seed42/43 public-3 b240/full-eval 共 6 个 run 全部未通过 claim gate，rolling pass rate 全部为 0。当前证据更强地支持降级 Stage III 性能叙事：它可以作为审计协议和候选机制保留，但不能作为已验证的稳定在线学习贡献。
```

## 9. 多 seed b240 追加：Panama seed44

运行：

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v2_sensitivity_audit.py `
  --config configs\stage3_v2_strong.yaml `
  --dataset panama `
  --run-id codex_stage3_v25_matrix_b240_pub3_seed44_panama_fg_20260525 `
  --seed 44 `
  --epochs 5 --invariant-epochs 5 --generator-epochs 10 `
  --device auto `
  --modes real zero shuffled sign_flipped amplified_2x amplified_5x random_matched_norm `
  --max-train-batches 240 --max-generator-batches 240 `
  --validation-windows 5 `
  --drift-source delta_phi_v --drift-polarity 1.0 `
  --signal-mode delta_vector --normalize-vector
```

输出目录：

```text
logs/load_benchmarks/stage3_v2/codex_stage3_v25_matrix_b240_pub3_seed44_panama_fg_20260525/panama/seed_44
checkpoints/stage3_v2/panama/codex_stage3_v25_matrix_b240_pub3_seed44_panama_fg_20260525/seed_44
```

耗时：

```text
约 2 小时 17 分钟，单 GPU RTX 5060，full test sample count = 16783。
```

### 9.1 Claim gate

| dataset | seed | budget | selected epoch | rolling pass rate | test gate | failure reasons |
| --- | ---: | --- | ---: | ---: | --- | --- |
| panama | 44 | b240 | 0 | 0.0000 | fail | `real_mse_gain;real_mae_gain;counterfactual;reject_rate` |

### 9.2 Test report-only metrics

| mode | base MSE | Stage III MSE | base MAE | Stage III MAE | use rate | reject risk count | reject shift count | pred shift | delta norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| real | 0.215734 | 0.215733 | 0.323364 | 0.323361 | 0.0002 | 16780 | 0 | 0.056670 | 0.274689 |
| zero | 0.215734 | 0.215734 | 0.323364 | 0.323364 | 0.0000 | 16783 | 0 | 0.000000 | 0.000000 |
| shuffled | 0.215734 | 0.215721 | 0.323364 | 0.323344 | 0.0011 | 16764 | 0 | 0.054728 | 0.264810 |
| sign_flipped | 0.215734 | 0.215736 | 0.323364 | 0.323278 | 0.0074 | 16659 | 0 | 0.056689 | 0.274689 |
| amplified_2x | 0.215734 | 0.215733 | 0.323364 | 0.323361 | 0.0002 | 16780 | 0 | 0.056670 | 0.274689 |
| amplified_5x | 0.215734 | 0.215733 | 0.323364 | 0.323361 | 0.0002 | 16780 | 0 | 0.056670 | 0.274689 |
| random_matched_norm | 0.215734 | 0.215738 | 0.323364 | 0.323366 | 0.0005 | 16774 | 0 | 0.052975 | 0.260377 |

Gate summary:

```text
real_mse_gain = 0.00000495
real_mae_gain = 0.00001155
best_counterfactual_mode = shuffled
real_vs_best_counterfactual_mse_gain_margin = -0.0000540
real_reject_rate = 0.9998
```

### 9.3 Rolling validation

| validation windows | pass count | pass rate | worst real MSE gain | worst real MAE gain | best counterfactual MSE gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0 | 0.0000 | 0.000000 | 0.000000 | 0.000329 |

Failure reasons:

```text
window_pass_rate;real_mse_gain;real_mae_gain;counterfactual;reject_rate
```

### 9.4 Panama seed44 诊断

Panama seed44 没有复现 seed43 的明显 report-only 改善。

1. real MSE/MAE 改善接近 0；
2. Stage III 实际使用率只有 0.018%；
3. real reject rate 达到 99.98%；
4. best counterfactual 是 `shuffled`，真实 drift 并不占优；
5. rolling validation 5 个窗口全失败。

当前 seed42/43/44 汇总：

| dataset | seed | test gate | rolling pass rate | real MSE gain | real MAE gain | reject rate | note |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| panama | 42 | fail | 0.0000 | 0.000858 | 0.001118 | 0.8258 | 小收益，高拒绝 |
| panama | 43 | fail | 0.0000 | 0.017843 | 0.009248 | 0.5804 | report-only 改善明显，但 counterfactual/reject 失败 |
| panama | 44 | fail | 0.0000 | 0.000005 | 0.000012 | 0.9998 | 几乎不使用 Stage III，shuffled 更好 |
| tetouan | 42 | fail | 0.0000 | 0.000225 | 0.000180 | 0.9991 | 几乎不使用 Stage III |
| tetouan | 43 | fail | 0.0000 | -0.000181 | -0.000243 | 0.9904 | 退化，几乎不使用 Stage III |
| spain_energy_weather | 42 | fail | 0.0000 | 0.003882 | 0.003264 | 0.3246 | 有方向信号，但 amplified 并列 |
| spain_energy_weather | 43 | fail | 0.0000 | -0.000075 | -0.000029 | 0.9987 | 退化，几乎不使用 Stage III |

```text
Panama seed44 说明 seed43 的明显 report-only gain 不稳定。当前 7 个 b240/full-eval run 全部 fail，rolling pass rate 全部为 0。
```

## 10. 多 seed b240 追加：Tetouan seed44

运行：

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v2_sensitivity_audit.py `
  --config configs\stage3_v2_strong.yaml `
  --dataset tetouan `
  --run-id codex_stage3_v25_matrix_b240_pub3_seed44_tetouan_fg_20260525 `
  --seed 44 `
  --epochs 5 --invariant-epochs 5 --generator-epochs 10 `
  --device auto `
  --modes real zero shuffled sign_flipped amplified_2x amplified_5x random_matched_norm `
  --max-train-batches 240 --max-generator-batches 240 `
  --validation-windows 5 `
  --drift-source delta_phi_v --drift-polarity 1.0 `
  --signal-mode delta_vector --normalize-vector
```

输出目录：

```text
logs/load_benchmarks/stage3_v2/codex_stage3_v25_matrix_b240_pub3_seed44_tetouan_fg_20260525/tetouan/seed_44
checkpoints/stage3_v2/tetouan/codex_stage3_v25_matrix_b240_pub3_seed44_tetouan_fg_20260525/seed_44
```

耗时：

```text
约 2 小时 26 分钟，单 GPU RTX 5060，full test sample count = 18312。
```

### 10.1 Claim gate

| dataset | seed | budget | selected epoch | rolling pass rate | test gate | failure reasons |
| --- | ---: | --- | ---: | ---: | --- | --- |
| tetouan | 44 | b240 | 0 | 0.0000 | fail | `real_mse_gain;counterfactual;reject_rate` |

### 10.2 Test report-only metrics

| mode | base MSE | Stage III MSE | base MAE | Stage III MAE | use rate | reject risk count | reject shift count | pred shift | delta norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| real | 0.101357 | 0.101014 | 0.202636 | 0.202134 | 0.1690 | 15217 | 0 | 0.038525 | 0.356240 |
| zero | 0.101357 | 0.101357 | 0.202636 | 0.202636 | 0.0492 | 17411 | 0 | 0.000000 | 0.000000 |
| shuffled | 0.101357 | 0.101537 | 0.202636 | 0.203028 | 0.1501 | 15564 | 0 | 0.036391 | 0.342422 |
| sign_flipped | 0.101357 | 0.102006 | 0.202636 | 0.203664 | 0.1317 | 15900 | 0 | 0.038345 | 0.356240 |
| amplified_2x | 0.101357 | 0.101014 | 0.202636 | 0.202134 | 0.1690 | 15217 | 0 | 0.038525 | 0.356240 |
| amplified_5x | 0.101357 | 0.101014 | 0.202636 | 0.202134 | 0.1690 | 15217 | 0 | 0.038525 | 0.356240 |
| random_matched_norm | 0.101357 | 0.101494 | 0.202636 | 0.202933 | 0.1449 | 15659 | 0 | 0.035890 | 0.340649 |

Gate summary:

```text
real_mse_gain = 0.003384
real_mae_gain = 0.002480
best_counterfactual_mode = amplified_2x
real_vs_best_counterfactual_mse_gain_margin = approximately 0
real_reject_rate = 0.8310
```

### 10.3 Rolling validation

| validation windows | pass count | pass rate | worst real MSE gain | worst real MAE gain | best counterfactual MSE gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0 | 0.0000 | 0.000367 | -0.000493 | 0.002012 |

Failure reasons:

```text
window_pass_rate;real_mse_gain;real_mae_gain;counterfactual;reject_rate
```

### 10.4 Tetouan seed44 诊断

Tetouan seed44 出现了正向 report-only MSE/MAE 收益，但仍未通过 claim gate。

1. rolling validation 5 个窗口全失败；
2. reject rate 为 83.10%，仍远高于协议阈值；
3. `amplified_2x/5x` 与 real 几乎完全并列；
4. validation worst-window MAE gain 为负，稳定性不足。

当前 seed42/43/44 汇总：

| dataset | seed | test gate | rolling pass rate | real MSE gain | real MAE gain | reject rate | note |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| panama | 42 | fail | 0.0000 | 0.000858 | 0.001118 | 0.8258 | 小收益，高拒绝 |
| panama | 43 | fail | 0.0000 | 0.017843 | 0.009248 | 0.5804 | report-only 改善明显，但 counterfactual/reject 失败 |
| panama | 44 | fail | 0.0000 | 0.000005 | 0.000012 | 0.9998 | 几乎不使用 Stage III，shuffled 更好 |
| tetouan | 42 | fail | 0.0000 | 0.000225 | 0.000180 | 0.9991 | 几乎不使用 Stage III |
| tetouan | 43 | fail | 0.0000 | -0.000181 | -0.000243 | 0.9904 | 退化，几乎不使用 Stage III |
| tetouan | 44 | fail | 0.0000 | 0.003384 | 0.002480 | 0.8310 | 有收益但 reject/counterfactual 失败 |
| spain_energy_weather | 42 | fail | 0.0000 | 0.003882 | 0.003264 | 0.3246 | 有方向信号，但 amplified 并列 |
| spain_energy_weather | 43 | fail | 0.0000 | -0.000075 | -0.000029 | 0.9987 | 退化，几乎不使用 Stage III |

```text
Tetouan 三个 seed 的共同结论不是“绝对没有数值收益”，而是“无法通过可归因和可部署 gate”。尤其 amplified drift 与 real 并列、rolling validation 全失败、reject rate 过高的问题仍然稳定存在。
```

## 11. 多 seed b240 追加：Spain seed44

运行：

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v2_sensitivity_audit.py `
  --config configs\stage3_v2_strong.yaml `
  --dataset spain_energy_weather `
  --run-id codex_stage3_v25_matrix_b240_pub3_seed44_spain_fg_20260525 `
  --seed 44 `
  --epochs 5 --invariant-epochs 5 --generator-epochs 10 `
  --device auto `
  --modes real zero shuffled sign_flipped amplified_2x amplified_5x random_matched_norm `
  --max-train-batches 240 --max-generator-batches 240 `
  --validation-windows 5 `
  --drift-source delta_phi_v --drift-polarity 1.0 `
  --signal-mode delta_vector --normalize-vector
```

输出目录：

```text
logs/load_benchmarks/stage3_v2/codex_stage3_v25_matrix_b240_pub3_seed44_spain_fg_20260525/spain_energy_weather/seed_44
checkpoints/stage3_v2/spain_energy_weather/codex_stage3_v25_matrix_b240_pub3_seed44_spain_fg_20260525/seed_44
```

耗时：

```text
约 1 小时 39 分钟，单 GPU RTX 5060，full test sample count = 12238。
```

### 11.1 Claim gate

| dataset | seed | budget | selected epoch | rolling pass rate | test gate | failure reasons |
| --- | ---: | --- | ---: | ---: | --- | --- |
| spain_energy_weather | 44 | b240 | 7 | 0.0000 | fail | `real_mse_gain;real_mae_gain;counterfactual;reject_rate` |

### 11.2 Test report-only metrics

| mode | base MSE | Stage III MSE | base MAE | Stage III MAE | use rate | reject risk count | reject shift count | pred shift | delta norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| real | 0.226007 | 0.225419 | 0.342695 | 0.342115 | 0.0854 | 7096 | 4097 | 0.221254 | 1.117543 |
| zero | 0.226007 | 0.226007 | 0.342695 | 0.342695 | 1.0000 | 0 | 0 | 0.000000 | 0.000000 |
| shuffled | 0.226007 | 0.226323 | 0.342695 | 0.343035 | 0.0728 | 7253 | 4094 | 0.216243 | 1.107366 |
| sign_flipped | 0.226007 | 0.227224 | 0.342695 | 0.343582 | 0.0910 | 6983 | 4141 | 0.220037 | 1.117543 |
| amplified_2x | 0.226007 | 0.225419 | 0.342695 | 0.342115 | 0.0854 | 7096 | 4097 | 0.221254 | 1.117543 |
| amplified_5x | 0.226007 | 0.225419 | 0.342695 | 0.342115 | 0.0854 | 7096 | 4097 | 0.221254 | 1.117543 |
| random_matched_norm | 0.226007 | 0.226373 | 0.342695 | 0.343001 | 0.0890 | 7255 | 3894 | 0.213953 | 1.112073 |

Gate summary:

```text
real_mse_gain = 0.002603
real_mae_gain = 0.001693
best_counterfactual_mode = amplified_5x
real_vs_best_counterfactual_mse_gain_margin = approximately 0
real_reject_rate = 0.9146
```

### 11.3 Rolling validation

| validation windows | pass count | pass rate | worst real MSE gain | worst real MAE gain | best counterfactual MSE gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0 | 0.0000 | 0.003608 | 0.001154 | 0.006734 |

Failure reasons:

```text
window_pass_rate;real_mse_gain;counterfactual;reject_rate;real_mae_gain
```

### 11.4 Spain seed44 诊断

Spain seed44 与 seed42 类似，出现了正向 report-only 收益和 real-vs-shuffled/sign-flipped 方向信号，但仍未通过 claim gate。

1. rolling validation 5 个窗口全失败；
2. reject rate 为 91.46%，真实使用率只有 8.54%；
3. `amplified_2x/5x` 与 real 几乎完全并列；
4. best window counterfactual MSE gain 高于 worst-window real gain。

## 12. 第一轮三数据集三 seed 汇总

本轮完成了 9 个 b240/full-eval run：

| dataset | seeds | gate pass | mean MSE gain | mean MAE gain | mean reject rate | summary |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| panama | 42, 43, 44 | 0/3 | 0.006235 | 0.003459 | 0.8020 | seed43 有明显 report-only gain，但 seed44 不复现 |
| tetouan | 42, 43, 44 | 0/3 | 0.001143 | 0.000806 | 0.9402 | seed44 有收益，但整体高拒绝且 counterfactual 并列 |
| spain_energy_weather | 42, 43, 44 | 0/3 | 0.002137 | 0.001642 | 0.7460 | seed42/44 有方向信号，seed43 不复现 |
| all | 9 runs | 0/9 | 0.003172 | 0.001969 | 0.8294 | 全部 rolling pass rate = 0 |

核心结论：

```text
Stage III V2.5 在部分 seed 上可以产生 report-only MSE/MAE 改善，但 9/9 claim-gated full-eval runs 全部失败。失败不是单一原因，而是 rolling validation 全失败、reject rate 高、amplified counterfactual 与 real drift 并列、以及 seed 依赖性共同造成的。
```

因此，当前证据只支持以下保守叙事：

```text
V2.5 是一个比 legacy Stage III 更严格、更可复现、更可审计的 drift-conditioned adaptation 候选机制和负面/部分正面诊断协议；它尚未被证明为稳定、可归因、可部署的在线学习性能贡献。
```

## 13. 下一步

继续按相同协议运行：

1. 暂停把 Stage III 写成性能核心创新；
2. 集中修复 generator sensitivity、amplified drift 并列、risk gate 过度拒绝和 rolling validation 全失败问题；
3. 只有在这些机制问题有明确修复方案后，再启动 b480/b960/b1920，以检查更大训练预算是否能改善稳定性和可归因性。
