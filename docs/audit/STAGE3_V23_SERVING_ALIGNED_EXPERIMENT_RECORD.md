# Stage III V2.3 Serving-Aligned 修复与 bounded retraining 记录

日期：2026-05-25

本记录对应 run id：

```text
codex_stage3_v23_serving_aligned_smoke_20260525
codex_stage3_v23_serving_aligned_panama_b60_20260525
```

## 1. 本轮要回答的问题

本轮不是为了证明论文结论，而是为了验证四个代码层面的科研完整性问题是否得到控制：

1. `AdaptationGenerator` 是否真正接入训练和评估闭环；
2. generator 的训练目标是否和 online serving 时的 `clamp -> pred_shift -> risk_gate -> blend` 逻辑一致；
3. `phi_v - phi_i` 的表示空间不严格对齐问题是否通过 V2 residual/projection/temporal loss 进一步缓解；
4. eval 阶段 VAE 是否仍然随机，以及这种随机性是否被 V2 路径拒绝。

本轮只修改 V2 路径，不覆盖原始 Stage III、原始训练代码、原始日志或 checkpoint。

## 2. 代码修复摘要

| 文件 | 修复内容 | 科研意义 |
| --- | --- | --- |
| `train/stage3_eval_policy.py` | 新增 deterministic eval guard | 拒绝 legacy stochastic VAE 或 `vae_eval_mode=sample` 直接进入 Stage III V2 online/generator 评估 |
| `train/online_loop_stage3_v2.py` | online replay 入口调用 deterministic guard | 避免同一模型同一输入因为 VAE 采样产生不同预测 |
| `train/stage3_generator_training.py` | generator training 使用 candidate/blended/serving-aligned metrics，加入 soft reject proxy | 训练目标更接近线上实际生效的输出，而不是只优化 raw candidate |
| `train/stage3_risk_gate_v2.py` | 新增 `Stage3RiskGateLogistic`，checkpoint load/save 兼容 MLP/logistic | 先使用更简单、可解释、低容量的风险门，降低小样本 gate 过拟合 |
| `train/stage3_v2_artifacts.py` | 新增 artifact bundle loader | generator/risk gate/delta standardizer 保存后必须 reload 再评估，避免 train-time 对象和 deploy artifact 不一致 |
| `models/aligned_representation_v2.py` | 新增 deployed temporal ranking 与 temporal contrastive loss | 将漂移定义从 embedding level 推向相邻时刻 `delta_phi_v[t] - delta_phi_v[t-1]` 的部署信号 |
| `train/offline_aligned_representation_v2.py` | 训练配置和日志接入新增 loss | 保证新 loss 可以进入 offline aligned representation 训练 |
| `train/run_stage3_v2_sensitivity_audit.py` | 接入新 loss、logistic gate、artifact reload manifest | 评估前重新加载 artifact，并在 manifest 中记录 `stage3_artifacts_reloaded_for_eval=true` |
| `configs/stage3_v2_strong.yaml` | 启用 deployed temporal loss、soft reject、logistic gate | 形成 V2.3 strong bounded retraining 配置 |

## 3. 验证

目标测试：

```powershell
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_online_loop_stage3_v2.py tests\test_stage3_generator_training_v2.py tests\test_stage3_risk_gate_v2.py tests\test_aligned_representation_v2.py tests\test_stage3_v2_artifacts.py -q
```

结果：

```text
27 passed in 2.66s
```

Stage III V2 回归套件：

```powershell
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_factor_mechanism_v2.py tests\test_adaptation_v2.py tests\test_hyperlora_generator_v2.py tests\test_aligned_representation_v2.py tests\test_offline_aligned_representation_v2.py tests\test_stage3_functional_v2.py tests\test_stage3_generator_training_v2.py tests\test_stage3_risk_gate_v2.py tests\test_stage3_sensitivity_v2.py tests\test_stage3_sensitivity_gate_v2.py tests\test_online_loop_stage3_v2.py tests\test_stage3_v2_protocol_safety.py tests\test_stage3_v2_split_safe_preprocessing.py tests\test_stage3_v2_artifacts.py tests\test_stage3_signals.py tests\test_stage3_ablation_benchmark.py tests\test_shapes.py -q
```

结果：

```text
75 passed in 3.90s
```

## 4. Smoke run

命令：

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v2_sensitivity_audit.py --config configs\stage3_v2_strong.yaml --dataset panama --run-id codex_stage3_v23_serving_aligned_smoke_20260525 --seed 42 --epochs 1 --invariant-epochs 1 --generator-epochs 1 --device cpu --modes real zero --max-train-batches 2 --max-generator-batches 2 --max-eval-batches 2 --validation-windows 2 --drift-source delta_phi_v
```

用途：只验证 pipeline 串联，不用于效果判断。

关键 manifest 事实：

| 字段 | 值 |
| --- | --- |
| `stage3_artifacts_reloaded_for_eval` | `true` |
| `stage3_risk_gate_train_metrics.model_type` | `logistic` |
| `stage3_test_gate_role` | `report_only` |
| `stage3_validation_gate_role` | `selection` |
| `stage3_delta_calibrated` | `true` |

## 5. Panama b60 bounded retraining

协议：

```text
dataset: panama
seeds: 42, 43
base epochs: 2
aligned epochs: 2
generator epochs: 4
max_train_batches: 60
max_generator_batches: 60
max_eval_batches: 60
validation_windows: 3
modes: real, zero, shuffled, sign_flipped, random_matched_norm
test role: report-only
selection role: validation / rolling validation
```

命令 seed42：

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v2_sensitivity_audit.py --config configs\stage3_v2_strong.yaml --dataset panama --run-id codex_stage3_v23_serving_aligned_panama_b60_20260525 --seed 42 --epochs 2 --invariant-epochs 2 --generator-epochs 4 --device cpu --modes real zero shuffled sign_flipped random_matched_norm --max-train-batches 60 --max-generator-batches 60 --max-eval-batches 60 --validation-windows 3 --drift-source delta_phi_v
```

命令 seed43：

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v2_sensitivity_audit.py --config configs\stage3_v2_strong.yaml --dataset panama --run-id codex_stage3_v23_serving_aligned_panama_b60_20260525 --seed 43 --epochs 2 --invariant-epochs 2 --generator-epochs 4 --device cpu --modes real zero shuffled sign_flipped random_matched_norm --max-train-batches 60 --max-generator-batches 60 --max-eval-batches 60 --validation-windows 3 --drift-source delta_phi_v
```

## 6. 结果表

### 6.1 Claim gate

| seed | test claim gate | rolling validation gate | rolling pass rate | failed windows | selected threshold |
| --- | --- | --- | ---: | --- | ---: |
| 42 | fail | fail | 0.6667 | 0 | 0.50 |
| 43 | pass | fail | 0.3333 | 0;2 | 0.65 |

### 6.2 Report-only test

| seed | base MSE | Stage III MSE | real MSE gain | base MAE | Stage III MAE | real MAE gain | best counterfactual | real reject rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 42 | 0.328935 | 0.329735 | -0.2431% | 0.485584 | 0.487998 | -0.4971% | zero | 0.0000 |
| 43 | 0.317831 | 0.315353 | +0.7797% | 0.476784 | 0.475797 | +0.2071% | zero | 0.0500 |

两 seed 平均：

| metric | mean |
| --- | ---: |
| mean test MSE gain | +0.2683% |
| mean test MAE gain | -0.1450% |
| mean rolling pass rate | 0.5000 |

### 6.3 Rolling validation

| seed | worst-window MSE gain | worst-window MAE gain | best-window counterfactual MSE gain | failure reasons |
| --- | ---: | ---: | ---: | --- |
| 42 | -0.3551% | -0.4121% | 0.0000% | `window_pass_rate;real_mse_gain;real_mae_gain;counterfactual` |
| 43 | +0.4094% | +0.1666% | 0.0000% | `window_pass_rate;real_mse_gain;real_mae_gain;reject_rate` |

## 7. 与 V2.1/V2.2 的公平比较

历史 bounded b60 结果：

| version | seed | test MSE gain | test MAE gain | rolling pass rate | conclusion |
| --- | ---: | ---: | ---: | ---: | --- |
| V2.1 split-safe | 42 | +0.4785% | -0.2272% | 0.6667 | test MSE 正，但 MAE 和 rolling 失败 |
| V2.1 split-safe | 43 | +0.2498% | +0.1117% | 0.6667 | 小幅趋势，未过 gate |
| V2.2 risk-gate | 42 | +0.2668% | -0.2374% | 0.6667 | 未过 gate |
| V2.3 serving-aligned | 42 | -0.2431% | -0.4971% | 0.6667 | 退化 |
| V2.3 serving-aligned | 43 | +0.7797% | +0.2071% | 0.3333 | test 通过但 validation 不稳 |

判断：

1. V2.3 修复了若干协议问题，但没有形成稳定性能收益；
2. seed42 从 V2.1/V2.2 的小幅正 MSE 趋势变成负收益；
3. seed43 test 表现增强，但 rolling validation 明显变差；
4. 因为 test 是 report-only，不能用 seed43 的 test pass 反向证明配置有效；
5. 当前不能声称“修改后优于旧版”或“显著提升”。

## 8. 诊断

### 8.1 AdaptationGenerator 是否已经接入训练

V2.3 中 generator 已经接入训练：

- `train/stage3_generator_training.py` 对 generator 参数反向传播；
- tests 覆盖 generator 更新且冻结 base model；
- runner 保存 `generator.pt` 后，通过 `load_stage3_v2_artifact_bundle()` 重新加载再评估。

但这只说明“训练闭环存在”，不说明“训练到了可靠 drift-conditioned adaptation”。b60 结果说明后者仍未成立。

### 8.2 网络结构设计如何

当前 HyperLoRA 结构比 legacy generator 更可辩护：

- zero drift 映射到 zero delta；
- 每个目标矩阵有独立 low-rank adapter；
- drift conditioner 无 bias；
- `lora_b` 零初始化，初始行为接近 identity。

主要问题：

- b60 下仍存在 seed 依赖；
- risk gate 高拒绝率会让 seed43 test 看起来安全，但 rolling validation 不稳；
- `target_modules: output_head` 只更新输出头，表达能力和稳定性可能不足；
- generator 训练没有真正的跨窗口 checkpoint selection，目前仍可能过拟合早期训练流。

### 8.3 `phi_v - phi_i` 表示空间不严格对齐如何解决

V2.3 已做的解决：

1. 用 shared trunk + residual branch 取代两个独立 encoder；
2. `phi_v = phi_i + delta_phi_v`，使差值本身成为同一坐标系内的 residual；
3. hard projection 使 `delta_phi_v` 尽量正交于 `phi_i`；
4. 新增 deployed temporal ranking，把在线漂移信号对齐到 `delta_phi_v[t] - delta_phi_v[t-1]`；
5. 新增 temporal contrastive calibration，用 residual-change similarity 校准 drift-transition similarity。

仍未解决：

- 这些 loss 只是约束表示，不保证 generator 能把表示变化转成有益参数更新；
- 当前证据没有证明 real drift consistently 优于 counterfactual；
- 需要后续用 rolling validation 做 representation loss ablation，不能只看 test。

### 8.4 eval 阶段 VAE 随机到底对不对

对于 Stage III online sensitivity audit，本轮结论是：不对。

原因：

1. 同一模型、同一输入、同一 drift 下，eval 随机采样会让 base/candidate/blended 都有额外噪声；
2. sensitivity audit 要比较 real/zero/shuffled/sign-flipped drift，如果 VAE 本身随机，差异就不一定来自 drift；
3. 因此 V2 路径必须使用 deterministic eval，例如 `vae_eval_mode="mean"`。

例外：

- 如果论文明确研究 Monte Carlo predictive uncertainty，可以保留 sampling；
- 但那必须单独报告 MC seeds、采样次数、置信区间；
- 当前 Stage III drift-conditioned adaptation 不是这个设定。

## 9. 是否更新论文

不要更新论文中的性能 claim。

可以更新论文的方法学/审计描述，例如：

```text
We introduce a split-safe and counterfactual-audited Stage III V2 protocol that enforces deterministic evaluation, artifact reload before online replay, and validation-only threshold selection.
```

不能写：

```text
Stage III V2.3 significantly improves load forecasting accuracy.
Stage III V2.3 reliably outperforms SOTA under concept drift.
The generator has learned a robust concept-drift-conditioned adaptation policy.
```

更安全的重构方向：

```text
Stage III is currently better treated as an audited online adaptation mechanism and diagnostic framework. The evidence supports protocol rigor and reveals seed-dependent transient adaptation behavior, but does not yet support a strong accuracy or SOTA claim.
```

## 10. 下一步修改方案

Tier 1：必须先做

1. 不更新论文主结果表和结论 claim；
2. 保留 V2.3 代码，但把实验结论标记为 `not yet supported`；
3. 增加 generator checkpoint selection：不能只按 train loss 选，必须按 rolling validation 的 served-output gate 选；
4. 对 risk gate 增加 forward-chaining label collection，不要在整段 train stream 上混合拟合；
5. 记录每个 validation window 的 use/reject/shift 分布，定位 window 0/2 失败原因。

Tier 2：增强竞争力

1. 做 `target_modules` ablation：`output_head` vs `temporal_encoder/output_head`；
2. 做 representation loss ablation：hard projection only、+deployed temporal ranking、+contrastive；
3. 做 risk gate ablation：none、logistic、MLP；
4. 做 budget ablation：b60 vs b120，但前提是同一代码快照、同一 seed 成对运行；
5. 扩到 public-3 dataset x multi-seed 后再谈统计显著。

Tier 3：可选

1. 把 train/serve policy 抽成共享 helper，减少 future drift；
2. 给 generator loss 增加 rolling validation early stopping；
3. 额外报告 delta norm、pred shift、reject rate 的机制图；
4. 在论文附录中说明 V2.1/V2.2/V2.3 的负结果和审计价值。

## 11. 最小安全下一步

最小安全下一步不是继续调 test，而是：

```text
实现 generator checkpoint selection on rolling validation，并只用 validation 决定是否继续 b120。
```

如果 rolling validation 不能稳定 3/3 通过，论文中 Stage III 只能写成审计型/诊断型贡献，不能写成可靠性能提升贡献。
