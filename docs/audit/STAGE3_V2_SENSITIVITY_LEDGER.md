# Stage III V2 Sensitivity Ledger

日期：2026-05-25

本文用于记录 Stage III V2 的 drift sensitivity audit。早期段落保留了从 smoke 到 V2/V2.3 的历史演进；最新 V2.5 证据以 `docs/audit/STAGE3_V25_CANONICAL_CKPT_SELECTION_EXPERIMENT_RECORD.md` 为准。

当前总判断：

```text
V2.5 已修复多项机制和协议问题，但 Panama seed42/43 b60/b120 均未通过 rolling/test claim gate。
因此 Stage III V2.5 仍不能支撑显著提升、可靠 concept-drift learning 或 SOTA 表述。
```

最新结果摘要：

| seed | budget | test MSE gain | test MAE gain | rolling validation pass rate | test gate |
| ---: | --- | ---: | ---: | ---: | --- |
| 42 | b60 | -0.0938% | -0.3505% | 0.3333 | fail |
| 42 | b120 | +1.2656% | -0.4907% | 0.3333 | fail |
| 43 | b60 | -0.3250% | +0.1074% | 0.3333 | fail |
| 43 | b120 | +1.6361% | +1.1517% | 0.3333 | fail |

---

## 1. 必测干预模式

| 模式 | 含义 | 期望诊断意义 | 当前状态 |
| --- | --- | --- | --- |
| `real` | 使用真实 drift | 主实验条件 | 干预函数已实现，真实数据未跑 |
| `zero` | drift 全零 | 检查模型是否不依赖 drift 内容 | 干预函数已实现，真实数据未跑 |
| `shuffled` | 打乱 drift 的时间/维度对应 | 检查时间和样本对应关系是否被利用 | 干预函数已实现，真实数据未跑 |
| `sign_flipped` | drift 取负 | 检查方向/符号是否被利用 | 干预函数已实现，真实数据未跑 |
| `amplified_2x` | drift 放大 2 倍 | 检查幅度敏感性 | 干预函数已实现，真实数据未跑 |
| `amplified_5x` | drift 放大 5 倍 | 检查饱和和 clamp 主导风险 | 干预函数已实现，真实数据未跑 |
| `random_matched_norm` | 随机方向但保持每个样本 norm | 检查方向信息是否重要 | 干预函数已实现，真实数据未跑 |

---

## 2. 已验证的机制行为

已通过测试：

```powershell
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_stage3_sensitivity_v2.py -q
```

结果：

```text
2 passed
```

验证内容：

1. 所有必测模式都能生成。
2. `zero` 确实为全零。
3. `sign_flipped` 确实为 `-drift`。
4. `amplified_2x` 和 `amplified_5x` 确实按倍数放大。
5. `shuffled` 保留原始数值集合但改变排列。
6. `random_matched_norm` 保持每个样本的 drift norm。

---

## 3. 真实实验通过标准

Stage III V2 只有满足以下条件，才可以在论文中称为 drift-conditioned adaptation：

1. `real` 与 `zero` 在 `delta_norm`、`candidate_shift` 或 prediction trajectory 上有可测差异。
2. `real` 与 `shuffled` 不应几乎完全相同。
3. `sign_flipped` 应改变 delta 方向或预测轨迹。
4. `amplified_2x/5x` 应改变 pre-clamp delta magnitude。
5. `real` 的 MSE/MAE 不应劣于 `zero` 和 `shuffled`。
6. 以上比较必须在相同 checkpoint、相同 stream、相同 seed 下 paired 完成。

---

## 4. 当前结论

当前只能确认 sensitivity audit 的干预函数已具备。

不能确认：

- Stage III V2 已经真正利用 drift；
- Stage III V2 比 zero / shuffled drift 更好；
- Stage III V2 比旧 Stage III 更好；
- Stage III V2 已有统计显著提升。

这些必须等待真实训练和审计 CSV。

---

## 5. 已完成的真实数据 smoke

运行 ID：

```text
codex_stage3_v2_batch1_smoke_20260525
```

数据集：

```text
panama
```

限制：

- `--max-train-batches 1`
- `--max-generator-batches 1`
- `--max-eval-batches 1`

输出：

- `logs/load_benchmarks/stage3_v2/codex_stage3_v2_batch1_smoke_20260525/panama/seed_42/sensitivity_summary.csv`
- `logs/load_benchmarks/stage3_v2/codex_stage3_v2_batch1_smoke_20260525/panama/seed_42/run_manifest.json`
- `checkpoints/stage3_v2/panama/codex_stage3_v2_batch1_smoke_20260525/seed_42/generator.pt`

结果摘要：

| mode | base MSE | Stage III V2 MSE | base MAE | Stage III V2 MAE | delta norm | shift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| real | 0.4775509834 | 0.4775510728 | 0.5510783792 | 0.5510783195 | 0.0045360276 | 0.0000111030 |
| zero | 0.4775509834 | 0.4775509834 | 0.5510783792 | 0.5510783792 | 0.0000000000 | 0.0000000000 |

解释：

- `zero` 符合 identity：delta norm 和 prediction shift 都为 0。
- `real` 产生了非零 delta 和极小 prediction shift。
- 该 smoke 没有显示 MSE 改进，且规模太小，不能用于论文结论。
- 下一步必须运行不带 batch 限制或更大 batch 限制的中等规模实验。

---

## 6. HyperLoRA strong public-3 bounded sensitivity

状态更新：

```text
以下结果产生于协议修复之前，现已降级为 exploratory / invalidated-for-claim evidence。
```

原因：

- 旧 V2 runner 使用窗口先构造再切分；
- online drift 在 batch 内平均，存在未来样本泄漏；
- aligned representation smoothness 使用 shuffled batch adjacency。

配置：

- `configs/stage3_v2_strong.yaml`
- generator: `LayerAwareHyperLoRAGenerator`
- target: `output_head`
- rank: `8`
- max scale: `0.5`
- train/eval/generator batches: `100`
- seed: `42`

结果：

| dataset | mode | MSE gain | MAE gain | delta norm | prediction shift |
| --- | --- | ---: | ---: | ---: | ---: |
| panama | real | 3.5116% | 2.0096% | 0.600959 | 0.092379 |
| panama | zero | 0.0000% | 0.0000% | 0.000000 | 0.000000 |
| panama | shuffled | 0.1532% | 0.0867% | 0.182942 | 0.026707 |
| panama | sign_flipped | -3.6770% | -2.1343% | 0.600959 | 0.095199 |
| tetouan | real | 3.4614% | 2.4628% | 1.529015 | 0.107228 |
| tetouan | zero | 0.0000% | 0.0000% | 0.000000 | 0.000000 |
| tetouan | shuffled | 0.1026% | 0.0768% | 0.276501 | 0.017362 |
| tetouan | sign_flipped | 1.4711% | 0.8352% | 1.529015 | 0.105644 |
| spain_energy_weather | real | 1.8114% | 0.9843% | 1.053539 | 0.178580 |
| spain_energy_weather | zero | 0.0000% | -0.0000% | 0.000000 | 0.000000 |
| spain_energy_weather | shuffled | 0.6008% | 0.3519% | 0.342715 | 0.049568 |
| spain_energy_weather | sign_flipped | -1.0357% | -0.4177% | 1.053539 | 0.187205 |

解释：

- `real` 在三个数据集上均优于 `zero`。
- `real` 在三个数据集上均明显优于 `shuffled`。
- `zero` 保持 identity。
- `sign_flipped` 在 panama 和 spain 上变差，支持方向信息被使用；在 tetouan 上仍有收益但低于 `real`，说明不是所有数据集都满足严格反向。
- `amplified_2x` 与 `real` 几乎相同，提示当前 generator 可能饱和，或 blend/clamp 后差异被压平。

结论等级：

```text
弱到中等正向证据，不能升级为最终论文 claim。
```

原因：

- 只有 seed 42。
- 是 100-batch bounded run，不是完整训练。
- 未与旧 Stage III 在完全相同 V2 deterministic protocol 下复跑。
- 未做 paired statistical test。

下一步：

1. 同协议跑 seeds `42,43,44`。
2. 对旧 Stage III / legacy full 做 deterministic-eval 对齐复跑。
3. 汇总 dataset-seed cell 的 paired differences。
4. 再决定是否更新论文创新点。

---

## 7. Causal-safe HyperLoRA public-3 bounded sensitivity

协议修复：

- raw timeline split before windowing；
- online batch size 1；
- ordered aligned representation training；
- required aligned embeddings；
- HyperLoRA checkpoint round-trip manifest。

测试证据：

```text
40 passed
```

结果：

| dataset | mode | MSE gain | MAE gain | use rate | reject count |
| --- | --- | ---: | ---: | ---: | ---: |
| panama | real | -7.9613% | -3.8491% | 0.500 | 10 |
| panama | zero | 0.0000% | -0.0000% | 1.000 | 0 |
| panama | shuffled | 0.3853% | 0.5146% | 0.650 | 7 |
| panama | sign_flipped | 7.8870% | 4.6744% | 0.650 | 7 |
| tetouan | real | -3.3956% | -1.5668% | 1.000 | 0 |
| tetouan | zero | -0.0000% | -0.0000% | 1.000 | 0 |
| tetouan | shuffled | -0.4487% | -0.2087% | 1.000 | 0 |
| tetouan | sign_flipped | 2.0409% | 0.7736% | 1.000 | 0 |
| spain_energy_weather | real | 0.4374% | 0.2576% | 0.200 | 16 |
| spain_energy_weather | zero | 0.0000% | -0.0000% | 1.000 | 0 |
| spain_energy_weather | shuffled | 0.4840% | 0.2805% | 0.250 | 15 |
| spain_energy_weather | sign_flipped | 0.4734% | 0.2041% | 0.150 | 17 |

结论：

```text
当前 causal-safe 结果不支持 real-drift-conditioned adaptation claim。
```

关键原因：

- panama 和 tetouan 上 `real` 明显变差；
- `sign_flipped` 在 panama 和 tetouan 上反而更好；
- spain 上 `real` 与 shuffled/sign_flipped 差异很小；
- 多个模式被 `max_pred_shift` 拒绝，说明当前 generator 或 drift 尺度仍不稳定。

下一步必须先解决：

1. drift 符号和 temporal-delta 定义是否与预测误差方向一致；
2. generator training 是否过拟合 train stream；
3. 是否应该改用 `delta_phi_v` 本身，而不是重新计算 `phi_v - phi_i`；
4. 是否需要把 generator target 从直接预测误差改为 improvement-aware / validation-aware objective。

---

## 8. delta_phi_v 与 polarity 追加诊断

### 8.1 新增诊断能力

新增显式配置：

```yaml
online:
  stage3:
    drift_source: delta_phi_v
    drift_polarity: 1.0
```

目的：

- 用 `delta_phi_v` 直接作为 drift source，避免在 runner 外层再次解释 `phi_v - phi_i`。
- 用 `drift_polarity` 测试 drift 全局符号是否可能反了。
- 保持训练 generator 和 online sensitivity 使用同一口径。

### 8.2 测试证据

命令：

```powershell
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_factor_mechanism_v2.py tests\test_adaptation_v2.py tests\test_hyperlora_generator_v2.py tests\test_aligned_representation_v2.py tests\test_offline_aligned_representation_v2.py tests\test_stage3_functional_v2.py tests\test_stage3_generator_training_v2.py tests\test_stage3_sensitivity_v2.py tests\test_online_loop_stage3_v2.py tests\test_stage3_v2_protocol_safety.py tests\test_stage3_signals.py tests\test_stage3_ablation_benchmark.py tests\test_shapes.py -q
```

结果：

```text
44 passed in 3.18s
```

### 8.3 Panama b30 结果

输出目录：

- `logs/load_benchmarks/stage3_v2/codex_stage3_v2_polarity_pos_panama_b30_20260525/panama/seed_42/`
- `logs/load_benchmarks/stage3_v2/codex_stage3_v2_polarity_neg_panama_b30_20260525/panama/seed_42/`

结果：

| polarity | mode | MSE gain | MAE gain | use rate | reject count |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1.0 | real | -3.5689% | -1.9644% | 0.900 | 3 |
| 1.0 | zero | 0.0000% | 0.0000% | 1.000 | 0 |
| 1.0 | shuffled | 0.2302% | 0.3962% | 1.000 | 0 |
| 1.0 | sign_flipped | 2.2552% | 1.3576% | 0.900 | 3 |
| -1.0 | real | -3.5689% | -1.9644% | 0.900 | 3 |
| -1.0 | zero | 0.0000% | 0.0000% | 1.000 | 0 |
| -1.0 | shuffled | 0.2302% | 0.3962% | 1.000 | 0 |
| -1.0 | sign_flipped | 2.2552% | 1.3576% | 0.900 | 3 |

### 8.4 诊断结论

1. `delta_phi_v` 直接作为 drift source 没有解决 real drift 变差的问题。
2. `sign_flipped` 仍优于 `real`，说明 drift 方向、temporal delta 定义或 generator 训练目标仍有问题。
3. 训练和评估同时整体翻转 `drift_polarity` 后，结果保持相同；这不是有效修复，而是 HyperLoRA 对全局符号约定的等价重参数化。
4. 下一步不应继续依赖全局符号调参，而应修改 generator training objective：
   - 引入 validation/holdout stream；
   - 优化 `adapted_mse - base_mse` 的相对改进，而不是只优化 adapted prediction loss；
   - 增加 real-vs-shuffled / real-vs-zero 对比损失；
   - 对 `max_pred_shift` 被拒绝样本做训练期约束；
   - 将 `real` 必须优于 `zero/shuffled` 作为 early stopping 选择标准。

---

## 9. Advanced V2 b50 claim-gate 记录

本轮新增了 claim gate：

- 文件：`train/stage3_sensitivity_gate_v2.py`
- 输出：`sensitivity_claim_gate.csv`

通过条件：

1. `real` 必须优于 base；
2. `real` 必须优于 zero/shuffled/sign_flipped/random 等 counterfactual；
3. real drift 必须产生非零 delta；
4. real drift 必须产生非零 prediction shift。

Panama b50 结果：

| mode | MSE gain | MAE gain | delta norm | pred shift |
| --- | ---: | ---: | ---: | ---: |
| real | -0.5096% | -0.5191% | 0.290135 | 0.067722 |
| zero | 0.0000% | 0.0000% | 0.000000 | 0.000000 |
| shuffled | 0.1542% | 0.0231% | 0.286658 | 0.076613 |
| sign_flipped | 0.3130% | 0.6810% | 0.290135 | 0.075366 |
| amplified_2x | -0.5096% | -0.5191% | 0.290135 | 0.067722 |
| random_matched_norm | -0.5721% | -0.2583% | 0.246417 | 0.059458 |

gate：

```text
claim_gate_pass: False
failure_reasons: real_mse_gain;counterfactual
best_counterfactual_mode: sign_flipped
```

解释：

1. 该结果不支持 drift-conditioned adaptation claim。
2. `real` 变差，而 `sign_flipped` 变好，说明 drift 语义或 generator update direction 仍不可靠。
3. `amplified_2x` 与 real 几乎相同，是因为 `normalize_vector=True` 抵消幅值放大；以后若要声称幅值敏感性，必须在未抵消幅值的配置下重新审计。
4. 当前 advanced V2 可以作为“严格诊断框架”和“negative evidence”，不能作为论文性能提升证据。

### 9.1 level_vector + no-normalization 对照

为检查 `delta_vector` 和 `normalize_vector=True` 是否导致 real drift 失败，又运行了一个较小 b30 对照：

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v2_sensitivity_audit.py --config configs\stage3_v2_strong.yaml --dataset panama --run-id codex_stage3_v2_adv_level_nonorm_panama_b30_20260525 --seed 42 --epochs 2 --invariant-epochs 2 --generator-epochs 4 --device cpu --modes real zero shuffled sign_flipped amplified_2x random_matched_norm --max-train-batches 30 --max-generator-batches 30 --max-eval-batches 30 --drift-source delta_phi_v --drift-polarity 1.0 --signal-mode level_vector --no-normalize-vector
```

结果：

| mode | MSE gain | MAE gain | use rate | reject count | delta norm | pred shift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| real | -0.9948% | -0.4538% | 0.1667 | 25 | 1.341413 | 0.551914 |
| zero | 0.0000% | 0.0000% | 1.0000 | 0 | 0.000000 | 0.000000 |
| shuffled | 0.2371% | 0.4306% | 0.6333 | 11 | 1.113881 | 0.283478 |
| sign_flipped | -0.5010% | -0.3470% | 0.3333 | 20 | 1.341413 | 0.345606 |
| amplified_2x | -0.9927% | -0.4527% | 0.1667 | 25 | 1.340966 | 0.551805 |
| random_matched_norm | 0.3769% | 0.0394% | 0.3000 | 21 | 1.194768 | 0.426873 |

claim gate：

```text
claim_gate_pass: False
failure_reasons: real_mse_gain;counterfactual
best_counterfactual_mode: random_matched_norm
```

解释：

1. `level_vector` 没有解决 real drift 失败。
2. 关闭 normalization 后，real drift 的 `pred_shift` 大到触发大量 reject，说明幅值路径不稳定。
3. 这表明问题不是单纯由 `delta_vector` 造成，也不是简单关闭归一化就能修复。
4. 下一步若继续探索幅值敏感性，必须加入 drift norm clipping 或 validation-gated adapter selection，而不是直接让未归一化 drift 进入 generator。

---

## 10. Validation gate 与 test report-only 记录

本轮修复后，Stage III V2 audit runner 同时输出 validation 和 test 两套 sensitivity：

| 文件 | 用途 |
| --- | --- |
| `validation_sensitivity_summary.csv` | 只能用于模型/配置选择 |
| `validation_sensitivity_claim_gate.csv` | selection gate |
| `sensitivity_summary.csv` | test final report |
| `sensitivity_claim_gate.csv` | test report-only gate |

新增关键字段：

- `split`
- `gate_role`
- `online_batch_count`
- `online_sample_count`
- `intervention_source_split`
- `runner_stream_materialized: false`
- checkpoint sha256

### 10.1 Panama b30 validation-gated 结果

运行 ID：

```text
codex_stage3_v2_valgate_panama_b30_20260525
```

Validation：

```text
claim_gate_pass: True
real_mse_gain: 0.016523
real_mae_gain: 0.015913
best_counterfactual_mode: zero
real_reject_rate: 0.1000
```

Test：

```text
claim_gate_pass: False
failure_reasons: real_mse_gain;real_mae_gain;counterfactual
real_mse_gain: -0.035996
real_mae_gain: -0.014569
best_counterfactual_mode: zero
```

### 10.2 解释

这说明：

1. 单一 validation split 上的 gate 通过并不代表 test 泛化。
2. 当前 Stage III V2 的 drift-conditioned signal 仍不稳定。
3. test gate 只能作为 report-only 证据，不能用于继续调参。
4. 论文不得写成“方法经验证能够泛化捕获 concept drift”。

下一步必须是 rolling validation 或多 validation window 稳定性测试，而不是继续用同一 test set 调参。
