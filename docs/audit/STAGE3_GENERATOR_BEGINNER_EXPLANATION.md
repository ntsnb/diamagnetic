# Stage III Generator 问题的初学者说明

本文解释前面审计中提到的五个问题：

1. `AdaptationGenerator` 没有被训练。
2. 即使 `drift = 0`，也会产生非零参数扰动。
3. 原始 Stage III 把向量 drift 压成了一个标量。
4. 模型在 `eval()` 阶段仍然有 VAE 随机采样。
5. `phi_v - phi_i` 的表示空间未严格对齐。

这份文档不是论文结论，也不是要否定整个项目。它的作用是帮助理解：原始 Stage III 代码为什么很难支撑“模型根据 concept drift 做了有意义的在线适应”这一说法。

## 2026-06-01 最新阅读入口：V12e 把“历史残差”改成逐 horizon 标签可用

V12d 发现了一个重要问题：`history_only` 很容易解释掉大部分表面修正收益。但 V12d 仍有一个在线协议弱点：多步预测的第 12 步误差，不能在下一条样本立刻知道。也就是说，上一行的完整 12 维 residual 向量不应该全部立刻进入历史 buffer。

V12e 做的事情可以直观理解为：

```text
只有当某个 horizon 的真实标签已经到达时，
对应的 residual 才能进入历史特征；
否则该 residual 必须被 mask 掉。
```

它还加入了更严格的 drift 对照：固定 history features，只打乱或随机化 drift signal。这样能检查 `phi_v - phi_i` 是否真的提供了超过历史状态的独立信息。

V12e public-3 条件置换矩阵结果：

| 检查项 | 结果 |
| --- | ---: |
| dataset-seed runs | `9/9` completed |
| feature-mode rows | `27/27` completed |
| `drift_only` gates | `0/9` |
| `history_only` gates | `0/9` |
| `drift_plus_history` gates | `0/9` |
| `drift_plus_history` rolling windows | `0/45` |
| rolling-min partial MSE positive runs | `2/9` |
| rolling-min partial MSE average | `-0.006893` |
| test partial MSE positive runs | `5/9` |
| test partial MSE average | `0.000954` |
| generator artifact | `0` |

这个结果比 V12d 更严格，也更接近真实在线部署。它说明：当我们不允许未来 horizon 的标签提前进入历史时，历史残差带来的表面收益大幅减弱；同时，真实 `phi_v - phi_i` drift 仍然没有稳定超过 conditional shuffled、conditional random matched norm 或 amplified drift controls。

因此当前结论没有变成“Stage III 更有效”，而是变得更清楚：

```text
Stage III 还没有证明自己学到了可部署的 drift-conditioned adaptation。
下一步必须重构表示学习/teacher 目标，而不是继续训练当前 generator。
```

详细记录见：
`docs/audit/STAGE3_V12E_LABEL_AVAILABILITY_CONTEXT_TEACHER_RECORD.md`。

## 2026-06-01 最新阅读入口：V12d 说明 drift 还没有超过历史残差

最新一轮 V12d 做了一个更容易理解的对照：

```text
只看 phi_v - phi_i drift；
只看过去预测错了多少，即历史 residual；
同时看 drift + 历史 residual。
```

直观上，如果 `phi_v - phi_i` 真的是有用的 concept drift 信号，那么
`drift + 历史 residual` 应该稳定超过 `只看历史 residual`。否则，模型表面上
“修正得更好”可能只是因为负荷预测误差本身有时间自相关，而不是 Stage III
真的学到了隐藏漂移。

V12d public-3 结果：

| 检查项 | 结果 |
| --- | ---: |
| dataset-seed runs | `9/9` completed |
| 三类 feature-mode summary rows | `27/27` completed |
| `drift_only` gates | `0/9` |
| `history_only` gates | `0/9` |
| `drift_plus_history` gates | `0/9` |
| `drift_plus_history` rolling windows | `6/45` |
| rolling minimum partial MSE positive runs | `3/9` |
| test partial MSE positive runs | `4/9` |
| generator artifact | `0` |

这说明：当前 `phi_v - phi_i` 还没有稳定提供超过历史残差自相关的增量信息。
因此现在仍不能训练 Stage III generator，也不能在论文中说 Stage III 已经可靠
学习 concept drift。详细记录见：
`docs/audit/STAGE3_V12D_CONTEXT_TEACHER_RECORD.md`。

## 2026-06-01 较早阅读入口：V12 大矩阵仍是安全闭门

本轮专家结论很明确：V11 不是性能修复，而是一次更严格的
fail-closed 审计。它把 teacher 校准改成 worst-block /
reliability-gated 形式，但 public-3 三数据集三种子仍然是 `0/9`
teacher gate、`0/45` rolling validation window 通过，所有主矩阵 run
的 `teacher_shrinkage_alpha=0.0`，并且没有生成任何可作为论文证据的
generator artifact。

最重要的直观结论是：

```text
alignment gate 通过，不等于 teacher 能产生有用预测修正。
teacher 能被安全地拒绝，比训练一个方向不可靠的 generator 更科学。
```

后续 `level_vector` 和 `block_horizon` 诊断也没有救回 V11：
`level_vector` 仍是 `0/9` gate、`0/45` window；`block_horizon`
虽然出现了部分非零 alpha，但仍是 `0/9` gate、`0/45` window。这说明问题
不只是 `delta_vector` 时间差分，也不只是是否按 horizon 单独缩放，而是
drift 表示到 residual correction 的可预测链条仍未建立。

V12 现在已经新增为独立 opt-in 路径，用来专门处理
`phi_v - phi_i` 空间未严格对齐的问题。它不是直接继续训练更大的
generator，而是先做更基础的事：

```text
把 phi_i 和 phi_v 放进 train-only shared latent anchor；
在这个共享锚空间里计算和清理 drift；
只有当 drift 能在 chronological calibration 上预测 forecast residual，
才保留对应维度，否则 fail closed 到 zero correction。
```

当前 V12 已完成代码级 smoke 和 public-3 teacher-only 大矩阵：

- `models/stage3_alignment_v12.py`
- `models/stage3_residual_teacher_v12.py`
- `train/run_stage3_v12_sensitivity_audit.py`
- `train/run_stage3_v12_teacher_audit_matrix.py`
- `configs/stage3_v12_outcome_aligned_teacher_gate.yaml`
- `tests/test_stage3_v12_alignment_protocol.py`
- `tests/test_stage3_v12_teacher_protocol.py`
- `tests/test_stage3_v12_artifacts.py`

代码级测试 `6 passed`，一个 Panama seed42 极小 teacher-only smoke 能完整写出
V12 diagnostics，并且在证据不足时写出 `teacher_only_abort.json` 阻止
generator training。这个 smoke 只证明 V12 路径可运行且能安全闭门，**不证明**
V12 已经提升预测精度，也不能升级论文 claim。

随后执行的 V12 public-3 teacher-only 大矩阵更关键：

| 检查项 | 结果 |
| --- | ---: |
| 数据集和种子 | `panama/tetouan/spain_energy_weather` x `42/43/44` |
| 完成 run | `9/9` |
| alignment gate | `9/9` 通过 |
| teacher gate | `0/9` 通过 |
| rolling validation window | `0/45` 通过 |
| 非零 teacher alpha | `0/9` |
| teacher alpha source | 全部 `fallback_block_gate` |
| generator artifact | `0` |

这是什么意思？

```text
V12 已经让 drift 表示更像一个可审计的低维信号；
但 teacher 仍然认为这个信号不能稳定改善预测；
所以系统选择 alpha=0，也就是不做 Stage III 修正。
```

用更直白的话说：V12 不是“修好了 Stage III”，而是更清楚地告诉我们
Stage III 现在卡在哪里。它把问题从“phi_v - phi_i 可能不对齐”推进到了
“对齐后的 drift 信号仍没有被 teacher 转化成稳定 residual correction”。
因此，下一步应该重构 V12b teacher，而不是训练 generator 或修改论文去声称
Stage III 已经有效。

详细记录见：

- `docs/audit/STAGE3_V11_TEACHER_MATRIX_RECORD.md`
- `docs/audit/STAGE3_V11_SIGNAL_AND_HORIZON_ABLATION_RECORD.md`
- `docs/audit/STAGE3_V12_OUTCOME_ALIGNED_TRANSPORT_RECORD.md`
- `docs/audit/STAGE3_V12C_MULTI_POLICY_TEACHER_RECORD.md`

## 2026-06-01 追加阅读入口：V12b/V12c 说明了什么

V12b 和 V12c 是在 V12 之后继续做的两轮 teacher-only 修复。它们仍然
没有训练或发布 generator，因为 teacher 没有通过安全门。

V12b 做了更严格的 out-of-fold teacher 训练：只用训练集前缀拟合，再在
后面的 chronological block 上验证 correction 是否真的有用。结果是：

```text
public-3: 9/9 completed, teacher gate 0/9, rolling window 0/45。
loose-alpha Panama probe: 即使允许非零 alpha，rolling 仍是 0/5，
而且 sign-flipped drift 比 real drift 更好。
```

这说明 V12b 的失败不是简单“门槛太高”。如果放松门槛后方向仍不对，
就不能把它写成模型学到了 concept drift。

V12c 进一步把 teacher 的 alpha 拆成三种诊断策略：

```text
ungated：只看 OOF 上最优 alpha；
relaxed：只要求足够多时间块通过；
strict：保留最严格 worst-block 安全条件。
```

V12c 的好处是能看见弱信号。Panama seed42 的 8 组合 probe 发现
`delta_vector` 比 `level_vector` 更好；V12c best-delta public-3 里
Panama seed43 甚至接近通过 MSE、MAE 和 counterfactual margin 门槛。
但是最终结果仍然是：

| 检查项 | V12c best-delta public-3 |
| --- | ---: |
| 完成 run | `9/9` |
| teacher gate | `0/9` |
| rolling window | `0/45` |
| generator artifact | `0` |

所以现在最准确的初学者结论是：

```text
V12c 让我们看到了“有一点信号”，但这个信号还不稳定、不够强、
跨数据集不可靠，也没有达到论文级 claim gate。
```

因此，当前 Stage III 仍然不能写成已验证的性能贡献。最多可以写成：
我们建立了一个更严格的 train-only 诊断路径，发现现有 drift-to-residual
机制还没有通过 prospective rolling validation。

## 2026-05-28 最新阅读入口：V10 部分修复但仍未过门

V10 是在 V9 teacher-only 失败后做的一轮更安全修复。它没有直接继续训练更大的 generator，而是先问一个更基础的问题：

```text
如果只让 teacher 直接修正预测 residual，不做参数扰动，
它能不能在 rolling validation 上稳定改善预测？
```

V10 做了两件关键改动：

1. **cross-fitted leakage projection**：不再用 V9 的 `delta - f(phi_i)` 方式扣除 invariant leakage，而是在 drift 向量空间里估计一个 leakage 子空间，然后用线性投影删掉它。这样 `delta=0` 时一定仍然输出 0。
2. **train-calibrated shrinkage**：teacher 在训练段先学 residual correction，再用训练段内部的 calibration slice 选择缩放系数。如果 calibration 不支持修正，就让 `alpha=0`，相当于不乱改。

同预算 Panama seed42 结果：

| 路径 | rolling pass | worst MSE gain | worst MAE gain | test MSE gain | test MAE gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| V9 teacher-only | `0/5` | `-0.197784` | `-0.118167` | `-0.097074` | `-0.052800` |
| V10 scalar shrinkage | `3/5` | `0.005937` | `-0.000710` | `0.026775` | `0.015189` |
| V10 signed alpha | `2/5` | `-0.001860` | `-0.009942` | `0.012472` | `0.009613` |

这说明 V10 确实比 V9 好很多：V9 是明显 harmful，V10 scalar 已经能在大多数窗口产生正收益。但它仍然没有过硬门，因为最后一个 validation window 的 MAE 仍是负的，最差信号方向一致性只有 `0.529688`，低于预设阈值 `0.55`。

所以当前结论是：

```text
V10 是一个有进展的机制修复，不是已经验证的 Stage III 性能贡献。
在 teacher-only rolling validation 5/5 通过之前，不应该训练或宣传 V10 generator。
```

详细记录见：

- `docs/audit/STAGE3_V10_CROSSFIT_TEACHER_RECORD.md`
- `docs/audit/STAGE3_V10_TEACHER_MATRIX_RECORD.md`
- `docs/audit/STAGE3_EXPERT_DIAGNOSIS_AND_REPAIR_STRATEGY.md`

## 2026-05-29 最新阅读入口：V10 扩大到三数据集三种子后仍失败

为了避免只看 Panama seed42 的偶然结果，本轮又跑了 V10 teacher-only
矩阵：`panama`、`tetouan`、`spain_energy_weather` 三个数据集，每个数据集
`seed=42/43/44`，仍使用 `5 epochs / 480 batches / 5 rolling windows`。

核心结果：

| 数据集 | teacher gate | rolling windows passed | 平均 worst MSE gain | 平均 worst MAE gain |
| --- | ---: | ---: | ---: | ---: |
| Panama | `0/3` | `7/15` | `+0.007158` | `+0.000809` |
| Tetouan | `0/3` | `1/15` | `-0.144104` | `-0.096893` |
| Spain | `0/3` | `0/15` | `-0.025551` | `-0.018493` |
| 全部 | `0/9` | `8/45` | `-0.054166` | `-0.038192` |

这说明：V10 在 Panama 上有一点机制信号，但跨数据集后不稳定；Tetouan 和
Spain 上明显不支持。所有九个 run 的 alignment gate 都通过，但 teacher gate
全失败。因此，“空间对齐做得更好”并不等于“Stage III 已经学会了有用 drift
适应”。

给初学者的直观解释是：

```text
V10 已经不像原始 Stage III 那样随机乱动，也不像 V9 那样明显有害。
但它还没有稳定学会什么时候、往哪个方向修正预测。
所以现在仍不能训练 generator 去模仿它，更不能把它写成论文创新点成功证据。
```

## 2026-05-28 最新阅读入口：V9 残差教师结论

先读本节，再读后面的 V8、V7 和历史诊断。本轮在不覆盖原模型代码和训练代码的前提下，新增了 V9 opt-in 分支。V9 做了三件事：

1. **加了 train-only residual teacher**：只用训练流学习“当前 drift 应该怎样修正预测残差”，避免 validation/test 泄漏。
2. **加了 teacher-aware drift alignment**：不再直接相信 raw `phi_v - phi_i`，而是先去掉 invariant-explainable 部分，再结合 residual teacher 信息做校准。
3. **加了 residual-basis hypergenerator**：在 V8 的 weight+bias 参数更新之外，又加入一个 bias-free residual correction head，并保证 zero drift 时不产生修正。

这些改动确实修复了不少旧的结构问题。例如，zero drift 现在基本不再产生预测扰动；失败候选也不会被保存成最终 `generator_served_policy.pt`。但是 V9 的 Panama seed42 同预算 canary 仍然失败：

| 指标 | V9 结果 |
| --- | ---: |
| selected epoch | `0` |
| rolling validation | `0/5` |
| worst real MSE gain | `-0.007082` |
| worst real MAE gain | `-0.005109` |
| worst real-vs-best-counterfactual margin | `-0.007082` |
| max reject rate | `0.0` |
| final served artifact | not saved |

最关键的现象是：在 selected epoch 的五个 validation windows 中，**zero drift 都是最好的 counterfactual**。这句话的含义是：

```text
V9 已经能做到“没有 drift 就不乱改”，
但只要输入真实 drift，它反而会把预测往错误方向推。
```

这说明当前 Stage III 的问题已经进入更深一层：不是简单地“generator 没训练”“drift 被压成标量”“zero drift 也会乱动”这些低级 bug，而是 `drift -> residual correction` 的映射在 train 上看起来能学到一点，在 rolling validation 上却不能泛化。

因此，截至 2026-05-28，最安全的论文定位仍然是：

```text
Stage III 是一个经过严格审计的在线瞬时适应候选机制。
V9 引入了 train-only residual teacher 和 zero-drift identity，
但当前证据仍未证明它能把 representation drift 稳定转换为有益的在线预测修正。
```

详细记录见：

- `docs/audit/STAGE3_V9_RESIDUAL_TEACHER_RECORD.md`
- `docs/audit/STAGE3_V9_TEACHER_ONLY_DIAGNOSTIC_RECORD.md`

teacher-only 诊断已经补做，结论更明确：如果完全绕开 generator，只做 `y_base + teacher(z)`，Panama seed42 同预算 rolling validation 仍然是 `0/5`，worst MSE gain 为 `-0.197784`，worst MAE gain 为 `-0.118167`，每个窗口都是 zero drift 最好。也就是说，当前 V9 不是“generator 没学好一个正确 teacher”，而是 **teacher 自己在验证流上就把预测往错误方向推**。

下一步如果继续研究 Stage III，不应马上把 V9 扩展到 public-3 当作性能实验。原因不是计算成本，而是单数据集 pre-declared canary 和 teacher-only 诊断都已经失败。更合理的下一步是重构 teacher/alignment：加入 train-calibrated shrinkage、residual-sign-aware teacher fitting、cross-fitted leakage-subspace projection，再看 teacher-only rolling validation 能否至少不伤害预测。

## 2026-05-28 阅读入口：V8 专家结论和下一步

先读本节，再读后面的 V7 和历史诊断。本轮继续测试了一个更强的 opt-in 版本 V8：它在 V7 的 residualized RRCCA drift 基础上，额外允许 Stage III generator 更新 `output_head` 的 bias 参数。这样做的直觉是：负荷预测误差里可能有一部分是 level / bias 型误差，只改 weight 可能太弱。

V8 的代码和实验结论如下：

1. **V8 确实接入了训练和在线回放**：新增的 `LayerAwareHyperAdapterGenerator` 可以同时生成 weight 和 bias delta，并且通过了 zero-drift identity、梯度、checkpoint round-trip、artifact reload、online replay 等测试。
2. **V8 没有提升效果，反而更差**：Panama seed42 同预算 canary 中，V8 最佳候选 epoch `3` 的 rolling validation 仍然是 `0/5`，worst-window MSE gain 为 `-0.008390`，MAE gain 为 `-0.005176`，real-vs-best-counterfactual margin 为 `-0.008390`。
3. **与 V7 对比，V8 更不适合作为论文证据**：V7 最佳 epoch 虽然也失败，但仍有很小正收益；V8 的真实 drift 在五个窗口全部是负收益。
4. **核心结论**：问题不是“generator 还不够大”这么简单。V8 证明了增加 bias adapter 容量会放大更新，但如果没有 residual teacher / meta-objective 告诉 generator 应该往哪个方向修正，更新会变得更有害。

最新证据入口：

- `docs/audit/STAGE3_V8_HYPERADAPTER_RECORD.md`
- `docs/audit/STAGE3_V7_ALIGNMENT_RECORD.md`
- `docs/audit/EXPERIMENT_EVIDENCE_LEDGER.md`
- `docs/audit/RISK_REGISTER.md`
- `docs/audit/REVISION_ROADMAP.md`

因此，V8 当时的结论是：论文仍不能把 Stage III 写成“已经验证的 concept-drift-conditioned online adaptation”；下一步应当测试 V9 残差教师更新、残差校正目标和更严格的 `phi_v/phi_i` 表示对齐。V9 已经在上节记录，结果仍未通过 claim gate。

## 2026-05-27 最新阅读入口：V5/V6/V7 专家结论

先读本节，再读后面的历史诊断。前面几轮修复已经把 Stage III 从“随机、未训练、评估不确定”的 legacy 实现，推进成了多个可审计的 opt-in 版本，但最新较大预算 canary 仍然没有证明 Stage III 可以作为论文性能贡献。

本轮新增了 V7：

1. **V7 解决的问题**：针对 `phi_v - phi_i` 空间未严格对齐的问题，新增 `residualized_rrcca_v7`。它先在训练集上去掉 `phi_v - phi_i` 中可由 `phi_i` 线性解释的部分，再只保留 train 内部校准稳定的 CCA drift 子空间。
2. **V7 的方法论结果**：V7 alignment gate 通过，active rank 为 `8`，说明它确实找到了一个更严格、更低维的 residual drift 子空间。
3. **V7 的预测结果**：Panama seed42 `g5_b480` served-policy rolling validation 为 `0/5`。最佳 epoch 的 worst-window MSE gain 只有 `0.000490`，MAE gain 只有 `0.000534`，real-vs-counterfactual margin 只有 `0.000267`，远低于预设门槛。
4. **最重要结论**：更先进的 alignment 方法并没有自动带来更好的在线预测。V7 支持“原始 `phi_v - phi_i` 确实有 invariant leakage 风险”，但不支持“Stage III 已经学会 concept drift 并提升预测”。

最新证据入口：

- `docs/audit/STAGE3_V7_ALIGNMENT_RECORD.md`
- `docs/audit/STAGE3_V5_V6_G5_CANARY_RECORD.md`
- `docs/audit/EXPERIMENT_EVIDENCE_LEDGER.md`
- `docs/audit/RISK_REGISTER.md`
- `docs/audit/REVISION_ROADMAP.md`

因此，截至 2026-05-27，论文中仍不应把 Stage III 写成核心已验证性能贡献。更安全的表述是：Stage III 是一个经过严格审计的 drift-conditioned adaptation 候选机制；当前实验证据揭示了从 representation shift 到可靠在线参数更新之间的困难。

## 2026-05-25 最新阅读入口

先读这一段，避免把早期诊断和后续 V2.5 修复混在一起：

1. **原始 Stage III** 确实存在本文件开头列出的五类问题：generator 未训练、zero drift 非零扰动、向量 drift 被压成标量、eval 阶段 VAE 随机、`phi_v - phi_i` 空间未严格对齐。
2. **Stage III V2/V2.5** 已经在不覆盖原始代码的前提下新增了 opt-in 修复路径：deterministic VAE eval、identity-preserving generator、显式 generator training、vector drift、train-only residual drift canonicalizer、rolling-validation generator checkpoint selection。
3. 但是，**机制修复不等于性能证明**。Panama seed42/43 的 b60/b120 较大预算实验均未通过 rolling validation gate，test gate 也全部失败。因此现在仍不能写“Stage III V2.5 显著提升精度”“可靠捕捉 concept drift”或“超过 SOTA”。
4. 目前最安全的论文定位是：Stage III V2.5 是一个严格审计过的 online adaptation 诊断模块；它揭示了 drift-conditioned parameter generation 的难点，但尚未成为可靠性能贡献。

最新证据入口：

- `docs/audit/STAGE3_V25_CANONICAL_CKPT_SELECTION_EXPERIMENT_RECORD.md`
- `docs/audit/STAGE3_V2_IMPLEMENTATION_LOG.md`
- `docs/audit/PAPER_CLAIMS_MATRIX.md`
- `docs/audit/EXPERIMENT_EVIDENCE_LEDGER.md`

## 本轮专家补充诊断与修改方案

本轮二次核查后，结论进一步明确：当前 Stage III 的核心风险不是某一个 drift 公式写得不够好，而是整条“drift -> generator -> parameter delta -> improved prediction”的证据链还没有建立起来。

核心判断如下：

1. `AdaptationGenerator` 当前没有接入训练流程。它在在线循环中被新建后直接用于生成参数扰动，但代码仓没有发现 `gen.parameters()` 对应的优化器或训练目标。因此它目前更像一个随机扰动器，而不是一个学到 drift-adaptation 映射的生成器。
2. 当前生成器结构不适合作为严谨的 drift-conditioned adapter。因为 `sigmoid(...) + 1.0` 和 Linear bias 会导致 `drift = 0` 时仍产生非零参数扰动；同时输出主要是正向缩放，缺少明确的有符号、可解释、identity-preserving 设计。
3. `phi_v - phi_i` 目前不是严格可信的 concept drift 表示。原因是 `encoder_i` 和 `encoder_v` 由不同目标训练，没有共享坐标系、投影校准、正交约束或对齐损失。直接相减可能只是两个不同表示空间的数值差。
4. VAE 在训练阶段随机采样是正确的，但普通验证/测试阶段继续随机采样是不合适的。除非论文明确做 Monte Carlo evaluation，否则 `eval()` 下应该默认使用 `mu_z`，否则同一模型、同一输入会得到不同预测，污染 MSE/MAE 和 Stage III sensitivity 结论。

更详细的专家诊断已写入：

- `docs/audit/STAGE3_EXPERT_DIAGNOSIS_AND_REPAIR_STRATEGY.md`
- `docs/audit/STAGE3_GENERATOR_EXPERT_REPAIR_PLAN_CN.md`

极其详细的实施级修改方案已写入：

- `docs/superpowers/plans/2026-05-24-stage3-generator-repair-plan.md`

建议最小安全下一步：先只修复 VAE eval 的确定性问题，再做 Stage III generator 的结构和训练修复。因为如果评估阶段仍然随机，后续任何“real / zero / shuffled / sign-flipped drift”对比都会被 VAE 噪声污染。

---

## 1. 先理解整个故事：Stage III 本来想做什么

这篇论文和代码的大方向是在线负荷预测。在线学习的核心问题是：

> 模型在过去数据上训练好了，但未来数据分布会变。模型需要在不完整重训练的情况下，快速适应新数据。

在这个项目里，在线适应大致分三层：

| 阶段 | 直观含义 | 代码中的作用 |
| --- | --- | --- |
| Stage I | 调整专家门控 | 微调 `gate_bias`，改变专家权重 |
| Stage II | 基于 relevance / LRP 做参数更新 | 用 drift 和 mask 更新部分参数 |
| Stage III | 用 invariant/variant embedding 估计 concept drift，生成一次性参数扰动 | 对当前 batch 做 transient adaptation |

Stage III 想表达的理想机制是：

```text
当前输入 x
  -> 得到 invariant embedding: phi_i
  -> 得到 variant embedding: phi_v
  -> 计算 drift = phi_v - phi_i
  -> generator 根据 drift 生成参数扰动 delta
  -> 临时把 delta 加到模型参数上
  -> 得到 adapted prediction
```

换成一句话：

> 如果模型检测到当前样本和稳定模式之间有差异，就根据这个差异临时调整预测模型。

这在概念上是合理的。问题出在当前代码实现还没有真正完成这个因果链条。

---

## 2. 什么是 embedding、invariant embedding、variant embedding

### 2.1 什么是 embedding

深度学习模型通常不会直接用原始数据做所有判断，而是先把输入变成一个向量表示。

例如，输入是一段负荷序列：

```text
x = 最近 24 小时的温度、湿度、负荷、电价等特征
```

模型会把它编码成一个向量：

```text
embedding = [0.12, -0.31, 0.88, ..., 0.05]
```

这个向量就叫 embedding。它可以理解为模型内部的“摘要”。

### 2.2 什么是 invariant embedding

`invariant` 的意思是“不容易随环境变化而变化”。在负荷预测里，它可以对应：

- 工作日和周末规律；
- 日周期、周周期；
- 某些稳定的温度-负荷关系；
- 用户行为中的长期稳定部分。

理想情况下：

```text
phi_i = 稳定、长期、可泛化的因素
```

### 2.3 什么是 variant embedding

`variant` 的意思是“随环境变化而变化”。它可能对应：

- 突发天气；
- 节假日异常；
- 临时生产活动；
- 传感器变化；
- 短期外生因素。

理想情况下：

```text
phi_v = 短期变化、环境变化、未观测因素
```

于是论文想用：

```text
drift = phi_v - phi_i
```

来表示 concept drift。

---

## 3. 什么是 concept drift

在时间序列预测中，concept drift 指的是：

> 过去学到的输入-输出关系，在未来发生了变化。

例如过去模型学到：

```text
温度升高 -> 空调负荷升高
```

但未来某段时间由于节假日、工厂停产、需求响应政策等因素，关系变成：

```text
温度升高 -> 负荷没有明显升高
```

这就是 drift。

但 drift 也分很多类型：

| 类型 | 例子 | 是否适合 Stage III |
| --- | --- | --- |
| 长期趋势 | 城市人口增加导致整体负荷升高 | 不一定适合一次性 transient update |
| 季节性变化 | 夏季空调用电增加 | 需要模型长期适应 |
| 短期扰动 | 突发高温、节假日、异常天气 | 更适合 Stage III |
| 噪声 | 传感器错误、随机波动 | 不应过度适应 |

你之前提出的想法是：

> 不应只看当前 `phi_v(t) - phi_i(t)`，因为它可能包含长期变化。应看 `[phi_v(t)-phi_i(t)] - [phi_v(t-1)-phi_i(t-1)]`，这样更像短期 fluctuation。

这个想法在概念上是合理的。它相当于把“当前偏离量”改成“偏离量的变化速度”。

---

## 4. 什么是 AdaptationGenerator

在这个项目里，`AdaptationGenerator` 是一个小神经网络。

它的作用是：

```text
输入: drift vector
输出: alpha, beta
再用 alpha, beta 生成参数扰动 delta
```

代码位置：

- `models/adaptation.py`
- `train/online_loop.py`

核心代码逻辑是：

```python
h = torch.sigmoid(self.w1(drift))
alpha_full = torch.sigmoid(self.w2_alpha(h)) + 1.0
beta_full = torch.sigmoid(self.w2_beta(h)) + 1.0
```

然后：

```python
scale = alpha.unsqueeze(1) * beta.unsqueeze(0)
delta = (scale - 1.0) * p
```

这里的 `p` 是模型原始参数。

直观解释：

```text
generator 读入 drift
  -> 生成一个缩放矩阵 scale
  -> 用 scale 改变模型参数
```

---

## 5. 问题一：AdaptationGenerator 没有被训练

### 5.1 什么叫“没有被训练”

一个神经网络刚创建时，参数是随机初始化的。

例如：

```python
gen = AdaptationGenerator(...)
```

这时候 `gen` 只是一个随机函数。

如果你希望它学会：

```text
什么样的 drift 应该导致什么样的参数更新
```

就必须有训练过程，例如：

```python
optimizer = torch.optim.Adam(gen.parameters(), lr=...)
loss = prediction_loss_after_adaptation
loss.backward()
optimizer.step()
```

但当前仓库里，我没有找到这样的训练流程。

### 5.2 为什么这是严重问题

如果 generator 没训练，它就不知道：

- 哪些 drift 是有害变化；
- 哪些 drift 是普通噪声；
- 哪些模型参数该改；
- 改多少才合适；
- 正方向改还是反方向改。

于是当前 Stage III 更像：

```text
随机初始化的 generator
  + 很小的 lr_scale
  + max_delta 限制
  + blend 融合
```

它可能偶尔带来小幅提升，但这个提升很难被解释为：

> 模型学会了根据 concept drift 做在线适应。

更谨慎的说法是：

> 当前 Stage III 产生了小幅 transient parameter perturbation，但尚未证明该扰动由有效 drift-conditioned generator 驱动。

---

## 6. 问题二：zero drift 也会产生非零扰动

### 6.1 初学者版解释

理论上，如果没有 drift：

```text
drift = 0
```

我们希望 Stage III 不要乱改模型：

```text
delta ≈ 0
```

但当前代码不是这样。

### 6.2 为什么 zero drift 也会动

generator 里有 `Linear` 层。`Linear` 层默认有 bias。

一个线性层是：

```text
y = W x + b
```

如果输入 `x = 0`，输出不是 0，而是：

```text
y = b
```

所以：

```text
drift = 0
```

经过 generator 后仍然可能产生非零输出。

再加上代码中有：

```python
torch.sigmoid(...) + 1.0
```

`sigmoid` 的输出范围是 `(0, 1)`，所以：

```text
alpha, beta 的范围大约是 (1, 2)
```

于是：

```python
delta = (scale - 1.0) * p
```

一般不会是 0。

### 6.3 这意味着什么

如果 zero drift 都会产生明显参数扰动，那么 Stage III 的改善可能来自：

```text
固定随机扰动 + 小步长 + blend
```

而不是来自：

```text
真实 drift 信息
```

这就是为什么需要做 zero drift 诊断。

---

## 7. 问题三：向量 drift 被压成标量

### 7.1 向量和标量的区别

一个标量只有一个数：

```text
0.37
```

一个向量有很多维：

```text
[0.2, -0.1, 0.8, 0.0, ...]
```

embedding 通常是向量，因为每一维可能代表不同方向的信息。

例如：

```text
第 1 维: 温度相关变化
第 2 维: 周期性变化
第 3 维: 异常负荷变化
...
```

### 7.2 当前原始代码怎么做

原始 Stage III 代码做了：

```python
drift_scalar = d_v.mean()
drift_vec = normalized_scalar.repeat(16)
```

也就是：

```text
[0.2, -0.1, 0.8, 0.0, ...]
  -> 求平均
  -> 0.05
  -> [0.05, 0.05, ..., 0.05]
```

这样会丢掉方向信息。

### 7.3 为什么这和论文可能不一致

如果论文里说的是：

```text
variant embedding 和 invariant embedding 之间的差异
```

读者通常会理解为向量差异：

```text
phi_v - phi_i 是一个向量
```

但代码实际使用的是：

```text
mean(phi_v - phi_i)
```

这就变成一个标量摘要。

所以你质疑“两个向量之间的插值应该是向量”是对的。

---

## 8. 问题四：eval 阶段 VAE 仍然随机

### 8.1 什么是 VAE

VAE 是 Variational Autoencoder，变分自编码器。

普通 encoder 可能输出一个确定向量：

```text
z = encoder(x)
```

VAE 输出的是分布参数：

```text
mu_z, log_var_z
```

然后采样：

```text
z = mu_z + eps * std
```

其中：

```text
eps 是随机噪声
```

### 8.2 为什么 eval 阶段还随机是问题

训练时随机采样是 VAE 的正常机制。

但评估时，我们通常希望模型稳定：

```text
同一个输入 -> 同一个输出
```

否则指标会有随机波动。

当前代码在 `eval()` 模式下仍然调用：

```python
eps = torch.randn_like(std)
return mu + eps * std
```

所以同一个模型、同一个输入，两次预测也可能不同。

### 8.3 它会污染什么

它会影响：

- frozen base prediction；
- Stage II prediction；
- Stage III candidate prediction；
- `stage3_pred_shift`；
- online MSE / MAE；
- 不同 drift signal 之间的对比。

这会让我们更难判断：

```text
预测变化到底来自 drift，还是来自 VAE 随机采样？
```

---

## 9. 问题五：phi_v - phi_i 的表示空间未严格对齐

### 9.1 什么叫表示空间对齐

假设有两个坐标系。

第一个坐标系里：

```text
x 轴 = 温度变化
y 轴 = 时间周期
```

第二个坐标系里：

```text
x 轴 = 节假日影响
y 轴 = 随机噪声
```

如果这两个坐标系没有对齐，那么直接相减：

```text
phi_v - phi_i
```

就不一定有明确含义。

### 9.2 当前代码的问题

当前 `encoder_i` 和 `encoder_v` 是两个不同的 MLP。

`encoder_i` 用 invariant learning 训练。

`encoder_v` 用 residual target 训练。

但是代码没有明显约束：

- 两个 encoder 输出的每一维含义一致；
- 两个 embedding 的尺度一致；
- 两个 embedding 的方向一致；
- `phi_v - phi_i` 能表示可解释的 drift；
- variant 部分和 invariant 部分真的分离。

所以：

```text
phi_v - phi_i
```

可能只是两个不同网络输出的差，而不是严格意义上的 concept drift。

---

## 10. 为什么 zero / shuffled / sign-flipped / amplified 诊断是合理的

这些诊断属于 intervention test，也就是主动改变输入，观察输出是否按预期变化。

目标不是提高结果，而是回答一个问题：

> Stage III 到底有没有真的使用 drift 信息？

### 10.1 real drift

正常输入：

```text
drift = phi_v - phi_i
```

这是基准。

### 10.2 zero drift

把 drift 改成 0：

```text
drift = 0
```

如果结果和 real drift 一样，说明：

```text
Stage III 不需要 drift 也能产生同样效果
```

这会削弱“drift-conditioned adaptation”的说法。

### 10.3 shuffled drift

把 drift 打乱：

```text
第 t 个样本使用别的时间点的 drift
```

如果结果和 real drift 一样，说明：

```text
drift 和当前样本之间的对应关系没有被利用
```

这会削弱“在线时序 drift 检测”的说法。

### 10.4 sign-flipped drift

把 drift 乘以 -1：

```text
drift -> -drift
```

如果结果几乎不变，说明：

```text
模型不关心 drift 的方向
```

这会削弱“embedding 差异方向有意义”的说法。

### 10.5 amplified drift

把 drift 放大：

```text
drift -> 5 * drift
```

如果结果仍然不变，说明可能存在：

- generator 饱和；
- clamp 截断；
- blend 主导；
- drift 输入没有有效影响下游参数。

---

## 11. 为什么要先控制 VAE 随机性和 generator 问题

如果不控制这些问题，诊断会混乱。

例如你比较：

```text
real drift vs zero drift
```

但模型本身每次 forward 都随机，那么观察到的差异可能来自：

```text
VAE 采样噪声
```

而不是 drift。

又比如 generator 没训练，那么观察到的差异可能来自：

```text
随机初始化参数
```

而不是学到的 drift 机制。

因此合理顺序是：

```text
第一步: 让 eval 阶段 deterministic
第二步: 做 sensitivity diagnostic
第三步: 如果确认 generator 不敏感，再考虑重构 generator
```

---

## 12. 当前证据支持什么，不支持什么

### 12.1 支持的说法

当前证据可以支持：

```text
Stage III 在若干数据集上产生了小幅 online prediction 改善。
```

也可以支持：

```text
当前 Stage III 的 transient parameter perturbation 对预测有一定影响。
```

### 12.2 暂时不支持的说法

当前证据不支持：

```text
Stage III 的改善来自有效的 concept drift representation。
```

也不支持：

```text
temporal-delta drift 比 current-level drift 更好。
```

也不支持：

```text
vector drift 被 generator 有效利用。
```

因为扩大实验显示：

```text
level_scalar / delta_scalar / level_vector / delta_vector
```

结果几乎一样。

---

## 13. 更适合论文的谨慎表述

如果现在必须写论文，不建议说：

```text
Our Stage III generator effectively captures concept drift and adapts parameters accordingly.
```

更稳妥的表述是：

```text
Stage III introduces a transient parameter perturbation mechanism conditioned on representation discrepancy. However, additional sensitivity analysis is required to confirm whether the current generator fully exploits the directional information in the drift representation.
```

中文理解：

```text
Stage III 引入了一个基于表示差异的临时参数扰动机制，但仍需要敏感性分析确认当前 generator 是否真正利用了 drift 表示中的方向信息。
```

---

## 14. 最小安全下一步

建议按下面顺序做，不要一上来改论文声称：

### Step 1：建立 deterministic evaluation 副本

在副本里修改 VAE eval 行为：

```text
训练时: z = mu + eps * std
评估时: z = mu
```

这样同一个输入在 eval 下输出固定。

### Step 2：做 Stage III sensitivity diagnostic

不重新训练主模型，只重放同一个 offline model。

比较：

| drift 类型 | 目的 |
| --- | --- |
| real | 正常 Stage III |
| zero | 检查是否不需要 drift |
| shuffled | 检查时间和样本对应关系是否重要 |
| sign-flipped | 检查方向是否重要 |
| amplified | 检查幅值是否重要 |

记录：

| 指标 | 含义 |
| --- | --- |
| `delta_norm` | generator 生成的参数扰动有多大 |
| `clamp_fraction` | 有多少参数扰动被 `max_delta` 截断 |
| `candidate_shift` | Stage III candidate 和原预测差多少 |
| `blend` | 最终融合比例 |
| `MSE/MAE` | 预测效果 |

### Step 3：判断机制是否成立

如果：

```text
real drift ≈ zero drift ≈ shuffled drift
```

说明 Stage III 目前没有真正利用 drift。

如果：

```text
real drift 明显优于 zero / shuffled / sign-flipped
```

才说明 drift 机制可能有效。

### Step 4：再决定是否重构

如果确认 generator 不敏感，可能需要：

- 给 generator 增加训练目标；
- 让 zero drift 对应 zero update；
- 保留向量 drift，不再强制 mean scalar；
- 对齐 `phi_i` 和 `phi_v` 的表示空间；
- 检查 `max_delta` 和 `blend` 是否过强导致输入差异被抹平。

---

## 15. 一句话总结

当前问题的核心不是“你的 temporal-delta 想法错了”。

相反，你的怀疑是合理的。

真正的问题是：

```text
Stage III 当前代码还没有证明 generator 学会了根据 drift 信息生成有意义的参数更新。
```

所以现在最科学的做法是：

```text
先做机制诊断，再决定是否重构 Stage III，最后再更新论文叙事。
```

---

## 16. 2026-05-25 V2 实测后的新结论

后续我已经在不覆盖原始代码的前提下，建立了 Stage III V2 副本路径，并做了更严格的 causal-safe 诊断。关键修复包括：

1. `eval()` 阶段 VAE 默认使用 `mu_z`，不再随机采样。
2. 新 generator 使用 identity-preserving 设计，`zero drift` 会产生 `zero delta`。
3. generator 已有可微训练路径，不再是随机初始化后直接使用。
4. 表示学习改为 shared residual 结构：

```text
phi_v = phi_i + delta_phi_v
```

所以：

```text
phi_v - phi_i = delta_phi_v
```

5. V2 runner 使用 raw timeline first split，避免滑窗边界泄漏。
6. online 诊断默认 batch size = 1，避免同一 batch 内未来样本污染当前 drift。
7. 新增 `drift_source: delta_phi_v`，可以直接使用 residual drift。
8. 新增 `drift_polarity`，可以显式测试 drift 符号方向。

但是，最新 Panama b30 诊断显示：

| drift 设置 | 现象 |
| --- | --- |
| `real` | MSE/MAE 变差 |
| `zero` | 完全 identity，符合预期 |
| `shuffled` | 反而有小幅提升 |
| `sign_flipped` | 明显优于 `real` |
| 同时把训练和评估的 polarity 设为 `-1.0` | 结果与 `1.0` 基本相同 |

这说明一个重要事实：

```text
仅仅把 phi_v - phi_i 换成 delta_phi_v，或者把 drift 全局乘以 -1，并不能修好 Stage III。
```

原因是：

1. `delta_phi_v` 解决的是“表示空间是否对齐”的问题。
2. 但当前失败还涉及“generator 学到的更新方向是否真的能降低未来预测误差”。
3. 全局 polarity 同时用于训练和评估时，HyperLoRA 可以学习到等价的反向参数，因此结果不会真正改变。
4. 真正需要改的是 generator 的训练目标和模型选择标准。

下一步更科学的修改方向是：

```text
不要只让 generator 最小化 adapted prediction loss。
```

而应该让它学习：

```text
真实 drift adaptation 必须比 base 更好，
必须比 zero drift 更好，
必须比 shuffled drift 更好，
并且不能产生过大的 prediction shift。
```

这可以理解为：

> generator 不只是学“怎么改参数”，还要学“什么时候这种 drift 值得改、改了以后是否真的比不改更好”。

所以当前最谨慎的论文结论仍然是：

```text
Stage III V2 的工程机制已经更合理，但真实 drift-conditioned adaptation 尚未被 causal-safe 实验证明。
```

### 16.1 再次增强后仍未通过的原因

我进一步加入了更强的修复：

1. generator 训练时不只看 adapted MSE，还加入：
   - 是否优于 base；
   - 是否优于 zero drift；
   - 是否优于 sign-flipped drift；
   - 是否优于 shuffled/previous drift；
   - prediction shift 是否过大。
2. `delta_phi_v` 表示加入 HSIC/VICReg 约束，减少它复制 invariant 信息或 collapse。
3. variant 分支支持 zero init，避免一开始就抢解释权。
4. drift 输入加入 train-only standardization，避免 test 数据参与校准。
5. 每次 sensitivity 自动输出 claim gate。

但 Panama b50 结果仍然是：

```text
real drift: MSE 变差约 0.51%
zero drift: 基本不变
shuffled drift: 小幅变好
sign-flipped drift: 比 real 更好
```

这说明：

```text
当前问题已经不是“代码有没有训练 generator”这么简单，
而是 real drift 表示和 generator 更新方向仍没有形成正确的预测改进关系。
```

更直白地说：

> 模型现在确实会根据 drift 产生参数扰动，但这个扰动的方向不一定是对的。

因此，论文现在不能说：

```text
Stage III successfully captures concept drift.
```

只能说：

```text
We implemented a diagnosed drift-conditioned adaptation pathway, but causal-safe sensitivity tests show that the current drift representation and generator objective do not yet yield reliable real-drift improvements.
```

### 16.2 为什么 validation 通过但 test 失败仍然不能写成贡献

后续我又补了一个更严格的实验协议：

```text
train split: 训练 base / representation / generator
validation split: 选择和判断是否通过 gate
test split: 只做最终报告，不允许回头调参
```

结果出现了一个很典型的问题：

```text
validation gate: 通过
test report-only gate: 失败
```

具体地说，在 Panama b30 中：

```text
validation real MSE gain: +1.6523%
validation real MAE gain: +1.5913%
test real MSE gain: -3.5996%
test real MAE gain: -1.4569%
```

这说明：

```text
模型在 validation 上看起来有效，但没有泛化到 test。
```

对初学者来说，这就像：

> 你在模拟考试上调出了一套方法，但正式考试表现反而变差。

所以论文不能只拿 validation 通过来写：

```text
Stage III works.
```

更严谨的说法是：

```text
Stage III V2 can pass a validation sensitivity gate in a bounded setting, but it fails to generalize to the held-out test stream under the same frozen configuration.
```

中文：

```text
Stage III V2 在有限设置下能通过 validation 敏感性门槛，但在冻结配置后的 test stream 上未能泛化。
```

这意味着下一步不能继续拿 test 结果调参，而应该做：

1. 多个 validation window；
2. rolling validation；
3. 多 seed；
4. 最终只在完全冻结配置后跑一次 test。

---

## 17. 2026-05-25 代码仓二次核查后的专家诊断

这一节专门回答你最新提出的四个小问题，并把它们放回完整的深度学习实验链条中解释：

1. `AdaptationGenerator` 在训练时到底有没有接入。
2. 当前 generator 网络结构设计得如何。
3. `phi_v - phi_i` 表示空间未严格对齐应该怎么解决。
4. `eval` 阶段 VAE 仍然随机到底对不对。

结论先说清楚：

```text
legacy Stage III 不能支撑论文中的强 claim。
V2 Stage III 已经修复了很多工程问题，但最新 validation/test 证据仍然不能证明 real drift-conditioned adaptation 能稳定泛化。
```

### 17.1 legacy `AdaptationGenerator` 是否真的没有训练

原始路径中的问题仍然成立：

```text
train/online_loop.py
models/adaptation.py
```

在 legacy 在线循环里，`AdaptationGenerator` 是在线阶段直接新建的：

```text
run_online_adaptation(...)
  -> gen = AdaptationGenerator(...)
  -> 用 gen 生成 delta
```

但这个 legacy 路径没有一条完整的训练链：

```text
optimizer = Adam(gen.parameters(), ...)
loss.backward()
optimizer.step()
save generator checkpoint
load generator checkpoint for online test
```

所以原始 Stage III 的科学含义不是：

```text
模型学会了 drift -> parameter update。
```

而更接近：

```text
模型把一个随机初始化的 generator 插入在线推理流程中，产生一次性参数扰动。
```

这在论文里不能写成“learned drift-conditioned generator”。如果没有训练目标和 checkpoint，generator 不是被学习出来的模块。

V2 路径已经补了训练接口：

```text
train/stage3_generator_training.py
train/run_stage3_v2_sensitivity_audit.py
models/adaptation_v2.py
```

V2 会构建 `LayerAwareHyperLoRAGenerator`，用 `stage3_generator_train_step(...)` 更新 generator 参数，并保存独立 checkpoint：

```text
checkpoints/stage3_v2/.../generator.pt
```

因此现在要区分两件事：

| 路径 | generator 是否训练 | 是否可用于论文强 claim |
| --- | --- | --- |
| legacy Stage III | 否 | 否 |
| Stage III V2 | 是 | 仍需看 sensitivity/test 是否通过 |

训练接入只是必要条件，不是充分条件。V2 虽然已经训练 generator，但最新实验显示 real drift 仍然不能稳定优于 zero/shuffled/sign-flipped drift。

### 17.2 legacy generator 网络结构为什么不理想

legacy `AdaptationGenerator` 的核心问题是：

```text
zero drift 也可能产生非零 delta。
```

直观解释：

如果 drift 表示“没有概念漂移”，那么 generator 应该输出：

```text
delta = 0
```

这叫 identity-preserving。意思是：

```text
没有漂移时，不要改模型。
```

但 legacy 结构使用了带 bias 的 Linear 和类似下面的缩放：

```text
scale = sigmoid(linear(drift)) + 1.0
delta = (scale - 1.0) * parameter
```

即使 `drift = 0`，`linear(0)` 仍然可能等于 bias，所以：

```text
scale != 1
delta != 0
```

这会造成严重解释问题：

```text
Stage III 的效果可能来自固定扰动或 bias，而不是来自 drift 内容。
```

legacy 结构还有两个问题：

1. 输出主要像正向缩放，不自然表达“向上调”和“向下调”的有符号方向。
2. 原始在线逻辑曾把向量 drift 做 mean 压成标量，再 repeat 成固定维度，丢失了 embedding 的方向信息。

V2 做了更合理的结构修正：

```text
models/adaptation_v2.py
```

V2 generator 的设计目标是：

1. zero drift 输出 zero delta。
2. drift 是向量输入，不再强行压成一个标量。
3. 支持 layer-aware HyperLoRA，对不同目标层生成不同低秩扰动。
4. delta 有 clamp 和 prediction shift gate，避免参数扰动过大。

但 V2 当前仍有科学风险：

```text
generator 会根据 drift 产生扰动，不等于扰动方向一定能降低预测误差。
```

最新 Panama 结果显示：

```text
validation real drift 可以通过 gate，
test real drift 反而变差。
```

所以现在的判断应是：

```text
V2 generator 结构比 legacy 合理，但它还没有被实验稳健证明为有效的 concept-drift adapter。
```

### 17.3 `phi_v - phi_i` 为什么不是天然可信的 drift

你可以把 embedding 想象成坐标系里的点。

如果两个向量处在同一个坐标系中：

```text
phi_v - phi_i
```

才有清楚含义。

但如果它们来自两个没有对齐的编码器，就像：

```text
phi_i 使用地图 A 的经纬度
phi_v 使用地图 B 的经纬度
```

这时直接相减可能没有物理意义。

legacy 代码中的风险是：

```text
encoder_i 和 encoder_v 目标不同
没有共享投影空间
没有显式 alignment loss
没有 train-only calibration
没有证明两个 embedding 的坐标轴可比较
```

因此论文不能直接写：

```text
phi_v - phi_i represents concept drift.
```

更谨慎的说法应该是：

```text
phi_v - phi_i is used as a proxy representation-shift signal.
```

也就是：

```text
它是一个表示漂移代理信号，不等于已经被证明的概念漂移真值。
```

V2 已经采用了更合理的方向：

```text
models/aligned_representation_v2.py
train/offline_aligned_representation_v2.py
```

核心思路是：

```text
shared trunk
  -> invariant representation phi_i
  -> variant residual delta_phi_v
  -> phi_v = phi_i + delta_phi_v
```

这样 `delta_phi_v` 的含义比 `phi_v - phi_i` 更清楚：

```text
variant 分支只学习相对于 invariant 表示的残差。
```

但这仍然只是第一步。更完整的解决方案应该包含：

1. shared trunk：让 invariant 和 variant 从同一个基础表示出发。
2. residual variant：让 variant 只解释 invariant 没解释的部分。
3. train-only standardization：只用训练集拟合 drift 均值和方差，不能用 validation/test。
4. orthogonality 或 HSIC：降低 invariant 和 variant 互相复制。
5. VICReg variance/covariance：防止表示 collapse 成常数或冗余维度。
6. contrastive drift calibration：让真实时序 drift 和错误配对 drift 在表示空间上可区分。
7. rolling validation：证明这种 drift 信号不是只在某一小段有效。

当前 V2 已经加入了其中一部分：

```text
delta_phi_v
TrainSetDeltaStandardizer
HSIC / VICReg
zero-init variant branch
```

但实验结果仍未通过 held-out test gate，所以现在不能把它升级成论文主贡献。

### 17.4 eval 阶段 VAE 随机采样到底对不对

这个问题需要分训练阶段和评估阶段。

训练阶段 VAE 随机采样是正常的：

```text
z = mu + std * epsilon
```

这叫 reparameterization trick。它的作用是让模型学习一个潜在分布，而不是只学习一个确定向量。

但普通 validation/test 阶段默认继续随机采样是不合适的。

原因很简单：

```text
同一个模型
同一个输入
同一个 checkpoint
```

如果每次评估都因为随机采样给出不同预测，那么 MSE/MAE 会混入随机噪声。Stage III sensitivity 更容易被污染，因为你要比较：

```text
real drift
zero drift
shuffled drift
sign-flipped drift
```

这些差异本来就可能很小。如果 VAE 自己每次都随机跳动，就很难判断变化来自 drift 还是来自 sampling noise。

所以标准做法是：

| 阶段 | 推荐做法 |
| --- | --- |
| train | 采样 `z = mu + std * epsilon` |
| validation/test 默认 | 使用 `mu`，确定性评估 |
| uncertainty analysis | 明确做 Monte Carlo 多次采样，并报告均值/方差 |

V2 中已经修成：

```text
model.train(): sample z
model.eval(): use mu_z by default
```

对应文件：

```text
models/factor_mechanism_v2.py
```

论文中应该这样写：

```text
During deterministic validation and testing, the VAE latent variable is set to its posterior mean.
```

如果以后想写不确定性分析，则必须明确：

```text
We perform Monte Carlo evaluation with K samples and report mean plus standard deviation.
```

否则不能把随机单次评估的结果当成稳定 MSE/MAE。

### 17.5 二次核查发现的实验协议风险

这次代码仓二次核查还发现两个和模型结构无关、但对论文可信度很重要的协议问题。

#### 风险 A：缺失值是在全时间轴上插补后再切分

当前 V2 已经修复了一个旧泄漏：

```text
先 raw split
再分别 build windows
```

对应文件：

```text
train/stage3_v2_data.py
```

这避免了滑窗跨 train/val/test 边界。

但还有一个更早发生的风险：

```text
data/readers.py
load_tabular_timeseries(...)
  -> sort full dataframe
  -> time_interpolate / ffill / bfill
  -> 返回完整 series
  -> 后面才 raw split
```

其中 `time_interpolate` 和 `bfill` 都可能使用未来观测值。

例如训练段最后一个缺失点，如果使用验证段的第一个真实值做插补，那么训练数据已经看到了未来。

因此当前不能写：

```text
The whole preprocessing pipeline is strictly causal and leakage-free.
```

只能写：

```text
The V2 windowing protocol avoids cross-split sliding-window leakage, while missing-value imputation remains a documented risk requiring split-safe preprocessing.
```

最小修复方案是：

1. 新增 split-safe preprocessing。
2. 先按 raw timestamp 切 train/validation/test。
3. 对 train 只用 train 内过去信息插补。
4. 对 validation/test 只允许使用该时刻之前已经可见的信息。
5. 禁用全局 `bfill()`。
6. 禁用跨 split `time_interpolate()`。
7. 在 manifest 中记录 imputation 是否 split-safe。

#### 风险 B：部分 sensitivity counterfactual 不是严格在线因果对照

`real` 模式本身是顺序在线的：

```text
online_batch_size = 1
stream is consumed batch by batch
```

但 `shuffled` 和 `random_matched_norm` 要先收集整段 eval split 的 drift，再做置换或随机匹配。这些是很有用的离线诊断，但不是可部署的在线 baseline。

所以它们应该被称为：

```text
offline diagnostic interventions
```

而不是：

```text
online causal baselines
```

我已经在 V2 runner 中新增字段区分：

```text
intervention_online_causal
intervention_protocol
```

含义是：

| mode | 协议含义 |
| --- | --- |
| real | streaming-causal diagnostic |
| zero | streaming-causal diagnostic |
| sign_flipped | streaming-causal diagnostic |
| amplified_* | streaming-causal diagnostic |
| shuffled | offline diagnostic |
| random_matched_norm | offline diagnostic |

这能防止论文把离线干预误写成真实部署可用的在线对照。

### 17.6 新增 rolling validation gate 的原因

之前出现了：

```text
validation gate pass
test gate fail
```

这说明单个 validation slice 可能太偶然。

因此 V2 runner 现在新增：

```text
rolling_validation_sensitivity_summary.csv
rolling_validation_claim_gate.csv
rolling_validation_claim_gate_aggregate.csv
```

对应代码：

```text
train/stage3_sensitivity_gate_v2.py
train/run_stage3_v2_sensitivity_audit.py
```

逻辑是：

```text
把 validation split 按时间顺序切成多个连续窗口
每个窗口独立跑 real / zero / shuffled / sign-flipped / ...
每个窗口独立过 gate
最后计算 pass rate 和 worst-window gain
```

只有当多个 validation windows 稳定通过，才允许把该配置冻结后拿去 test。

这一步解决的问题是：

```text
不要因为某一个 validation 小片段表现好，就误以为机制已经稳定。
```

当前 `configs/stage3_v2_strong.yaml` 已设置：

```text
validation_windows: 3
rolling_min_window_pass_rate: 1.0
```

含义是：

```text
3 个 validation windows 必须全部通过，才算 rolling validation 通过。
```

### 17.7 极其详细的修改方案

下面是我建议的下一阶段修改路线。注意：这些不是论文措辞，而是工程和科研验证任务。

#### Tier 0：先保证实验协议可信

优先级最高。因为如果协议有泄漏，模型再好也不能写成可信结论。

任务 0.1：实现 split-safe missing-value preprocessing。

验收标准：

```text
train segment 的缺失值不能使用 validation/test 的值。
validation/test 的缺失值不能使用未来时刻的值。
manifest 明确记录 split_safe_imputation=true。
增加边界缺失值单元测试。
```

任务 0.2：保留 raw split before windowing。

验收标准：

```text
train/validation/test 的 target index 不重叠。
窗口输入不能跨 split 边界。
```

任务 0.3：继续强制 online batch size = 1。

验收标准：

```text
causal_online=true 时 batch_size > 1 必须报错。
```

任务 0.4：区分 online-causal 和 offline diagnostic interventions。

验收标准：

```text
CSV 中每个 mode 有 intervention_protocol 字段。
论文中 shuffled/random_matched_norm 只能叫 diagnostic interventions。
```

#### Tier 1：修复 Stage III 机制本身

任务 1.1：彻底弃用 legacy untrained generator 作为论文证据。

验收标准：

```text
论文表格不再引用 legacy Stage III 作为正向主结果。
如果保留，只能作为 diagnosed flawed baseline 或 ablation。
```

任务 1.2：只使用 V2 trained generator。

验收标准：

```text
generator 有训练 loss。
generator 有 optimizer。
generator checkpoint 可复现加载。
online replay 只加载已训练 checkpoint。
```

任务 1.3：保证 zero drift identity。

验收标准：

```text
zero drift -> delta norm = 0 或数值上接近 0。
zero drift -> prediction shift = 0 或数值上接近 0。
```

任务 1.4：保证 drift direction sensitivity。

验收标准：

```text
real 和 sign_flipped 的 delta direction 不同。
real 和 shuffled 的 prediction shift 不同。
amplified drift 在 clamp 前产生更大 delta norm。
```

任务 1.5：训练目标加入反事实比较。

验收标准：

```text
real adapted loss < base loss
real adapted loss < zero drift adapted loss
real adapted loss < sign-flipped adapted loss
real adapted loss < shuffled adapted loss
prediction shift 不超过预算
```

#### Tier 2：修复 representation drift 的语义

任务 2.1：默认使用 `delta_phi_v`，而不是未校准的 `phi_v - phi_i`。

验收标准：

```text
drift_source=delta_phi_v
calibrate_delta=true
delta standardizer 只在 train split 上 fit
```

任务 2.2：增强 representation disentanglement。

可选技术：

```text
orthogonality loss
HSIC loss
VICReg variance/covariance
contrastive drift ranking
temporal smoothness
```

验收标准：

```text
variant 不 collapse。
variant 不简单复制 invariant。
delta_phi_v 对真实时序变化有响应。
```

任务 2.3：加入漂移事件级分析。

验收标准：

```text
按季节、预测步长、负荷高峰、天气突变、节假日或分布漂移强度分组报告。
不能只报告全局平均 MSE/MAE。
```

#### Tier 3：重建证据链

任务 3.1：rolling validation selection。

验收标准：

```text
至少 3 个 validation windows。
最好 5 个 validation windows。
selection 只看 validation，不看 test。
```

任务 3.2：多 seed。

验收标准：

```text
至少 seed = 3 个。
报告 mean +/- std。
如果样本足够，做 paired test 或 bootstrap CI。
```

任务 3.3：多数据集。

验收标准：

```text
Panama / Tetouan / Spain 分开报告。
不要只取平均掩盖某个数据集失败。
每列同时报告 MSE 和 MAE。
```

任务 3.4：消融必须包含 sensitivity controls。

最低表格：

```text
base
Stage I
Stage II
Stage III V2 zero
Stage III V2 real
Stage III V2 sign-flipped
Stage III V2 shuffled diagnostic
```

如果 real 不能优于 zero/sign-flipped/shuffled，就不能说 drift 被有效利用。

#### Tier 4：论文 claim 改写规则

如果 Stage III V2 在 rolling validation 和 held-out test 上都通过，可以写：

```text
drift-conditioned transient adaptation
```

如果它只优于 base，但不优于 zero/shuffled/sign-flipped，只能写：

```text
transient adaptation improves predictions, but current evidence does not isolate drift conditioning as the cause.
```

如果它在 test 上不稳定或变差，应写：

```text
Stage III is retained as a diagnostic ablation and reveals the difficulty of learning reliable drift-conditioned updates under temporal distribution shift.
```

现在最安全的创新点重构是：

```text
The current evidence supports a rigorously audited online adaptation framework with explicit drift-sensitivity diagnostics, rather than a confirmed SOTA drift-conditioned generator.
```

中文：

```text
当前证据更适合支持“带显式漂移敏感性诊断的在线适应审计框架”，而不是“已经被证实的 SOTA 漂移条件生成器”。
```

### 17.8 当前最小安全下一步

不要立刻改论文主结论，也不要继续用 test 调参。

最小安全下一步是：

```text
实现 split-safe imputation
运行 rolling validation smoke test
如果 rolling validation 通过，再冻结配置跑一次 test report-only
```

如果 rolling validation 仍失败，那么论文应该主动降级 Stage III：

```text
从核心贡献改为 negative diagnostic ablation。
```

这样反而比强行写 SOTA 更有科研完整性。

---

## 18. V2.1 修复后的实验结果：为什么仍不能写成显著提升

本轮我继续做了两类实质性修复：

1. 修复实验协议：新增 split-safe preprocessing。
2. 修复表示对齐：新增 Projected Residual Alignment V2.1。

### 18.1 split-safe preprocessing 解决了什么

之前的风险是：

```text
全序列先插补 -> 后切分 train/validation/test
```

这可能让训练段缺失值用到未来 validation/test 的信息。

现在 V2 路径改成：

```text
先按时间顺序确定 train/validation/test 边界
再只用历史可见值做 ffill
段首缺失只用 train-only median
```

这意味着 Stage III V2 runner 的 preprocessing manifest 已能记录：

```text
split_safe_imputation = true
preprocessing_uses_future_values = false
imputation_uses_bfill = false
```

这是一个真实修复。

### 18.2 Projected Residual Alignment V2.1 解决了什么

原来 `delta_phi_v` 虽然是 residual，但仍可能和 `phi_i` 混在一起。

V2.1 增加了硬投影：

```text
delta_raw -> 去掉沿 phi_i 方向的平行分量 -> delta_phi_v
```

所以 `delta_phi_v` 更像是 invariant 表示之外的正交残差信号。

实验中 `parallel_ratio` 约为：

```text
seed 42: 3.84e-08
seed 43: 2.84e-08
```

这说明几何正交约束确实生效。

### 18.3 但预测效果没有通过门槛

我运行了较大的 bounded retraining：

```text
dataset: Panama
seeds: 42, 43
base epochs: 2
aligned epochs: 2
generator epochs: 4
train/generator/eval cap: 60 batches
validation windows: 3
modes: real / zero / shuffled / sign_flipped / random_matched_norm
```

结果：

| seed | rolling validation pass rate | validation gate | test MSE gain | test MAE gain | test gate |
| --- | ---: | --- | ---: | ---: | --- |
| 42 | 0.6667 | fail | 0.4785% | -0.2272% | fail |
| 43 | 0.6667 | fail | 0.2498% | 0.1117% | fail |

这是什么意思？

1. 两个 seed 的 rolling validation 都只通过 2/3 个窗口。
2. validation 总 gate 都失败。
3. test MSE 比之前的负结果有所改善，但 test 是 report-only，不能用来反向证明方法成立。
4. MAE 不稳定，seed 42 仍然变差。

所以现在最诚实的判断是：

```text
V2.1 让 drift representation 更干净了，
也让实验协议更可信了，
但还没有把这些优势稳定转化成在线预测收益。
```

### 18.4 论文应该怎么写

不能写：

```text
Stage III V2.1 significantly improves online load forecasting.
Stage III V2.1 reliably captures concept drift.
```

可以写：

```text
We introduce a split-safe and counterfactual-audited Stage III variant with projected residual alignment. Although the representation geometry is improved, bounded rolling-validation experiments show that the current drift-conditioned adaptation pathway does not yet yield reliable held-out gains.
```

中文含义：

```text
我们提出了一个更严格、更可审计的 Stage III 变体；它改善了表示几何和实验协议，但当前还没有稳定证明能带来 held-out 在线预测增益。
```

### 18.5 当前最小安全下一步

不要继续用 test 调参。

下一步应该只看 validation：

```text
定位 failed validation window 0 为什么失败。
```

具体要看：

1. 该窗口的 drift norm 是否异常；
2. real / zero / shuffled 的 prediction shift 差异；
3. generator 是否在该窗口产生方向错误的 delta；
4. 该窗口是否对应特殊季节、负荷水平或天气状态；
5. 是否需要让 generator 学会“何时不更新”，而不是每次都强行 perturb。

---

## 19. 2026-05-25 二次全仓核查：四个核心问题的专家诊断与修改方案

本节专门回答你提出的四个问题：

1. 当前 `AdaptationGenerator` 在训练时是否真的接入？
2. 当前 generator 网络结构设计得如何？
3. `phi_v - phi_i` 表示空间未严格对齐应该如何解决？
4. `eval()` 阶段 VAE 仍然随机到底对不对？

本节的判断基于 Stage III 相关代码路径的重新核查，包括：

- `models/adaptation.py`
- `models/adaptation_v2.py`
- `models/factor_mechanism.py`
- `models/factor_mechanism_v2.py`
- `models/aligned_representation_v2.py`
- `train/run_experiment.py`
- `train/online_loop.py`
- `train/online_loop_stage3_v2.py`
- `train/run_stage3_v2_sensitivity_audit.py`
- `train/stage3_generator_training.py`
- `train/stage3_risk_gate_v2.py`
- `train/stage3_signals.py`
- `configs/stage3_v2_strong.yaml`
- Stage III V2 tests and latest V2.1/V2.2 experiment logs.

### 19.1 最核心结论

一句话结论：

```text
legacy Stage III 不能被描述为“训练好的 drift-conditioned generator”；
V2 已经补上了训练、identity-preserving、deterministic eval、aligned residual 等关键基础，
但 V2 仍未通过 rolling validation 和 counterfactual gate，
因此还不能写成“已证明可靠捕获 concept drift 并显著提升在线预测”。
```

更具体地说：

| 问题 | legacy 状态 | V2 当前状态 | 论文 claim 边界 |
| --- | --- | --- | --- |
| generator 是否训练 | 未接入训练，在线中新建后直接用 | V2 runner 有训练 step 和 checkpoint 保存 | legacy 不可用于“learned generator”叙事；V2 只能说“具备训练机制” |
| zero drift 是否 identity | 不是，zero drift 也会产生非零 delta | 是，V2 设计上 zero drift -> zero delta | V2 可写为 identity-preserving；legacy 不能 |
| drift 是否保留向量方向 | legacy 主路径压成标量 | V2 vector mode 保留方向 | 只能用 V2 路径支持向量 drift 叙事 |
| `phi_v - phi_i` 是否同空间 | legacy 不是严格同空间 | V2 shared residual 改善同空间风险 | V2.1 改善几何，但未证明 drift 语义稳定 |
| eval VAE 随机性 | legacy eval 仍随机 | V2 默认 eval 用 `mu_z` | 历史 legacy 指标需谨慎；V2 主实验应使用 deterministic eval |
| 最新预测证据 | 机制证据不足 | V2.2 risk gate 仍未通过 validation/test gate | 不能写显著有效，只能写审计与负结果 |

---

## 20. 问题一：`AdaptationGenerator` 当前是否在训练时接入？

### 20.1 legacy 路径：没有真正接入 generator 训练

legacy generator 定义在：

```text
models/adaptation.py:76-116
```

核心逻辑是：

```python
class AdaptationGenerator(nn.Module):
    def __init__(self, drift_dim, rank, max_dim):
        self.w1 = nn.Linear(drift_dim, rank)
        self.w2_alpha = nn.Linear(rank, max_dim)
        self.w2_beta = nn.Linear(rank, max_dim)

def build_param_delta(params, drift, generator):
    alpha, beta = generator(drift, din, dout)
    scale = alpha.unsqueeze(1) * beta.unsqueeze(0)
    delta = (scale - 1.0) * p
```

但是 legacy online loop 中，generator 是在在线评估函数里直接新建的：

```text
train/online_loop.py:97-102
```

代码含义是：

```python
gen = AdaptationGenerator(...).to(device)
```

然后在 Stage III 中直接使用：

```text
train/online_loop.py:200-204
```

也就是：

```python
drift_scalar = d_v.mean()
drift_vec = normalized_scalar.repeat(...)
delta = build_param_delta(dict(model.named_parameters()), drift_vec, gen)
```

这条 legacy 链路缺少以下关键环节：

```text
optimizer = Adam(gen.parameters())
loss = prediction_loss_after_adaptation
loss.backward()
optimizer.step()
save generator checkpoint
load generator checkpoint
online replay uses loaded generator
```

所以 legacy Stage III 的真实含义不是：

```text
generator 学会了 drift -> parameter update 的映射。
```

而是：

```text
在线评估时临时插入一个随机初始化 generator，
让它根据 drift 产生一次性参数扰动。
```

这就是为什么 legacy 论文表述中类似下面的说法必须删除或降级：

```text
The generator parameters are optimized using the post-update prediction error.
The learned generator proactively adapts parameters under variant fluctuations.
```

更诚实的 legacy 表述只能是：

```text
Stage III applies a transient parameter perturbation based on a representation-discrepancy signal.
```

甚至还要补一句：

```text
The legacy implementation does not provide evidence that the perturbation generator is separately trained.
```

### 20.2 V2 路径：已经有训练机制，但还不是完整生产闭环

V2 的 generator 训练路径在：

```text
train/stage3_generator_training.py
train/run_stage3_v2_sensitivity_audit.py
```

其中 `stage3_generator_train_step()` 做了几件重要事情：

```text
train/stage3_generator_training.py:60-159
```

1. 冻结 base model。
2. 用当前 drift 生成 delta。
3. 通过 differentiable functional forward 得到 adapted prediction。
4. 用 adapted MSE、delta 正则、zero drift 正则、prediction shift 正则、counterfactual compare loss 等训练 generator。
5. 检查 base model 参数没有被误改。

V2 runner 中真正训练 generator 的入口是：

```text
train/run_stage3_v2_sensitivity_audit.py:503-610
```

并保存 checkpoint：

```text
train/run_stage3_v2_sensitivity_audit.py:1117-1123
```

这说明 V2 已经修复了 legacy 中最严重的“generator 完全不训练”问题。

但 V2 仍有三个重要缺口。

#### 缺口 1：主实验入口仍是 legacy

`train/run_experiment.py` 仍然构建 legacy `FactorMechanism`，并调用 legacy `run_online_adaptation`。也就是说，如果论文表格仍来自旧主入口，就不能把 V2 的修复自动解释为论文结果已经修复。

安全处理方式：

```text
legacy results 和 V2 results 必须分开记录。
不能用 V2 代码解释 legacy 表格。
```

#### 缺口 2：V2 runner 保存了 checkpoint，但在线评估主要传的是内存对象

V2 有：

```text
save_stage3_generator_checkpoint()
load_stage3_generator_checkpoint()
```

但当前 V2 audit runner 的主要流程是：

```text
训练 generator -> 保存 checkpoint -> 继续把内存里的 generator 传给 online loop
```

还不是严格的：

```text
训练 generator -> 保存 checkpoint -> 重新加载 checkpoint -> 在线 replay
```

这在科研复现上很重要。因为真正部署或复现实验时，应该证明：

```text
同一个 checkpoint 文件
同一个模型
同一个配置
可以重新加载并得到同样结果。
```

#### 缺口 3：`online.stage3.generator_checkpoint` 配置目前更像记录字段

配置文件里有：

```text
configs/stage3_v2_strong.yaml:152
online.stage3.generator_checkpoint: ""
```

但当前在线 loop 接口是直接接收 `generator` 对象：

```text
train/online_loop_stage3_v2.py:47-55
```

如果 `generator=None` 且 `allow_untrained_generator=false`，它会报错；如果允许未训练，它会新建诊断 generator。它本身没有负责读取 `generator_checkpoint`。

所以后续应新增一个 V2-only 的 artifact loading wrapper，而不是让在线 loop 自己变得混乱。

建议新增：

```text
train/stage3_v2_artifacts.py
```

职责：

```text
load base_model checkpoint
load aligned_representation checkpoint
load delta_standardizer checkpoint
load generator checkpoint
load risk_gate checkpoint
validate schema and parameter shapes
return Stage3V2ArtifactBundle
```

---

## 21. 问题二：当前 generator 网络结构设计得如何？

### 21.1 legacy generator 的结构问题

legacy generator 的输入是 drift vector，输出是 `alpha` 和 `beta` 两个向量，再生成缩放矩阵：

```text
scale = alpha outer beta
delta = (scale - 1.0) * parameter
```

结构上有四个严重问题。

#### 问题 1：zero drift 不是 zero update

legacy 中：

```python
h = torch.sigmoid(self.w1(drift))
alpha_full = torch.sigmoid(self.w2_alpha(h)) + 1.0
beta_full = torch.sigmoid(self.w2_beta(h)) + 1.0
```

如果：

```text
drift = 0
```

因为 `Linear` 默认有 bias：

```text
Linear(0) = bias
```

所以：

```text
alpha != 1
beta != 1
scale != 1
delta != 0
```

这会导致一个很严重的解释问题：

```text
即使模型没有检测到 drift，Stage III 也会改模型参数。
```

所以 legacy Stage III 的改善即使出现，也可能来自：

```text
固定偏置扰动
随机初始化扰动
blend/clamp 带来的弱正则效果
```

而不是：

```text
drift-conditioned adaptation
```

#### 问题 2：输出主要是正向乘性缩放，不适合表达方向

因为 `sigmoid + 1.0` 的输出大于 1，legacy generator 更自然地表达：

```text
把某些参数放大
```

而不是表达：

```text
这个 drift 方向需要正向更新
那个 drift 方向需要反向更新
```

对 concept drift 来说，方向是核心信息。

例如：

```text
高温导致负荷升高
```

和：

```text
高温但需求响应导致负荷降低
```

可能需要相反方向的参数调整。legacy 的正值缩放结构不适合清晰表达这种方向差异。

#### 问题 3：legacy 主路径把向量 drift 压成标量

legacy Stage III 做了：

```text
d_v = phi_v - phi_i
drift_scalar = d_v.mean()
drift_vec = normalized_scalar.repeat(...)
```

这会丢掉 embedding 的维度方向。

如果 `d_v` 原本是：

```text
[温度响应变化, 周期变化, 节假日变化, 噪声变化, ...]
```

求平均以后就变成：

```text
一个全局强度数
```

这不能支撑论文里“向量 embedding difference 被 generator 利用”的说法。

#### 问题 4：对所有 2D 参数统一生成缩放，层语义不清楚

legacy `build_param_delta()` 遍历所有二维参数矩阵：

```text
models/adaptation.py:105-116
```

这有两个风险：

1. 不同层的参数语义不同，但 generator 没有 layer-aware 设计。
2. 如果目标参数维度超过 `max_dim=512`，会有潜在维度风险。

### 21.2 V2 generator 的结构改进

V2 有两个 generator。

#### V2 轻量版：`IdentitySignedSeparableAdaptationGenerator`

位置：

```text
models/adaptation_v2.py:19-88
```

它的关键改进是：

```text
bias=False
tanh
zero drift -> zero output
signed scale
```

所以它解决了 legacy 的两个基本问题：

1. `drift=0` 时不会乱改模型。
2. `drift` 符号改变时，输出可以改变方向。

但它仍是比较弱的结构：

```text
row scale + column scale
```

表达力有限，适合作为轻量 ablation，不适合作为最终强贡献的唯一 generator。

#### V2 主力版：`LayerAwareHyperLoRAGenerator`

位置：

```text
models/adaptation_v2.py:107-214
```

它更合理，因为：

1. 每个目标参数矩阵有自己的 LoRA adapter。
2. drift conditioner 是共享的，但 layer update 是 layer-specific。
3. 生成的是加性低秩 delta，而不是单纯按原参数乘性缩放。
4. `lora_b` 零初始化，所以初始状态是 identity。
5. 非零 drift 仍然能给 `lora_b` 产生梯度。

这比 legacy 结构更符合深度学习中 hypernetwork / LoRA adapter 的常见设计。

### 21.3 V2 generator 仍然没有完全解决的问题

V2 的结构比 legacy 好，但最新实验说明：

```text
结构合理 != 已经证明有效
```

#### 问题 1：训练目标和在线服务行为不完全一致

训练中主要优化：

```text
adapted_mse
delta regularization
zero regularization
shift MSE
intervention compare loss
```

但在线服务真正执行：

```text
clamp delta
compute y_candidate
if pred_shift too high: reject
if risk gate says unsafe: reject
y_final = (1-blend)*y_base + blend*y_candidate
```

训练里主要看的是 `y_adapted`，而部署用的是 `y_final`。这会造成 train/serve mismatch。

最小修复：

```text
generator training metrics 中同时记录：
base_mse
candidate_mse
blended_mse
base_mae
candidate_mae
blended_mae
pred_shift_l1
pred_shift_mse
```

进一步修复：

```text
把 loss 从 adapted_mse 改成 serving-aligned blended loss：
L = MSE(y_blended, y)
  + lambda_candidate * MSE(y_candidate, y)
  + lambda_improve * relu(MSE(y_blended,y)-MSE(y_base,y)+margin)
  + counterfactual compare loss
  + delta/shift regularization
```

#### 问题 2：counterfactual loss 还不够强

当前 V2 训练中已经有：

```text
negative_drifts = zero, sign_flipped, previous_shuffled, random_matched_norm
```

但最新 V2.1/V2.2 结果显示：

```text
sign_flipped 或 shuffled 仍可能优于 real。
```

这说明 generator 还没有稳定学到：

```text
real drift 的方向和时间配对比 counterfactual 更有用。
```

下一步不应继续只调 `drift_polarity`，而应把训练目标改成更明确的排序目标：

```text
MSE(real) + margin < MSE(zero)
MSE(real) + margin < MSE(sign_flipped)
MSE(real) + margin < MSE(shuffled)
```

并在 validation windows 上检查是否成立。

#### 问题 3：risk gate 方向科学，但当前实现仍偏早期

V2.2 已经加入 risk gate，最近一次 Panama b60 结果：

```text
run_id: codex_stage3_v22_riskgate_panama_b60_20260525
dataset: Panama
seed: 42
threshold selected: 0.8
rolling validation pass rate: 2/3
rolling validation gate: fail
```

验证集整体：

| mode | base MSE | stage3 MSE | use rate | risk rejects |
| --- | ---: | ---: | ---: | ---: |
| real | 0.5812715 | 0.5824411 | 0.1000 | 54/60 |
| zero | 0.5812715 | 0.5812715 | 0.0000 | 60/60 |
| shuffled | 0.5812715 | 0.5814775 | 0.0833 | 55/60 |
| sign_flipped | 0.5812715 | 0.5810177 | 0.1333 | 52/60 |

测试集 report-only：

| mode | base MSE | stage3 MSE | base MAE | stage3 MAE | use rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| real | 0.3289351 | 0.3280576 | 0.4855843 | 0.4867372 | 0.7500 |
| zero | 0.3289351 | 0.3289351 | 0.4855843 | 0.4855843 | 0.0000 |
| shuffled | 0.3289351 | 0.3286968 | 0.4855843 | 0.4851912 | 0.8333 |
| sign_flipped | 0.3289351 | 0.3313268 | 0.4855843 | 0.4857533 | 0.4833 |

这说明：

1. risk gate 的确学会了拒绝大量 candidate。
2. 但 validation window 0 仍失败。
3. test 的 real MSE 有小幅改善，但 MAE 变差。
4. shuffled 在 test 上 MAE 反而更好。
5. 因为 test 是 report-only，不能用 test 反向调参。

所以 risk gate 的科学方向是对的，但当前实现不能支撑强 claim。

更安全的下一步：

```text
不要继续扩大 MLP risk gate；
先实现 logistic-first / linear gate；
再做 train 内 forward-chaining cross-fit；
最后用 rolling validation 选择 threshold。
```

原因：

```text
当前 gate 训练样本只有 train stream 中同一个 generator 下的 in-sample accept label，
容易学到过于乐观的“这个 generator 在训练段什么时候好用”。
```

---

## 22. 问题三：`phi_v - phi_i` 表示空间未严格对齐，应该如何解决？

### 22.1 legacy 的根本问题：两个 encoder 同构不等于同空间

legacy 中：

```text
train/run_experiment.py:63-66
encoder_i = FlattenMLPEncoder(...)
encoder_v = FlattenMLPEncoder(...)
```

这两个网络结构相同，但参数不共享。

`encoder_i` 通过 MTIL/MTEI 路径训练：

```text
train/run_experiment.py:68-92
train/offline_invariant.py
models/invariant_learning.py
```

`encoder_v` 后续单独训练，用来预测 residual：

```text
train/run_experiment.py:93-114
```

训练目标是：

```text
residual = y - y_inv
y_var = reg_v(phi_v)
loss = MSE(y_var, residual)
```

这里没有任何约束保证：

```text
phi_i 第 k 维 和 phi_v 第 k 维含义一致
phi_i 与 phi_v 尺度一致
phi_i 与 phi_v 方向一致
phi_v 只包含 variant 信息
phi_i 只包含 invariant 信息
phi_v - phi_i 是 calibrated drift
```

深度学习里还有一个可识别性问题：

```text
如果 A 是任意可逆矩阵，
phi_v 和 A phi_v 都可能被后面的 regressor 吸收，
预测 residual 的损失不一定变差。
```

也就是说：

```text
phi_v 的坐标轴本身不唯一。
```

如果坐标轴不唯一，直接做：

```text
phi_v - phi_i
```

就没有严格语义。

所以 legacy 中更准确的说法应该是：

```text
phi_v - phi_i is a proxy representation-discrepancy signal.
```

不能写：

```text
phi_v - phi_i is the concept drift.
```

### 22.2 V2 shared residual 是正确方向

V2 的改法是：

```text
models/aligned_representation_v2.py
```

核心结构：

```text
shared trunk
  -> invariant_head -> phi_i
  -> variant_head -> delta_phi_v

phi_v = phi_i + delta_phi_v
```

这时：

```text
phi_v - phi_i = delta_phi_v
```

这个设计至少解决了“坐标系不一致”的核心问题，因为 `delta_phi_v` 是从同一个 shared trunk 出发、作为 `phi_i` 的残差学习出来的。

V2.1 又加入 projected residual：

```text
delta_raw -> remove component parallel to phi_i -> delta_phi_v
```

最近 Panama b60 中 `parallel_ratio` 接近 0，说明几何约束确实生效。

但要注意：

```text
几何正交不等于因果 drift。
```

它只说明：

```text
delta_phi_v 在向量空间中没有沿 phi_i 方向复制太多信息。
```

它不说明：

```text
delta_phi_v 一定对应真实 concept drift。
delta_phi_v 一定能提高 online prediction。
```

### 22.3 推荐解决路径

#### 路径 A：保留 shared residual 作为主路径

建议继续以 V2 为主：

```text
phi_i = invariant_head(shared_trunk(x))
delta_phi_v = variant_residual_head(shared_trunk(x))
phi_v = phi_i + delta_phi_v
drift = delta_phi_v 或 delta_phi_v(t)-delta_phi_v(t-1)
```

这比 legacy 的两个独立 MLP 相减更可辩护。

#### 路径 B：hard projection 和 soft orthogonal 都要做 ablation

当前 V2.1 用 hard projection：

```text
delta_phi_v = delta_raw - projection(delta_raw on phi_i)
```

优点：

```text
强制减少 variant residual 复制 invariant 信息。
parallel_ratio 非常低，容易审计。
```

风险：

```text
真实 drift 可能沿着原来的季节性/温度敏感方向发生。
如果硬投影太强，可能把有用信号删掉。
```

所以建议做两臂对比：

```text
V2.1-hard: hard projected residual
V2.1-soft: no hard projection, but use lambda_orth / HSIC / Barlow / VICReg
```

选择标准必须是 rolling validation，不是 test。

#### 路径 C：temporal ranking 要和下游信号一致

当前 representation loss 中的 temporal ranking 约束的是：

```text
||delta_phi_v|| 与 |y - y_inv| 的批内相邻排序
```

但在线 generator 看到的是：

```text
build_stage3_drift_vector(...)
mode = delta_vector
drift = temporal difference of delta_phi_v
normalize by abs mean
```

这意味着 representation 训练目标和 deployed drift signal 不完全一致。

建议新增一个 deployment-aligned ranking：

```text
drift_deployed(t) = normalize(delta_phi_v(t) - delta_phi_v(t-1))
score(t) = ||drift_deployed(t)||
need(t) = base residual or invariant residual
ranking loss: score(t) should be larger when need(t) increases
```

但必须只在 train split 内做，不得用 validation/test residual 训练表示。

#### 路径 D：contrastive drift calibration 是下一步最有价值的增强

推荐新增 train-only 对比学习：

正样本：

```text
(delta_phi_v(t), x_t) 的真实时间配对
```

负样本：

```text
shuffled delta_phi_v
sign-flipped delta_phi_v
random matched norm delta
different regime 的错配 delta
```

目标不是直接追求 embedding 好看，而是让模型学会：

```text
真实时间配对 drift 比错配 drift 更能解释当前 residual / adaptation need。
```

注意：

```text
不能用 full validation/test split 做 pair mining。
不能为了让 test 好看而调整 contrastive pairs。
```

#### 路径 E：CCA / Procrustes / affine aligner 只能做 legacy 诊断

`models/aligned_representation_v2.py` 中已有 `TrainSetAffineAligner`，可以把 legacy `phi_v` 映射到 `phi_i` 空间。

这适合作为诊断：

```text
如果线性校准后 legacy phi_v - phi_i 效果仍不稳定，
说明问题不只是尺度和旋转。
```

但不建议把它作为主方法，因为：

```text
它只是后处理校正，
不能保证 variant residual 的语义；
也不能证明 drift-conditioned adaptation 成立。
```

---

## 23. 问题四：`eval()` 阶段 VAE 随机到底对不对？

### 23.1 训练阶段随机是对的

VAE 的训练依赖 reparameterization trick：

```text
z = mu_z + epsilon * std_z
epsilon ~ Normal(0, I)
```

这在训练阶段是合理的，因为模型要学习一个 latent distribution，而不是只学习一个确定向量。

所以：

```text
model.train(): sample z
```

是正确的。

### 23.2 普通 validation/test 随机是不对的

legacy `FactorMechanism` 中：

```text
models/factor_mechanism.py:93-95
z = self._reparameterize(mu_z, log_var_z)
```

而 `_reparameterize()` 中：

```text
models/factor_mechanism.py:134-137
eps = torch.randn_like(std)
return mu + eps * std
```

这里没有判断：

```text
if self.training:
```

所以即使：

```python
model.eval()
```

它仍然随机采样。

这会污染：

1. offline validation MSE/MAE；
2. best checkpoint 选择；
3. online base prediction；
4. Stage III candidate prediction；
5. `pred_shift`；
6. Stage I/II rollback/accept 判断；
7. real/zero/shuffled/sign-flipped sensitivity 对比。

对于本文这种主指标是：

```text
MSE / MAE
```

的负荷点预测任务，普通验证/测试应采用：

```text
model.eval(): z = mu_z
```

也就是 deterministic evaluation。

### 23.3 什么时候 eval 随机是合理的？

只有在论文明确做不确定性分析时才合理，例如：

```text
Monte Carlo evaluation with K samples
report mean prediction and uncertainty interval
report metric mean +/- std
```

如果只是单次随机 forward，然后把结果写成主表 MSE/MAE，这是不严谨的。

### 23.4 V2 的修复是正确的，但需要硬护栏

V2 修复在：

```text
models/factor_mechanism_v2.py:36-39
```

逻辑是：

```python
if self.training or self.vae_eval_mode == "sample":
    sample
return mu
```

这很合理：

| 阶段 | 行为 |
| --- | --- |
| train | 采样 |
| eval + mean | 使用 `mu_z` |
| eval + sample | 显式 Monte Carlo / uncertainty 模式 |

但当前还有三个遗漏。

#### 遗漏 1：legacy 入口仍使用 V1

`train/run_experiment.py` 和 `train/run_experiment_stage3_delta.py` 仍然构建 legacy `FactorMechanism`。

因此历史结果如果来自这些入口，就可能受 eval 随机性影响。

#### 遗漏 2：V2 online loop 接受任意 model

`run_online_adaptation_v2()` 只是：

```text
model.eval()
```

但如果传进去的是 legacy V1，`model.eval()` 并不会停止 VAE sampling。

应新增硬护栏：

```text
if getattr(model, "use_vae", False) and not deterministic_vae_eval(model):
    raise ValueError
```

#### 遗漏 3：generator training step 也需要确定性检查

`stage3_generator_train_step()` 中也会调用 base model：

```text
base_output = model(x)
adapted_output = functional_call_with_param_delta(...)
```

如果传入 V1，训练目标本身就有随机噪声。

所以也应在 generator training step 中检查：

```text
训练 generator 时，base model 必须是 deterministic eval。
```

---

## 24. 当前 V2.2 risk gate 实验说明了什么

最新 V2.2 Panama b60 结果很有信息量。

实验设置：

```text
dataset: Panama
seed: 42
base epochs: 2
aligned epochs: 2
generator epochs: 4
train/generator/eval cap: 60 batches
validation windows: 3
risk gate: enabled
threshold grid: 0.35 / 0.5 / 0.65 / 0.8
selected threshold: 0.8
```

risk gate 训练指标：

```text
initial_loss: 0.7649
final_loss: 0.0642
train_accuracy: 0.9667
positive_rate: 0.8167
sample_count: 60
```

这个结果说明：

```text
risk gate 在训练样本上能区分 candidate 是否比 base 好。
```

但 rolling validation 结果：

```text
validation windows: 3
pass windows: 2
pass rate: 0.6667
failed window: 0
rolling gate: fail
failure reasons: window_pass_rate; real_mse_gain; real_mae_gain; counterfactual
```

这说明：

```text
训练段 gate 可分，不代表 validation 的时序 regime 中可靠。
```

尤其是 validation window 0：

```text
real_mse_gain = -0.2012%
real_mae_gain = -0.2059%
best counterfactual = sign_flipped
```

这仍然击中了 Stage III 的核心问题：

```text
real drift 不稳定优于 counterfactual drift。
```

因此 V2.2 不能写成：

```text
Risk gate solves Stage III and proves drift-conditioned adaptation.
```

只能写成：

```text
Risk-aware gating reduces some unsafe updates but does not yet establish reliable drift-conditioned gains under rolling validation.
```

---

## 25. 极其详细的修改方案

以下方案遵循一个原则：

```text
不覆盖 legacy 原始模型和训练代码；
所有机制修复继续放在 V2 路径；
先做可验证的工程护栏，再做更贵的实验。
```

### Tier 0：立即冻结论文强 claim

#### 任务 0.1：冻结 Stage III 强叙事

暂时不要写：

```text
Stage III learns concept drift.
Stage III reliably captures variant unobserved factors.
Stage III significantly outperforms SOTA due to drift-conditioned generation.
```

临时安全表述：

```text
Stage III is investigated as a transient adaptation mechanism conditioned on representation-shift signals. Current V2 audits improve the mechanism's reproducibility and geometry, but rolling-validation evidence is still insufficient to claim reliable drift-conditioned gains.
```

验收标准：

```text
PAPER_CLAIMS_MATRIX.md 中 Stage III 相关 claim 标记为 weak support / unverified。
论文主文暂不把 Stage III 写成核心已验证贡献。
```

### Tier 1：确定性和 artifact 复现护栏

#### 任务 1.1：新增 deterministic eval guard

建议新增文件：

```text
train/stage3_eval_policy.py
```

提供函数：

```python
def assert_deterministic_eval_model(model, *, context: str) -> None:
    ...
```

规则：

```text
如果 model.use_vae=False：通过。
如果 model.use_vae=True 且 model.vae_eval_mode == "mean"：通过。
如果 model.use_vae=True 但没有 vae_eval_mode：拒绝。
如果 model.vae_eval_mode == "sample"：拒绝，除非显式 mc_eval=true。
```

修改：

```text
train/online_loop_stage3_v2.py
train/stage3_generator_training.py
train/run_stage3_v2_sensitivity_audit.py
```

验收测试：

```text
tests/test_factor_mechanism_v2.py 保留。
新增 test_online_loop_stage3_v2_rejects_stochastic_vae_eval。
新增 test_stage3_generator_training_rejects_stochastic_vae_eval。
新增 test_stage3_v2_runner_rejects_sample_eval_without_mc。
```

#### 任务 1.2：新增 V2 artifact bundle loader

建议新增文件：

```text
train/stage3_v2_artifacts.py
```

职责：

```text
load generator checkpoint
load risk gate checkpoint
load delta standardizer
validate generator target shapes against model
validate feature schema
validate config hash / model shape / drift_dim
fail closed if artifact missing
```

新增数据结构：

```python
@dataclass
class Stage3V2ArtifactBundle:
    generator: nn.Module
    generator_metadata: dict
    risk_gate: Stage3RiskGateBundle | None
    delta_standardizer: TrainSetDeltaStandardizer | None
```

修改：

```text
run_stage3_v2_sensitivity_audit.py
```

让 report-only test 使用：

```text
save checkpoint -> reload checkpoint -> online replay
```

而不是只用内存对象。

验收测试：

```text
train -> save -> load -> run_online_adaptation_v2
loaded result equals in-memory result within tolerance
shape mismatch raises ValueError
missing checkpoint raises ValueError
```

### Tier 2：训练目标和在线服务一致

#### 任务 2.1：加入 train/serve parity metrics

修改：

```text
train/stage3_generator_training.py
```

训练 step 记录：

```text
base_mse
candidate_mse
blended_mse
base_mae
candidate_mae
blended_mae
candidate_shift_l1
candidate_shift_mse
delta_norm_pre_clamp
delta_norm_post_clamp
clamp_fraction
zero_delta_norm
counterfactual_mse_zero
counterfactual_mse_sign_flipped
counterfactual_mse_previous
counterfactual_mse_random_matched_norm
```

验收标准：

```text
generator_train_metrics.csv 中有 base/candidate/blended 三套指标。
online_sensitivity_summary.csv 中同名字段可对齐。
```

#### 任务 2.2：把 generator loss 改成 serving-aligned

当前主 loss 中 `adapted_mse` 是核心。建议改为：

```text
y_candidate = adapted model prediction
y_blended = (1 - blend_alpha) * y_base + blend_alpha * y_candidate

L_main = MSE(y_blended, y)
```

再加入：

```text
L_candidate = MSE(y_candidate, y)
L_improve = relu(MSE(y_blended,y) - MSE(y_base,y) + margin)
L_shift = max(0, L1(y_candidate,y_base) - shift_budget)^2
L_delta = normalized delta norm
L_zero = ||G(0)||^2
```

counterfactual ranking：

```text
L_cf = sum relu(MSE(real_blended,y) - MSE(cf_blended,y) + margin)
```

验收标准：

```text
在 train metrics 中 real_blended_mse 应逐步下降。
zero_delta_norm 保持 0。
real vs sign_flipped / shuffled 的 training margin 应提高。
```

### Tier 3：risk gate 从 MLP 改为保守、可解释、时序安全

#### 任务 3.1：新增 logistic risk gate

当前：

```text
Stage3RiskGateMLP
```

建议新增：

```text
Stage3RiskGateLogistic
```

配置：

```yaml
stage3_v2:
  risk_gate:
    model_type: logistic
```

默认先用 logistic，不用 MLP。

原因：

```text
样本少；
论文需要可解释；
MLP 容易在 train stream 上过拟合。
```

验收测试：

```text
logistic gate learns separable toy labels
checkpoint round-trip preserves model_type
MLP/logistic config 都能加载
```

#### 任务 3.2：train 内 forward-chaining cross-fit

当前 risk gate label 来自同一 train stream 和同一已训练 generator，容易 optimistic。

更科学方案：

```text
train prefix A: train base/aligned/generator
train prefix B: collect risk gate examples
validation: select threshold
test: report only
```

或者：

```text
train split 按时间切 K 个 folds
每个 fold 用过去 folds 训练 generator
在当前 fold 采 gate labels
合并 labels 训练 final gate
```

验收标准：

```text
risk_gate_train_examples.csv 中记录 label_source=forward_chaining
manifest 中记录 gate_label_split_start/end
不能用 validation/test 标签训练 gate
```

### Tier 4：表示学习与 drift 语义增强

#### 任务 4.1：soft projection vs hard projection ablation

新增配置：

```text
use_projected_residual: false
lambda_orth: >0
lambda_barlow: >0
lambda_hsic: optional
```

与当前 hard projection 对比。

判据：

```text
rolling validation pass rate
real vs zero/sign_flipped/shuffled margin
parallel_ratio
delta variance/covariance
```

#### 任务 4.2：deployment-aligned temporal ranking

新增 loss：

```text
ranking on deployed_drift = delta_phi_v(t)-delta_phi_v(t-1)
```

而不是只在原始 `delta_phi_v` norm 上排名。

验收标准：

```text
deployed_drift_norm 与 later adaptation need 的 train-only rank correlation 提升。
validation 中 real drift 不再系统性输给 sign_flipped。
```

#### 任务 4.3：train-only contrastive drift calibration

新增 train-only counterfactual pairs：

```text
real temporal pair
shuffled pair
sign-flipped pair
random matched norm pair
neighbor regime mismatch pair
```

目标：

```text
real pair score > counterfactual pair score + margin
```

严禁：

```text
用 validation/test 生成训练 pair。
用 test 反复选择 pair 策略。
```

### Tier 5：实验协议

#### 任务 5.1：最小可信确认实验

先不要跑完整高成本训练，建议：

```text
datasets: panama, tetouan, spain_energy_weather
seeds: 42, 43, 44, 45
base epochs: 3
aligned epochs: 3
generator epochs: 6
batch caps: 120
validation windows: 4
modes: real, zero, shuffled, sign_flipped, random_matched_norm
```

统计单元：

```text
dataset x seed
```

不要把 validation windows 当独立样本做显著性。

通过标准：

```text
至少 9/12 dataset-seed runs 通过 rolling validation。
pooled real_vs_best_counterfactual MSE margin 的 bootstrap 95% CI 下界 > 0。
pooled real MSE gain 的 bootstrap 95% CI 下界 > 0。
MAE 不得系统性变差。
```

#### 任务 5.2：test 只能 report-only

规则：

```text
validation 选择 threshold / config。
test 只跑一次。
test 失败不能回头改 config 后再声称成功。
```

manifest 必须记录：

```text
selection_split = validation
test_gate_role = report_only
config hash
checkpoint hash
preprocessing manifest
git status
```

### Tier 6：论文改写规则

如果 V2 后续通过：

```text
rolling validation
counterfactual margins
held-out test report-only
multi-seed public-3
```

可以写：

```text
The trained Stage III V2 module performs drift-conditioned transient adaptation under a split-safe and counterfactual-audited protocol.
```

如果只优于 base，但不优于 zero/shuffled/sign-flipped：

```text
The transient adaptation module improves prediction in some settings, but the current evidence does not isolate drift conditioning as the causal source of the gain.
```

如果 rolling validation 继续失败：

```text
Stage III should be reframed as a diagnostic ablation that exposes the difficulty of converting representation shift into reliable online parameter adaptation.
```

当前最安全的论文创新点重构是：

```text
The paper contributes a rigorously audited online adaptation framework for load forecasting, including split-safe preprocessing, deterministic evaluation, aligned representation diagnostics, and counterfactual drift-sensitivity tests. Current evidence supports the audit framework and some transient adaptation behavior, but not yet a strong claim of reliable SOTA drift-conditioned generation.
```

---

## 26. 最小安全下一步

下一步不要直接改论文主结论，也不要继续用 test 调参。

最小安全下一步是：

```text
1. 在 V2 边界加入 deterministic eval guard。
2. 加入 generator checkpoint reload -> online replay 的端到端测试。
3. 给 generator training 增加 base/candidate/blended 三套 train-serve parity metrics。
4. 把 risk gate 改成 logistic-first，并设计 forward-chaining gate label 收集。
5. 只在 validation rolling windows 上比较是否修复 window 0 失败。
```

只有当这些通过以后，才值得扩大到：

```text
public-3 x multi-seed
```

再决定是否更新论文中的 Stage III 叙事。

---

## 27. 2026-05-25 V2.3 实施后的新结论

详细实验记录：

```text
docs/audit/STAGE3_V23_SERVING_ALIGNED_EXPERIMENT_RECORD.md
```

### 27.1 已经修掉了什么

本轮 V2.3 已经把几个“代码是否可信”的问题向前推进了一步：

1. `AdaptationGenerator` 不再只是 online loop 里随机新建的模块，V2.3 会训练 generator、保存 checkpoint，并在评估前 reload；
2. generator training 不再只看 raw candidate，而是记录并优化更接近线上逻辑的 `candidate/blended/shift/clamp` 指标；
3. V2 路径拒绝 stochastic VAE eval，要求 `vae_eval_mode="mean"` 这类确定性评估；
4. `phi_v - phi_i` 不再只是两个独立 encoder 的直接相减，而是 shared residual 表示，并额外加入相邻时刻 drift transition 的约束；
5. risk gate 切到 logistic-first，降低小样本 gate 的过拟合风险；
6. runner 会在 manifest 中记录 `stage3_artifacts_reloaded_for_eval=true`，说明评估使用的是落盘 artifact。

这些是工程和科研协议修复，不等于性能贡献已经成立。

### 27.2 实验结果是否支持更新论文

不支持。

Panama b60、seed42/43 的结果是：

| seed | test MSE gain | test MAE gain | test gate | rolling validation pass rate |
| --- | ---: | ---: | --- | ---: |
| 42 | -0.2431% | -0.4971% | fail | 0.6667 |
| 43 | +0.7797% | +0.2071% | pass | 0.3333 |

这说明：

1. seed42 明显退化；
2. seed43 的 test 是正的，但 test 是 report-only，不能用来反向选择配置；
3. seed43 的 rolling validation 只有 1/3 通过；
4. 两个 seed 的平均 MAE 仍为负；
5. 因此不能写“显著提升”“稳定优于 SOTA”或“generator 已可靠学习 concept drift”。

### 27.3 当前最准确的专家判断

V2.3 让 Stage III 从“代码和论文机制不一致”变成了“机制更可审计，但性能尚未被证明”。

更具体地说：

```text
已证明：V2.3 有训练闭环、确定性评估、artifact reload、split-safe protocol、counterfactual audit。
未证明：V2.3 能稳定改善预测误差。
未证明：real drift signal 比 zero/shuffled/sign-flipped 等 counterfactual 稳定更有效。
未证明：当前 Stage III 可以支撑 SOTA 或显著提升叙事。
```

### 27.4 下一步不要做什么

不要因为 seed43 的 test pass 就改论文主结论。

原因是：

```text
validation failed
rolling validation failed
seed42 failed
test is report-only
```

如果现在更新论文，就会把一个没有通过选择协议的结果写成贡献，这属于科研完整性风险。

### 27.5 下一步应该做什么

下一步应该先实现：

```text
generator checkpoint selection on rolling validation
```

也就是说，不要只用训练流上的 loss 保存最后一个 generator，而是：

1. 每个 generator epoch 后保存 candidate checkpoint；
2. 只在 validation rolling windows 上评价；
3. 选择 rolling gate 最好的 checkpoint；
4. 只用选出的 checkpoint 跑 test report-only；
5. 如果 validation 仍不能稳定 3/3 通过，就停止性能 claim。

这一步比继续扩大 b120 更安全。因为如果 b60 的 validation 不稳，直接跑 b120 很可能变成“用更多计算寻找偶然 test 好结果”。

---

## 28. 2026-05-25 V2.5 后续：更高级对齐和 checkpoint selection 以后怎样了

详细记录：

```text
docs/audit/STAGE3_V25_CANONICAL_CKPT_SELECTION_EXPERIMENT_RECORD.md
```

### 28.1 新增了什么

V2.5 又补了两件关键事：

1. **generator checkpoint selection**：每个 generator epoch 保存一个 candidate，不再默认用最后一个 epoch；只在 rolling validation 上选择。
2. **residual drift canonicalizer**：只用训练集拟合一个 canonical 坐标变换，让 `delta_phi_v[t]-delta_phi_v[t-1]` 尽量和 residual transition 对齐。

这解决的是“机制是否更可审计”的问题，不自动解决“效果是否更好”的问题。

### 28.2 seed42/43 b60/b120 结果

| seed | budget | test MSE gain | test MAE gain | rolling validation pass rate | 结论 |
| ---: | --- | ---: | ---: | ---: | --- |
| 42 | b60 | -0.0938% | -0.3505% | 0.3333 | 失败 |
| 42 | b120 | +1.2656% | -0.4907% | 0.3333 | 仍失败 |
| 43 | b60 | -0.3250% | +0.1074% | 0.3333 | 失败 |
| 43 | b120 | +1.6361% | +1.1517% | 0.3333 | 仍失败 |

读法：

```text
b120 的 test MSE 在两个 seed 上都变好了。
但 rolling validation 没有变好，所有 run 都只有 1/3。
seed42 的 b120 MAE 仍然更差。
seed43 的 b120 虽然 test MSE/MAE 都正，但 reject rate 超过阈值。
所以这不是“显著提升”，而是“更大 budget 带来 report-only test MSE 局部改善，但没有通过预注册验证协议”。
```

### 28.3 当前最诚实的结论

V2.5 已经比原始 Stage III 更严谨：

```text
有训练 generator
有 zero-drift identity 约束
有 deterministic VAE eval
有 train-only canonical drift
有 validation-only checkpoint selection
有 artifact reload
有 counterfactual audit
```

但 V2.5 仍没有证明：

```text
稳定提升 MAE
rolling validation 稳定通过
real drift 稳定优于 counterfactual
多 seed 稳定有效
```

因此，论文中仍不能把 Stage III 写成可靠性能核心创新。

### 28.4 下一步

seed43 已经补跑完，rolling validation 仍然失败。因此应该停止把 Stage III 当性能贡献，改成：

```text
一个严格审计过的 online adaptation 诊断模块；
一个揭示 drift-conditioned parameter generation 难点的负结果；
一个用于规范 split-safe、deterministic、counterfactual-audited online evaluation 的实验框架。
```

---

## 29. 2026-05-25 大规模 full-eval seed42 三数据集结果

详细记录：

```text
docs/audit/STAGE3_V25_LARGE_SCALE_EXPERIMENT_RECORD.md
```

这轮把 evaluation 放大到完整 test split，不再只看前 60 或 120 个 online samples。训练和 generator replay budget 为 b240。

| dataset | test MSE gain | test MAE gain | rolling pass rate | reject rate | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| Panama | +0.0858% | +0.1118% | 0.0000 | 82.58% | 失败 |
| Tetouan | +0.0225% | +0.0180% | 0.0000 | 99.91% | 失败 |
| Spain Energy--Weather | +0.3882% | +0.3264% | 0.0000 | 32.46% | 失败 |
| Panama seed43 | +1.7843% | +0.9248% | 0.0000 | 58.04% | 失败 |
| Tetouan seed43 | -0.0181% | -0.0243% | 0.0000 | 99.04% | 失败 |
| Spain seed43 | -0.0075% | -0.0029% | 0.0000 | 99.87% | 失败 |
| Panama seed44 | +0.0005% | +0.0012% | 0.0000 | 99.98% | 失败 |
| Tetouan seed44 | +0.3384% | +0.2480% | 0.0000 | 83.10% | 失败 |
| Spain seed44 | +0.2603% | +0.1693% | 0.0000 | 91.46% | 失败 |

通俗解释：

```text
Panama 和 Tetouan 的改善太小，而且大多数时候 Stage III 被 gate 拒绝使用。
Spain 有一点更像真实信号：real drift 比 shuffled/sign-flipped 更好。
但 Spain 仍然没过关，因为收益小、rolling validation 全失败、reject rate 偏高，而且 amplified drift 和 real drift 几乎一样。
Panama seed43 的数值改善更明显，但也没过关，因为 rolling validation 全失败、reject rate 过高，而且 amplified drift 仍然和 real drift 几乎一样。
Tetouan seed43 则直接退化，并且仍然几乎完全被 gate 拒绝使用。
Spain seed43 也没有复现 seed42 的正向信号，说明 Spain seed42 不能单独当成稳定证据。
Panama seed44 没有复现 Panama seed43 的较大收益，说明 Panama seed43 也不能单独当成稳定证据。
Tetouan seed44 有收益，但仍然因为 reject rate 高、rolling validation 全失败、amplified drift 和 real drift 几乎一样而失败。
Spain seed44 也有收益和方向信号，但同样被 reject rate、rolling validation、amplified drift 并列卡住。
这 9 个 run 的总结果是：0/9 通过 claim gate，0/9 通过 rolling validation，平均 reject rate 约 82.94%。
```

所以目前最准确的判断是：

```text
Stage III V2.5 比原始版本更可审计，但还不是一个已经被验证有效的性能模块。
如果论文继续保留 Stage III，应该把它写成“严格审计下的 drift-conditioned adaptation 候选机制”，而不是“已证明可靠提升预测精度的核心创新”。
```

---

## 30. 2026-05-26 V3 修复和 b120 探针结果

V3 不是覆盖原始模型，而是在 V2/V2.5 的副本路径上继续增强审计能力。

新增内容：

1. **temporal residual CCA alignment**：只用训练集，把 `delta_phi_v[t]-delta_phi_v[t-1]` 和预测残差变化对齐，生成更可审计的 drift 坐标。
2. **generator-input counterfactual**：以前 `amplified_2x/5x` 先放大 raw drift，但又被 `normalize_vector=True` 除回去；现在反事实直接在 generator 输入空间做，放大不会被归一化抹掉。
3. **no-overwrite protection**：同一个 run 目录或 checkpoint 默认不允许覆盖。
4. **resolved config manifest**：每次 run 保存实际生效配置 `resolved_config.json` 和 hash。
5. **alignment diagnostics**：输出 train/validation/test 的 drift-transition 与 residual-transition 相关性、effective rank、collapse fraction。

一次 Panama seed42 b120/full-eval probe 的结果：

| 指标 | 数值 |
| --- | ---: |
| real MSE gain | 0.9338% |
| real MAE gain | 0.6660% |
| reject rate | 16.86% |
| use rate | 83.14% |
| best counterfactual | `amplified_2x` |
| rolling validation pass rate | 0/5 |
| test claim gate | fail |

这说明 V3 有一个积极变化：

```text
高拒绝率问题明显缓解，而且 real drift 在这个 seed 上确实优于 zero/shuffled/sign-flipped/random。
```

但它仍然不能作为论文强证据，因为：

```text
amplified_2x 比 real drift 更好；
rolling validation 5 个窗口全部没通过；
只有一个数据集、一个 seed；
没有统计显著性。
```

最通俗的解释：

```text
V3 让 Stage III 更像一个能工作的 adapter，但它还没有证明“真实 drift 的方向和幅值正好是最佳信息”。
如果把 drift 放大 2 倍更好，说明 generator 可能需要更强的幅值校准，而不是已经学会了真实 concept drift。
```

所以当前结论是：

```text
V3 比 V2.5 更可审计，也出现了更好的单次 report-only 数值，但仍不能写成“显著提升”或“可靠捕捉 concept drift”。
```

---

## 31. V4 又修了什么？小白版解释

V3 full run 后发现一个很关键的问题：

```text
把 drift 放大 5 倍后，输入信号确实变大了，
但 generator 产生的参数扰动几乎没有变大，
risk gate 反而觉得 amplified drift 更安全。
```

这像什么？

```text
你告诉一个学生：“正常声音读这句话。”
又告诉他：“用 5 倍音量读这句话。”
但他说出来的声音几乎一样大。
同时评分器还觉得 5 倍音量的输入更可靠。
```

这说明系统并没有真正理解“幅值”。

V4 做的事情是：训练 generator 时，不只告诉它 real drift 是正例，还把 `amplified_2x` 和 `amplified_5x` 当成 hard negative。也就是明确告诉模型：

```text
真实 drift 应该比这些被人为放大的 drift 更合理。
```

V4 的结果：

| 指标 | V3 | V4 |
| --- | ---: | ---: |
| reject rate | 24.67% | 0.80% |
| real MSE gain | 4.19% | 3.65% |
| best counterfactual | amplified_5x 更好 | real 略好 |
| rolling validation | 0/5 | 0/5 |

所以 V4 是进步吗？

```text
是。它明显修好了“经常拒绝使用 Stage III”的问题，也让 real drift 不再输给 amplified drift。
```

但 V4 够不够写进论文当强创新？

```text
还不够。
```

原因是 real 比 amplified 只好了一点点，margin 只有 `0.000138`，而我们预先设定的安全门槛是 `0.002`。同时 rolling validation 五个窗口还是全部没通过。

最安全的理解：

```text
V4 说明我们的修复方向是对的，但还没有达到论文可以声称“Stage III 可靠学习 concept drift”的程度。
```

---

## 32. 2026-05-27/28 最新 V7 结果：为什么“更干净的 drift”还是没有通过？

V7 做的是一件更严格的事：

```text
先把 phi_v - phi_i 中能被 invariant embedding 解释的部分去掉，
再只保留训练集上稳定、可校准的少数 drift 方向。
```

这相当于不再直接相信“两个 embedding 一减就是 concept drift”，而是先问：

```text
这个差值里有多少其实还是 stable/invariant 信息？
剩下的变化方向是否真的和预测误差变化有关？
```

结果发现：

| 项目 | 结果 |
| --- | ---: |
| residualizer R2 mean | 0.931512 |
| active rank | 8 |
| served-policy rolling pass | 0/5 |
| 是否保存最终 served generator | 否 |

`residualizer R2 mean = 0.931512` 的小白解释是：

```text
phi_v - phi_i 里面有很大一部分可以被 phi_i 解释。
所以你最开始担心“这个差值混入 stable long-term changes”是合理的。
```

但是，V7 把 drift 变干净以后，预测性能没有变强。最新同等预算重跑 `v7ps42g5b480diag1` 记录了所有候选 generator、所有 validation window、所有 drift mode 的细节。最佳候选是 epoch 4，但五个窗口仍然全部失败：

| Window | real MSE gain | real MAE gain | best counterfactual | margin | 结果 |
| ---: | ---: | ---: | --- | ---: | --- |
| 0 | 0.000561 | 0.000534 | amplified_2x | 0.000556 | fail |
| 1 | 0.002325 | 0.001505 | zero | 0.002325 | fail |
| 2 | 0.001524 | 0.001651 | amplified_2x | 0.000900 | fail |
| 3 | 0.000490 | 0.000634 | amplified_2x | 0.000267 | fail |
| 4 | 0.001710 | 0.000719 | amplified_2x | 0.001684 | fail |

为什么 fail？

```text
我们预先设定的 MSE gain 门槛是 0.005。
最佳窗口也只有 0.002325。

我们预先设定的 MAE gain 门槛是 0.002。
最佳窗口也没有稳定超过这个门槛。

我们还要求 real drift 明显优于 counterfactual drift。
多个窗口中 real 和 amplified_2x 的差距仍然太小。
```

所以当前最准确的理解是：

```text
V7 证明了原始 phi_v - phi_i 的确存在表示混杂问题；
但 V7 没有证明 Stage III 能可靠提升预测性能。
```

你现在不能把 Stage III 写成：

```text
我们提出的方法可靠捕捉 concept drift，并显著优于 baseline。
```

更安全的写法是：

```text
Stage III 目前更适合被写成一个经过审计的在线适应诊断模块。
它揭示了 representation drift 到 parameter adaptation 之间的困难，
但当前证据还不足以支持强性能贡献。
```
