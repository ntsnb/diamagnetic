# 第三阶段（Stage III）长周期实验总报告

生成日期：2026-06-02。  
报告性质：这是科研完整性审计报告，不是论文正文，也不是正向实验结果汇编。  
证据范围：本报告只整理当前仓库中已经存在的审计文档、实验日志路径、运行清单（manifest）、测试记录和实验台账。本次整理没有启动新的训练，也没有重新生成任何实验结果。

## 0. 阅读说明和术语翻译约定

为了避免中英混杂造成理解困难，本报告采用“中文解释在前，英文术语在括号中保留”的写法。英文术语保留的原因是代码、日志、指标列名和论文方法部分都使用这些名称。

| 英文术语 | 中文含义 | 本报告中的解释方式 |
| --- | --- | --- |
| `Stage III` | 第三阶段在线适应模块 | 写作“第三阶段（Stage III）”。 |
| `generator` | 生成参数扰动或预测修正的生成器 | 写作“生成器（generator）”。 |
| `teacher` | 用来提供残差修正方向的教师模型 | 写作“教师模型（teacher）”。 |
| `teacher-only` | 只评估教师模型，不训练或服务生成器 | 写作“只评估教师模型的诊断（teacher-only diagnostic）”。 |
| `drift` | 漂移信号，通常指表示随时间变化的信号 | 写作“漂移信号（drift signal）”。 |
| `counterfactual` | 反事实干预，用来检查模型是否真的使用了真实漂移 | 写作“反事实干预（counterfactual intervention）”。 |
| `gate` | 预先定义的通过门槛 | 写作“论断门槛（claim gate）”或“教师门槛（teacher gate）”。 |
| `rolling validation` | 按时间顺序滚动切分的验证 | 写作“滚动验证（rolling validation）”。 |
| `public-3` | 三个公开数据集组合 | 写作“三个公开数据集（public-3）”。 |
| `abstain` | 当候选模型未通过门槛时主动不发布该模型 | 写作“弃权（abstain）”。 |
| `manifest` | 记录命令、配置、路径和结果摘要的运行清单 | 写作“运行清单（manifest）”。 |
| `zero` | 零漂移反事实输入 | 写作“零漂移干预（`zero`）”。 |
| `identity` | 恒等映射，即输入不会造成预测或参数变化 | 写作“恒等映射（identity mapping）”。 |

例如，原先过短的中英混写表述已经改为完整句子：零漂移干预（`zero` intervention）在该测试中保持恒等映射（identity mapping），也就是它不会改变基线模型的预测输出。

## 1. 总结论

第三阶段目标正在向更严谨的科研方向发展，但目前还没有发展成“已经证明模型显著提升预测精度”的正向结论。更准确地说，长周期实验完成了大量必要的审计和修复工作，但这些工作主要证明了当前第三阶段机制仍然没有通过严格验证。

第一，早期第三阶段实现中存在多个会影响论文可信度的根问题。这些问题包括：生成器（generator）没有被训练、零漂移输入会产生非零扰动、向量漂移被压缩成标量、评估阶段的变分自编码器（VAE）仍然随机采样，以及不变表示和可变表示的差值（`phi_v - phi_i`）没有处在严格对齐的表示空间中。

第二，从 V2.1 到 V12e 的工作显著提高了实验协议的严谨性。这些改进包括确定性评估（deterministic evaluation）、不覆盖旧结果的运行策略（no-overwrite policy）、训练后重新加载产物再评估（artifact reload）、只用验证集选择生成器（validation-only selection）、反事实漂移敏感性检查（counterfactual sensitivity audit）、只评估教师模型的门槛（teacher-only gate），以及按预测步长控制标签可用性的约束（per-horizon label availability）。

第三，截至 V12e，所有严格的三个公开数据集（public-3）、多随机种子（multi-seed）、滚动验证（rolling validation）和反事实门槛（counterfactual gate）都没有证明当前第三阶段可以作为可靠的漂移条件在线适应模块（drift-conditioned online adaptation module）。

第四，当前最强的证据不是正向性能证据，而是负面诊断证据。V12e 表明，在按预测步长控制标签到达时间（horizon-aware label availability）并固定历史残差信息后，当前以 `phi_v - phi_i` 及其时间差分为基础的漂移管线没有稳定提供超过历史残差的增量预测价值。

第五，因此，当前论文不能声称第三阶段已经学习到概念漂移（concept drift），不能声称第三阶段显著提升预测精度，也不能把生成器作为已经验证成功的核心创新点。

当前工具中的长期目标状态为“受阻（blocked）”。这个状态不是因为实验报告没有完成，而是因为 V12e 已经把当前路径推到科学停止条件：继续扩大现有生成器训练缺乏证据基础，下一步必须先批准并实施 V13 级别的表示学习目标和教师模型目标重构。

## 2. 本报告使用的主要证据文件

下表列出本报告依赖的核心证据文件。这些文件中的数字和结论构成本报告的证据来源。

| 文件 | 中文用途说明 |
| --- | --- |
| `docs/audit/EXPERIMENT_EVIDENCE_LEDGER.md` | 这是统一的实验结果证据台账，用来追踪实验数字来自哪些日志、表格和运行目录。 |
| `docs/audit/RISK_REGISTER.md` | 这是论文与实验风险登记表，用来标记哪些实验或论文表述存在高风险。 |
| `docs/audit/PAPER_CLAIMS_MATRIX.md` | 这是论文论断与证据的对应矩阵，用来判断每个 claim 是否被证据支持。 |
| `docs/audit/REVISION_ROADMAP.md` | 这是论文修改路线图，用来规划哪些问题必须先修复。 |
| `docs/audit/STAGE3_EXPERT_DIAGNOSIS_AND_REPAIR_STRATEGY.md` | 这是第三阶段专家诊断总文档，用来汇总跨版本的机制判断。 |
| `docs/audit/STAGE3_GENERATOR_BEGINNER_EXPLANATION.md` | 这是面向初学者的第三阶段机制解释文档。 |
| `docs/superpowers/plans/2026-05-24-stage3-generator-repair-plan.md` | 这是长周期实施计划与阶段状态记录。 |

下表列出各阶段的审计记录文件。

| 阶段 | 记录文件 |
| --- | --- |
| V2、V2.1、V2.3、V2.5 | `STAGE3_V2_IMPLEMENTATION_LOG.md`、`STAGE3_V21_SPLITSAFE_EXPERIMENT_RECORD.md`、`STAGE3_V23_SERVING_ALIGNED_EXPERIMENT_RECORD.md`、`STAGE3_V25_CANONICAL_CKPT_SELECTION_EXPERIMENT_RECORD.md`、`STAGE3_V25_LARGE_SCALE_EXPERIMENT_RECORD.md` |
| V3 到 V7 | `STAGE3_V3_REPAIR_AND_SMOKE_RECORD.md`、`STAGE3_V4_COUNTERFACTUAL_HARD_NEGATIVE_RECORD.md`、`STAGE3_V5_SERVED_POLICY_COUNTERFACTUAL_GATE_RECORD.md`、`STAGE3_V5_V6_G5_CANARY_RECORD.md`、`STAGE3_V7_ALIGNMENT_RECORD.md` |
| V8 到 V10 | `STAGE3_V8_HYPERADAPTER_RECORD.md`、`STAGE3_V9_RESIDUAL_TEACHER_RECORD.md`、`STAGE3_V9_TEACHER_ONLY_DIAGNOSTIC_RECORD.md`、`STAGE3_V10_CROSSFIT_TEACHER_RECORD.md`、`STAGE3_V10_TEACHER_MATRIX_RECORD.md` |
| V11 到 V12e | `STAGE3_V11_TEACHER_MATRIX_RECORD.md`、`STAGE3_V11_SIGNAL_AND_HORIZON_ABLATION_RECORD.md`、`STAGE3_V12_OUTCOME_ALIGNED_TRANSPORT_RECORD.md`、`STAGE3_V12C_MULTI_POLICY_TEACHER_RECORD.md`、`STAGE3_V12D_CONTEXT_TEACHER_RECORD.md`、`STAGE3_V12E_LABEL_AVAILABILITY_CONTEXT_TEACHER_RECORD.md` |

## 3. 最初发现的第三阶段根问题

下表说明最初的代码问题、它们为什么会影响论文结论，以及后续修复到什么程度。

| 问题 | 代码或机制含义 | 科学后果 | 后续处理状态 |
| --- | --- | --- | --- |
| 生成器（generator）没有被训练 | 旧版 `train/online_loop.py` 直接实例化生成器，但没有发现对 `gen.parameters()` 的优化器更新。 | 如果论文写成“学习到的生成器”，实现与论文就不一致。 | V2 之后的可选新路径已经加入生成器训练，但严格实验仍然失败。 |
| 零漂移输入产生非零扰动 | 旧版生成器包含偏置项，并且使用 `sigmoid(...) + 1.0` 形式的缩放，因此零输入也可能产生参数变化。 | 零漂移干预（`zero` intervention）不能作为可信的无效对照。 | V2 之后通过无偏置结构和恒等初始化逐步修复。 |
| 向量漂移被压缩成标量 | 旧版逻辑把漂移向量做平均，然后复制成固定维度输入。 | 这会丢失向量方向信息，也与论文中的向量漂移叙述不一致。 | V2 之后使用完整漂移向量（`delta_vector`）。 |
| 评估阶段的 VAE 仍然随机 | 模型在 `eval()` 状态下仍可能使用重参数化采样。 | 相同输入可能得到不同预测，漂移敏感性检查会被随机噪声污染。 | V2 之后的审计路径使用确定性评估，通常使用均值向量。 |
| `phi_v - phi_i` 表示空间和语义未严格成立 | 旧路径中不变表示和可变表示来自不同目标，直接相减未必有稳定语义；即使在 V2.1 之后改成共享残差表示，这个差值也更像“可变长期状态相对不变规律的偏移”，不应自动等同于短期波动。 | `phi_v - phi_i` 不一定能代表第三阶段需要响应的 variant unobserved factor driven fluctuation；更接近该目标的候选信号应是相邻时刻的变化量，例如 `delta_phi_v(t)-delta_phi_v(t-1)`。 | V2.1 到 V12e 做了多轮对齐、时间差分、教师模型和历史残差对照，但仍没有验证成功。 |

这些修复是必要条件，但不是充分条件。它们只能让第三阶段机制“可以被严肃检验”，不能自动证明第三阶段机制已经成功。

## 4. 长周期实验总表

下表把每一轮实验的目标、证据和结论压缩到同一张表中。表中的“门槛”指预先定义的论断门槛或教师门槛，不能在实验后为了让结果好看而放宽。

| 阶段 | 主要改进 | 实验规模或证据 | 关键结果 | 结论 |
| --- | --- | --- | --- | --- |
| V2 原始修复 | 加入确定性评估、零漂移恒等映射、对齐残差表示、生成器训练和可选在线循环。 | 早期烟雾测试（smoke test）和有预算上限的三个公开数据集实验（bounded public-3）。 | 极小烟雾测试只证明链路可运行；早期看似正向的 bounded public-3 后来被因果安全检查（causal-safe check）降级。 | V2 建立了可检验链路，但不能作为论文正向性能证据。 |
| V2.1 | 加入安全切分预处理（split-safe preprocessing）和投影残差对齐（projected residual alignment）。 | Panama b60，seed 42 和 seed 43；测试记录为 `63 passed`。 | 两个 seed 的滚动验证通过率都是 `0.6667`，但总门槛失败；测试集 MSE 小幅正向，MAE 和反事实归因不稳定。 | 数据协议和表示几何更安全，但性能未验证。 |
| V2.3 | 加入确定性评估保护、与服务逻辑一致的生成器损失、逻辑回归风险门和产物重新加载。 | Panama b60，seed 42 和 seed 43；测试记录为 `75 passed`。 | seed 42 的测试 MSE 和 MAE 都变差；seed 43 的测试通过，但滚动验证通过率只有 `0.3333`。 | 训练闭环更真实，但效果依赖随机种子。 |
| V2.5 | 加入滚动验证选择生成器检查点（generator checkpoint）、训练集残差漂移规范化和拒绝率口径修正。 | Panama b60/b120，随机种子 42/43；后续扩展到三个公开数据集 b240。 | b120 有时改善测试 MSE，但四个 Panama run 的滚动通过率都只有 `0.3333`；public-3 b240 的 `9/9` 测试门槛全部失败。 | 工程修复有效，但科学论断失败。 |
| V3 | 加入时间残差典型相关分析（temporal residual CCA）和生成器输入空间反事实干预（generator-input counterfactual intervention）。 | Panama seed 42 b120 完整评估。 | 测试 MSE 和 MAE 分别为 `+0.9338%` 和 `+0.6660%`，但放大两倍漂移（`amplified_2x`）反事实优于真实漂移，滚动验证为 `0/5`。 | 审计性增强，但真实漂移归因失败。 |
| V4 | 加入 amplified hard negatives 和逐步指标追踪。 | Panama seed 42 full run。 | 测试 MSE 和 MAE 分别为 `+3.6507%` 和 `+3.0443%`，拒绝率降到 `0.80%`，但反事实 margin 只有 `0.000138`，低于 `0.002`，滚动验证为 `0/5`。 | 这是部分机制修复，不是投稿级证据。 |
| V5/V5.1 | 加入服务策略选择（served-policy selection）、反事实感知风险门和弃权机制。 | Panama seed 42 g3_b240。 | V5 过度保守，真实漂移拒绝率为 `1.0`；V5.1 测试 margin 为 `0.005633`，但滚动验证只有 `2/5`。 | 方向有价值，但还没有通过滚动验证。 |
| V5 g5/g8 与 V6 | 增加生成器预算，并测试 whitened-Procrustes 对齐。 | Panama seed 42 g5_b480、g8_b960 和 V6 g5_b480。 | V5 g5 最好达到 `4/5`，但仍弃权；V5 g8 降到 `3/5`；V6 只有 `1/5`。 | 更多训练量和更复杂对齐没有稳定救回模型。 |
| V7 | 加入残差化降秩正则 CCA（residualized RRCCA），对 `phi_v - phi_i` 中由 `phi_i` 可解释的部分做残差化。 | Panama seed 42 g5_b480。 | 内部表示对齐门槛（alignment gate）通过，活跃秩（active rank）为 `8`，但服务策略（served-policy）的滚动验证为 `0/5`。 | 证明了原始差值存在泄漏风险，但性能不成立。 |
| V8 | 生成器从只改权重扩展到同时改权重和偏置。 | Panama seed 42 的同预算先导验证实验（same-budget canary）。 | 滚动验证为 `0/5`，最差窗口 MSE 和 MAE 分别为 `-0.008390` 和 `-0.005176`。 | 更大生成器容量使结果更差。 |
| V9 | 加入只用训练集拟合的残差教师模型（train-only residual teacher）、教师感知 RRCCA（teacher-aware RRCCA）和残差基生成器（residual-basis generator）。 | Panama seed 42 的同预算先导验证实验。 | 训练残差 R2 只有 `0.0926`，滚动验证为 `0/5`，零漂移常是最好反事实。 | 结构更合理，但教师方向不迁移。 |
| V9 teacher-only | 不经过生成器，直接评估教师模型修正预测。 | Panama seed 42 的同预算实验。 | 滚动验证为 `0/5`，最差窗口 MSE 和 MAE 分别为 `-0.197784` 和 `-0.118167`。 | 失败发生在生成器上游。 |
| V10 | 加入交叉拟合泄漏投影（cross-fitted leakage projection）和训练集校准缩放（train-calibrated shrinkage）。 | Panama seed 42 和三个公开数据集矩阵实验。 | Panama seed 42 达到 `3/5`，但 public-3 是 `0/9` teacher gates 和 `8/45` rolling windows。 | 局部修复不能泛化。 |
| V11 | 加入最差时间块校准（worst-block calibration）和失败时归零的缩放策略（fail-closed shrinkage）。 | 三个公开数据集的 seeds 42/43/44。 | `0/9` teacher gates，`0/45` rolling windows，所有 alpha 都为 `0.0`。 | 这是安全弃权，不是性能修复。 |
| V11 ablations | 检查水平向量（`level_vector`）和分块预测步长 alpha（`block_horizon`）是否能救回结果。 | 三个公开数据集。 | `level_vector` 仍为 `0/9` 和 `0/45`；`block_horizon` 有 `4/9` 非零 alpha，但仍为 `0/9` 和 `0/45`。 | 失败不是简单信号模式或单一 alpha 的问题。 |
| V12 | 加入结果对齐传输（outcome-aligned transport）和共享潜在锚点（shared latent anchor）。 | 三个公开数据集。 | 表示对齐门槛（alignment gate）为 `9/9`，但教师门槛（teacher gate）为 `0/9`，滚动窗口（rolling windows）为 `0/45`。 | 表示对齐通过不等于预测修正通过。 |
| V12b/V12c | 加入按时间顺序的折外教师模型（chronological OOF teacher）和多策略 alpha 诊断（multi-policy alpha diagnostics）。 | 三个公开数据集和 Panama 八组合探针。 | V12b 是 `0/9` 和 `0/45`；V12c best-delta public-3 仍是 `0/9` 和 `0/45`。 | 诊断更细，但仍不能训练生成器。 |
| V12d | 加入历史残差对照（residual-history controls），比较只用漂移、只用历史残差和漂移加历史残差。 | 三个公开数据集，27 行特征模式记录。 | 所有特征模式都是 `0/27` gates；drift-plus-history 的 rolling windows 为 `6/45`，但相对 history-only 的 partial gain 平均为负。 | 表面修正（apparent correction）多数可由历史残差解释。 |
| V12e | 加入按预测步长控制标签可用性（per-horizon label availability）和固定历史条件漂移对照（fixed-history conditional drift controls）。 | 三个公开数据集，27 行特征模式记录。 | 所有特征模式是 `0/27` gates；drift-plus-history rolling windows 为 `0/45`。 | 这是当前最严格的负面证据。 |

## 5. 分阶段实验记录和改进细节

### 5.1 V2 原始修复：把旧版第三阶段变成可检验路径

V2 的目标不是证明模型已经有效，而是先把旧版第三阶段从不可审计状态改成可检验状态。该阶段加入了确定性评估（deterministic evaluation）、零漂移恒等映射（zero-drift identity）、对齐残差表示（aligned residual representation）、可微分暂态前向计算（differentiable transient forward）、生成器训练步骤、漂移敏感性干预，以及可选的在线循环。

早期测试记录为 `44 passed in 3.18s`。烟雾测试运行（smoke run）的编号为 `codex_stage3_v2_batch1_smoke_20260525`，目录位于 `logs/load_benchmarks/stage3_v2/.../panama/seed_42`。

这个极小烟雾测试运行中，真实漂移干预（`real` intervention）的 MSE 从 `0.4775509834` 变为 `0.4775510728`，这个变化几乎可以忽略。零漂移干预（`zero` intervention）保持恒等映射（identity mapping），也就是它没有改变基线预测。这个结果只能说明代码链路已经接通，不能说明性能提升。

V2 早期有一组有预算上限的三个公开数据集实验（bounded public-3）结果一度看似正向，其中 Panama、Tetouan 和 Spain 的真实漂移 MSE 收益（real MSE gain）分别为 `3.5116%`、`3.4614%` 和 `1.8114%`。但是后续因果安全的三个公开数据集检查（causal-safe public-3 check）显示，Panama 和 Tetouan 分别变成 `-7.9613%` 和 `-3.3956%`，Spain 只有 `+0.4374%`。因此，这些早期正向数字已经被审计文件明确降级，不能用于论文论断（claim）。

Panama b50 的论断门槛（claim gate）为 `False`，其中 `real_mse_gain=-0.005096`。使用验证门槛选择的 b30 实验（validation-gated b30）在验证端通过，`real_mse_gain=0.016523`，但在测试端失败，`real_mse_gain=-0.035996`。这个结果说明，仅在验证端看到正向趋势并不能证明测试端也会稳定改善。

V2 敏感性台账（sensitivity ledger）还固定了七类漂移干预：真实漂移（`real`）、零漂移（`zero`）、打乱漂移（`shuffled`）、符号翻转漂移（`sign_flipped`）、放大两倍漂移（`amplified_2x`）、放大五倍漂移（`amplified_5x`）和随机匹配范数漂移（`random_matched_norm`）。这些干预后来成为检查第三阶段是否真正使用真实漂移的核心工具。

V2 的最终意义是建立可运行、可审计、可证伪的链路。它不支持“显著提升”“稳定漂移条件在线适应”或“达到当前最先进方法水平（SOTA）”的论文表述。

### 5.2 V2.1：安全切分预处理和投影残差对齐

V2.1 的重点是修复数据预处理和表示几何问题。新增的 `data/readers_stage3_v2.py` 和 `tests/test_stage3_v2_split_safe_preprocessing.py` 用来实现 split-safe preprocessing。这里的中文含义是：先在原始时间轴上确定训练集、验证集和测试集，再只用当前可见历史做缺失值填充，避免未来信息进入过去。

V2.1 还加入了投影残差对齐（Projected Residual Alignment）。这里需要按照你刚刚指出的逻辑重新解释：前一版报告把 `phi_v` 直接说成“短期扰动表示”，这个说法不够严谨，也容易误导后续的理论判断。

更符合论文设定的理解是这样的。第一类是不变表示，记作 `phi_i`。它应该包含相对稳定、长期有用的负荷规律，例如一天内的用电周期、工作日和周末差异、季节性规律等。第二类是可变表示，记作 `phi_v`。它不应该被简单等同于短期波动；它更像一种会随阶段、环境、运行状态或负荷制度而改变的可变状态表示。这个可变状态可以发生变化，但变化本身未必是高频扰动，也可能属于文章里所说的 stable, long-term changes。

因此，`phi_v - phi_i` 的理论含义也要更谨慎地表述。它更像“可变长期状态相对不变长期规律的偏移”，而不应直接写成“第三阶段需要追踪的短期波动”。如果论文中的第三阶段目标是处理 fluctuations driven by variant unobserved factors，那么更接近这个目标的信号应该是时间上的变化量，而不是某一个时刻的表示水平值。也就是说，候选信号应优先考虑 `phi_v(t)-phi_v(t-1)`，或者在 V2.1 共享残差表示下使用 `delta_phi_v(t)-delta_phi_v(t-1)`。

代码核查也支持这个区分。`models/aligned_representation_v2.py` 中的共享残差编码器定义了 `phi_v = phi_i + delta_phi_v`，所以 V2.1 之后的 `phi_v - phi_i` 在数值上等于 `delta_phi_v`。这确实比旧版两个独立编码器直接相减更可审计，因为它把差值放在同一个共享残差结构里。但是，这只解决“能不能相减”和“是否尽量减少与 `phi_i` 的几何重叠”两个问题，并不能自动证明 `delta_phi_v` 就是短期波动。

投影残差对齐的目的，是把 `delta_phi_v` 中沿着 `phi_i` 方向的成分去掉，只保留更不像 `phi_i` 的部分。这里的“正交”可以理解为减少重复解释：如果 `phi_i` 已经表达了某些稳定负荷规律，那么 `delta_phi_v` 不应该再大量复制这些规律。与此同时，V2.1 还要求不变分支 `phi_i` 本身保留独立预测能力，避免模型把所有有用信息都塞进 `delta_phi_v`。

但是，投影残差对齐不能完成另一件事：它不能把“可变长期状态偏移”自动变成“未观测可变因素驱动的短期波动”。所以更严格的说法应当是：V2.1 让 `delta_phi_v` 更适合作为候选漂移表示，但第三阶段真正服务时应关注这个候选表示的时间变化，例如 `delta_phi_v(t)-delta_phi_v(t-1)`。仓库后续代码也确实沿着这个方向发展：`train/stage3_signals.py` 的 `delta_vector` 模式使用相邻样本和上一批次尾部样本构造时间差分；`TrainSetResidualDriftCanonicalizer` 和 `_deployed_temporal_ranking_loss` 也都显式使用 `delta_phi_v[t]-delta_phi_v[t-1]` 这类 transition signal。

因此，我同意你的宏观纠正：如果论文把 `phi_v` 放在 stable, long-term changes 的语义下，那么 `phi_v - phi_i` 更适合被解释为长期可变状态偏移，而不是第三阶段最终要响应的 fluctuation 本身。第三阶段的更合理输入应是这个偏移随时间的变化量。前一版报告中把 `phi_v` 和 `delta_phi_v` 直接讲成“短期变化备忘录”的说法已经在这里修正。

需要强调的是，即使使用 `delta_phi_v(t)-delta_phi_v(t-1)`，它仍然只是更合理的候选信号，而不是已经被证实的成功机制。后续实验之所以还要做零漂移、打乱漂移、符号翻转漂移、放大漂移、历史残差对照和滚动验证，就是为了检查这个时间差分信号是否真的能提供稳定预测收益。

该阶段涉及 `models/aligned_representation_v2.py`、`train/offline_aligned_representation_v2.py`、`train/run_stage3_v2_sensitivity_audit.py` 和 `configs/stage3_v2_strong.yaml`。

验证命令的记录为 `63 passed in 3.40s`。主实验编号为 `codex_stage3_v21_splitsafe_panama_b60_20260525`，覆盖 Panama 数据集的 seed 42 和 seed 43。

几何对齐确实在数值上生效。seed 42 和 seed 43 的 validation parallel ratio 分别为 `3.84e-08` 和 `2.84e-08`。这个指标接近零，说明投影正交约束确实起作用。

但是，预测证据没有通过门槛。两个随机种子（seed）的滚动验证通过率（rolling validation pass rate）都是 `0.6667`，但总门槛仍然失败。测试集只报告、不用于选择配置的结果（report-only result）中，seed 42 的 MSE gain 为 `+0.4785%`，MAE gain 为 `-0.2272%`；seed 43 的 MSE gain 为 `+0.2498%`，MAE gain 为 `+0.1117%`。这说明 MSE 有小幅趋势，但 MAE 和滚动验证不稳定。

V2.1 可以写成协议安全和表示审计的改进，不能写成已经确认的性能贡献。

### 5.3 V2.3：与服务逻辑一致的训练和确定性评估

V2.3 的目标是让训练和最终在线服务逻辑更一致。该阶段新增 `train/stage3_eval_policy.py`，用于拒绝随机评估进入第三阶段审计。`train/online_loop_stage3_v2.py` 在在线回放入口加入确定性评估保护。`train/stage3_generator_training.py` 把训练目标改得更接近最终服务时的候选预测、混合预测和风险门控输出。`train/stage3_risk_gate_v2.py` 加入更简单、可解释的逻辑回归风险门（logistic risk gate）。

该阶段还加入 `train/stage3_v2_artifacts.py`，用于重新加载训练好的生成器、风险门和漂移标准化器后再评估。这样可以避免“训练时对象有效，但保存后加载的部署产物不一致”的问题。

验证记录包括 targeted tests 的 `27 passed in 2.66s`，以及 Stage III V2 回归套件的 `75 passed in 3.90s`。

Panama b60 实验中，seed 42 的测试 MSE gain 为 `-0.2431%`，MAE gain 为 `-0.4971%`，说明真实漂移使结果变差。seed 43 的测试 MSE gain 为 `+0.7797%`，MAE gain 为 `+0.2071%`，测试门槛通过，但是滚动验证通过率（rolling validation pass rate）只有 `0.3333`，仍然失败。两个 seed 的平均测试 MSE gain 为 `+0.2683%`，平均 MAE gain 为 `-0.1450%`。

V2.3 证明生成器训练闭环和重新加载评估已经接入，但它没有证明训练出的生成器是稳定可靠的漂移条件策略。

### 5.4 V2.5：规范化漂移和滚动验证选择生成器

V2.5 的目标是避免“使用最后一个训练轮次（epoch）的生成器”这种隐式选择方式。该阶段新增 `train/stage3_generator_selection_v2.py`，用于保存多个候选生成器，并只根据滚动验证选择候选。测试集仍然只是只报告（report-only），也就是只能报告结果，不能用来反向选择模型。

该阶段还加入 `TrainSetResidualDriftCanonicalizer`。它的中文含义是：只用训练集中的有序漂移变化和残差变化拟合一个漂移规范坐标系，然后在验证集和测试集冻结使用。这样做是为了让漂移坐标更可审计。

验证记录包括 `82 passed in 3.63s`。Panama b60/b120 成对实验覆盖 seed 42 和 seed 43。

| seed | budget | 选择的生成器 epoch | 滚动验证通过率 | 测试门槛 | MSE gain | MAE gain | 拒绝率 |
| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |
| 42 | b60 | 0 | `0.3333` | 失败 | `-0.0938%` | `-0.3505%` | `0.0833` |
| 42 | b120 | 3 | `0.3333` | 失败 | `+1.2656%` | `-0.4907%` | `0.0750` |
| 43 | b60 | 3 | `0.3333` | 失败 | `-0.3250%` | `+0.1074%` | `0.0500` |
| 43 | b120 | 0 | `0.3333` | 失败 | `+1.6361%` | `+1.1517%` | `0.3750` |

b120 的测试 MSE 在两个 seed 上都有改善，但滚动验证没有改善，seed 42 的 MAE 仍然变差，seed 43 的拒绝率达到 `37.5%`，超过预设阈值。由于测试集是只报告结果（report-only），这些测试改进不能反过来证明配置选择正确。

V2.5 的三个公开数据集（public-3）b240 完整评估（full-eval）进一步扩大到 Panama、Tetouan 和 Spain，并覆盖随机种子（seeds）42、43 和 44。结果是 `9/9` 个测试论断门槛（test claim gates）全部失败，`9/9` 个滚动验证汇总（rolling validation aggregates）的通过率（pass rate）都是 `0.0000`。整体平均 MSE gain 为 `+0.3172%`，整体平均 MAE gain 为 `+0.1969%`，整体平均拒绝率为 `82.94%`。

这个结果是强负面证据。V2.5 可以作为审计型修复，不能作为第三阶段性能贡献。

### 5.5 V3：时间残差 CCA 和生成器输入空间反事实干预

V3 的核心修复是把反事实干预移动到生成器输入空间。这样可以避免早期放大漂移（amplified drift）在原始漂移空间（raw-drift space）被归一化（normalize）后失去放大效果。V3 还加入 `TrainSetTemporalResidualCCAAligner`，中文含义是：只用训练集拟合漂移变化与预测残差变化之间的典型相关结构。

该阶段保存 `alignment_diagnostics.csv`、`resolved_config.json` 和配置哈希，增强了可追溯性。验证记录为 `88 passed in 3.75s`。

V3 的烟雾测试运行（smoke run）只用于协议检查。它的测试集真实漂移 MSE 收益（test real MSE gain）为 `0.001976`，测试集真实漂移 MAE 收益（test real MAE gain）为 `0.000978`，最佳反事实模式（best counterfactual）是放大两倍漂移（`amplified_2x`），滚动通过率（rolling pass rate）为 `0.0`，论断门槛（claim gate）失败。

Panama seed 42 b120 完整评估探针（full-eval probe）的结果更有信息量。真实漂移的测试 MSE 收益（test MSE gain）为 `+0.9338%`，MAE gain 为 `+0.6660%`，拒绝率为 `16.8563%`，低于 `20%` 的阈值。但是放大两倍漂移（`amplified_2x`）的 MSE gain 为 `+1.0273%`，高于真实漂移，因此真实漂移相对最佳反事实的边际（real-vs-best-counterfactual margin）为 `-0.000935`。滚动验证仍为 `0/5`。

V3 改善了审计性和拒绝率，但没有证明真实漂移特异性，也不能写成第三阶段成功。

### 5.6 V4：反事实 hard negative 训练

V4 的目标是直接处理 V3 中放大漂移（amplified drift）优于真实漂移的问题。这里的困难负样本（hard negative）指那些很容易欺骗模型、但实际上不应该优于真实漂移的反事实输入。V4 把放大两倍漂移（`amplified_2x`）和放大五倍漂移（`amplified_5x`）纳入训练期负样本，并且让负分支（negative branch）的 MSE 不再从反向传播中分离（detach）。这样生成器会被训练成让反事实漂移不容易优于真实漂移。

该阶段还在 `train/online_loop_stage3_v2.py` 中加入逐步指标记录（step-level metrics），包括 base MSE、candidate MSE、最终 MSE、使用状态、拒绝原因、漂移范数、delta 范数、预测位移和风险分数。

验证记录为 `97 passed in 3.67s`。Panama seed 42 full run 的编号为 `codex_stage3_v4_full_panama_seed42_trace_20260526_04`。

V4 测试集结果为：真实漂移 MSE gain `+3.6507%`，真实漂移 MAE gain `+3.0443%`，拒绝率 `0.8044%`。这比 V3 的高拒绝率明显更好。最佳反事实仍然是 `amplified_2x`，但真实漂移已经略好于反事实，margin 为 `+0.0001382`。

问题在于，预声明的反事实边际（counterfactual margin）阈值是 `0.002`，而 V4 的 margin 只有 `0.0001382`。此外，V4 的滚动验证（rolling validation）仍然是 `0/5`。窗口级诊断显示，五个验证窗口全部因为反事实 margin 不足而失败。

V4 可以写成部分机制修复。它不能写成已经证明第三阶段学习到了真实概念漂移。

### 5.7 V5 和 V5.1：服务策略选择和反事实感知风险门

V5 的重点是让生成器选择过程和最终服务过程一致。该阶段新增 `score_served_policy_candidate()`，并且在选择生成器时把真实漂移收益、拒绝率、滚动通过率和真实漂移相对最佳反事实的 margin 都纳入评分。

V5 还加入了真实弃权机制。如果没有任何候选通过预定义门槛，runner 会保存诊断文件，但不会保存最终服务用的生成器产物。这一点很重要，因为它可以防止把失败候选误当作成功模型发布。

验证记录为 `34 passed`。V5 首个有预算上限的探针实验（bounded probe）的风险门正类比例（risk gate positive rate）只有 `0.1072`，选择的阈值为 `0.8`，测试集真实漂移拒绝率达到 `1.0`。这意味着所有第三阶段更新都被拒绝，滚动验证为 `0/5`。这个结果说明风险门过度保守。

V5.1 加入正类权重（positive class weight）和更低阈值网格后，情况明显改善。测试 MSE gain 为 `+0.7531%`，MAE gain 为 `+0.1228%`，真实漂移相对最佳反事实的边际（real-vs-best-counterfactual margin）为 `+0.005633`，拒绝率为 `6.25%`，选择阈值为 `0.05`。但是滚动验证（rolling validation）只有 `2/5`，最差窗口 MAE gain 为 `-0.001290`，并且仍有窗口中放大反事实（amplified counterfactual）优于真实漂移。

V5.1 是有价值的修复方向，但仍不能进入论文主结果。

### 5.8 V5 g5/g8 和 V6：更大预算与 whitened-Procrustes 对齐

V5 g5_b480 是最接近通过的一次早期结果。运行编号（run id）为 `v5g5s42r1`，输出目录为 `logs/s3v5g5/v5g5s42r1/panama/seed_42`。最佳候选是 epoch 4，滚动验证通过 `4/5`，最差窗口 MSE gain 为 `+0.004620`，最差窗口 MAE gain 为 `+0.004778`，最差真实漂移相对反事实的边际为 `+0.004575`，最大拒绝率为 `9.5833%`。

这个结果接近通过，但没有通过。原因是 worst MSE gain 低于预声明阈值 `0.005`，并且滚动窗口要求是 `5/5`。runner 正确地弃权，没有保存最终服务生成器。

用户要求不要降低计算规模，因此后续运行了 V5 g8_b960。这个更大预算版本的最佳滚动通过率（rolling pass rate）反而降到 `3/5`，说明简单增加生成器训练预算不能解决问题。

V6 加入 `TrainSetWhitenedProcrustesTemporalResidualCCAAligner`。它的中文含义是：先用训练集白化（whitening）和 CORAL 对齐表示分布，再用正交 Procrustes 方法对齐方向，然后再做时间残差 CCA。V6 g5_b480 的最佳滚动通过数只有 `1/5`，最差窗口 MSE gain 为 `-0.000633`。

因此，V5 g5 是 near-miss，也就是接近但失败的修复信号。它不能被写成成功证据，因为更大预算和 V6 对齐都没有复现或改善它。

### 5.9 V7：residualized RRCCA 对齐

V7 的目标是更严肃地处理 `phi_v - phi_i` 的表示空间问题。V7 先移除 `delta_phi_v` 中可以由 `phi_i` 线性解释的部分，再在训练集内部做 reduced-rank CCA。中文可以理解为：先去掉不变表示能够解释的泄漏，再保留与残差变化较稳定相关的低维漂移子空间。

新增文件包括 `models/stage3_alignment_v7.py` 和 `configs/stage3_v7_residualized_rrcca_served_policy_gate.yaml`。验证记录为 `120 passed in 4.76s`，诊断重跑前为 `121 passed in 3.99s`。

V7 的内部对齐诊断通过。active rank 为 `8`，inactive dimensions 为 `56`，residualizer R2 mean 为 `0.931512`，residualizer R2 max 为 `0.970671`。这说明 `delta_phi_v` 中很大一部分确实可以由 `phi_i` 解释，原始相减存在 invariant leakage 风险。

但是，服务策略预测门槛（served-policy forecasting gate）失败。初次先导验证实验（canary）中，最佳 epoch 4 的滚动通过数仍为 `0/5`，最差窗口 MSE gain 为 `+0.000490`，最差窗口 MAE gain 为 `+0.000534`，margin 为 `+0.000267`。诊断重跑保存了 `125` 行候选窗口模式记录，最佳窗口的真实漂移 MSE gain 最高也只有 `0.002325`，MAE gain 最高只有 `0.001651`，仍然低于阈值。

V7 的意义是证明表示泄漏问题真实存在。它不是性能修复。

### 5.10 V8：LayerAware HyperAdapter

V8 测试了一个直观假设：V7 失败可能是因为生成器只能修改权重，不能修改偏置。为此，V8 新增 `LayerAwareHyperAdapterGenerator`，允许更新 `output_head` 中的权重（weight）和偏置（bias）。中文含义是：生成器既可以改变线性层权重，也可以改变偏置项，从而有能力修正电力负荷水平偏移。

V8 是可选分支，没有覆盖旧版第三阶段。相关修改位于 `models/adaptation_v2.py`、`train/stage3_generator_training.py`、`train/online_loop_stage3_v2.py` 和 `configs/stage3_v8_hyperadapter_residualized_rrcca_served_policy_gate.yaml`。

验证记录为 `127 passed in 5.59s`。smoke run 确认目标参数包括 `output_head.0.weight`、`output_head.0.bias`、`output_head.2.weight` 和 `output_head.2.bias`。

同预算先导验证实验（same-budget canary）的日志目录为 `logs/s3v8diag/v8ha_s42_g5b480_01/panama/seed_42`。所选 epoch 为 3，滚动通过数为 `0/5`，最差窗口 MSE gain 为 `-0.008390`，最差窗口 MAE gain 为 `-0.005176`，最差 margin 为 `-0.008390`。没有保存 `generator_served_policy.pt`。

V8 的训练指标显示，它确实增大了参数更新和预测位移。因此 V8 不是因为“没有动起来”而失败，而是因为更强的更新方向不可靠。这个结果否定了“只要增加生成器容量就能解决第三阶段”的简单假设。

### 5.11 V9：残差教师模型和残差基生成器

V9 的设计基于一个更深的判断：第三阶段失败可能不是因为生成器太小，而是因为生成器没有明确的残差修正教师信号。V9 因此加入只用训练集拟合的残差教师模型（train-only residual teacher）、教师感知 RRCCA（teacher-aware RRCCA）和残差基超生成器（residual-basis hypergenerator）。

新增文件包括 `models/stage3_residual_teacher_v9.py`、`models/stage3_alignment_v9.py`、`models/adaptation_v9.py` 和 `configs/stage3_v9_residual_teacher_rrcca_served_policy_gate.yaml`。V9 测试记录为 `7 passed`，完整测试套件记录为 `134 passed`。

同预算先导验证实验的日志目录为 `logs/s3v9diag/v9rt_s42_g5b480_01/panama/seed_42`。教师模型使用 `480` 个训练样本，漂移维度（drift dimension）为 `64`，特征维度（feature dimension）为 `193`，输出维度（output dimension）为 `12`。训练残差 MSE 从 `0.1608959436` 降到 `0.1450631768`，教师残差 R2（teacher residual R2）为 `0.0926146507`。

这个 R2 说明教师模型在训练集上只有很弱的残差解释能力。服务策略（served-policy）结果也失败：所选 epoch 为 0，滚动验证为 `0/5`，最差窗口 MSE gain 为 `-0.007082`，最差窗口 MAE gain 为 `-0.005109`，最差 margin 为 `-0.007082`。多个验证窗口中零漂移是最好反事实。

V9 修复了结构问题，但没有让教师模型学到可以迁移的修正方向。

### 5.12 V9 只评估教师模型：确认失败发生在生成器上游

V9 只评估教师模型的诊断（teacher-only diagnostic）的中文含义是：完全绕过生成器，不生成参数扰动，只直接测试教师模型给出的残差修正是否有用。如果教师模型本身都不能改善滚动验证，那么继续训练更大的生成器通常没有意义。

该阶段新增 `train/stage3_v9_teacher_diagnostics.py`，并输出多类只评估教师模型的 CSV 文件（teacher-only CSV）。新增测试记录为 `8 passed`。

烟雾测试运行的只评估教师模型滚动验证（teacher-only rolling validation）为 `0/5`，最差窗口 MSE gain 为 `-1.116660`，最差窗口 MAE gain 为 `-0.482254`。

同预算诊断先导实验（same-budget diagnostic canary）的日志目录为 `logs/s3v9teacher_diag/v9teacher_s42_g5b480_01/panama/seed_42`。结果显示，只评估教师模型的滚动验证仍为 `0/5`，最差窗口 MSE gain 为 `-0.197784`，最差窗口 MAE gain 为 `-0.118167`，最差真实漂移相对最佳反事实的边际为 `-0.197784`，最差信号一致率（worst sign agreement）为 `0.515625`。测试集 MSE gain 为 `-0.097074`，MAE gain 为 `-0.052800`。

五个滚动窗口中，最佳反事实都是零漂移。这意味着真实漂移教师修正在每个窗口都伤害预测。这个实验把失败定位到生成器上游，也就是从漂移到残差修正的教师模型（drift-to-residual teacher）本身不可靠。

### 5.13 V10：交叉拟合且保持零漂移恒等的教师模型

V10 尝试修复 V9 教师模型不能迁移的问题。它加入交叉拟合且保持零漂移恒等的泄漏投影（cross-fitted zero-preserving leakage projection），中文含义是：用交叉拟合方式去除泄漏子空间，并保持零漂移不会产生修正。V10 还加入训练集校准的残差教师缩放（train-calibrated residual-teacher shrinkage），中文含义是：只用训练内部校准来决定教师修正的缩放强度。

Panama seed 42 的标量缩放教师模型先导实验（scalar shrinkage teacher-only canary）中，训练残差 R2 为 `0.1190261841`，选择的 alpha 为 `0.5`。滚动验证从 V9 的 `0/5` 提升到 `3/5`，最差窗口 MSE gain 为 `+0.0059367432`，但是最差窗口 MAE gain 为 `-0.0007096970`，最差信号一致率为 `0.5296875238`，低于 `0.55` 的阈值。测试 MSE 和 MAE gains 分别为 `+0.0267751812` 和 `+0.0151892459`。

V10 的逐输出有符号 alpha 对照（signed per-output alpha）没有改善，滚动验证降到 `2/5`，最差窗口 MSE gain 为 `-0.0018595963`，最差窗口 MAE gain 为 `-0.0099424396`。

随后 V10 扩展到三个公开数据集的只评估教师模型矩阵实验（public-3 teacher-only matrix）。主运行清单是 `logs/s3v10teacher_matrix/_manifests/v10teacher_matrix_20260529_112849_e5_ie5_b480_ew480`，Spain 重跑运行清单是 `logs/s3v10m/_manifests/v10m_20260529_121906_e5_ie5_b480_ew480`。

三个公开数据集结果是 `0/9` 个教师门槛通过，`8/45` 个滚动窗口通过。Panama 有 `7/15` 个窗口，Tetouan 只有 `1/15` 个窗口，Spain 为 `0/15` 个窗口。整体平均最差窗口 MSE gain 为 `-0.054166`，整体平均最差窗口 MAE gain 为 `-0.038192`，整体平均测试 MSE gain 为 `+0.000599`，整体平均测试 MAE gain 为 `-0.015011`。没有训练或保存生成器。

V10 是第一个显示 Panama 局部正信号的教师修复，但三个公开数据集结果不支持泛化。因此生成器仍然不能训练，论文也不能升级论断（claim）。

### 5.14 V11：worst-block calibration 和 fail-closed teacher

V11 的目标是防止教师模型因为早期容易窗口表现好而通过整体校准。它加入按时间顺序的最差块校准（worst-block chronological calibration），中文含义是：按时间块检查最差校准块，只有每个关键块都稳定，才允许非零修正。

V11 新增 `models/stage3_residual_teacher_v11.py`、`models/stage3_alignment_v11.py`、`train/run_stage3_v11_sensitivity_audit.py`、`train/run_stage3_v11_teacher_audit_matrix.py` 和 `configs/stage3_v11_block_calibrated_teacher_gate.yaml`。

验证记录包括 `18 passed in 2.58s` 和项目本地测试 `145 passed in 5.49s`。public-3 manifest 是 `logs/s3v11m/_manifests/v11tm_20260601_100708_e5_ie5_b480_ew480`。

结果非常明确：教师门槛（teacher gates）为 `0/9`，滚动窗口（rolling windows）为 `0/45`，所有运行（runs）的 `teacher_alpha_source` 都是 `fallback_block_gate`，所有 alpha 都是 `0.0`。Panama、Tetouan 和 Spain 的平均训练 R2（mean train R2）分别为 `-0.012011`、`-0.027989` 和 `-0.008380`。没有生成器产物。

V11 的意义不是性能改善，而是证据不足时自动归零的安全机制（fail-closed safety）。中文含义是：当证据不足时，系统宁愿选择零修正，也不让未经支持的漂移修正进入服务模型。

### 5.15 V11 signal and horizon ablations

V11 后续做了两个消融检查。第一个检查是水平向量（`level_vector`）是否能替代差分向量（`delta_vector`）。这里 `level_vector` 指直接使用表示水平值，`delta_vector` 指使用表示变化量。第二个检查是按时间块和预测步长划分的 alpha（block-horizon alpha）是否能避免单一 alpha 把局部有用的预测步长（horizon）压制掉。

`level_vector` 结果仍然是 `0/9` 个教师门槛、`0/45` 个滚动窗口和 `0/9` 个非零 alpha。`block_horizon` 虽然产生了 `4/9` 个非零 alpha，但仍然是 `0/9` 个门槛和 `0/45` 个滚动窗口，平均最差窗口 MSE gain 为 `-0.000762`，平均最差窗口 MAE gain 为 `-0.000592`，平均测试 MSE 和 MAE 分别为 `-0.000101` 和 `-0.000217`。

这些结果排除了两个简单解释：失败不是因为只用了 `delta_vector`，也不是因为单一 alpha 过于保守。

### 5.16 V12：outcome-aligned transport

V12 进一步处理表示对齐问题。它先把 `phi_i` 和 `phi_v` 放入训练集拟合的共享潜在锚点（shared latent anchor）；然后在该空间中做保持零漂移恒等的传输（zero-preserving transport）、正交泄漏移除（orthogonal leakage removal）和按时间顺序的残差迁移校准（chronological residual-transfer calibration）。中文含义是：对齐表示坐标，去除不变表示可解释的泄漏，并只保留在时间校准中对残差迁移有支持的维度。

新增文件包括 `models/stage3_alignment_v12.py`、`models/stage3_residual_teacher_v12.py`、`train/stage3_v12_artifacts.py` 和 V12 的 sensitivity audit 与 matrix runner。测试记录为 `6 passed`。

三个公开数据集实验的运行清单（public-3 manifest）是 `logs/s3v12m/_manifests/v12tm_20260601_134623_e5_ie5_b480_ew480`。V12 完成运行数（completed runs）为 `9/9`，表示对齐门槛（alignment gate）为 `9/9`，活跃秩（active rank）在 `2-3` 之间。部分折外迁移 MSE 收益（OOF transfer MSE gain）很高，例如 Panama 达到 `0.640201`，Tetouan 达到 `0.476276`。

但是，教师门槛（teacher gate）仍为 `0/9`，滚动窗口（rolling windows）为 `0/45`，非零 alpha 数量为 `0/9`，所有服务时使用的 alpha（served alpha）都是 `0.0`。没有生成器产物。

V12 的结论非常重要：表示对齐门槛通过并不等于预测修正门槛通过。V12 可以作为更强的 representation audit，不能作为成功的 online adaptation 证据。

### 5.17 V12b 和 V12c：chronological OOF teacher 与 multi-policy diagnostics

V12b 把 V12 的包装教师模型（wrapper teacher）替换成真正的按时间顺序折外残差教师模型（chronological out-of-fold residual teacher）。中文含义是：按时间顺序做前向折外训练，只用过去拟合未来校准块，避免泄漏。V12b 的三个公开数据集实验仍然是 `0/9` 个教师门槛和 `0/45` 个滚动窗口。V12b 的放宽 alpha Panama seed 42 实验即使强制非零修正（nonzero correction），也仍然是 `0/5`，而且符号翻转漂移（sign-flipped）反事实最好。

V12c 的目标是把 alpha 策略（alpha policy）拆开，不让严格门槛（strict gate）一清零就完全看不到弱信号。它引入 `ungated`、`relaxed` 和 `strict` 三种 alpha 策略。中文含义分别是：不加最终门槛的探索策略、放宽内部条件的诊断策略，以及保留论文级安全条件的严格策略。

Panama seed 42 的八组合探针（8-combination probe）检查了水平向量（`level_vector`）、差分向量（`delta_vector`）、漂移方向正负，以及有符号或非负 alpha（signed/nonnegative alpha）。结果显示 `delta_vector` 比 `level_vector` 更好，但所有 8 个组合的教师门槛都失败。差分向量家族（delta-vector family）的最差滚动 MSE 和 MAE 分别为 `0.000909` 和 `0.000707`，最差信号一致率（worst sign）为 `0.466146`，测试 MSE 和 MAE 分别为 `0.001881` 和 `0.000834`。

V12c 最佳差分向量三个公开数据集运行清单（best-delta public-3 manifest）是 `logs/s3v12cm/_manifests/v12ctm_delta_nonneg_20260601_190259_e5_ie5_b480_ew480`。结果为 `9/9` 个运行完成、`0/9` 个教师门槛通过、`0/45` 个滚动窗口通过。最接近通过的是 Panama seed 43，它的最差窗口 MSE 为 `0.005604`，最差窗口 MAE 为 `0.003026`，最差 margin 为 `0.002918`，这三个指标都过了阈值，但最差信号一致率为 `0.440451`，低于 `0.55`。

V12c 的贡献是提高诊断分辨率。它没有证明第三阶段捕捉了概念漂移，也没有解锁生成器训练。

### 5.18 V12d：加入 residual-history context teacher

V12d 的核心问题是：如果基线模型残差（residual）本身存在时间自相关，那么看似有效的漂移修正可能只是利用了历史残差信息，而不是利用了 `phi_v - phi_i` 的概念漂移信息。

为此，V12d 新增 `train/stage3_v12d_context_features.py` 和 `models/stage3_residual_teacher_v12d.py`，并比较三种输入模式：只用漂移（`drift_only`）、只用历史残差（`history_only`）以及同时使用漂移和历史残差（`drift_plus_history`）。

目标测试（targeted tests）记录为 `4 passed`。三个公开数据集运行清单（public-3 manifest）是 `logs/s3v12dm/_manifests/v12dm_delta_20260601_203114_e5_ie5_b480_ew480`。数据集和随机种子的运行数（dataset-seed runs）为 `9/9`，特征模式行数（feature rows）为 `27/27`。

三种模式的教师门槛（teacher gates）全部为 `0/9`。`drift_only` 的滚动窗口为 `3/45`，`history_only` 为 `0/45`，`drift_plus_history` 为 `6/45`。但是，`drift_plus_history` 相对 `history_only` 的滚动最小部分 MSE 收益（rolling-min partial MSE gain）只有 `3/9` 为正，平均为 `-0.043138`。测试部分 MSE 收益（test partial MSE gain）只有 `4/9` 为正，平均为 `-0.019044`。

V12d 的结论是，表面修正（apparent correction）很大程度上可以由历史残差自相关解释。当前 `phi_v - phi_i` 没有稳定提供超过历史残差的增量预测价值。

### 5.19 V12e：按预测步长控制标签可用性和条件漂移空模型

V12e 是当前最严格的第三阶段诊断。它修复了 V12d 的一个现实性问题：多步预测中，某些预测步长（horizon）的真实标签在当前时刻其实还不可见。如果把这些还没有到达的残差放进历史特征，就会让实验过于乐观。

V12e 新增 `train/stage3_v12e_label_availability.py`。它按以下条件控制某个 residual 是否可以进入历史缓冲区：

```text
t + h + label_delay <= k
```

中文含义是：对于 forecast origin `t` 和 horizon `h`，只有当该标签在当前时刻 `k` 之前已经到达，才允许使用它。

V12e 还新增 `train/stage3_v12e_conditional_permutation.py`，用于固定历史残差信息，只打乱漂移信号。这样可以检查真实漂移是否真的提供了超过固定历史条件的增量信息。

目标测试记录为 `11 passed in 1.34s`。烟雾测试输出目录（smoke output）是 `logs/s3v12e_cond_smoke/smoke_v12e_cond_panama_s42_20260601_01/panama/seed_42`。最终三个公开数据集运行清单（public-3 manifest）是 `logs/s3v12emc/_manifests/v12emc_delta_20260601_214834_e5_ie5_b480_ew480`。

V12e 的数据集和随机种子运行数（dataset-seed runs）为 `9/9`，特征模式行数（feature-mode rows）为 `27/27`。所有特征模式门槛（feature-mode gates）都是 `0/27`，`drift_plus_history` 的滚动窗口为 `0/45`，没有生成器产物。

| 特征模式 | gate 通过情况 | rolling window 通过情况 | 平均 worst MSE gain | 平均 worst counterfactual margin |
| --- | ---: | ---: | ---: | ---: |
| 只用漂移（`drift_only`） | `0/9` | `2/45` | `-0.004451` | `-0.010971` |
| 只用历史残差（`history_only`） | `0/9` | `0/45` | `-0.000911` | `0.000000` |
| 漂移加历史残差（`drift_plus_history`） | `0/9` | `0/45` | `0.001000` | `-0.005649` |

`drift_plus_history` 相对历史残差的滚动最小部分 MSE 收益只有 `2/9` 为正，平均为 `-0.006893`。测试集部分 MSE 收益有 `5/9` 为正，平均为 `0.000954`，但这不足以支持论文论断（claim），因为滚动验证和反事实归因都失败。

`drift_plus_history` 的 rolling windows 中，最佳反事实主要是放大两倍漂移（`amplified_2x`，15 次）、条件随机匹配范数（`conditional_random_matched_norm`，10 次）、条件打乱漂移（`conditional_shuffled`，9 次）、零漂移（`zero`，7 次）和符号翻转漂移（`sign_flipped`，4 次）。这说明真实漂移没有稳定胜过反事实漂移。

V12e 是当前最接近部署现实的负面证据。它说明，在遵守 horizon-aware label release 和 fixed-history conditional drift controls 的情况下，当前 `phi_v - phi_i` 漂移信号没有稳定增量价值。

## 6. 按问题归纳的改进链条

| 原始问题 | 已做改进 | 当前结论 |
| --- | --- | --- |
| 生成器没有被训练 | V2 之后显式训练生成器，V9 之后加入残差教师模型。 | 训练闭环已经存在，但训练信号没有通过科学门槛。 |
| 零漂移产生非零扰动 | 使用无偏置结构、恒等初始化和 zero-drift 测试。 | 工程上基本修复，但真实漂移没有稳定胜过零漂移或条件反事实。 |
| 向量漂移被压成标量 | V2 之后使用完整向量，并区分 `level_vector` 和 `delta_vector`。 | `delta_vector` 更符合第三阶段响应短期 fluctuation 的理论目标，但实验上仍不足以通过门槛。 |
| VAE 评估阶段随机 | 加入 deterministic eval guard，并使用均值模式。 | 对第三阶段漂移敏感性审计而言，这是正确修复。 |
| `phi_v - phi_i` 空间未对齐 | 逐步加入 projection、canonicalizer、temporal CCA、Procrustes、RRCCA、teacher-aware alignment 和 outcome-aligned transport。 | 表示越来越可审计，但表示对齐通过不等于预测修正通过。 |
| 反事实干预不充分 | V3 之后使用 generator-input counterfactuals，V4 加入 hard negatives，V12e 加入 conditional permutation。 | 真实漂移多次不能稳定胜过反事实漂移。 |
| 失败候选可能被发布 | V5 之后加入 served-policy selection 和 abstain。 | 弃权机制正确阻止了失败生成器被保存为服务模型。 |
| 教师模型信号不可靠 | V9 teacher-only、V10 到 V12e teacher gates 逐步定位问题。 | 当前教师路径无法在 public-3 上通过。 |
| 历史残差混淆漂移贡献 | V12d 和 V12e 加入 history-only、drift-plus-history 和 label availability controls。 | apparent correction 多数可以由历史残差或标签时序解释。 |

## 7. 当前能够证明什么，不能证明什么

### 7.1 当前可以证明的内容

第一，旧版第三阶段实现与论文中“学习到的漂移条件生成器”叙述不一致，必须修正或降级。

第二，新增可选路径显著提高了科研完整性。具体表现包括确定性评估、运行不覆盖旧结果、产物重新加载、验证集选择、反事实敏感性检查、教师模型门槛和标签可用性约束。

第三，`phi_v - phi_i` 不能直接当成干净的概念漂移向量。更谨慎的解释是，它可能包含可变长期状态相对不变规律的偏移，也可能包含不变表示可解释的结构；只有它的时间变化量才更接近第三阶段所需的短期 fluctuation 候选信号，但这个候选信号也没有稳定转化为有益的残差修正。

第四，严格安全门槛可以阻止没有足够验证证据支持的预测修正（unsupported correction）进入服务模型。

第五，V12e 证明当前第三阶段漂移信号在更接近真实部署场景的延迟反馈（realistic delayed-feedback）和按预测步长控制标签可用性（per-horizon label availability）下没有稳定增量价值。

### 7.2 当前不能证明的内容

第一，不能证明第三阶段显著提升负荷预测精度。

第二，不能证明第三阶段学习到了概念漂移。

第三，不能证明 `phi_v - phi_i` 或 `delta_phi_v(t)-delta_phi_v(t-1)` 是稳定、可泛化、因果语义清晰的漂移表示。

第四，不能证明生成器可以作为论文主贡献进入主结果表。

第五，不能声称相较当前最先进方法（SOTA）有明显优势。尤其不能把只允许报告、不能用于选择配置的测试收益（report-only gain）写成跨数据集、跨随机种子普遍成立的提升（universal improvement）。

## 8. 对论文的直接影响

下列英文 claim 是当前不能写进论文的说法。每条英文都配有中文含义和处理建议。

| 原英文说法 | 中文含义 | 当前处理建议 |
| --- | --- | --- |
| `Stage III learns concept drift.` | 第三阶段学习到了概念漂移。 | 删除或降级，因为 V12e 不支持这个结论。 |
| `The invariant/variant embedding difference provides robust online adaptation.` | 不变表示和可变表示的差值提供了稳健在线适应能力。 | 删除或重写，因为 `phi_v - phi_i` 没有通过历史残差对照（history control）和条件漂移对照（conditional drift control）。 |
| `The adaptation generator significantly improves forecasting accuracy.` | 适应生成器显著提升预测精度。 | 删除，因为没有生成器版本通过严格滚动验证。 |
| `Stage III is a validated contribution over SOTA.` | 第三阶段是相较最先进方法已经验证成功的贡献。 | 删除，因为当前证据不是 SOTA 性能证据。 |
| `The method consistently outperforms baselines under drift.` | 该方法在漂移下持续优于基线方法（baselines）。 | 删除或严格限定，因为多数据集和多种子结果不支持“持续优于”。 |

更安全的中文改写方向如下：

第三阶段可以被重新表述为一个严格的在线适应审计与候选机制。它的贡献不是已经证明生成器提升性能，而是揭示了在真实时间顺序、反事实漂移干预、教师模型门槛和标签可用性约束下，把表示漂移转化为可靠预测修正是困难的。当前证据不支持把第三阶段写成已经验证成功的概念漂移生成器。

如果需要保留英文论文表达，可以使用下面的英文句子，并紧跟中文解释：

> `The Stage III experiments are reported as an audit of drift-conditioned adaptation rather than as validated performance evidence.`
>
> 中文含义：第三阶段实验应被报告为对漂移条件在线适应机制的审计，而不是已经验证成功的性能证据。

> `Under horizon-aware label availability and fixed-history conditional drift controls, the current drift representation does not show robust incremental residual-correction value.`
>
> 中文含义：在按预测步长控制标签可用性，并且固定历史残差信息后，当前漂移表示没有显示出稳健的增量残差修正价值。

## 9. 最高风险实验、表格和图像

| 风险项 | 风险原因 | 当前处理建议 |
| --- | --- | --- |
| 旧版第三阶段结果 | 旧版生成器未训练、评估随机、漂移被压成标量，与论文设定冲突。 | 不再作为正向证据，只能作为旧版路径（legacy path）的负面背景。 |
| V2.5 public-3 | `9/9` 测试门槛失败，拒绝率很高。 | 只能作为负面审计证据。 |
| V4 测试集汇总结果 | 测试集 MSE 和 MAE 很好，但滚动验证为 `0/5`，反事实边际（counterfactual margin）过小。 | 不能单独进入论文论断（claim）。 |
| V5 g5 接近通过但失败的结果 | 达到 `4/5`，接近通过但仍然弃权。 | 可以作为接近但失败的诊断，不能放宽阈值。 |
| V10 Panama seed 42 | 单个 seed 有局部改善，但 public-3 失败。 | 不能泛化到论文主结论。 |
| V12 alignment gate | alignment 为 `9/9`，但 teacher 为 `0/9`。 | 不能把表示对齐成功写成在线适应成功。 |
| V12d 绝对收益 | 只用历史残差（history-only）和漂移加历史残差（drift-plus-history）的绝对收益可能来自历史残差自相关。 | 必须用相对 history-only 的部分收益（partial gain）判断。 |
| V12e 测试集部分收益 | 测试集部分 MSE 收益（test partial MSE gain）有 `5/9` 为正，但滚动验证（rolling validation）和反事实归因（counterfactual attribution）失败。 | 不能作为稳定增量价值证据。 |

## 10. 当前最小安全下一步

最小安全下一步不是继续扩大现有生成器训练，也不是把 V12e 之前的局部正结果写进论文主结论。更安全的下一步是先做 V13 设计确认和实验预注册。

第一，目标应该从“让当前生成器更大”改为“先证明表示能够在延迟标签可用条件下预测每个预测步长的残差创新（residual innovation）”。这里的残差创新指历史残差无法解释的新残差信息。

第二，可以保留共享潜在锚点（shared latent anchor），但不能只做几何对齐。下一版应该加入对比式残差状态表示（contrastive residual-state representation）或残差转移目标（residual-transition objective）。中文含义是：让表示学习直接服务于残差状态或残差变化的可预测性，而不是只让两个表示空间看起来更对齐。

第三，教师模型必须先通过三个公开数据集（public-3）、随机种子 42/43/44（seeds 42/43/44）、按预测步长释放标签（per-horizon label release）、只用历史残差的对照（history-only control），以及固定历史条件下的漂移对照（fixed-history conditional drift controls）。

第四，只有只评估教师模型的三个公开数据集实验（teacher-only public-3）通过后，才允许训练生成器先导验证实验（generator canary）。这里的先导验证实验指小规模但严格的预验证。

第五，论文应该先按负面诊断证据改写第三阶段，而不是等待一个尚未验证的正向生成器结果。

## 11. 当前 goal 距离最终目标的距离

如果最终目标是“让论文在科研完整性上可防守”，当前已经完成了很大一部分。我们已经找出了不能写的论文论断（claim），建立了可追踪的实验台账，并且明确了哪些结果只能作为负面诊断。

如果最终目标是“让第三阶段成为正向性能贡献”，当前还没有达到。原因不是实验规模不够，而是最严格的 V12e 证据已经说明当前漂移表示和教师模型路径本身不稳定。

距离正向模型目标至少还差五步。第一步是设计 V13 表示目标。第二步是让 V13 的只评估教师模型的三个公开数据集实验（V13 teacher-only public-3）通过。第三步是让 V13 生成器先导验证实验（V13 generator canary）通过。第四步是让三个公开数据集、多随机种子和完整评估（full-eval）全部通过预声明门槛。第五步是重新更新论文论断矩阵（claim matrix）和实验表格。

因此，当前最诚实的状态是：诊断链条基本完成，正向第三阶段机制尚未验证。下一阶段不应该继续堆训练量，而应该重构表示目标和教师模型目标，并重新预注册验证门槛。
