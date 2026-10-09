# 新版结构、控制器与三阶段消融

2026-10-07；实施与验证记录。当前交付是可运行配置、结构测试和审计定义，未执行全数据长训练，未产生性能优越性结论。

## 当前矩阵

`tools/build_full_experiment_matrix.py` 已生成 `configs/full_experiment/structural/matrix.json` 及 **1794 个训练配置**。矩阵包含：

- 13 个真实数据任务：Panama、Tetouan（Zone 1 单负荷目标）、已修复时钟的 Spain，以及丽水 v2 九县与指定 5 分钟市级源组成的 10 个独立小时任务。候选 15 分钟市级文件不重复计数。
- 8 个完整作者预测器：iTransformer、Crossformer、Autoformer、Transformer、PatchTST、Informer、FEDformer、Pyraformer，另有原生 `native` 参考。使用 `model.expert_backbone`，不以 `*_style` 编码器代替完整骨干。
- 默认种子 42；可通过 `--seeds` 扩展。没有执行多种子实验，不支持方差、显著性或稳定性结论。
- 每个 Full 配置训练一次，再在同一个离线 checkpoint 上评估 8 种三阶段组合；结构/控制器消融单独生成训练配置。共 **2613 项阶段评估**。
- 六个基线 GD、Head-GD、SOLID、SOLID++、OneNet、PROCEED 共 **624 项计划**（13×8×6）。这里记录类名、骨干、数据、种子和共同时间规则；基线项不是传给 `paper_impl.run` 的训练配置，需要由基线运行器调度。

OneNet 正式清单显式采用 `onenet_mode=onlinetsf_cross_backbone`：每个目标作者骨干配独立官方 channel FSNet 分支（`paper_impl.official_fsnet.OfficialFSNet`）。此前双 Transformer 等显式双骨干组合属于命名变体，不计为这个正式组合；两分支的架构、预训练和检查点须分别记录。清单交由 `paper_impl.baseline_run` 调度。

Tetouan 主任务保持原稿的 `Zone 1 Power Consumption` 单目标；Zone 2、Zone 3 与其余数值列仍可作为历史输入。首轮生成时的三目标配置已在本轮未训练配置内更正，以保持原实验任务可比性；未修改数据或结果。三目标版本应另用明确的可选任务名称登记，当前矩阵未默认增加它。

完整作者源码和在线基线源码清单的 SHA-256 已纳入矩阵，来源和边界分别见 `OFFICIAL_BACKBONES.md`、`ONLINE_BASELINES.md`。

矩阵逐项记录配置 SHA-256，并记录构建脚本与来源清单哈希。Tetouan 目标更正后，全部 1794 项重新通过当前 `_normalize` 契约解析，配置哈希已刷新；OneNet 组合元数据更新后的专项矩阵测试也已通过。

所有配置使用完整数据轴，`data.max_rows=null`、`replay.max_windows=null`、`test_windows=null`；默认 L=24、H=12、训练/验证比例 .55/.10、完整测试后缀，额外标签延迟 2 步。Tetouan 每步 10 分钟，其余当前任务每步 1 小时，因此相同 L/H 是相同步数，不是相同物理时长。各任务独立切分；丽水单位仍为 `null`，不能跨任务直接汇总有量纲误差。

主模型配置采用 3 专家、hidden=64、heads=8、depth=2。固定起始预算为主预测器 5 epoch，FOIL 2 round×2 MIL epoch、每轮 1 MEI epoch，生成器 2 次完整训练前缀回放。这是可复现的起始预算，不是已经调优或证明充分收敛的论文预算。原生 Full 显式启用 RevIN；模型类历史默认仍是关闭，矩阵不依赖该默认。

## 消融的精确定义

每项都保留完整标签时钟 t+H+delay 和原始真值掩码，除非该项明确改变额外延迟。

| 配置名称 | 真正改变的行为 | 保留项与解释 |
| --- | --- | --- |
| `full` | 全结构与三阶段 | 测试阶段评估全部 `stages_000` 到 `stages_111` |
| `noSpec` | `training.specialization_weight=0` | 专家数、VAE、分支和三阶段保留；训练日志分工项严格为 0 |
| `noVAE` | 删除 `mu`、`log_var`、`decoder` 模块，以 MLP 确定性特征送入 router | 不是只把 KL 权重设 0；重构/KL 输出为 None，训练相应损失为 0 |
| `Single` | `n_experts=1`，确实只创建一个专家、单输出 router | gates 恒为 1，没有专家对；分工损失为 0，Stage I 自然没有可学习的路由竞争 |
| `woInvariantDisabled` | `foil.mode=disabled` 删除**两条**辅助预测分支；`generator.mode=identity` 没有生成器参数；Stage III 关闭 | 主 MoE 仍训练，评估 `stages_110`；这是整体移除辅助不变模块，不能冒称只去掉一个损失 |
| `ERMpair` | 保留独立两分支，分别用普通 MSE 训练，取消环境推断和 FOIL 环境目标 | 生成器仍接收 variant−invariant 预测差；是辅助预测器容量对照，不是同计算量对照 |
| `VariantOnly` | 删除 invariant 预测器，只有一个用 ERM 训练的 variant 预测器 | 生成器接收 variant 相对固定零锚的预测；与 ERMpair、woContrast 均不同 |
| `woRevIN` | 原生模型关闭 RevIN；PatchTST 关闭 `patch_revin`；iTransformer 关闭 `itransformer_use_norm` | 仅这三个模型有此定义，共 39 配置；其余六个官方骨干不生成“无变化”的伪消融 |
| `woContrast` | 保留并训练两分支，`foil.contrast=variant`，将生成器输入由预测差改为 variant 预测 | 辅助预测器容量和训练保留；不是旧代码“潜变量 z 替代”的数值复刻 |
| `DenseStageII` | `stage_ii_scope=all_forecaster`，Stage II 的可更新集合扩大到全部预测器参数 | 只有实际有梯度的参数发生变化；保留释放时钟与步长半径 |
| `DenseExperts` | `stage_ii_scope=dense_experts`，专家全部参数可更新，不做稀疏输出通道选择 | router、共享输出头与门控编码器保持在此范围之外 |
| `SyncAllExperts` | `stage_ii_scope=experts_router_head`，联合更新专家、router 和预测头 | “同步”指参数范围，不表示提前获取或同步释放未来标签 |
| `NoEMA_current` | `reference_mode=current`，LRP 打分时复制当前模型，并在同一快照下重算历史和当前归因 | 无 EMA；不是冻住初始参考，也不是让历史图保留旧坐标 |
| `ReferenceFrozen` | `reference_mode=frozen`，一直保留初始化参考 | 单独检验不刷新参考的作用，与 NoEMA_current 分开 |
| `woAcceptanceCheck` | `rollback_on_worse=false` 关闭 Armijo/劣化拒绝 | 有限但更差的候选步可接受；非有限损失仍被拒绝，半径约束保留；新版没有 acceptance buffer，不能宣称删除了一个不存在的 buffer |
| `NoExtraDelay` | `data.extra_delay=online.extra_delay=0` | 完整 H 步标签仍在 t+H 释放；Full 的额外延迟为 2，因而有真正对照 |

`FoilConfig.mode` 与 `ForecastBranches(mode=...)` 必须匹配。`FoilConfig.contrast` 与 `ForecastBranches(contrast=...)` 同样必须匹配；支持 difference、variant、invariant 三种预测坐标，其中矩阵选择 variant 作为 w/o difference 对照。无 invariant 分支时不能请求 invariant contrast。

官方骨干保持自己的完整预测头和归一化路径；不能在八个完整模型外面统一包一层原生 RevIN。`woRevIN` 的 PatchTST/iTransformer 两项是它们各自的归一化消融，命名对应“去掉该模型归一化”，不意味着两者机制完全相同。

## 三阶段与标签时钟

位序固定为 I/router、II/持久专家更新、III/临时调制。`stages_000` 是冻结离线预测器；其余七种组合按位启用。Full 的八个阶段组合共享已经训练的专家、分支和生成器，目的是隔离测试期操作；**不为每一个阶段组合单独再训练生成器**。如果研究目标变为各组合各自最优性能，需要另建预注册的独立训练对照。

结构消融通过独立配置重新训练，因此 checkpoint 中的模块/参数集合确实不同。`woInvariantDisabled` 只能评估第三位为 0 的组合，当前矩阵采用 I+II；开启第三位会被拒绝。

`DataConfig.extra_delay` 与在线设置保持一致。主线按首次后继分段发行时刻设置离线训练/验证截止，移除尚未完整释放的尾部监督窗口，保持原始时间索引，不重新压缩时钟。掩码缺失窗口不进入监督拟合；在线评分与完整标签释放遵守各自掩码。额外延迟置 0 仍然必须等待整个预测区间完成，不能拿 t+1…t+H 在 t 训练。

## 专项验证

命令：

```powershell
python -m pytest tests/paper_impl/test_structural_ablations.py -q
```

结果：**42 passed in 11.15s**。测试是 48 点人工序列上的有界功能检查，原生小模型 hidden=4；这些检查不代表真实数据预测性能。

16 个配置分别走过真实主模型训练、辅助分支训练或明确跳过、生成器延迟回放或 identity 跳过、在线发行/释放、checkpoint 保存与再评估。每个再评估的 issue_times、predictions、targets 与首次评估逐数组一致；回放与在线记录的每次标签释放都不早于 issue+H+delay。

附加检查验证：

- K=1 真实模块数、单路由、无专家对、分工项为零。
- noVAE 的参数名里确实没有 mu/log_var/decoder，MLP gate 接收梯度，`stochastic=True` 仍为确定性输出。
- RevIN 的输入/参数梯度、显式目标通道逆归一化、LRP 有限性与有符号守恒；外包官方模型的原生 RevIN 被拒绝。
- 四种分支模式的真实模块数量、参数不共享、梯度与训练后冻结；disabled 分支和 identity 生成器的 checkpoint 均无参数。
- difference/variant/invariant 三种对比量都保留两分支容量，并选择准确的预测坐标。
- 稀疏专家、稠密专家、专家+router+head、全预测器四种范围的实际参数变化；范围外参数保持不变。
- EMA 只按系数更新参考专家权重；current 使用当次评分前当前快照；frozen 不改变初始参考。
- 人为构造有限劣化与 NaN 候选损失：关闭接受检查只允许前者，NaN 仍拒绝并恢复参数。
- 矩阵无重复阶段名称、不截断数据/回放/测试，六基线齐备，归一化消融只生成有意义的模型组合，已有输出不能被构建器覆盖。

未发现需要修改核心模型/FOIL/训练/控制器语义的失败。测试编写期间修正过一个测试自身的 LRP 维度顺序假设，采用项目既有 `[batch,target,expert,variable,lag]` 契约，没有为通过测试更改模型。

完整官方八骨干的前后向、头部范围和权重载入验证见已有 `test_official_backbones.py`；本文件中的 16 项闭环主要验证原生结构/控制器开关，不把它等同于 1794 项配置均完成了官方模型的全量训练。

## 旧消融名称与新版差异

旧配置入口主要是 `train/run_lishui_component_ablation.py:37–67` 与 `train/run_ablation_load_efficient.py:119–168`。旧在线行为见 `train/online_loop.py:107–121,189,393,465–479,608`。

| 旧名称 | 新版对应 | 需要避免的误解 |
| --- | --- | --- |
| `single_expert` | Single | 必须确实构造 K=1，不能只改变显示名称；新版已测 |
| `mlp_gate_no_vae` | noVAE | 新版删除完整 VAE 模块；旧重构误差门控规则不沿用 |
| `wo_invariant` / `no_invariant` | woInvariantDisabled、ERMpair、VariantOnly 三个清晰对照 | “去不变模块”同时可能改变容量和第三阶段功能，必须分别报告 |
| `wo_revin` / `no_revin` | woRevIN | 基准必须开启对应归一化；只在 native/PatchTST/iTransformer 生成 |
| `dense_stage2_all_layers` | DenseStageII | 旧脚本清空 target_modules；新版明写 all_forecaster；不是只调大 top-k |
| `dense_stage2`（丽水旧入口） | DenseExperts 或 DenseStageII，须按问题选定 | 旧入口同时调整 mask_threshold/topk/target_modules，不能视为一个无歧义的范围名称 |
| `sync_all_experts` | SyncAllExperts | 旧 alpha_head 对应新版 router；没有去掉完整标签等待 |
| `no_ema_reference` | NoEMA_current；另加 ReferenceFrozen | 关闭 EMA 不等于使用冻结初始参考；两者已分开 |
| `no_rollback_accept`，报告显示 w/o Rollback/Buffer | woAcceptanceCheck | 新版没有旧 accept_buffer，不能伪称复现“删除 buffer” |
| `no_delay_feedback` | NoExtraDelay | 只移除额外延迟，不能越过 H 步观测等待 |
| `no_invdelta_stage3` | woContrast | 旧开关可能走 latent-z 替代；新版使用 variant 预测坐标，明确是新定义而非数值复刻 |
| `grad_stage2_no_lrp` | `online.selector=gradient_only` | 已有运行器与 evaluation_overrides 支持；当前结构矩阵没有单列此项，可追加为选择器专项对照 |

精简建议：保留 Full/ERMpair/woContrast 来区分 FOIL 目标与预测差的作用；将 Single、noVAE、woInvariantDisabled 的参数量和训练成本一并报告；以 DenseExperts 与 SyncAllExperts 区分“专家内稀疏性”与“额外更新 router/head”；将梯度选择器对照单独加入选择器实验，避免混同为去掉模型的全部 LRP 功能。历史结果不能直接贴到这些新定义下。

## 重建命令

```powershell
python tools/build_full_experiment_matrix.py `
  --dataset-root 'C:\Users\admin\Desktop\最近\1.28\IEEE-Transactions-LaTeX2e-templates-and-instructions\dataset\load_benchmarks' `
  --lishui-registry data/lishui_online/v2/registry.json `
  --spain-source data/paper_impl_prepared/spain_energy_weather_clock_corrected.csv `
  --output configs/full_experiment/structural/rebuild_20261007 `
  --seeds 42 --extra-delay 2
```

构建器只生成配置并解析当前 API 契约，不自动开始长实验。未来正式运行前应冻结预算与种子计划，记录实际训练/验证/测试元数据和所有方法共同发行索引，分别报告预测误差、训练代价、在线更新时间与缺失真值覆盖率。
