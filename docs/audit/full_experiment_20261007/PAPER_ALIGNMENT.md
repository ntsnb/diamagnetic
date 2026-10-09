# 论文方法与新实验实现同步建议

日期：2026-10-07。最终状态：父任务已将下述替换包集成为独立27页审阅稿，完成主稿/便携稿编译、引用与视觉验收；原稿和历史数值未覆盖。英文替换包为 [paper_protocol_revision.tex](/C:/Users/admin/Documents/杂活/LoadForecasting_Paper_Implementation_20261007/docs/audit/full_experiment_20261007/paper_protocol_revision.tex)，不含新版正式性能结果。最终PDF SHA256为 `ff0f7958c53331625c95c6e55b1c54c1a5d41507f0359976e2ee228534e1b076`；编译与视觉范围见 REVIEW 目录的 `full_implementation_review_20261007_validation.json`。

核对稿件：[manuscript_stagewise_review.tex](/C:/Users/admin/Documents/杂活/论文前三部分修订_20260921/Methodology图件紧凑新罗马版_20261004/stagewise_adaptation_review_20261006/manuscript_stagewise_review.tex)。读取时 SHA-256：`2decdda8606d8eac00c3af5c38bdc668dde66e5c68bc587dcf436a5c4e74677e`。下列行号针对该版本；同时给出原句锚点，避免后续编辑使行号失效。英文包以 `% BEGIN <ID>` 分块，**只复制所需块，不能整份 `\input`**。

## 结论与可行性

方法、源码与实验协议能够同步，但必须把“实现完整且经过有限功能检查”与“新版完整数据性能已评估”分开。当前论文保留的表图来自更早实现，不能沿用它们支持完整作者模型、修正反馈规则或新丽水十任务的性能结论。混合预测、按释放标签更新、逐算子相关性比较已经有明确实现；相关性分离本身仍不足以识别空调、工业等物理负荷机制。正文可以保留这些业务例子作为建模动机，结果解释应使用 expert / predictive pattern。

最接近的已核验工作仍是 OneNet（两分支在线集成）、SOLID（相似历史上的预测头适配）及 PROCEED（漂移条件参数生成）。因此创新论证应集中在共同参考下的专家相关性比较、受约束的更新范围以及与临时调制的组合；不能再把前人统一描述为“更新整个模型的一种固定规则”。是否带来额外预测收益和可接受的成本，需要新协议下的实验。相关性传播还增加了较高计算成本，原有“lightweight”结论暂不适用于新实现。

## 与当前代码一致的边界

| 事项 | 当前可写的定义与证据 | 必须避免的旧读法 |
| --- | --- | --- |
| 原生预测器 | `native` 的变量注意力专家输出表示；按 gate 混合、时间池化，再经过**一个共享预测头**。[model.py:232](/C:/Users/admin/Documents/杂活/LoadForecasting_Paper_Implementation_20261007/paper_impl/model.py:232)、[model.py:286](/C:/Users/admin/Documents/杂活/LoadForecasting_Paper_Implementation_20261007/paper_impl/model.py:286) | 用一个完整网络替换 summary encoder 就代表完整官方骨干接入 |
| 完整预测器变体 | 同一架构创建多个完整专家，每个保留原 embeddings / encoder / decoder（若存在）/ normalization / head；选择相同目标列后加权其最终 forecast。[model.py:236](/C:/Users/admin/Documents/杂活/LoadForecasting_Paper_Implementation_20261007/paper_impl/model.py:236)、[model.py:282](/C:/Users/admin/Documents/杂活/LoadForecasting_Paper_Implementation_20261007/paper_impl/model.py:282) | 与原生非线性共享头模型完全同构；八个作者模型具有完全相同的模块、参数量或运算预算 |
| 真实来源 | 七个作者仓库；Transformer 是 THUML Time-Series-Library 的完整时间序列参考实现。FEDformer 此处选完整 Fourier 变体。[OFFICIAL_BACKBONES.md](/C:/Users/admin/Documents/杂活/LoadForecasting_Paper_Implementation_20261007/docs/audit/full_experiment_20261007/OFFICIAL_BACKBONES.md) | 八个均来自原论文作者发布的同类预测代码；Transformer 源码属于 Vaswani 等人原仓库 |
| 时间特征与归一化 | 统一runner不传独立calendar marks；需要该接口的作者路径用zero marks，iTransformer默认为None，Crossformer/PatchTST无该接口。日历只来自显式观测input columns。统一的是训练前缀z-score，各作者实例归一化保留自己的options。[official_backbones.py:334](/C:/Users/admin/Documents/杂活/LoadForecasting_Paper_Implementation_20261007/paper_impl/official_backbones.py:334) | 完整结构等于作者原实验全部输入协议；所有模型外面统一套同一RevIN |
| 预测头训练与在线范围 | 主预测损失离线训练原预测头；specialization 归因固定共享头/各专家头。父任务本轮已统一默认 `expert_weights()`：仅专家内部 **非预测头** `nn.Linear.weight`。默认 Stage II/III 采用该集合，头均固定。[model.py:328](/C:/Users/admin/Documents/杂活/LoadForecasting_Paper_Implementation_20261007/paper_impl/model.py:328) | specialization 的 `freeze_head` 自动等于所有更新路径都冻结；扩大范围消融也必须冻结头 |
| LRP 与通道 | 对完整预测路径逐算子传播，先每个预测步取绝对值再平均；gate 不进入单专家归因；mixture channels 只暴露 eligible weights。FFT/complex 是明确的实虚坐标扩展，残余单列。[official_lrp.py:358](/C:/Users/admin/Documents/杂活/LoadForecasting_Paper_Implementation_20261007/paper_impl/official_lrp.py:358)、[OFFICIAL_LRP.md](/C:/Users/admin/Documents/杂活/LoadForecasting_Paper_Implementation_20261007/docs/audit/full_experiment_20261007/OFFICIAL_LRP.md) | 用全网 input×gradient 代替 LRP；仅凭残余记账闭合声称归因忠实或因果机制已识别；把频域扩展全归于 AttnLRP |
| 预算与阈值 | 对应层的 `channel_budget` 由候选专家竞争共享；先用预训练模型validation误差阈值驱动训练prefix replay，再用训练后generator的同一validation预测重校准部署阈值。两值留档。归因 `training.attribution_batch_size=1` 逐块覆盖所有样本/H/target/专家对，累积梯度后统一更新。[adaptation.py:590](/C:/Users/admin/Documents/杂活/LoadForecasting_Paper_Implementation_20261007/paper_impl/adaptation.py:590)、[training.py:86](/C:/Users/admin/Documents/杂活/LoadForecasting_Paper_Implementation_20261007/paper_impl/training.py:86) | 每专家每层各得到完整budget；重放与部署必用同一个数值阈值；归因分块意味着下采样或减小H |
| 丽水数据身份 | `v2` 为九县区 15 分钟来源 + 用户指定独立市级 5 分钟来源，均形成独立小时任务，共 10 个；不同年份、独立时间轴。没有新增可追溯气象文件。[LISHUI_AND_SHIFT_PROTOCOL.md](/C:/Users/admin/Documents/杂活/LoadForecasting_Paper_Implementation_20261007/docs/audit/full_experiment_20261007/LISHUI_AND_SHIFT_PROTOCOL.md) | 新十任务等于旧小时清洗 Lishui-9 加一列；市级等于九县之和；单位已经确认为 MW |
| 主协议 | 按时间行切分 55% / 10% / 全部剩余后缀；目标区间完整落在各段；标签成熟条件 `issue+H+extra_delay`；训练/验证尾部加释放 embargo。[data.py:151](/C:/Users/admin/Documents/杂活/LoadForecasting_Paper_Implementation_20261007/paper_impl/data.py:151) | 只要训练/验证/测试按窗口顺序分开就没有目标重叠；额外延迟为 0 时当前 H 步标签立即可用 |
| 缺失数据 | 固定发行时刻不取决于未来 mask；缺失目标不插值；完整监督窗才更新；已有有效点仍评分。输入前向填充，首次不可用输入在模型 loader 使用训练前缀中位数。[data.py:132](/C:/Users/admin/Documents/杂活/LoadForecasting_Paper_Implementation_20261007/paper_impl/data.py:132)、[evaluation.py:16](/C:/Users/admin/Documents/杂活/LoadForecasting_Paper_Implementation_20261007/paper_impl/evaluation.py:16) | 将已插补旧目标当作新观测真值；因为未来目标缺失而不发行预测；所有预处理窗口数等于实际训练数 |
| 旧选段 | 旧规则基于窗口数先分 55%/10%，在约 35% 后部候选池枚举约 20% 长度块，最大化输入均值/方差偏移，先见所有候选输入。可作事后压力测试。[stream_protocols.py](/C:/Users/admin/Documents/杂活/LoadForecasting_Paper_Implementation_20261007/paper_impl/stream_protocols.py) | 严格不知道未来的在线检测；选中块代表整个部署后缀；新版 causal loader 与旧污染处理逐位复现 |
| 基线与检查点 | 六种自适应方法；OneNet 明确完整 FSNet 第二分支，PROCEED 有必要离线适配器训练；源码与兼容差异独立记录。[ONLINE_BASELINES.md](/C:/Users/admin/Documents/杂活/LoadForecasting_Paper_Implementation_20261007/docs/audit/full_experiment_20261007/ONLINE_BASELINES.md) | OneNet 单网络；SOLID 自监督；SOLID++ 只把阈值调激进；所有基线与本文 K 专家模型共享最终同一 checkpoint |

父任务在本轮论文核对后修改了默认 head 范围。本子任务同步修改 `official_lrp.py` 的 channel 过滤及对应测试；完整头仍在归因路径中。全部八模型 mixture + 预测头保护新增回归已通过，适配层与 LRP 合并为 **106 passed in 32.16s**，含 CPU/CUDA。该测试结果只说明实现行为。本轮此前允许头参与调制的短闭环记录与修正后的短闭环必须分版本保存；二者也都不等于旧论文表格的来源，更不等于新版正式性能实验。

## 逐处替换清单

英文完整内容见同目录 `.tex` 的对应 ID。为减少公式改动，C1/C2/C4/C6/D2 只替换文字，保留原公式及标签；C3 增加一个有独立标签的完整预测融合公式。

| ID / 位置 | 需要替换的原句或锚点 | 建议与理由 |
| --- | --- | --- |
| A0，摘要 92 | “The framework supports plug-and-play integration, with the same controller applied to eight Transformer-based backbones without redefinition.”；“Experiments on four load-forecasting benchmarks ... show improvements ...” | 替换整个摘要为中性方法摘要。结尾明确新协议覆盖 3 个公共任务 + 10 个丽水任务，而旧数值仅供追溯，新版性能尚待实验；同时移除“所有前人都更新全模型”这一不成立概括。 |
| A1，贡献 113–114 | “A plug-and-play implementation evaluated on eight ...”；“Empirical evidence of improvements ...” | 改为实现与实验协议贡献，不再把待执行的完整矩阵写成已得到的实证贡献。 |
| C1，241 | “Other Transformer backbones can replace $m_{\psi}$ while retaining its summary-embedding output.” | 说明 summary encoder 供 latent/gate，完整骨干进入 expert bank。 |
| C2，262 | “Each expert ... is implemented as a Transformer block ... embeddings are fused ... output head” | 保留 native 定义，句首明确 native；不把它当作全部作者模型共同结构。 |
| C3，269 后 | 原式 `eq:expert_fusion_forecast` 后没有完整网络变体定义 | 插入完整 `F_k(X)` 输出加权公式；明确每专家保留原预测头，不能把非线性共享头移出/移入求和当作等价变换。 |
| C4，279 | “For each expert, the shared output head maps its time-averaged embeddings ...” | 分别定义 native 单专家预测和 complete-network 单专家预测；两者均排除 gate 再做 LRP。 |
| C5，287 后 | “attention and normalization layers use their corresponding propagation rules” | 加短段解释卷积/频域覆盖及残余；频域规则是此实现扩展，不能由一条 AttnLRP 引用代替。 |
| C6，370 | “parameters of the shared output head are held fixed for this term” | 改为 native shared head 或完整专家各自 head；只限定分工损失，预测损失继续训练头。 |
| D1，386–388 | “The shared encoders, output head, and invariant--variant branches remain fixed online.” | 明确默认候选是排除预测头后的专家线性权重；Stage II/III 同集合，别把 dense 范围也写成头冻结。 |
| D2，417 | “Each output channel corresponds to a column of its layer's weight matrix” | 说明同层跨专家共享budget；在数学记号 `W_{de}` 下保留列选择，避免与 PyTorch 的 `W[e,d]` 存储方向混淆。正文不需加入库级实现细节。 |
| E0，467–469 | 仅在 `\expnote{H1...}` 中承认旧实现；随后“We evaluate the proposed framework...” | 把历史来源写成普通可见正文，使删审阅颜色/脚注后仍成立。 |
| E1，471–477 | “new Lishui-9 ... matched meteorological covariates ... all key experimental modules”；“No random shuffling ... temporal axis” | 分开公共任务、新十任务与历史 Lishui-9；说明各自频率、单位状态；shuffle 规则放到处理段准确表述。 |
| E2，478–481 | “Standard cleaning and interpolation are applied when needed” | 以真值 mask、输入前向填充、小时右闭时间、严格目标边界和释放 embargo 代替笼统插值。 |
| E3a / E3b，484 / 513 | “same framework ... expert fusion module ... unchanged”；“comparison isolates the effect of the forecasting backbone itself” | 区分架构变体和共同控制器；旧实现表保留历史身份；新配置预算仅作为起点，说明来源、验证选模和计算成本。 |
| E3c，Implementation Details后或附录 | 原稿没有重放/部署阈值顺序和归因分块定义 | 简短交代两个validation校准阈值、默认归因batch=1而全样本计量；完整结构但独立marks置零的约定已纳入E3a。 |
| E4，515–525 | “Labels arrive with delayed feedback”；只列单阶段及 Full | 给出完整时钟与8种 stage 位组合；限定同 checkpoint 的适用层级，明确新未来 mask 不决定发行、接受测试仅用已释放标签。 |
| E5，527–528 | “whether each individual stage improves ... joint ... achieves the best” | 改为中性评分定义：已发行预测、有效目标点、任务聚合规则、warmup、更新次数与成本；不预设 Full 最优。 |
| E6，530–531 | “we additionally evaluate the strongest backbone ... search ... maximizes ... shift score” | 改名为事后选择的 shift stress test；说明 55/10/35 候选、20%块、枚举选择；旧分数保留在旧结果，不迁移到新数据。 |
| B1，534–549 | “All methods start from the same frozen offline checkpoint”；OneNet / SOLID / SOLID++ 的三条定义 | 改为六算法真实定义及 FSNet 模式、必要训练和共同反馈协议；Frozen 是额外部署参考，不算六个自适应算法之一。 |
| S1，建议置于新协议段或替换现有结构消融定义 | 668 的“w/o Invariant / Dense Stage II / MLP Gate”及旧标签 | 新结构定义单列；不能只换旧表的 row label 后声称相同消融。 |
| H1，551 或历史结果入口 | 原文直接据旧表宣称新 controller 的优势 | 可见历史说明，保留旧数值与图表原貌。 |
| H2，可用于 579 / 609 / 638 / 670 / 866–877 的历史总结 | “current results support ... replacing only temporal encoder”；“robust and lightweight improvement” | 若保留旧结果讨论，则改成历史实现范围内描述。H2 是一次性保守总述，不必在每个小节重复粘贴。 |

三个短处还需父任务随集成核对：107 的 “difference between variant and invariant representations” 应改为 “difference between variant and invariant predictions”；160 的 “interchangeable ... backbones” 宜改为 “complete forecasting experts with architecture-specific prediction paths”；96 的 “batch-delayed online adaptation” 宜改为 “delayed-feedback online adaptation”。新主协议是观测时钟，不是整批未来目标提前可用的时钟。

## 结构开关与公平比较的明确含义

替换包 S1 给可放入正文的两段摘要；详细配置和可执行证据以 [ABLATIONS.md](/C:/Users/admin/Documents/杂活/LoadForecasting_Paper_Implementation_20261007/docs/audit/full_experiment_20261007/ABLATIONS.md) 与 [build_full_experiment_matrix.py](/C:/Users/admin/Documents/杂活/LoadForecasting_Paper_Implementation_20261007/tools/build_full_experiment_matrix.py) 为准。

| 开关 | 必须写清楚的实际差异 |
| --- | --- |
| `Single` | 真正 K=1；gate 恒1；无专家对，spec=0。不能保留多专家再说“单专家”。 |
| `noVAE` / `noSpec` | 前者移除 mu/log-var/reconstruction 模块，换确定性 gate features；后者只置 specialization 权重0。 |
| `woInvariantDisabled` / `ERMpair` / `VariantOnly` | 分别是移除两分支及生成器、保留两个普通ERM分支、仅保留variant相对零锚。三者容量和Stage III功能不同。 |
| `woContrast` | 仍训练两分支，生成器只接收variant预测；区别于删除invariant分支。 |
| `woRevIN` | 仅 native / PatchTST / iTransformer 有明确对象；iTransformer的归一化不应冒称其就是同一个RevIN模块。 |
| `DenseExperts` / `SyncAllExperts` / `DenseStageII` | 分别扩大为全部专家参数、专家+router+head、全部预测器参数。官方专家自带头包含在专家内，因此前两者不能声称“所有头仍固定”；实际无梯度参数不保证发生变化。 |
| `NoEMA_current` / `ReferenceFrozen` | 当前快照重算所有历史和当前图，与一直使用初始参考，是两个独立对照。 |
| `woAcceptanceCheck` | 去掉有限劣化步的拒绝，仍保留范数界和非有限拒绝；新实现没有旧 acceptance buffer，不能命名成已经删除了 buffer。 |
| `NoExtraDelay` | 只令额外delay=0，仍等到issue+H；当前Full额外delay=2，所以有真实对照。 |
| `stages_000`…`stages_111` | 位序为I/II/III；同结构的8项共享一个完成离线训练的checkpoint，含同一辅助分支/生成器；结构消融另训。 |

六基线与本文模型的公平性应写成同数据、目标、发行索引、可见标签、验证预算及预算透明，而不是所有模型相同。当前框架的 K 个完整预测专家及两辅助分支与基线单骨干的参数量不同。OneNet 又有独立 FSNet，PROCEED 又有必需的离线生成器训练；这些容量与训练代价都应列报。新矩阵中的种子42与未截断预算是计划配置，不构成多种子或收敛证据。

第二轮来源核对后，B1 仅补充两句复现设置。原 SOLID 的 [eval 模式](C:/Users/admin/Documents/杂活/LoadForecasting_Paper_Implementation_20261007/third_party/official_online/SOLID/exp/exp_main.py:600)与[逐样本梯度求和](C:/Users/admin/Documents/杂活/LoadForecasting_Paper_Implementation_20261007/third_party/official_online/SOLID/exp/exp_main.py:806)对应默认 `solid_adaptation_mode='eval'`、`solid_gradient_reduction='sum'`；OnlineTSF 的 [train 模式](C:/Users/admin/Documents/杂活/LoadForecasting_Paper_Implementation_20261007/third_party/official_online/OnlineTSF/exp/exp_solid.py:174)配合该文件第143行的默认 mean MSE，通过显式 `train`/`mean` 组合提供，SOLID++ 固定采用该组合。PROCEED 离线合成漂移训练的历史概念来自上一随机 mini-batch，与当前 [fit_offline 实现及说明](C:/Users/admin/Documents/杂活/LoadForecasting_Paper_Implementation_20261007/paper_impl/online_baselines.py:908)一致；这两句不新增性能结论或引用 key。

## review PDF 保留旧结果的建议

1. 保留旧 `.tex` 中表格、图像资源和数字的原始内容。将实验前六小节定位为“revised implementation and evaluation protocol”，历史结果从单独可见的 `Historical results retained for review` 段开始。E0 和 H1 应为普通正文，不能只藏在可关闭的 `\expnote` 内。
2. 旧 `tab:impl_details` 同样是历史表，其 4 experts、hidden128/4heads、统一RevIN和旧更新门控不能配置新版。E3a 已明确该表身份；也可把整张旧表原样移到历史部分。新设置引用当前配置清单，暂不造一张“已完成主实验设置”表。
3. 表图原注、标签和数值保持不变；在历史结果开头说明旧 architecture / baseline 标签是该版本标签，包括旧代理算法。若浮动使表图跑到新协议前，使用稿件已有的 `\FloatBarrier` 限制历史段边界，或在表前加简短来源说明。不要为方便阅读把 Lishui-9 改名为 Lishui-10。
4. 保留 E5/E10/E11 的历史警告或把它们归入普通历史限制段：当前批目标用于接受、旧代理基线、不同checkpoint对比等疑点，不会因新实现已修复而使旧数值自动有效。旧最大shift分数也只属于旧文件和旧选段。
5. 579、609、638、670、794以后的Lishui讨论和866–877仍有“current/newly curated/confirmed/robust/lightweight”等跨版本结论，不能只改前六小节就保留为新版证明。建议用 H2 作历史总述，逐处删去跨版本泛化；保留旧数值本身不要求保留旧因果推断或排名解释。
6. 新数值齐备后，用新run ID、配置/源哈希、目标掩码和统一checkpoint计划生成新的表图，再决定正式稿保留哪些历史结果。本轮有限CPU/CUDA检查和短真实CSV闭环只放实现验证记录，不放性能比较表。

## 引用核验与新增来源

替换包只使用原稿已存在的6个 cite keys：`lrp`、`dtd`、`attnlrp`、`onenet`、`solid`、`pad`。本轮核对了下面的一手来源，未新增 `\bibitem`，未把 PROCEED 另起一个不存在的 cite key。算法行为还对照了本地固定提交，具体兼容差异见来源审计。

| 已存在key | 一手来源与核验信息 | 本次用途 |
| --- | --- | --- |
| `lrp` | S. Bach et al., *On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation*, PLOS ONE, 2015，[期刊原页](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140) | 原始LRP背景，不用来声称所有新频域规则已有证明。 |
| `dtd` | G. Montavon et al., *Explaining NonLinear Classification Decisions with Deep Taylor Decomposition*，[作者预印本](https://arxiv.org/abs/1512.02479)，与原稿2017期刊条目的题名/作者对应 | 沿用原有LRP/Taylor背景。 |
| `attnlrp` | R. Achtibat et al., ICML 2024, PMLR 235:135–168，[官方论文页](https://proceedings.mlr.press/v235/achtibat24a.html) | 注意力/归一化传播。源码频域扩展另外说明。 |
| `onenet` | Y.-F. Zhang et al., NeurIPS 2023，[官方论文](https://papers.neurips.cc/paper_files/paper/2023/file/dd6a47bc0aad6f34aa5e77706d90cdc4-Paper-Conference.pdf)、[作者预印本](https://arxiv.org/abs/2309.12659) | 两互补分支与动态组合；不要写单网络。 |
| `solid` | M. Chen et al., *Calibration of Time-Series Forecasting: Detecting and Adapting Context-Driven Distribution Shift*，[作者预印本](https://arxiv.org/abs/2310.14838)、[作者仓库](https://github.com/HALF111/calibration_CDS/tree/3c74e5f8b3c921a2e7cda30f36117c295bf8c112) | 选择相似上下文、监督微调预测头；不是自监督。 |
| `pad` | L. Zhao and Y. Shen, *Proactive Model Adaptation Against Concept Drift for Online Time Series Forecasting*, KDD 2025，[作者预印本](https://arxiv.org/abs/2412.08435)、[作者仓库](https://github.com/SJTU-DMTai/OnlineTSF) | PROCEED；SOLID++和跨骨干FSNet组合也需明确该来源。 |

FSNet 原稿无对应 key。可列为待插入来源：Q. Pham, C. Liu, D. Sahoo, S. C. H. Hoi, *Learning Fast and Slow for Online Time Series Forecasting*, ICLR 2023，[OpenReview原文](https://openreview.net/pdf?id=q-PbpHD3EOk)、[作者仓库](https://github.com/salesforce/fsnet/)。本任务不擅造 `\cite{fsnet}`。若父任务不新增参考文献，可用源代码注或现有 `\url` 包脚注暂时标明；正式稿宜补核验后的文献条目。

Transformer 的实现来源需要独立于 `attention` 论文引用标注：[Time-Series-Library 固定提交](https://github.com/thuml/Time-Series-Library/tree/4e938a1767106324dd753b2a44832bf870a0252e)。现有 `attention` 只能说明架构出处，不能代替该预测代码出处。其余七个完整骨干的已有论文引用保留，实际运行提交、许可证、兼容补丁以 `OFFICIAL_BACKBONES.md` 和 manifest 为准。

## 剩余风险与集成核查

- **结果尚未齐备。** 当前全矩阵、脚本和功能检查不能支撑新版优越性、全架构稳健性、物理机制识别或低时延结论；单种子配置也没有训练方差证据。不要把最新测试计数写入摘要当作实证性能。
- **数据含义尚有待核。** 九县来源先前加工历史、负荷单位和业务统计边界未完全确认；市级与九县年份不同。新十任务不适合计算未经单位确认的物理量汇总，亦不能直接和旧Lishui-9数值作配对比较。新 Spain 时钟已与旧派生文件区分，旧数值不会自动获得新时轴保证。
- **预处理与监督计数有不同层次。** v2构建器的有效窗口统计早于训练/验证释放embargo；模型loader又允许训练中位数初始化缺失输入。论文应以每次run的实际发行/可监督计数为准，不能原样把构建统计写成训练样本数。
- **归因理论和成本。** FFT/complex 实虚坐标规则是实现扩展；正文与附录已摘要说明卷积、频域、离散选择及残余记账约定，详细逐算子实现见 `OFFICIAL_LRP.md` 和代码。完整图参数梯度和残余检查已测，但未完成扰动忠实性实验；记账守恒不能替代这一证据。完整数据上的训练显存和时间尚待实测。
- **扩大范围消融。** 默认头冻结已经统一；DenseExperts/SyncAllExperts/DenseStageII有意扩大范围，不能套用默认冻结结论。尤其含复数权重和非线性模块的完整模型消融，需要与普通Stage II分开检查实际变更集合。
- **随机策略差异。** 主框架官方eval采样使用固定 expert seed；在线基线运行器使用可恢复的事件seed计数。正式比较应固定并公开这些策略，不能仅凭同一个整数 seed 声称相同的随机路径。
- **来源许可。** 在线来源审计记录 OneNet 对应LICENSE材料缺失、OnlineTSF固定版本未找到仓库级LICENSE。正文可以准确说明来源；发布源码前仍需处理原记录中的许可材料问题，不应填入猜测许可证。
- **审阅中代码可能继续变化。** 当前引用的实现和默认head集合以本轮读到的版本为准。父任务若改变scope、mask处理或seed策略，应刷新矩阵来源hash、检查点来源及相应文字。本文件不宣布ZCode正在进行的审核已经完成。

英文包保留原native融合、相关性差异、Stage I–III方程，不重写旧表图。本任务对新增包做静态引用、标签、环境与括号检查；最终版面、公式换行、浮动顺序及PDF视觉检查由父任务在集成后完成。
