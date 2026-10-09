# 新版区域负荷预测方法实现

当前入口是 `paper_impl`，对应2026-10-07方法审阅稿。原有 `models/`、`train/` 和历史实验记录保留，不能将它们的旧结果作为新版方法的结果。新实现所在分支为 `codex/paper-complete-20261007`。在Linux服务器运行请先阅读 [nanyang部署说明](docs/NANYANG_SETUP_CN.md)。

**2026-10-07扩展已完成实现验收：结构消融、八种完整官方骨干、六种在线基线及丽水十任务进入新入口；整套600项回归及48种骨干/方法组合通过，独立论文审阅版已同步，ZCode GLM-5.3 max三轮审核闭环。** 最新逐项证据见 [全文实验验收](docs/audit/full_experiment_20261007/REQUIREMENTS.md) 与 [验证记录](docs/audit/full_experiment_20261007/VALIDATION_REPORT.md)。旧就绪性文档保留为历史状态。短流程验证只说明程序能运行，不能代替正式论文的性能实验。

## 完整实验入口

官方骨干额外依赖列在 `third_party/official_forecasters/requirements-runtime.txt`。完整网络以固定提交快照置于本仓库，不在运行时下载。主方法与基线共同使用 `paper_impl.data` 的数据、标签时钟和评分真值；`paper_impl.experiment` 提供统一调度。

**2026-10-09：当前矩阵为 `configs/full_experiment/controlled_seeds42_43/matrix.json`。** 已加入独立官方骨干Frozen、同预算参数定位对照及专家诊断。42/43各有2522项运行，共3588份框架配置与1456份参照配置。原 `structural` 和 `structural_seeds42_43` 保留为历史版本。[本轮修改与完整清单](docs/audit/controlled_comparisons_20261009/IMPLEMENTATION_RECORD.md)；[此前时间索引修复](docs/audit/experiment_fixes_20261009/IMPLEMENTATION_RECORD.md)。

```powershell
# 先生成可审核计划：市级任务、iTransformer、Full与Frozen及六在线基线，42/43共16项
python -m paper_impl.experiment --matrix configs/full_experiment/controlled_seeds42_43/matrix.json --datasets lishui_city --backbones itransformer --ablations full --seeds 42 43 --output results/full_experiment/my_plan

# 不加任务筛选，生成完整5044项运行计划；默认不启动训练
python -m paper_impl.experiment --matrix configs/full_experiment/controlled_seeds42_43/matrix.json --seeds 42 43 --output results/full_experiment/my_full_plan

# 指定一份主方法配置进行训练和在线评价
python -m paper_impl.experiment --kind framework --config configs/full_experiment/controlled_seeds42_43/lishui_city/itransformer/full_seed42.json --output results/full_experiment/my_framework

# 单独执行基线配置；OneNet默认由目标骨干与官方FSNet组成
python -m paper_impl.experiment --kind baseline --config configs/full_experiment/controlled_seeds42_43/baselines/lishui_city/itransformer/OneNet_seed43.json --output results/full_experiment/my_baseline

# 完整官方骨干在真实市级小样本上的功能验收，非正式性能结果
python tools/smoke_full_framework.py --device cpu --models itransformer --output results/full_experiment/my_backbone_check
```

矩阵命令默认只写计划。加 `--execute` 才逐项运行，可用 `--families framework` / `--families baseline`、`--methods`、`--ablations`、`--seeds` 缩小实验集合。生成器可通过 `tools/build_full_experiment_matrix.py --help` 查看如何冻结多个种子。不同结构需重新训练；Full的八种阶段组合从同一预训练checkpoint分别启动，其余结构按各自配置指定的阶段组合评价。

种子请求按每个适用的“数据任务×骨干×结构/基线”逐项检查；例如缺少43会报 `种子 43 缺少配置`，不会只执行42。主配置、FOIL、专家训练和官方骨干采样的根种子必须与任务一致。FOIL两个分支保留 `seed`、`seed+1009`，官方专家采样保留 `seed+专家索引`；这些确定性偏移均随根种子改变。旧checkpoint重评估保留原保存的采样种子，不据此生成新的种子实验。

缺失标签窗口仍从监督训练中排除。保留窗口的 `issue_times` 传入FOIL，平滑邻居满足原始时间索引差不超过 `smoothing_radius`，不把缺口两侧压缩成相邻时刻。现有数据采用固定采样间隔，因此该半径以原始采样步计。`foil_history.json` 和checkpoint记录窗口索引与环境标签的逐行对应关系。

Full配置另含 `selection_comparisons: ["matched_lrp", "matched_gradient", "matched_random", "matched_dense"]`：同一个离线checkpoint分别评价四个只启用Stage II的控制分支。前三者每层选相同数量的完整权重行，候选集合、标量参数预算和更新规则相同，仅排序规则不同；密集控制更新同一候选集合的所有权重。原Full的自适应稳定性筛选和八阶段组合不变。`selection_comparison.json`逐释放事件校验预算，更新记录分别保存selected/changed/eligible参数量。旧 `gradient_only` 单独使用时不等于该严格预算控制。

新矩阵默认 `expert_diagnostics: {"enabled": true, "distance_interval": 1}`，逐发行窗保存路由、持久专家预测/误差和分工距离。各评价分支输出 `*_expert_diagnostics.npz/json`，并汇总为 `expert_metrics.csv`、`expert_pair_metrics.csv`；`metrics.csv`同时有总体预测误差、路由熵和平均分工距离。利用率同时报告平均路由权重与最高权重占比；当前是稠密专家计算，不能把它解释成只执行部分专家。分工诊断采用未叠加Stage III调制的持久专家，与分工损失定义一致；实际发行预测的误差另行保留。诊断时间/峰值另记，纯运行成本测量可显式关闭诊断。旧配置默认不开启。

完整官方网络通过 `model.expert_backbone` 选择：iTransformer、Crossformer、Autoformer、Transformer、PatchTST、Informer、FEDformer和Pyraformer均保留原预测网络与输出头。此路径用门控权重组合各完整专家的预测；原生路径在共享输出头之前组合专家表示。两种实现形式在配置中明确区分。卷积/FFT等操作的LRP规则、官方代码兼容修改和适用变体见 [骨干来源](docs/audit/full_experiment_20261007/OFFICIAL_BACKBONES.md) 与 [归因实现](docs/audit/full_experiment_20261007/OFFICIAL_LRP.md)。

逐样本、逐预测步的可微LRP开销明显高于单次前向。`training.attribution_batch_size=1` 默认分块反传全部样本的分工损失，再与整批预测梯度合并，只裁剪和更新一次；没有减少预测步或专家对。等价性和有界成本见 [归因分块核验](docs/audit/full_experiment_20261007/ATTRIBUTION_BATCHING.md)。先在目标设备上测量计划中的L/H、专家数和batch_size，再决定完整实验预算。不要将短样本的误差或运行时间直接写入正式结果表。

独立时间标记接口采用固定约定：需要marks的作者模型收到零值，optional marks保持缺省，日历信息仅使用显式输入列。训练前缀列标准化在所有任务中一致，额外的实例归一化按各架构的作者选项保留。两者不能统称为“完全相同的归一化网络”。直接构造官方子集目标时必须显式指定 `target_indices`；统一入口会自动从数据列推导。

## 运行

在本目录执行。已验证环境为Python 3.11、PyTorch 2.7.1、NumPy 2.4.6、Pandas 2.3.3；版本清单见 `requirements-paper-implementation.txt`。代码默认float32，不启用半精度LRP。CUDA运行若启用严格确定性，应在启动Python前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。

```powershell
# 数学、梯度、状态、数据边界、模型保存加载测试
python -m pytest tests/paper_impl -q

# 完整短流程：FOIL双分支 -> 主预测器 -> 延迟回放generator -> 八种阶段组合
python -m paper_impl.run --config configs/paper_impl/panama_smoke.json --output results/paper_impl/my_panama_run

# 从完整checkpoint重评估，继承数据、模型和校准阈值
python -m paper_impl.run --checkpoint results/paper_impl/my_panama_run/model.pt --output results/paper_impl/my_panama_reevaluation

# 24个有界验证配置：12真实数据任务、其他native/style编码器与选择器对照
python tools/paper_impl_validate_suite.py --configs configs/paper_impl/validation_suite --output results/paper_impl/my_validation_suite
```

输出目录必须不存在。每个运行保存实际配置、数据/源码哈希、三阶段训练记录、checkpoint、原发行预测、逐次更新记录和汇总指标。`config.json`是请求配置，`effective_config.json`是实际完整配置，`online_config.json`含验证集校准后的部署阈值。`replay_history.json`另存生成器训练前的回放阈值和训练后的部署阈值；二者均由验证集确定。`resource_counts.json`记录去重参数量，`backbone_input_protocol.json`记录时间标记/归一化约定。不能只复制一个权重文件而丢失其数据协议。

重新评估可以改变设备、线程、评估变体和窗口上限，也可以迁移内容哈希相同的数据文件。模型、数据列/目标顺序、划分、归一化、生成器强度、训练种子不允许静默改变。旧版本缺少完整统计契约的checkpoint明确拒绝，需新跑完整流程。

## 已完整实现的方案

| 配置 | 含义 |
| --- | --- |
| `variants: ["stages_000", ..., "stages_111"]` | 三阶段全部8种开关组合；每种从相同预训练状态独立启动 |
| `online.selector: "relevance_error"` | 固定参考归因定位，已释放误差触发同预算梯度定位 |
| `online.selector: "gradient_only"` | 原梯度排序变体；严格同预算对照使用selection_comparisons |
| `generator.share_matching_shapes: true` | 相同矩阵形状共用生成器，对照其容量与成本 |
| `generator.mode: "identity"` | 恒等调制消融；无需训练generator，也支持已训练checkpoint的显式重新评估 |
| `model.encoder` | 三种native encoder与八种保留原实现的style encoder |
| `model.n_experts: 1` | 只构造一个专家，专家间分工损失为零；保留适用的在线参数更新 |
| `model.use_vae: false` | 确定性MLP路由特征；无VAE编码、采样、重构或KL项 |
| `foil.mode: "disabled"` | 移除两分支，搭配identity generator及不含Stage III的阶段组合 |
| `foil.mode: "erm_pair"` / `"variant_only"` | 同容量双分支ERM对照 / 删除invariant分支的单分支ERM对照 |
| `foil.contrast: "variant"` | 保留两分支训练，用variant预测替代分支差；与删除模块分开报告 |
| `model.use_revin` | 原生路径的可逆实例归一化；官方骨干采用各自原实现的归一化开关 |
| `online.stage_ii_scope` | sparse_experts、dense_experts、all_forecaster、experts_router_head，分别控制真实参数集合 |
| `online.reference_mode` | ema、current、frozen；同一比较中的输入均由同一个参考快照解释 |
| `online.rollback_on_worse: false` | 关闭下降接受条件；非有限值和异常仍恢复参数 |

在既有checkpoint上做算法对照，使用以下配置加 `--config`；其余设置从checkpoint继承，变更会单独留痕：

```json
{
  "variants": ["stages_111"],
  "evaluation_overrides": {
    "online": {"selector": "gradient_only"},
    "generator": {"mode": "identity"}
  }
}
```

通常分别改变其中一项以形成清楚的消融。测试期固定的是超参数、shared modules、双分支和generator；Stage I/II仍按协议利用已释放标签更新，不能将此称作“测试时完全不训练”。

Native编码器名称：`temporal_transformer`、`inverted_transformer`、`patch_transformer`。

复用旧工程的编码器名称：`transformer_style`、`itransformer_style`、`patchtst_style`、`informer_style`、`autoformer_style`、`fedformer_style`、`pyraformer_style`、`crossformer_style`。它们替换summary encoder，公共专家、LRP与在线控制器不变；它们不是八篇论文官方模型的复现。其源码依赖也纳入每次运行清单。

## 数据与协议

输入为 `[batch, lookback, variables]`，预测为 `[batch, horizon, targets]`。目标列必须明确列出，并通过 `target_input_columns` 指定对应输入列（默认同名）。标准化只使用训练前缀，输入缺失仅前向填充后使用训练中位数补初值。缺失真值需要明确的 `target_mask_columns`；真值不插补，测试发行不依未来mask筛选，逐点评分仅纳入实际观测值，完整目标不足的窗口不做监督更新。训练及验证窗口按下一阶段启动时钟剔除尚未释放的标签。检查严格时间顺序和等间隔；近常量训练列的单位尺度处理明确记录。

丽水主数据为 `data/lishui_online/v2/registry.json` 中十个独立任务。九县源自2020—2023年的15分钟原始负荷；全市采用用户确认的2024—2026年5分钟文件。以完整小时结束时刻聚合，记录原始频率和缺失覆盖。没有强行对齐两段时间，也没有为新数据补造气象变量。负荷单位暂为null，禁止把跨任务原始单位误差直接平均。另一份15分钟市级文件仅是候选来源。详见 [数据与旧shifted协议说明](docs/audit/full_experiment_20261007/LISHUI_AND_SHIFT_PROTOCOL.md)。

`data.protocol="chronological"` 使用完整后缀作为主在线流。`legacy_shifted` 保留旧窗口均值/方差偏移选段规则，明确记录其使用候选测试区间的未来输入；该压力测试不能称为实时检测到的分布变化，也不能用来替代完整后缀结果。新入口同时保留标签可用性约束，不能假称复现旧插值和跨边界处理得到的数值。

Panama、Spain、丽水数据为每小时观测；Tetouan为每10分钟观测。同样的H表示相同观测步数，并不表示相同物理时长。配置中的400行、L=12、H=4、短训练与12/16个在线窗口仅是实现验证预算。正式实验应独立冻结数据范围、特征集合、预测时长、超参数和多个种子，不依据这组短流程测试结果调参。

Spain已下载清洗CSV的部分ISO日期有日/月错置。审计副本为 `data/paper_impl_prepared/spain_energy_weather_clock_corrected.csv`，旁边的 `.provenance.json` 记录来源与修复。修复仅改变时间字段，全部35,064行恢复连续小时顺序，数值和行顺序不变。该副本未恢复上游清洗可能丢失的原始缺失掩码。原始文件保持不动。

异机部署先复制或挂载授权数据，再改配置路径。重新生成配置时显式给出校正副本：

```powershell
python tools/paper_impl_configs.py --dataset-root <负荷数据目录> --spain-source <已审计的Spain副本> --output <新的配置目录>
```

发行于t的完整预测覆盖t+1到t+H，最早在t+H+extra_delay释放。先处理可用标签再发出当前预测；指标始终使用原发行值。生成器离线回放也遵循该顺序，episode重置持久模型与队列，临时增益不写回。

## 代码与证据导航

- `paper_impl/model.py`、`lrp.py`、`discrepancy.py`：网络、专家归因和互补距离。
- `paper_impl/foil.py`、`training.py`、`replay.py`：三个训练环节。
- `paper_impl/adaptation.py`、`stream.py`：在线控制和延迟反馈。
- `paper_impl/data.py`、`evaluation.py`、`run.py`：数据、固定发行预测评分、完整入口。
- `docs/audit/paper_implementation_20261007/REQUIREMENT_MATRIX.md`：论文—实现—测试对应表。
- 同目录 `METHOD_REVIEW.md`：最新文献定位、可证明性质与经验假设。
- 同目录 `PROJECT_INVENTORY.md`：原本机工程、历史版本与nanyang已搜索范围。
- 同目录 `ZCODE_*REVIEW.md`、`ZCODE_DECISIONS.md`：GLM-5.3最高推理强度复核原意见与Codex裁决。

六个方法由 `paper_impl/online_baselines.py` 与 `baseline_run.py` 执行：GD、Head-GD、OneNet、SOLID、SOLID++和PROCEED。OneNet含显式官方FSNet分支及其梯度记忆状态；PROCEED执行所需离线概念编码器/生成器学习。SOLID每个上下文恢复预训练预测头，SOLID++持续更新全模型。完整定义及来源见 [基线审计](docs/audit/full_experiment_20261007/ONLINE_BASELINES.md)。

新增 `Frozen` 也使用这一入口：单个完整官方骨干离线训练并按验证集选模，测试期参数、归一化缓冲与优化器状态不更新。它与 `stages_000` 冻结整个MoE框架的含义不同。本轮没有加入Naive或其他简单模型。

全数据、多随机种子实验尚未由新版入口重新完成；旧稿表图属于旧实现结果。函数齐备、算子性质或短流程通过均不能代替这些性能证据。
