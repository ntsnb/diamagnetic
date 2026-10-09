# 归因分块的目标等价与有界成本检查

2026-10-07；run_id：`20261007_attribution_batching_v1`。本次只新增专项测试与本记录，没有修改核心训练实现，没有进行真实数据训练或产生预测性能结论。

## 问题与结论

父侧在 `TrainingConfig.attribution_batch_size` 中加入归因分块大小，默认 1。`batch_size` 仍控制监督训练批次与优化器更新频率；归因块大小只控制分工损失对应的高阶图每次驻留多少个样本。每块分工梯度累积完，再合并整批预测与 VAE 梯度，随后统一裁剪并执行一次 Adam 更新。

验证结果为 **11 passed, 2 skipped in 46.61s**。默认跳过的两个检查是需要显式启用的成本简测；独立启用 CUDA 后，结果为 **2 passed, 11 deselected in 9.04s**。未发现本次分块实现的目标、梯度或更新次数偏差。

同初始化、固定样本下，原生模型与官方完整 Transformer 均通过 CPU float64 对照：归因块大小 1、2、整批 5 的分工损失与逐参数梯度相符；单 epoch 的每步合并梯度、最终参数和训练日志也与旧整批目标参考相符。这里“相符”指下述高精度容差，允许浮点求和次序造成的舍入差异，不主张逐比特相同。

## 数学依据及适用边界

`paper_impl/discrepancy.py:148` 的 `specialization_loss` 对全部样本、目标及固定上三角专家对取均值。令批大小为 B、目标数为 T、专家数为 K、P=K(K−1)/2，单项罚值为 ℓ：

\[
L_{\mathrm{spec}} = \frac{1}{BTP}\sum_{b=1}^{B}\sum_{t=1}^{T}\sum_{p=1}^{P}\ell_{btp}.
\]

有效地图对使用 `max(margin−distance,0)²`；任一地图总质量为零的对固定罚 `margin²`，**仍保留在分母中**。没有按各块有效对数量重新归一化。对于互不重叠的块 Cⱼ，Bⱼ=|Cⱼ|，有：

\[
L_{\mathrm{spec}} = \sum_j\frac{B_j}{B}L_{\mathrm{spec}}(C_j),\qquad
\nabla_\theta L_{\mathrm{spec}} = \sum_j\frac{B_j}{B}\nabla_\theta L_{\mathrm{spec}}(C_j).
\]

因此 B=5、块大小=2 时，权重必须是 2/5、2/5、1/5，不能对三个块简单平均。全部块共用同一组更新前参数，不能在块间执行 optimizer step，不能逐块裁剪后再相加。最终合并的梯度对应原始预测、重构、KL 与分工项的同一个加权目标。

这一可加性也要求每个样本的归因不随同批其他样本变化。当前原生路径的 LayerNorm/RevIN 按样本计算；官方归因按单样本执行，使用 eval 模式及受保护的固定采样流。专项模型选择官方 Transformer、dropout=0，以排除随机性。若未来引入依赖当前批统计的归因规则或跨样本损失，需要重新验证，不能直接沿用本结论。

实现位置为 `paper_impl/training.py:91–103`：块内分工损失按 Bⱼ/B 加权后立即反传，日志只累积 detach 后的标量，释放该块归因引用；整批监督损失最后反传；每批一次 clip 和 step。该改动降低同时保留的分工高阶图数量，整批监督预测图仍然存在。

## 专项测试覆盖

测试文件：`tests/paper_impl/test_attribution_batching.py`。

| 检查 | 固定条件 | 验证内容 |
| --- | --- | --- |
| 固定专家对的数学可加性 | B=5、T=2、K=3、不同样本有不同零质量专家对 | 块大小 1/2/5 的标量值与地图梯度；标量 rtol/atol=1e−14，梯度 rtol=1e−13、atol=1e−14 |
| 真实分工参数梯度 | native + 官方 Transformer；float64；L=4、H=2、T=2、K=3、hidden=4、depth=1 | 块大小 1/2/5；分工损失为正，专家参数存在非零梯度；逐参数梯度 rtol=1e−9、atol=1e−10 |
| 单 epoch 对照 | 7 个固定样本，batch_size=5，批次为 5+2；相同 VAE 随机种子；归因块 1/2/5 | 独立旧目标参考将全部损失相加后一次 backward；对比每步 clip 前合并梯度、最终全部参数、五项损失日志；rtol=1e−9、atol=1e−10 |
| 更新与裁剪次数 | grad_clip=.02，并确认参考梯度范数实际超过阈值 | 每个批次只有一次 clip 和一次 Adam step，共两次；不能靠裁剪未触发而掩盖顺序问题 |
| 输入契约 | 归因块大小 0、−1、1.5、True | 均被 `TrainingConfig` 拒绝 |

真实参数检查使用 `freeze_head=True`，同时比较参数梯度是否为 None；没有将缺失梯度一律改成零来隐藏错误。单 epoch 基准保留 VAE 重构/KL 正权重和随机潜变量采样，因而覆盖分工梯度与主目标合并的路径。

普通检查命令：

```powershell
& 'C:/Users/admin/anaconda3/python.exe' -m pytest tests/paper_impl/test_attribution_batching.py -q
```

## 默认矩阵尺寸的单样本成本简测

设备为 NVIDIA GeForce RTX 5060，物理显存 8151 MiB；Python 3.11.14、PyTorch 2.7.1+cu128、CUDA 12.8。CPU 线程数设为 1。启动前确认 `results/full_experiment/20261007_full_framework_cuda_v2/progress.json` 已包含八个 passed 项，父侧同时确认 CPU/CUDA v2 任务及其 shell 均已结束。测试仍在调用前检查该进度文件。

简测前 `nvidia-smi` 记录其他背景占用 5864 MiB、GPU 利用率 7%；没有停止或清理其他进程。以下显存是**本测试进程的 PyTorch 分配器统计**，不是整张卡的总占用。统计在模型和输入已分配后重置，报告包含初始分配的峰值以及相对该初始值的增量；reserved 是缓存分配器保留量，不能当作存活张量内存。

两项均只做一次归因及分工反传，不做 optimizer step。配置为 B=1、L=24、H=12、N=8、T=1、K=3、hidden=64、heads=8、depth=2、latent_block=8、seed=20261007、margin=.3。模型/输入为 float32；差异函数按已有实现使用 float64。native 启用 RevIN，官方 Transformer dropout=0。数据为合成随机样本。

| 模型 | 归因时间（秒） | 分工反传（秒） | 合计（秒） | peak allocated（MiB） | 增量 peak allocated（MiB） | peak reserved（MiB） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native | 0.637951 | 0.248522 | 0.886473 | 305.795 | 304.128 | 314 |
| 官方 Transformer | 5.814098 | 0.532022 | 6.346120 | 160.132 | 130.677 | 334 |

native 分工损失为 0.08084526387856811，107 个参数张量有非零梯度；Transformer 为 0.0011617248797589868，192 个参数张量有非零梯度。地图和梯度均有限；最大带符号守恒误差分别为 1.2665987e−7、1.0741254e−7。

原始显存字节记录：native initial=1747968、peak_allocated=320649216、incremental_peak=318901248、peak_reserved=329252864；Transformer initial=30885888、peak_allocated=167910912、incremental_peak=137025024、peak_reserved=350224384。

命令：

```powershell
$env:ATTRIBUTION_PROFILE='1'
$env:ATTRIBUTION_PROFILE_DEVICE='cuda'
& 'C:/Users/admin/anaconda3/python.exe' -m pytest tests/paper_impl/test_attribution_batching.py -q -s -k default_matrix_single_sample_profile
```

时间包含首次归因调用及官方路径的 FX tracing，没有预热或重复测量；两个模型顺序执行，缓存和设备背景状态也不同。**不能用此表比较模型优劣、声明稳定延迟、推算全数据训练时间，或声称实测了某个分块显存降低比例。** 本次只验证默认尺寸单样本归因可反传并记录一次运行成本；未运行整批 32 的显存对照，也未为八种官方骨干分别建立成本曲线。

## 验证时的代码哈希

| 文件 | SHA-256 |
| --- | --- |
| `paper_impl/training.py` | `e294c8ff8208e11ada6746690524e1601519526e42c490857efcba5f1b7f32bf` |
| `paper_impl/discrepancy.py` | `14b2191eb01270325a3c4f9818c2456f2f6c69952857005a6a5144e36d5cba51` |
| `paper_impl/model.py` | `9fd19bf0b9da7142ab1f863a4df14c9f36bd6079965d32e4fb6f3db714761a79` |
| `paper_impl/official_lrp.py` | `208afab1025fabf7826b81df0a6c12b8bc0bb071bfb6192e0c78ded94111247d` |
| `tests/paper_impl/test_attribution_batching.py` | `747e76038704ad77ba2a7503d77e66d123ff3f65d8c8b0b0c812b06a03ffbb4a` |

本证据用于支持归因分块的数学和实现等价性及单样本可运行性，不替代正式实验的训练预算、误差指标或重复测量。

## 后续补充：八个官方骨干默认尺寸预算检查 v1

run_id：`20261007_attribution_budget_v1`，执行时间为 2026-10-07 15:13:15–15:14:52 UTC（北京时间 23:13:15–23:14:52）。本节是独立的新运行，**不替换或修改前面 native/Transformer 两项记录**。脚本为 `tools/profile_full_attribution.py`，证据目录为 `results/full_experiment/20261007_attribution_budget_v1/`。

脚本直接读取本轮丽水市级 Full 配置中的模型结构，不读取其真实数据：B=1、L=24、H=12、N=8、T=1、K=3、hidden=64、heads=8、depth=2、latent_block=8。保留矩阵原有 `official_options`：dropout=.1、moving_avg=25、patch_len=16、patch_stride=8、crossformer_seg_len=6、fedformer_modes=12，以及各完整作者模型的其他默认配置。归因仍遵守现有官方 eval 模式规则，不因训练配置中的 dropout 改成随机归因。模型/输入为 float32，分工差异计算为 float64，margin=.3，freeze_head=True。

八个模型均接收同一个独立 CPU RNG 生成的合成输入，seed=20261007，输入 SHA-256 为 `11942d56ae2e4eeed316defe5ff3784ffd58c254989755a7711e1447078bcbda`。每模型只做一次归因、一次分工损失及反传，没有优化器更新、预热、重试、减小结构或参数调优。

每个模型由脚本创建的独立子进程顺序执行；结束前 gc/empty_cache，只控制本任务创建的进程。单模型上限 90 秒、脚本总预算 570 秒，超时只结束对应自有子进程。部分框架缓存对象在 gc 后仍保持少量分配，`allocated_after_cleanup_bytes` 如实记录；子进程退出提供最终资源隔离，未宣称 empty_cache 会释放所有存活张量。背景进程没有被结束。

结果为 **7 passed、1 failed**，总墙钟 **96.759588 秒**。脚本按存在失败返回退出码 1，并完整生成 summary/progress；这不是脚本中止或漏跑。未出现 OOM、单模型超时或总预算耗尽。七个通过项均得到有限地图、有限分工损失、有限梯度及非零分工梯度。

| 模型 | 状态 | 全模型参数数 | 归因（秒） | 分工反传（秒） | 归因＋分工前向/反传（秒） | peak allocated（MiB） | 非零分工梯度参数张量数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| iTransformer | passed | 425247 | 4.122591 | 0.330739 | 4.475990 | 40.104 | 108 |
| Crossformer | **failed** | 2156913 | — | — | — | — | — |
| Autoformer | passed | 630099 | 9.933768 | 0.862575 | 10.818122 | 190.813 | 132 |
| Transformer | passed | 631059 | 5.932111 | 0.574163 | 6.527483 | 160.044 | 192 |
| PatchTST | passed | 428565 | 4.072954 | 0.276873 | 4.382656 | 87.157 | 111 |
| Informer | passed | 669267 | 7.357322 | 0.663748 | 8.041416 | 129.786 | 216 |
| FEDformer | passed | 685395 | 31.786049 | 2.545408 | 34.356458 | 153.988 | 99 |
| Pyraformer | passed | 507867 | 4.253351 | 0.366255 | 4.642853 | 182.016 | 141 |

参数数包含 K=3 的完整专家和共享门控/VAE；分工梯度只来自专家归因路径，所以不要求全部参数均有该项梯度。分工损失前向耗时已计入合计列，原始秒数和显存字节保存在每模型 JSON，避免把四舍五入表当作原始记录。

显存仍为本进程 PyTorch 分配器峰值，不是整卡占用。开始时背景查询为 5870 MiB 已用、2026 MiB 空闲；结束时为 6077 MiB 已用、1819 MiB 空闲。这个整卡变化包含未受本任务控制的背景状态，不能归因给本脚本。每个子进程均在模型/输入分配后重置峰值；完整 JSON 同时记录 initial、incremental peak、peak reserved 和清理后分配量。

### 唯一失败与证据冻结

Crossformer 在默认 L=24、segment_len=6、depth=2 的归因 tracing 阶段产生 `aten.add_.Tensor`，现有闭合算子集合拒绝了它：

```text
UnsupportedLRPOperator: no explicit LRP rule for aten.add_.Tensor
profile_full_attribution.py:132
  -> model.py:350
  -> official_lrp.py:425 / 373
  -> _trace:159
  -> _kind:65
```

完整配置、失败时显存和 traceback 保存在 `crossformer/result.json`，子进程墙钟 4.786 秒。失败发生在完整归因产生之前，所以不把其部分分配峰值或耗时填为成功成本，也不填 0 冒充有效测量。该失败证明较小结构的既有检查不足以保证默认结构可归因；已交父任务修复。本 v1 不重跑、不覆盖 Crossformer 失败证据。后续修复必须在新结果目录验证，不能将本轮汇总写成八个通过。

八个模型记录的 `official_lrp.py` 都是同一 SHA-256：`208afab1025fabf7826b81df0a6c12b8bc0bb071bfb6192e0c78ded94111247d`。脚本 SHA-256 为 `1a5596dd23743905d42f5f1597353ef7cb1d575d53c9f8e1bcb898dae0ce5400`。各模型 JSON 另存矩阵配置哈希、完整 ModelConfig/OfficialForecastConfig、核心代码哈希、执行的 runtime 文件哈希、源码提交与补丁哈希。

`audit.json` 核验了八个不重复身份、固定维度、同输入、单次运行、无真实数据/优化器、脚本和配置哈希、runtime 来源一致、总预算，以及七个成功项的有限性和非零梯度；还保存逐模型 `result.json` 哈希。`summary.json` 汇总全部原始记录，`progress.json` 提供精简状态，各模型目录保留 stdout/stderr。

复现命令（输出必须是新目录，脚本拒绝覆盖现有目录）：

```powershell
& 'C:/Users/admin/anaconda3/python.exe' tools/profile_full_attribution.py `
  --output results/full_experiment/20261007_attribution_budget_v1 `
  --device cuda --max-wall-seconds 570 --per-model-seconds 90
```

本补充为归因成本风险提供了实际单样本记录，并暴露了一个默认结构算子缺口。它仍只有一次冷调用，含 tracing、Python 调度和设备状态影响；没有批次、epoch、数据规模或多次重复的系统测量，**不能线性外推正式全数据训练时间，也不能据此声称不同模型的稳定成本排名或预测性能优劣**。

## Crossformer 修复后的单项验证 v2

父任务随后在 `paper_impl/official_lrp.py` 中加入受限的 `_functionalize_linear_bias_additions`：只处理独立新分配 `mm` 输出经单一消费者 view 链后的线性 bias `add_`，改为等值的非原地 add，并保留 Linear 注解。可观察的 alias、输入或 buffer 修改仍拒绝；没有把所有 in-place 操作加入通用允许集合。

只在新目录 `results/full_experiment/20261007_attribution_budget_v2/crossformer/` 对 Crossformer 重做一次默认 CUDA 检查，其余七个模型没有重跑。运行时间为 2026-10-07 15:17:54–15:18:14 UTC。与 v1 失败项相比，ModelConfig、OfficialForecastConfig、合成输入和 profile 脚本完全相同，来源记录中仅 `paper_impl/official_lrp.py` 哈希改变：

- 修复前：`208afab1025fabf7826b81df0a6c12b8bc0bb071bfb6192e0c78ded94111247d`。
- 修复后：`6b850d4b7c9653db7ff41dbc47cd73d7488755a47e466d138dbc2aed3ecd6e61`。

修复后 **passed**：归因 17.432597 秒、分工前向 0.023080 秒、分工反传 1.686476 秒，合计 **19.142153 秒**；包含进程启动与记录的外部墙钟为 22.212492 秒，低于 90 秒上限。peak allocated 为 **397128704 bytes（378.731 MiB）**，增量峰值 388371968 bytes（370.380 MiB），peak reserved 398458880 bytes（380 MiB）。

完整模型参数数仍为 2156913。分工损失为 0.0778162784448397；789 个参数张量、2035498 个参数元素有非零分工梯度；地图、损失和梯度均有限，最大带符号守恒误差 3.0174851e−7。没有优化器更新或真实数据评价。

`v2/audit.json` 验证了 v1 **全部八个结果文件哈希保持不变**、Crossformer 原失败仍存在、默认配置与输入一致、仅归因实现来源变化，以及 v2 的有限性与非零梯度。v2 结果文件 SHA-256 为 `c6e2ce9e8db44460a474bec40e479f9a72588641f22a8844d63abe97185c089b`。v1 七个旧来源通过项与 v2 修复后 Crossformer 共同构成逐模型可运行证据；不能将它们改写成同一来源版本、同一轮的“八项全部通过”。

本次通过父进程的 `subprocess.run(..., timeout=90)` 监督执行以下 worker 命令，输出目录提前新建且拒绝复用：

```powershell
& 'C:/Users/admin/anaconda3/python.exe' tools/profile_full_attribution.py `
  --worker crossformer `
  --output results/full_experiment/20261007_attribution_budget_v2/crossformer `
  --device cuda --seed 20261007
```

该单项复验解决本次发现的默认 Crossformer 算子覆盖缺口，不补造正式训练性能或全量预算结论。前述单次成本测量和不可线性外推的限制继续成立。
