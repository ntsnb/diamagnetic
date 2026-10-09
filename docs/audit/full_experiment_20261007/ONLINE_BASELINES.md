# 正式在线基线来源与实现审计

日期：2026-10-07。范围：`paper_impl/online_baselines.py`、`official_fsnet.py`、独立 `baseline_run.py`、对应小规模测试、`configs/full_experiment/baselines/` 和 `third_party/official_online/` 来源快照。已运行真实 Panama CSV 的有限闭环；这些结果用于验证实现，不构成数据集长训练或论文性能证据。

## 固定来源

| 方法 / 来源 | 原论文 | 作者仓库固定提交 | 许可证证据 |
|---|---|---|---|
| OneNet | [NeurIPS 2023, arXiv:2309.12659](https://arxiv.org/abs/2309.12659) | [yfzhang114/OneNet](https://github.com/yfzhang114/OneNet/tree/65eed9d6c878133a4d81d9c381c69e742ad47fd0), `65eed9d6c878133a4d81d9c381c69e742ad47fd0` | README 声明 MIT，但对应 LICENSE 文件在该提交缺失；不补造许可证 |
| SOLID | [KDD 2024, arXiv:2310.14838](https://arxiv.org/abs/2310.14838) | [HALF111/calibration_CDS](https://github.com/HALF111/calibration_CDS/tree/3c74e5f8b3c921a2e7cda30f36117c295bf8c112), `3c74e5f8b3c921a2e7cda30f36117c295bf8c112` | MIT，原 LICENSE 完整保留 |
| PROCEED / SOLID++ | [KDD 2025, arXiv:2412.08435](https://arxiv.org/abs/2412.08435) | [SJTU-DMTai/OnlineTSF](https://github.com/SJTU-DMTai/OnlineTSF/tree/0a8646199f6dcbf5346f933508b33c5e35b5708b), `0a8646199f6dcbf5346f933508b33c5e35b5708b` | 该提交无仓库级 LICENSE / COPYING，不推定授权 |

三个源码快照未修改；已去除嵌套 `.git` 元数据。`third_party/official_online/SOURCE_MANIFEST.json` 保存逐文件 SHA256、提交日期、来源地址及许可证状态（185 / 93 / 83 个源文件）；`audit_sources.py` 在已有清单下核验字节和提交，变化时拒绝覆盖证据。运行时基线模块不导入这些仓库的旧训练入口，也不依赖当前工程 `train/` 中的代理基线。FSNet 在独立私有导入命名空间加载原作者网络，兼容补丁保存在本地模块。正式发表或再分发前，仍须解决标明缺失的许可证材料。

## 算法与来源对应

| 本地类 | 实际行为 | 主要源码证据 |
|---|---|---|
| `GradientDescent` | 每条完整释放标签驱动持久全参数 MSE 更新；默认 Adam，可明确配置 SGD / AdamW | OnlineTSF `exp/exp_online.py::Exp_Online`、`exp/exp_basic.py::_select_optimizer` |
| `HeadGradientDescent` | 只更新真实 `head_parameters()`；骨干参数与运行统计冻结，参数持续保留 | 明确的 head-only GD 实验对照，不声称来自一个独立正式算法论文 |
| `OneNet` | 两预测分支以各自 MSE 训练；全局通道权重由独立 Adam 更新；三层 Tanh 决策网络根据已释放的两分支预测和真实目标产生下一步短期偏置 | OneNet `exp/exp_onenet_fsnet.py::_ol_one_batch` / `_process_one_batch`；OnlineTSF `models/OneNet.py`、`exp/exp_online.py::Exp_OneNet` |
| `SOLID` | 周期相位过滤 → 输入欧氏距离 Top-K → 每个测试输入从预训练状态开始、只 SGD 微调预测头 | SOLID `exp/exp_main.py::select_with_distance`，特别是逐样本梯度求和和一次头更新 |
| `SOLIDPlusPlus` | 相同选择逻辑，累积保留适配状态、更新全部模型参数 | PROCEED 论文 §5.1；OnlineTSF `exp/exp_solid.py` 的 `whole_model=True, continual=True` |
| `PROCEED` | 历史 `(X,Y)` 与当前 `X` 两个概念 MLP；概念差 → 共享瓶颈生成器 → 权重输入/输出方向乘性缩放、偏置加性调整；随机历史批次离线联合训练后，在线固定概念编码器和生成权重 | OnlineTSF `adapter/proceed.py`、`adapter/module/generator.py`、`adapter/module/down_up.py`、`adapter/module/up.py`、`exp/exp_proceed.py` |

`SOLID++` 不是另一篇 SOLID 原作：名称及持续全模型更新定义来自 PROCEED。OneNet 年份使用原论文 NeurIPS 2023；OnlineTSF README 中的 “NeurIPS 2024” 不照抄。

## 因果边界与统一接口

```python
config = OnlineBaselineConfig(lookback=L, horizon=H, target_indices=(0,),
                              extra_delay=delay, learning_rate=lr)
method = make_online_baseline("GD", pretrained_forecaster, config)
released = ReleasedBatch(x=historical_x, y=fully_observed_y,
                         issue_times=(s,), available_at=(s + H + delay,),
                         prediction=fixed_original_prediction)
method.update(released, time=t, current_context=current_x)
prediction = method.forecast(current_x, time=t)
```

`X=[B,L,N]`、`Y=[B,H,T]`，时间是外部统一协议的整数观测时钟。`forecast/predict` 不接受 Y，返回已分离并复制的预测。`update/observe/release` 首先验证整个批次，再开始任何状态变更，要求每个 `issue+H+extra_delay <= available_at <= time`。重复标签和倒退的时钟被拒绝。

预测时真正观测到的特征由父级调度器负责；算法不持有完整数据集、测试真值数组或未释放标签缓存。`prediction` 仅为原发行预测证据，梯度训练重新前向历史 X，不把重算结果覆盖原预测。

`seed_history` 只接纳已验证释放的训练/验证历史，不做更新。SOLID 将历史放入有界记忆；PROCEED 保存最近训练样本作为概念参照。SOLID 的 `prepare_context` / `forecast` 可用已经释放的历史进行测试样本级适配；同一输入同一时刻只适配一次，无新标签时也可重新选择历史。通过 `last_update` 获取真实步骤和选中样本。预测流仍必须逐时发行，不能把未来特征作为同批当前输入。

`state_dict/load_state_dict` 与 `save_checkpoint/load_checkpoint` 保存模型、所有优化器、已释放记忆、OneNet 权重梯度和短期偏置、PROCEED 概念参照、生成器、离线训练状态及随机计数。构造时复制模型，保护输入的预训练检查点。`set_reset_point/reset` 为每个阶段建立/恢复完整基准状态，文件检查点同时保存该基准。

## 必需离线阶段

OneNet 控制器构造时必须显式传入独立 `secondary_model`，不能默默复制同一预训练模型。运行器默认 `onenet_mode="onlinetsf_cross_backbone"`，按 OnlineTSF 正式跨骨干比较构造目标官方骨干 + 完整 FSNet。`author_architecture` 使用原 OneNet 时间/通道双 FSNet；`explicit_backbone_variant` 才允许用户明确选择另一个官方骨干作为组合变体。三种模式在 `protocol.json::source_variant`（同时保留 `onenet_variant`）和检查点中区分，不把两 Transformer 组合代称原双 FSNet 架构。

OneNet `fit_offline` 联合训练两分支以及组合权重、决策网络。PROCEED `fit_offline` 是实际执行的必要阶段：对训练历史随机洗牌，**上一完整随机 mini-batch 的聚合历史概念**供历史编码器、当前批各样本供当前编码器；每样本生成调制、计算 MSE 并共同更新骨干/编码器/生成器。这里历史参照按批次聚合，没有实现逐样本的一一随机配对；首次尚无历史参照时沿用作者零参照初始化。输出层从零初始化，因此训练后才出现学习到的调制。未训练 PROCEED 调用在线预测会报错，不能冒充可运行的正式基线。

两个离线接口均可传 `issue_times` 和 `cutoff`，拒绝监督区间超出训练边界。训练后先在验证集选择超参/检查点，再进入测试；本模块不根据测试结果选超参。PROCEED 在线优化骨干和作者代码保留的生成器最终输出偏置，其余适配器参数冻结。无需也不使用本文主方法的 Stage III。

PROCEED 入口参数组明确对应固定 `OnlineTSF/run.py`：第 60 行 `--freeze` 为 `store_true, default=False`，第 63 行 `--act` 默认 `sigmoid`，第 64 行 `--tune_mode` 默认 `down_up`，第 65 行 `--ema` 默认 0，第 68 行 `--individual_generator` 默认 False，第 73 行 `--wo_clip` 默认 False。因此本实现使用 `freeze=False / sigmoid / down_up / shared generators / drift clipping`。`adapter/proceed.py:28-37` 据此保持骨干可训练、令 `more_bias = not args.freeze` 并选择 Sigmoid；`exp/exp_proceed.py:26` 在线调用 `freeze_adapter(True)`，冻结两个概念 MLP 与生成权重，保留骨干和最后输出偏置的更新。第二轮评审声称该快照没有 `act/freeze` 的 argparse 定义，经上述原文件逐行核查不成立；这里记录实际默认值，没有补造来源参数。

## 完整 FSNet 与记忆持久化

`OfficialFSNet` 执行固定提交中的 `TSEncoder`、`DilatedConvEncoder` 和 `ConvBlock`，默认 hidden=64、output=320、depth=10，即 11 个完整残差块、22 个校准卷积；没有改成轻量代理网络。`OfficialFSNetConfig` 可显式缩小尺寸用于单元测试，真实 CSV FSNet 闭环使用完整默认架构。`author_onenet_branches` 提供原时间轴/通道轴两分支。

实际核验：上游 `grads`、`f_grads`、`q_ema`、`trigger`、`try_trigger` 都不是注册 buffer，原始 `state_dict` 会遗漏；`W` 虽保存为参数但 `requires_grad=False`。本地 `PersistentSamePadConv` 将上述状态及首次查询标志全部注册为 buffer，保留 W 的冻结属性，并在梯度更新后、清梯度前执行 `store_grad`。设备跟随模型 `.to()`，修复源代码硬编码 `.cuda()` 导致 CPU 不可用的问题。

两个来源的具体差异没有合并：OnlineTSF 在快梯度更新后、慢梯度更新前检测负相关，首次查询保持零 EMA，触发读取实际采用 top-k **索引**乘记忆；原 OneNet 在慢梯度更新后检测、首次就更新查询 EMA，读取采用 top-k **attention 值**。本实现分别保留这些源码语义，包括前者可疑但实际存在的索引加权行为；未静默“修正”成另一公式。源代码数值 oracle 覆盖强制触发、读取、写回和梯度均值。

原 OneNet 在线编码器在 eval 模式更新，保留其触发条件；OnlineTSF 在 train 模式更新并开启显式 trigger。原时间分支离线保留源码 binomial 掩码，PyTorch 和 NumPy 随机流均以可恢复事件计数隔离。跨骨干版 FSNet 保留作者 RevIN，最多使用最近 60 点；无日历特征接口时追加 7 个零通道，沿用 OnlineTSF fallback。目标子集在完整多变量输出及 RevIN 逆变换后选择。原双 FSNet 版保留全部 lookback 且不添加 RevIN。这些输入兼容约定写入配置和状态元数据。

## 明确兼容差异

1. 统一外部延迟协议替代原 OneNet 即时全标签在线入口。算法只接纳完整释放的 H 步目标；额外延迟由同一配置强制检查。
2. 原 SOLID 在 eval 模式对所选样本的梯度求和，OnlineTSF 在 train 模式使用平均 MSE。本实现默认的完整组合为 `solid_adaptation_mode="eval"`、`solid_gradient_reduction="sum"`，对应 `source_variant="solid_original_eval_sum"`。显式 `train + mean` 记录为 `onlinetsf_train_mean_full_state_reset`；混搭 `train + sum` 或 `eval + mean` 记录为 `explicit_solid_mode_reduction_variant`，不混称原入口。SOLID++ 无论共享配置如何都固定 `train + mean`、全模型持续更新，记录实际生效模式。
3. 设当前发行时刻为 t，历史发行时刻为 s，实际释放时刻为 a_s，H 为 horizon，δ 为额外反馈延迟，M 为 memory_size。记忆先仅保留最近 M 条已释放样本；候选还须同时满足 **a_s ≤ t、t − s ≤ H + δ + M − 1、(t − s) % period ≤ period × lambda_period**。M 同时限制条数和时效窗口，存在缺失反馈或不规则发行时不会以任意遥远样本补足候选。一侧取模条件沿用两作者代码；Top-K 不足时用全部可用候选；无候选时保持相应当前/预训练状态并记录 0 步。
4. Head-GD 和默认原 SOLID 使用 eval 前向，dropout 关闭、BN 运行统计冻结；显式 SOLID train 模式开启 dropout 和 BN 统计更新，但梯度仍仅允许预测头参数。所有非持续 SOLID 模式在每个新输入前恢复完整预训练 `state_dict`（包括全部 BN buffers），预测时切回 eval。同一输入/时刻不重复适配。OnlineTSF 原头适配入口仅恢复预测头，因此这里的 train 选项在 BN 状态恢复范围上有明确扩展，并以 `full_state_reset` 标签披露。SOLID++ / GD 按 train 模式持续更新。默认 eval + sum 在线性头与固定表示上等价于原作逐样本梯度求和。
5. PROCEED 采用作者默认 `down_up`：`nn.Linear` 和 `nn.Conv1d`，保留按层末级名称及输出维度共享的瓶颈权重、每层偏置、Sigmoid、概念差范数裁剪。PyTorch 新式逐样本激活缩放避免永久覆盖权重；已验证与显式参数映射一致。
6. PROCEED 论文对 bias 描述为缩放，而作者代码实际为加性 shift；本实现依照代码，并把此差异明示。作者默认不对 LayerNorm/BatchNorm 做 `all_down_up` 扩展，此实现也不扩展。
7. 当任务含外生变量且只预测部分通道时，两概念 MLP 均使用可观测目标通道（历史编码器使用其 X+已释放 Y）；预测骨干仍接收全部 N 个输入。不会为概念编码捏造未来天气等外生标签。这是多输入、目标子集任务的明确兼容选择。
8. OneNet 原代码决策网络反向同时在全局组合权重上积累梯度，再与后续组合误差梯度一起更新。这里保留该行为，包括离线批次间残留全局权重梯度，并在状态中保存，便于精确恢复。
9. 调度器可一次释放多条完整样本；GD/Head-GD/OneNet/PROCEED 在模块内按发行顺序逐条更新。SOLID 先将整批加入历史，然后针对当前输入选择。
10. `fit_offline` 的 CPU shuffle generator、PyTorch/NumPy 训练与预测随机上下文被隔离，重载后续更新精确恢复且不扰动其它对照方法随机流。Informer 预测采用明确记录的每事件 seed+counter，而主框架可选择固定评估采样 seed；这不是共享隐式全局 RNG。保留明确配置，不声称本地数据超参等于作者五个原始数据集最优超参。

SOLID 模式的最小源证据（均为上表固定提交、未修改的本地文件）：

| 来源 / 行号 | 实际代码与语义 |
|---|---|
| `third_party/official_online/SOLID/exp/exp_main.py:600-602` | `cur_model = copy.deepcopy(self.model)`；`# cur_model.train()`；`cur_model.eval()`。`select_with_distance` 每输入复制模型，原头适配在 eval 模式执行。 |
| 同文件 `:817-818`、`:855-856`、`:881` | all-ones 权重汇总所选样本梯度、手动 GD 更新预测头；发行前再次 `cur_model.eval()`。 |
| `third_party/official_online/OnlineTSF/exp/exp_solid.py:174`、`:182-185`、`:191` | 适配前 `self.model.train()`；MSELoss 默认 mean 后 SGD；预测前 `self.model.eval()`。 |
| 同文件 `:201-205` | 非持续且非 whole_model 时仅恢复 `final_head`；whole_model 路径恢复完整状态。 |

因此第二轮评审观察到相对 OnlineTSF 的模式差异是事实，但原 SOLID 使用 eval 并非实现缺陷。新增明确配置与来源标签，默认维持原 SOLID 的一致组合；没有把两个来源的模式/梯度口径混合作为默认。

## 独立训练与延迟流运行器

入口 `python -m paper_impl.baseline_run`；不修改也不依赖旧基线训练入口。`--config FILE --out NEW_DIRECTORY` 跑一臂；`--matrix --models ... --methods ... --seeds ...` 跑显式有限矩阵，逐臂独立输出。`--stop-after-observations N` 在观测边界保存；`--resume RUN/stream_checkpoint.pt --out NEW_DIRECTORY` 从同一流继续，不重复离线训练或已发行预测。输出目录已存在时拒绝覆盖。

配置中的 `data` 直接使用公共 `DataConfig/prepare_data`。`data.extra_delay` 未指定时继承 `online.extra_delay`，显式不一致报错。官方骨干的输入/目标维度来自 `PreparedData.metadata`，而不是独立猜测。离线训练窗口先经过公共数据层的标签释放 embargo，再只取完整可监督目标；`max_train_windows` 如设置则取最早的有限前缀，`validation_windows` 限制独立验证前缀。以验证 MSE 选择骨干和 OneNet/PROCEED 方法检查点，记录各 epoch；测试数据不参与选模。主骨干训练 seed 与第二分支分离，第二分支的创建不改变主骨干 dropout / Informer 训练采样。

`DelayedBaselineStream` 仅逐时读取 `data.targets[t]`。固定 `split.issue_times` 决定发行，不参考未来 mask；保存原发行预测，在后续观测到达时逐点计分。到 `issue+H+extra_delay` 才检查完整目标窗：含缺失点则整窗不用于监督，已经可见的有效点仍参与评分。每时刻先释放成熟标签、更新，再发行当前预测。`target_input_columns`、`target_mask_columns` 以及独立目标标准化均通过公共 PreparedData 接入，逆尺度误差使用 `target_std`。

测试前已观测历史只作为 SOLID 记忆/PROCEED 概念参照，无梯度更新；不在验证流额外预热。`legacy_shifted` 仍使用原行号时钟，协议清单明确 `diagnostic_only=true`；未来输入参与选择测试区段的旧协议不能作为主因果比较。它沿用公共数据层修复后的训练前缀标准化和监督 embargo，不声称重现旧污染处理数值。

每臂输出 `configuration.json`、`data_metadata.json`、`source_manifest.json`（包括实际运行代码、全部完整官方骨干源码、FSNet 原文件哈希及软硬件版本）、`offline_checkpoint.pt`、`training_log.json`、`protocol.json`、`metrics.json`、`updates.json`、`forecasts.npz`、`stream_checkpoint.pt` 和 `run_record.json`。后者保存精确命令、UTC 时间、seed、设备和恢复路径。流检查点包含所有方法状态/优化器/FSNet 记忆/随机计数、未成熟输入窗、仅已观测的目标、固定预测和累计统计；恢复时严格验证配置、源数据 SHA256、时钟计划、尺度和实现源码哈希。

独立基线运行器的 CUDA 字段现名为 **`last_segment_peak_cuda_bytes`**：每次 `evaluate_baseline_stream` 进入在线片段前重置 CUDA allocator 峰值，记录本次片段的最大已分配量，包含当时驻留模型/优化器张量与当前片段临时分配；它不统计早先离线阶段或先前暂停片段的峰值。恢复后的输出只代表最后执行片段，不对多个片段取全程最大值；CPU 值为 0。`protocol.json::cuda_memory_metric` 明示 `aggregated_across_resumes=false`。旧结果中的 `peak_cuda_bytes` 保留原字段和原始数值，不能解读成跨恢复的全程峰值。

运行器默认 `deterministic=true`，设置 PyTorch 确定性算法、cuDNN deterministic、关闭 benchmark；`paper_impl` 包导入时提供 `CUBLAS_WORKSPACE_CONFIG=:4096:8`（既有环境值不静默覆盖）。包初始化本身不导入 PyTorch、不初始化 CUDA。manifest 保存这些设置；选择确定性内核可能改变速度和显存开销。这样处理是因为仅固定随机种子或关闭 benchmark 并不能保证 CUDA 卷积内核本身确定，参见 [PyTorch 2.7 reproducibility 官方说明](https://docs.pytorch.org/docs/2.7/notes/randomness.html)。仍不承诺跨设备、跨 PyTorch 版本逐位相同。

同进程集成曾暴露一个初始化缺陷：在基线入口才设置 cuBLAS 环境值，会晚于其他模型首次创建 cuBLAS handle；同时确定性标志会留给后续调用者。修复将环境设置前移到包导入，并在 `run_baseline` 的成功和异常出口恢复调用者的 deterministic / warn-only、cuDNN deterministic / benchmark、CPU 线程数、Python / NumPy / CPU Torch 随机状态及进入时已经初始化的 CUDA 随机状态。明确保留的环境默认值必须早于本包首次 CUDA 工作；外部应用若在导入本包之前已创建 cuBLAS handle，仍须在进程启动时自行设置兼容值，不能在 handle 建立后追溯修复。

同进程回归不依赖 pytest 或 shell 预设环境：子进程先删除继承的 workspace 环境值，导入本包并验证未初始化 CUDA，再进行无关 GEMM、导入运行器、开启确定性并再次 GEMM。CUDA PROCEED / OneNet 测试也先执行无关 GEMM 后完成训练和精确断点恢复；成功/失败 API 测试验证全局状态恢复。

## 验证记录

运行标识：`20261007_official_online_baselines_unit_v2`。

命令：`python -m pytest -q -p no:cacheprovider tests/paper_impl/test_online_baselines.py tests/paper_impl/test_baseline_run.py tests/paper_impl/test_official_fsnet.py`。

最终结果：**115 passed, 2 warnings in 15.09s**，CPU/CUDA、确定性合成序列和小训练闭环。两条 warning 来自未修改作者 oracle 中隐式 softmax dim，不是本地实现失败。覆盖：

- GD/Head-GD 实际参数更新范围；原模型不被改动；原发行预测保持固定。
- 六方法未释放标签批次被原子拒绝，包含伪造 release time 和额外反馈延迟。
- 六方法检查点恢复后的下一次更新、预测及内部状态逐项完全一致；重置恢复完整算法状态。
- SOLID 每次回到预训练起点而 SOLID++ 持续累积；周期选择及同一时刻不重复适配；SOLID 更新与独立逐样本求和 SGD 数值一致。
- OneNet 两分支、全局权重、决策网络都发生真实学习；离线阶段运行。
- PROCEED 离线训练实际改变两个概念 MLP、生成器和骨干；在线继续优化骨干，适配器内部仅最终输出偏置可更新，临时调制不污染持久参数。
- PROCEED 生成器与固定源码中未修改的作者 `Bottleneck` 类数值一致；Linear/Conv1d 激活缩放与显式逐样本参数映射一致。
- 离线训练目标或其额外反馈延迟穿过训练 cutoff 被拒绝。
- 八个完整官方骨干 × 六方法的真实前向/反向更新；另加八个骨干 + 官方 FSNet 的 OneNet 更新和强制记忆触发后的精确恢复。
- FSNet 原始源码与兼容实现数值 parity、完整默认 11 残差块、所有梯度记忆进入 state_dict、冻结 W 不被优化器解冻。
- 六方法真实离线训练 → 验证选模 → 流预测/更新 → 断点继续，与不中断运行的预测/目标/掩码逐元素一致；原双 FSNet 架构也通过。
- 活流禁止目标切片和未来索引访问；修改未观测未来目标不改变已发行预测；缺失真值不压缩发行时刻，逐点 mask MSE 正确。
- Informer 采样及原 FSNet NumPy 掩码随机流精确恢复；改变源 CSV、延迟不一致或重复输出目录被拒绝；旧协议标明诊断。
- 完整 CUDA PROCEED / OneNet+FSNet 训练、适配、流更新与显存统计；两方法 CUDA 中断/恢复的预测逐元素一致。

资源/输入协议日志及同进程初始化修复后的新增回归（2026-10-07）：上述三文件加 `tests/paper_impl/test_resource_report.py`，**126 passed, 2 warnings in 17.93s**；全项目 CUDA 命名测试 `python -m pytest -q -p no:cacheprovider tests/paper_impl -k cuda`，**36 passed, 545 deselected in 14.09s**。FSNet 两个原源码 parity 也在该 126 项中通过；全套 CUDA 测试未配置外部 workspace 环境 workaround。

真实 CSV 有限闭环使用 Panama `nat_demand` + 4 个天气输入，前 320 行、L=24、H=8、extra_delay=2、16 个离线训练窗、8 个验证窗、12 个在线发行窗。所有行均完成 12 次发行 / 12 次完整释放 / 96 个有效目标点，未修改源 CSV。输出如下：

| 运行目录（相对项目根） | 已执行范围 | 检查结果 |
|---|---|---|
| `results/full_experiment/20261007_online_baseline_cpu_v4/` | CPU iTransformer × GD / Head-GD / OneNet(完整官方 FSNet) / SOLID / SOLID++ / PROCEED，单 seed=2026 | 六臂全部完成，PROCEED 必需离线适配阶段实际执行 |
| `results/full_experiment/20261007_onenet_author_cpu_v3/` | CPU 原时间/通道双 FSNet，64/320/10 完整架构 | 完成；两分支、权重、决策网络和梯度记忆保存 |
| `results/full_experiment/20261007_onenet_fsnet_cuda_v3/` | RTX 5060 上 iTransformer + 完整 FSNet | 完成，确定性在线阶段进程峰值显存 109,493,760 bytes |
| `results/full_experiment/20261007_online_baseline_cuda_proceed_v3/` | RTX 5060 上 PROCEED，2 轮离线适配 | 完成，在线阶段进程峰值显存 68,404,736 bytes |

真实 CSV CUDA FSNet 还在第 14 个观测后暂停，当时 12 个预测已经发行，4 个窗口已更新，8 个仍等待释放；从检查点继续后，发行时刻、96 个预测、真值和 mask 与不中断运行逐元素一致，最大预测差为 0。对应 `20261007_onenet_fsnet_cuda_partial_v3/`、`..._resumed_v3/`，后者保存 `resume_verification.json`，并验证三路实际源码哈希等于该次运行的实现。早先未开启确定性内核的 v2 检查曾在断点之前发现 1/96 个预测差 2.98e-8；这条失败证据保留，确定性设置后的 v3 才作为恢复验证。后续新增资源/输入协议日志及同进程状态隔离改变了运行器源码哈希；旧运行保持原始证据，不改写 manifest，也不能直接通过当前严格源码校验恢复。

环境：Python 3.11.14、PyTorch 2.7.1+cu128、NumPy 2.4.6，NVIDIA GeForce RTX 5060；各输出的 manifest 保存精确版本。`panama_cpu_smoke.json`、`panama_cuda_proceed_smoke.json`、`panama_onenet_author_smoke.json` 提供复现配置。CPU 矩阵命令为：

```text
python -m paper_impl.baseline_run --config configs/full_experiment/baselines/panama_cpu_smoke.json --out NEW_DIRECTORY --matrix --models itransformer --methods GD Head-GD OneNet SOLID SOLID++ PROCEED --seeds 2026
```

早先 `20261007_online_baseline_cpu_v1/` 中 OneNet 是明确记录的 iTransformer+PatchTST 变体，保留为开发验证，不作为正式 FSNet 结果；其他较旧版本保留各自真实源码哈希和设置，当前结果以本表为准。未运行完整数据集、全 8×6 的正式多 seed 长训练、调参公平预算或统计显著性比较，不能把上述有限闭环数字用于方法优劣结论。日历特征目前使用显式零 fallback，跨设备浮点结果不承诺逐位一致；同设备同流恢复已验证。

## 八骨干 × 六基线真实 CSV 集成复核

新增 `tools/smoke_official_baselines.py`，实际运行结果位于 `results/full_experiment/20261007_official_baseline_crosscheck_v1/`。2026-10-07 在同一个 CPU 进程（PID 74304）顺序执行八个完整作者网络 × 六种方法，共 **48 个组合全部完成**；先执行 PROCEED / OneNet 的 16 个高风险组合，再执行其余方法，没有修改核心实现或因失败替换结构。全部运行连同首轮检查耗时 23.59 秒，低于十分钟预算；此耗时仅描述有界检查成本，不是正式速度基准。

共同配置：Panama 原 CSV 前 80 行，输入 `nat_demand` 及 4 个已明确列出的天气变量；L=16、H=2、extra_delay=1，chronological 60% / 20% / 20% 划分；8 个最早可训练窗、4 个验证窗、4 次发行，骨干训练均 1 epoch，OneNet / PROCEED 的方法离线阶段亦为 1 epoch，batch size=4。八骨干均为作者完整计算路径的显式小配置：hidden=16、heads=8、depth=1、decoder_depth=1、d_ff=32、label_len=8、dropout=0，FEDformer Fourier modes=4、Pyraformer FC、Crossformer seg_len=4、PatchTST patch_len=4 / stride=2。OneNet 全部使用 OnlineTSF 正式通道 FSNet，维持 64 / 320 / 10 默认大小（11 个残差块、22 个梯度记忆层），没有用另一种 Transformer 替代。

每组合均核验：骨干训练与验证选模日志存在；OneNet / PROCEED 必需离线方法训练完成；4 次发行全部释放并监督、8 个目标点全部评分，发行预测有限、未成熟队列为空；反馈不早于 `issue+H+extra_delay`；在线优化实际发生；模型及完整流检查点可以读取；源码、参数量、输入协议等 14 类输出齐全。合计 **192 次预测、192 个完整反馈窗、384 个有效目标点**。OneNet 检查点证实骨干/权重/决策三个优化器各执行 2 次离线步骤，再达到各 6 次总步骤；22 个 FSNet 梯度记忆全部存在且有非零更新。PROCEED 检查点证实已离线拟合且有 2 次适配训练步骤。全部组合保存同一份数据 SHA256 与同一组核心实现源码哈希。

首次驱动脚本把 OneNet 的 `epochs/losses/samples/cutoff` 报告也当成含 PROCEED 专用 `training_steps`，因此 `summary.json` 原样保留 **40 passed / 8 validator KeyError**。这 8 个 OneNet 的实际训练、预测和反馈当时已经完成。修正检查器字段解释后，在原有文件上独立追加 `revalidation.json`，结果 **48 passed / 0 failed**，没有重新训练、覆盖初始日志或改写方法结果。原驱动快照为 `smoke_driver_executed.py`，修正后的复核驱动快照为 `revalidation_driver.py`；8 条原异常栈仍在 `logs/`。此集成结果支持网络与方法接口可运行，不支持精度排名或正式泛化结论。

复现命令（输出目录必须全新）：

```text
python tools/smoke_official_baselines.py --out NEW_DIRECTORY --budget-seconds 540
```

## 第二轮评审后的最终修订与 v2 复核

修改仅涉及 `online_baselines.py`、`baseline_run.py`、两者测试、对应 smoke 检查器与本审计文档：新增显式 SOLID 模式与真实来源标签，保持原 `eval + sum` 默认；补记忆时效公式；CUDA 峰值字段改为末片段语义；补充 PROCEED 上一 mini-batch 与真实 argparse 默认来源。未修改作者源码、骨干结构或数据协议。

针对性命令 `python -m pytest -q -p no:cacheprovider tests/paper_impl/test_online_baselines.py tests/paper_impl/test_baseline_run.py`：**118 passed in 14.38s**。新增覆盖两个 SOLID 模式的真实 dropout 激活/关闭、BN buffers、头参数唯一梯度范围、每输入恢复完整预训练 state、随机/BN 状态精确恢复、SOLID++ 始终 train/mean、记忆时效边界、配置/来源标签透传，以及 CUDA 恢复后只报告最后片段峰值（受控 allocator 读数 900→300，确认不误聚合为 900；模型 CUDA 前向/反向和状态恢复真实执行）。该受控数字仅用于字段语义测试，不是测得的 GPU 成本。

随后以冻结核心源码重新运行同一 48 组合真实 Panama 闭环，独立输出 `results/full_experiment/20261007_official_baseline_crosscheck_v2/summary.json`：**48 passed / 0 failed / 0 skipped**，同一 CPU PID 82044，总时间 **24.64 秒**；192 次预测、192 个完整反馈窗、384 个有效目标点，全部组合使用相同源码文件集合哈希和数据 SHA256。v2 检查器直接核验 SOLID 的 `eval/sum` 与 SOLID++ 的 `train/mean` 生效模式及新显存字段。v1 的驱动错误、复核与原始模型结果全部保留，未覆盖。

另将 v1/v2 全部 48 组合的 `issue_times`、`predictions`、`targets`、`target_mask` 逐元素对账，**全部完全相同**；证据为 v2 下 `default_behavior_comparison_v1.json`。这验证本轮明确配置和日志语义没有改变原默认运行结果；显式 train 模式的行为差异由上述 dropout / BN 测试覆盖。
