# Frozen、同预算定位对照与专家分工记录

- run_id：20261009_controlled_comparisons_and_diagnostics_v1。
- 范围：新增独立完整官方骨干Frozen；同一离线模型上比较LRP/梯度/随机固定参数预算与同范围密集更新；保存误差、路由、利用率和分工距离。暂不加入Naive/其他简单模型，不扩展长任务恢复。
- 对照：上一轮646项通过的paper_impl及42/43矩阵。旧源码成果、矩阵、结果保留。
- 改动模块：baseline_run/online_baselines；adaptation；evaluation/run/新增diagnostics；实验配置生成器及专项测试。
- 变量隔离：Frozen单独训练一个官方骨干；定位对照复用Full的同一离线checkpoint，均仅启用Stage II，区别为参数排序或是否密集更新。原Full与八阶段组合保留。
- 参数预算：三个稀疏对照每个对应层选相同数量的完整权重行；同层行宽一致，故选中的标量参数数量一致。实际非零变化数、接受/拒绝另行记录，不把掩码数量冒充实际更新量。
- 专家诊断：发行时记录路由和持久专家预测，按预先配置的固定间隔计算持久专家相关性距离；标签仅在事后评分使用。诊断开销另记，不能混入算法运行成本比较。
- 数据/种子：沿用13数据任务、42/43、既有切分/标签掩码/释放协议，不运行正式长训练。
- 验证：Frozen参数/缓冲冻结、精确预算、局部随机种子、范围隔离、诊断不影响发行预测或更新、指标形状/缺失掩码、同checkpoint评价和全套回归。
- 分支：codex/paper-complete-20261007现有独立实现目录；保留未提交工作。子agent各自拥有基线模块、适应模块；父agent负责评价诊断和整合。
- 状态：实现完成，717项回归通过、2项成本测试未开启；正式长训练未启动。

## 1. 独立官方骨干Frozen

`baseline_run`新增 `method: "Frozen"`。它与其他基线使用相同的数据、预处理、标签释放协议和离线训练/验证选模流程，预测器是一个完整官方骨干。测试期不建优化器，不进行反馈参数更新；保持eval和冻结参数，缓冲状态不变。标签释放仍进入公共时钟与评分记录。

`stages_000`继续表示完整专家混合框架的三个在线阶段全部关闭。二者分开报告：前者是独立骨干参照，后者是完整框架的冻结对照。没有加入季节性Naive或其他简单模型。

## 2. 同预算参数定位对照

Full配置新增 `selection_comparisons`，共用其离线检查点，分别从相同参数重新开始在线评价：

| 评价名称 | 选择依据 | 可更新范围 |
|---|---|---|
| matched_lrp | 当前与参考窗口的通道相关性变化 | 每个对应层固定K行权重 |
| matched_gradient | 已释放样本损失的梯度行范数 | 相同层、相同K行权重 |
| matched_random | 由任务seed和释放时刻确定的局部随机排序 | 相同层、相同K行权重 |
| matched_dense | 不排序，密集更新 | 同一候选集合全部权重 |

四者均只开启Stage II，Stage I/III关闭，使用相同标签、损失、步长上限、范数约束和接受规则。候选集合统一为专家内部允许更新的线性权重矩阵，排除预测头。不同专家的对应层具有相同行宽，选择K行即严格匹配K乘以行宽个标量参数。跨层预算分别匹配，不能用不同层行宽凑相同“通道数”。`matched_dense`更新同一候选集合的全部权重；原 `DenseExperts` 对更多专家参数的更新仍作为既有结构对照保留。

定位实验只比较排序规则，因此固定预算LRP不额外依据稳定性阈值排除专家，也不在高误差时切换为梯度选择。原Full的自适应逻辑未改变。合法零漂移使用确定性并列规则；参考不足、输入/通道相关性无效或无法归一化时，固定LRP明确抛出 `SelectionUnavailableError` 并回滚，不伪造足额排序。随机对照不消耗主训练或推理随机序列。

更新日志分别记录：

- `selected_parameter_count`：本次掩码允许更新的标量参数数目。
- `changed_parameter_count`：本次更新后数值确实改变的参数数目，拒绝更新时为0。
- `eligible_parameter_count`：候选集合总参数数目。
- `layer_budgets`：各对应层的通道预算、行宽、选中数和候选数。
- `selection_event_seed`：随机选择的释放事件种子。

`selection_comparison.json`逐次核对三个稀疏对照的释放时间、样本和各层预算。相同选中容量不保证实际非零改变量相同：零梯度、浮点舍入和更新拒绝都会影响实际变化数，必须连同接受率一起报告。小型功能测试中三个稀疏控制每次都选37个参数，密集控制选392个；这些是测试模型的容量核验，不是正式模型规模或准确率结果。

## 3. 专家分工的联合证据

新矩阵默认 `expert_diagnostics.enabled=true`、`distance_interval=1`，每个发行窗口均记录。间隔可显式设置，但必须在实验前固定；取样不依据误差或未来标签。旧配置默认关闭诊断。

| 产物 | 保留内容 |
|---|---|
| metrics.csv | 实际发行预测的误差、平均路由熵、有效专家数、平均分工距离及运行成本 |
| expert_metrics.csv | 每个专家的MSE/MAE、原尺度误差、平均路由权重、最高权重次数/占比 |
| expert_pair_metrics.csv | 每个专家对的平均分工距离、有效目标窗口数 |
| 各分支 *_expert_diagnostics.npz | 原时间索引、逐窗路由、每专家预测/误差、实际发行误差、标签观测掩码、逐目标专家对距离和有效性、总相关性 |
| 各分支 *_expert_diagnostics.json | 诊断范围、采样间隔、汇总指标、无效对数量与诊断耗时 |
| evaluation_protocols.json | 每个阶段/定位分支实际使用的完整在线配置 |

平均路由权重反映专家承担的相对贡献，最高权重占比反映其成为主导专家的频率。当前所有专家均参与计算，这两项不能解释为计算稀疏率。分工距离使用论文的归一化相关性差异，保留变量身份与滞后信息；零总相关性图的距离保存为无效值及显式掩码，不当作距离0。单专家没有专家对，距离汇总为null。

专家预测与分工距离针对当前持久专家参数，不叠加Stage III临时调制、不乘路由权重，与训练分工损失的对象一致；整体预测误差使用真实发行的预测，包括已启用的临时调制。诊断不接收当前未来标签，真实标签只在全部发行完成后的评分中使用，缺测点不计入误差。

诊断不更新参数、不累积参数梯度，并保留随机状态。额外时间和显存峰值独立记录；诊断不需要用于训练的高阶梯度图。正式纯成本对照可显式关闭诊断，以免额外计算影响运行环境。

## 4. 新任务清单

矩阵：`configs/full_experiment/controlled_seeds42_43/matrix.json`，原两个矩阵目录保留。

| 项目 | 42和43合计 |
|---|---:|
| 框架训练配置 | 3588 |
| Frozen配置（8骨干×13任务×2种子） | 208 |
| 既有六在线基线配置 | 1248 |
| 独立运行任务 | 5044 |
| 框架评价记录（含936个新增定位控制） | 6162 |
| 全部评价记录 | 7618 |

42、43各2522项运行。Full每份配置含原8阶段组合和4个定位控制，后四者共用离线检查点；其他结构仍按声明的阶段组合评价。新增诊断覆盖各框架配置。所有配置、数据来源哈希和13任务的时间索引核验通过。

- [完整运行任务CSV](generated_plan/training_tasks.csv)
- [展开评价任务CSV](generated_plan/evaluation_tasks.csv)
- [可执行计划JSON](generated_plan/plan.json)
- [配置与数据验证JSON](generated_plan/validation.json)

## 5. 验证及边界

最终整套：**717 passed, 2 skipped, 2 warnings in 128.63s**，原始日志 [pytest_full.log](pytest_full.log)。两项跳过仍是需显式开启的归因成本测试；两条警告来自保留的官方FSNet softmax代码。

Frozen验证覆盖8官方骨干CPU/CUDA：反馈标签值不影响其预测、参数和缓冲；提前反馈被拒绝；短完整离线训练、验证选模、流式评价及文件保存通过。固定预算测试覆盖不同层宽度、零分并列、随机重试、精确掩码、拒绝/接受计数、无效输入/通道图与原子回滚。新诊断在native/iTransformer/Informer连续短流中保持发行预测、更新记录和随机状态完全一致；另有native与全部8官方骨干的CPU单窗只读复核。保存后重评估的全部诊断数组与定位结果一致。

初轮专项的3项失败均来自新测试设置：官方子集目标漏填target_indices，以及模拟旧checkpoint时对共享字典重复pop。修正后专项15项通过，最终整套通过。审核另发现固定LRP未拒绝无效输入图，已补严格检查与回滚测试；诊断多余高阶图已关闭。

最终补强短流测试至5个发行窗口，要求在后续预测发行前确实已有接受的反馈更新；诊断开启/关闭后的后续预测、更新记录与随机状态仍完全一致，四个定位分支预算及checkpoint重评估仍一致。补强后的15项测试全部通过（8.29秒），见 [pytest_feedback.log](pytest_feedback.log)。该步仅扩展测试观察范围，生产代码未再修改。

命令（项目根目录）：

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
C:/Users/admin/anaconda3/python.exe -m pytest tests/paper_impl -q -p no:cacheprovider --basetemp=docs/audit/controlled_comparisons_20261009/pytest_full_tmp

C:/Users/admin/anaconda3/python.exe tools/build_full_experiment_matrix.py --dataset-root 'C:/Users/admin/Desktop/最近/1.28/IEEE-Transactions-LaTeX2e-templates-and-instructions/dataset/load_benchmarks' --seeds 42 43 --output configs/full_experiment/controlled_seeds42_43

C:/Users/admin/anaconda3/python.exe docs/audit/experiment_fixes_20261009/verify_generated_matrix.py --matrix configs/full_experiment/controlled_seeds42_43/matrix.json --output docs/audit/controlled_comparisons_20261009/generated_plan
```

这些目录已存在；复核时请选新目录，保留原始日志。本轮没有全长正式性能结果，尚不能判断LRP胜过梯度/随机或专家分工提高准确率。已补齐产生与检验这些证据的实现、配置和输出。原始数据、论文源稿、旧结果未改动，长任务恢复未扩展。
