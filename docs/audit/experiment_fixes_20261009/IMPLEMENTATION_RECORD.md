# FOIL 时间索引与双种子实验计划修复

- run_id：20261009_foil_time_and_two_seed_plan_v1。
- 用户确认范围：保留缺失标签窗口剔除；用原始时间索引选取平滑邻居；真正生成42/43配置；请求不齐时报错；暂不处理长任务恢复。
- 类型：实现、回归测试、计划生成。不是正式训练或性能研究。
- 基线：当前paper_impl，上一轮审查600测试通过；FOIL缺口索引与种子部分缺失已证实。
- 变化A：原始issue_times沿FOIL训练调用链传递，保持真实时间间隔。
- 变化B：完整双种子配置与任务清单、对应训练seed核验、计划请求完整性验证。
- 数据范围：正式13任务；不改变数据值、缺失掩码、划分、标签释放时钟或评分。
- 工作目录/分支：现有独立实现目录LoadForecasting_Paper_Implementation_20261007，codex/paper-complete-20261007；保留现有未提交工作，不额外创建丢失未提交实现的工作树。
- 文件责任：子agent负责foil.py/run.py及时间索引专项测试；父agent负责experiment.py、配置生成器/配置、调度专项测试与文档；另一路只读检查随机种子传播。
- 验证：专项缺口/缺失seed反例、全部配置与每个训练阶段seed一致性、13真实数据索引检查、现有全套回归；精确命令及结果完成后追加。
- 预期产物：代码修复、42/43完整任务清单和配置、验证日志与中文说明；原单种子冻结矩阵及旧实验结果保留。
- 状态：两项修复完成，验证通过。未启动正式性能训练。

## 1. 时间索引修复

含缺失标签的监督窗口继续被剔除，原始标签中的NaN保持不变。统一入口将每个剩余训练窗口的 `data.train.issue_times` 传入 `train_foil_branches`，再传至环境推断、时间平滑及空环境修复。小批次打乱仅改变计算顺序，环境风险仍写回对应样本行。

平滑邻居采用 `|issue_s - issue_t| <= smoothing_radius`，不再使用剔除窗口后的行号距离。现有数据入口要求固定采样间隔，所以索引距离对应原始采样步的真实时间距离；半径2在小时数据上即2小时。FOIL的目标、交替优化与多数投票规则未改变。直接调用辅助接口而未传索引时仍兼容连续行号，并明确记录来源；正式入口始终传原始索引。

`foil_history.json` 与checkpoint保存 `training_sample_ids`、`sample_id_source`、`smoothing_radius_unit`、`environment_label_order`。索引与每个分支最终环境标签逐行对应，可回查每个窗口的原始时刻。

实际13个数据任务已验证全部保留训练标签等于原时间位置的目标切片。九个丽水县区任务各剔除29个不完整训练窗口，仍保留 `13877→13895`、`14046→14059` 的18/13步间隔；全市任务剔除12个窗口，保留 `7590→7603` 的13步间隔。三个公开数据任务的当前训练索引连续。此检查仅验证现有输入文件，不补充Spain上游缺测来源证据。

## 2. 独立种子配置与请求完整性

新矩阵为 `configs/full_experiment/structural_seeds42_43/matrix.json`；原 `structural` 单种子矩阵保留。每个任务均实际生成42、43的JSON文件，基线也保存独立JSON及其SHA-256，而不只列出一个待派生条目。

| 环节 | 种子42 | 种子43 |
|---|---|---|
| Python/NumPy/Torch初始化及主预测器训练 | 42 | 43 |
| FOIL invariant训练 | 42 | 43 |
| FOIL variant训练（原有分支偏移） | 1051 | 1052 |
| 官方专家固定采样 | 42 + 专家索引 | 43 + 专家索引 |
| 六种基线训练 | 根种子42及方法既有确定性派生 | 根种子43及方法既有确定性派生 |

生成器继承相应种子的初始化和预测器，按固定时间顺序回放；没有增加虚构的独立回放seed。OneNet保留第二骨干 `seed+1` 的既有设计。新运行中，未显式设置的 `official_sampling_seed` 继承根种子，修复了根种子43但官方采样仍用默认42的问题；旧checkpoint重评估则保留保存时的采样值，不悄悄改写历史模型。

调度先按声明构建完整适用任务集合，再逐项核验种子和文件。只缺某个任务的43、某个任务的42/43同时缺失、清单有条目但JSON文件缺失、配置内仍写旧种子，都会报错。错误例如 `种子 43 缺少配置`；不存在“42可以找到就静默忽略43”的行为。合法适用范围保持原设计：native不生成外部基线，woRevIN仅适用于native/PatchTST/iTransformer。旧版基线条目仍兼容从Full配置派生。

## 3. 完整任务清单

| 项目 | 每个种子 | 42与43合计 |
|---|---:|---:|
| 框架训练配置 | 1794 | 3588 |
| 基线训练配置 | 624 | 1248 |
| 可独立运行任务 | 2418 | 4836 |
| 框架在线评价分支 | 2613 | 5226 |
| 含基线的评价记录数 | 3237 | 6474 |

清单覆盖13个数据任务、8个官方骨干及native、适用结构消融和6种在线基线。Full的8种阶段组合共用同一离线检查点，因此评价记录数大于训练任务数。以上均为待执行数量，不是已完成实验数。

- [完整运行计划JSON](generated_plan/plan.json)：每项内含可执行配置。
- [4836项训练及运行任务CSV](generated_plan/training_tasks.csv)：数据、骨干、结构/方法、种子、配置路径、配置哈希、评价数量。
- [6474项评价任务CSV](generated_plan/evaluation_tasks.csv)：进一步展开阶段组合。
- [配置与13个数据任务验证记录](generated_plan/validation.json)：所有配置哈希、数据来源哈希、原始时间缺口及标签对应检查均通过。

## 4. 验证

最终整套回归：**646 passed, 2 skipped, 2 warnings，131.41秒**。两项跳过为需显式开启的归因成本测试；两条警告来自保留的官方FSNet softmax代码。原始记录：[pytest_full.log](pytest_full.log)。没有改动这些第三方代码。

新增回归覆盖：缺口距离反例、打乱行顺序、非法索引、旧调用兼容、缺测数据完整短训练链、日志/checkpoint索引、官方采样种子继承及旧checkpoint兼容、任务/种子/文件缺失、内层seed不匹配和骨干适用范围。旧单种子基线派生路径也通过回归。

运行命令（在项目根目录）：

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
C:/Users/admin/anaconda3/python.exe -m pytest tests/paper_impl -q -p no:cacheprovider --basetemp=docs/audit/experiment_fixes_20261009/pytest_full_tmp

C:/Users/admin/anaconda3/python.exe tools/build_full_experiment_matrix.py --dataset-root 'C:/Users/admin/Desktop/最近/1.28/IEEE-Transactions-LaTeX2e-templates-and-instructions/dataset/load_benchmarks' --seeds 42 43 --output configs/full_experiment/structural_seeds42_43

C:/Users/admin/anaconda3/python.exe docs/audit/experiment_fixes_20261009/verify_generated_matrix.py --matrix configs/full_experiment/structural_seeds42_43/matrix.json --output docs/audit/experiment_fixes_20261009/generated_plan
```

上述产物已经存在；复核时应换用新的输出目录，保留原配置和日志。验证脚本只生成/检查任务计划和加载数据，不训练模型。

## 5. 修改文件与边界

- 运行实现：`paper_impl/foil.py`、`paper_impl/run.py`、`paper_impl/experiment.py`。
- 配置：`tools/build_full_experiment_matrix.py`、`configs/full_experiment/structural_seeds42_43/`。
- 测试：`test_foil_time_indices.py`、`test_experiment_seeds.py`、`test_experiment_dispatch.py`。
- 说明：README、本目录、项目目标/实验/术语/审查记录。

本轮完成的是实现修复和实验计划准备，没有正式训练结果或性能提升结论。按用户范围暂不修改长任务恢复、失败隔离、自动跳过或在线状态续跑。论文源稿、原始数据和既有结果未改写。
