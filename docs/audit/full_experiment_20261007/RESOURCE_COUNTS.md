# 完整矩阵参数与存储计数审计

日期：2026-10-07。已统计 **1,794 个结构训练配置和 624 个计划在线基线**，按结构等价性缓存，仅实际构造 **384 个 CPU 模型组**。没有训练，没有预测前向，也没有启动 CUDA 计算；这不是 1,794 次实验完成记录。

## 接口与计数口径

公共接口位于 `paper_impl/resource_report.py`：

```python
from paper_impl.resource_report import framework_parameter_counts
report = framework_parameter_counts(model, branches, generator)
```

三个位置参数可接收 `nn.Module`、`nn.Parameter`、它们的命名映射/序列或 `None`。`include_parameter_details=True` 额外导出每个参数的名字、shape、dtype、device、requires_grad 和计数，不执行 forward，也不修改权重、梯度、训练模式或随机状态。

| 字段 | 定义 |
|---|---|
| `logical_parameter_count` | 去重 Parameter 的 `numel()` 之和；一个复数元素算一个逻辑参数 |
| `real_scalar_count` | 实数元素算 1，复数元素算 2，即实部和虚部；FEDformer 的两个口径不能混写 |
| `trainable_logical_parameter_count` / `trainable_real_scalar_count` | 仅统计检查时 `requires_grad=True` 的参数 |
| `parameter_bytes` | 每个去重 Parameter 的 `numel * element_size` 之和，按实际 dtype 计算 |
| `logical_buffer_count` / `buffer_real_scalar_count` / `buffer_bytes` | 注册 buffer 的对应计数，包括 `persistent=False` buffer |
| `components.*.inclusive` | 一个组件单独包含的所有去重参数/注册 buffer |
| `components.*.exclusive_contribution` | 按 model → branches → generator 的顺序，将跨组件共享参数分配给第一次出现者；各组件贡献之和等于顶层总量 |
| `components.*.direct_children_inclusive` | 组件直属子模块分项，例如 experts、router、invariant/variant、generator blocks |

去重键是 **Parameter 对象身份**，不是注册路径；同一个模块或参数被多处引用不会重复计数。两个不同 Parameter 即使共享底层 storage，仍是两个逻辑参数；因此 `parameter_bytes` 是张量元素存储口径，不是 allocator 实测占用。注册 buffer 也按对象身份去重。lazy 未物化参数报错，不错误记作零。

`requires_grad` 不是在线阶段实际被某个 mask 选中的参数数，也不是每个样本会更新的参数数。矩阵同时报告构造阶段及与 `run.py` 一致、冻结辅助分支后的 `online_requires_grad_snapshot`；Stage II 稀疏通道/专家选择另行记录。生成器的 eval 模式不等于其 requires_grad 标志为 False。

**不包含的成本**：优化器状态、梯度、激活、临时调制张量、控制器重置快照/历史缓存、检查点复制、FLOPs、训练/推理时延、峰值内存或显存。这些项目不能从参数总数直接推断，需单独实测。

## 完整矩阵报告

命令：

```text
python tools/report_full_experiment_resources.py --matrix configs/full_experiment/structural/matrix.json --out results/full_experiment/20261007_resource_counts_v1 --allow-dtype-scan
```

产物：

- `results/full_experiment/20261007_resource_counts_v1/resource_counts.json`：完整组件分项、所有作业到缓存结构的映射、计数口径、源码 SHA256、矩阵 SHA256。
- `framework_parameter_counts.csv`：1,794 个结构作业逐行计数，保留 dataset/backbone/ablation/seed/config/structure_id。
- `baseline_parameter_counts.csv`：624 个计划基线逐行计数。
- `dataset_schemas.json`：实际列清单、目标到输入的索引映射和 dtype 推断证据；不保存原始数据行。

十个丽水任务从 registry 的显式 feature_columns 取得输入，再只读 CSV 表头核对。Panama、Tetouan、Spain 原配置为 `feature_columns=null`，表头不能证明数值 dtype；经父任务明确授权，工具对这三个 CSV 各执行一次 `read_csv` dtype 推断，并严格使用与 `prepare_data` 相同的 `select_dtypes(include="number")` / timestamp / target mask / target-input mapping 排除规则。未启用 `--allow-dtype-scan` 时，工具拒绝替这些任务猜输入维数。

| 任务 | 输入变量数 | 目标输入索引（从 0 开始） |
|---|---:|---|
| Panama | 16 | `[0]` |
| Tetouan | 8 | `[5]`，Zone 1 |
| Spain | 9 | `[0]` |
| 丽水九县及独立市级任务 | 每任务 8 | 每任务 `[0]` |

报告保留 matrix 中声明的数据 SHA256，但不宣称本工具重新认证了原始数据来源。结构键包含输入变量数、目标索引、L/H、完整模型选项、辅助分支数量和生成器结构。seed、优化器/损失权重、Stage 选择不改变参数形状时复用计数；相同参数量不表示这些实验行为相同。

框架结构直接调用公共 `paper_impl.run.build` 的 CPU 构造路径，显式提供 target_indices。基线采用同一个 full 配置的 L/H、输入 schema、官方骨干尺寸/选项；OneNet 添加完整默认官方 FSNet 和作者决策网络/组合权重，PROCEED 构造真正的概念编码器和生成器。基线 inventory 记录任务索引，统一入口在生成计划时派生完整训练配置；适配器尺寸使用当前 `OnlineBaselineConfig` 构造默认值，并在每个缓存项的 `adapter_defaults` 记录。将来改变 concept/rank/FSNet 等尺寸后必须重新计数。这只是结构实例化，不跳过实际执行时必需的 PROCEED 离线训练。

## 参数量示例：正式 Panama full 配置

该表使用正式矩阵 L=24、H=12、三个专家及完整框架组件，不是此前 hidden=16 的小烟雾配置。

| 完整专家骨干 | 逻辑参数总数 | 实数标量总数 | 在线 requires_grad 逻辑数 | 参数元素字节 |
|---|---:|---:|---:|---:|
| iTransformer | 700,903 | 700,903 | 497,231 | 2,803,612 |
| Crossformer | 3,244,201 | 3,244,201 | 3,040,529 | 12,976,804 |
| Autoformer | 978,211 | 978,211 | 774,539 | 3,912,844 |
| Transformer | 974,563 | 974,563 | 770,891 | 3,898,252 |
| PatchTST | 767,821 | 767,821 | 564,143 | 3,071,284 |
| Informer | 1,012,771 | 1,012,771 | 809,099 | 4,051,084 |
| FEDformer | 1,033,507 | **1,088,803** | 829,835 | 4,355,212 |
| Pyraformer | 933,763 | 933,763 | 730,091 | 3,735,052 |
| 原生完整框架 | 764,639 | 764,639 | 560,967 | 3,058,556 |

对应 Panama 单个 iTransformer 正式基线：

| 基线 | 逻辑参数总数 | 在线 requires_grad 逻辑数 |
|---|---:|---:|
| GD | 102,476 | 102,476 |
| Head-GD | 102,476 | 780 |
| SOLID | 102,476 | 780 |
| SOLID++ | 102,476 | 102,476 |
| OneNet（+完整 FSNet） | 1,465,412 | 1,182,788 |
| PROCEED | 313,596 | 106,172 |

OneNet 总数包含作者冻结的 FSNet W 记忆参数；其额外梯度/查询 buffer 单独报告。PROCEED 总数包含离线训练后在线冻结的概念/生成权重，在线 requires_grad 只保留骨干和作者最终输出偏置。这些差异证明不能只用“相同主干”假定总容量相同。

## 每次基线运行产物与输入协议

`baseline_run.py` 现在自动写 `resource_counts.json`，从实际控制器的主模型、第二分支、PROCEED 适配器或 OneNet 决策网络/组合参数统计，不再用外部估算代替运行实例。恢复运行同样生成当前实例计数。主框架可直接使用上述 helper 接入自己的日志。

同时写 `backbone_input_protocol.json`：日历信息只来自明确输入列；不另行向作者接口传日历张量。iTransformer 保持 `x_mark_enc=None`；Crossformer/PatchTST 的作者 forward 没有独立 marks 参数；需要 encoder/decoder marks 的作者路径使用 zero marks；FSNet 通道分支采用七个零日历通道的明确 fallback。全局标准化为 PreparedData 的训练前缀 z-score，作者内部 normalization 按完整配置保留。

## 验证与成本证据边界

`python -m pytest -q -p no:cacheprovider tests/paper_impl/test_resource_report.py`：**8 passed in 1.91s**。覆盖复数/冻结参数、跨组件共享身份去重、同 storage 不同 Parameter 的口径、lazy 拒绝、权重/梯度/模式无副作用、Python/NumPy/Torch RNG 和默认设备恢复（包括异常路径）、等价结构缓存、显式 schema 不读数据行也不 forward、dtype 推断必须明确启用。

资源测试与六方法真实训练/同流恢复的针对性集成检查：**14 passed, 11 deselected in 4.99s**。六方法实际运行都产生参数报告和输入协议，Head-GD/SOLID 冻结范围、OneNet 额外组件分项得到核验。

完整计数工具恢复 CPU 线程设置，并在每组构造时隔离/恢复 Python、NumPy、CPU Torch RNG 和默认 device。实际报告记录 `device="cpu"`、`forward_calls=0`、`training_steps=0`。没有把 384 次构造当作耗时训练，没有外推 1,794 项矩阵执行成本。

已有有限运行的耗时/显存证据位于 [ONLINE_BASELINES.md](ONLINE_BASELINES.md) 所列原始 `metrics.json`，它们具有不同的 L/H/宽度和有限窗口预算，不能直接外推正式矩阵总耗时。完整框架或算法的每阶段运行成本应由独立 profile 记录实际输入尺寸、设备、确定性设置、同步和预热方式后报告；本交付不虚构 FLOPs、吞吐率或完整矩阵资源预算。
