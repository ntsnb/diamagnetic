# 论文方法—实现—验收对应表

本表针对2026-10-07新版方法。历史论文实验、八个官方模型、外部基线的完整复现不由函数存在或短流程通过推出。

后续范围复核：全文实验仍有实际代码缺项，包括外部基线接入、结构组件消融和原shifted-block协议；不是单纯“还没跑”。逐项见 [EXPERIMENT_READINESS_20261007.md](EXPERIMENT_READINESS_20261007.md)。本表的“完整训练/CLI”仅指主方法和表内已列出的对照。

| 论文要求 | 当前实现 | 有意义的验证 |
| --- | --- | --- |
| B：IRN、环境分组、MEI/MIL交替 | `paper_impl/foil.py` | 全局目标与分批精确梯度一致；按原始时间索引平滑；参数组冻结、空环境修复；CPU/CUDA |
| B：独立两分支及共用目标坐标 | `model.py:ForecastBranches` | 独立参数、同目标预测形状、冻结、分支差诊断 |
| C：summary→VAE→router | `model.py:SummaryEncoder, PaperForecaster` | 重构、KL、确定性后验均值、离线随机重参数路径 |
| C：独立专家及加权融合 | `model.py:VariableExpert` | 输出步/目标索引；单专家预测与完整共享head计算一致 |
| C：逐专家LRP | `lrp.py`, `model.py:attribute` | 单步/局部闭式参照，gradcheck，专家参数有限差分；有符号残差账本单独记录 |
| C：变量与滞后互补距离 | `discrepancy.py` | 区间界、三角不等式、变量置换、缩放不变、L=1、零图处理 |
| C：固定专家对分工目标 | `discrepancy.py:specialization_loss`, `training.py` | 固定分母、零图最大惩罚、共享head梯度边界，实际训练更新 |
| D-I：仅router bias持久更新 | `adaptation.py:OnlineController` | 参数支持集不越界、已释放标签时序、Armijo下降/失败回退 |
| D-II：固定参考、EMA、归因通道筛选 | `adaptation.py` | 同参数参考、完整历史集合、无效图处理、同层跨专家预算及输出通道方向 |
| D-II：误差触发梯度选择 | `adaptation.py` | 发行时预测误差触发；与归因定位同预算；gradient_only对照 |
| D-III：中心化有界通道调制 | `adaptation.py:CenteredBoundedGenerator` | 零对比精确恒等、增益界、非对称out×in方向、无写回 |
| 生成器离线回放 | `replay.py` | 每episode恢复状态、仅训练前缀、generator梯度独占、之后Stage I/II梯度通过固定gain |
| 多步反馈完整释放 | `stream.py` | t+H+delay、缺时刻/重复观测处理、队列有界、异常回退 |
| 新数据协议 | `data.py` | 训练统计独占、目标不插补、输入仅前向填充、完整目标跨段边界、等间隔与数据哈希 |
| Spain原时间字段问题 | `tools/paper_impl_repair_spain_clock.py` | 只逆转已确认歧义日月；逐行小时轴核对、所有其他字段字符串不变、禁止覆盖 |
| 完整训练/重新评估CLI | `run.py` | 配置拒绝、种子继承、checkpoint配置/数据一致性、逐预测和更新记录往返相等 |
| 三阶段全部8组合 | `evaluation.py`, `run.py` | 每组合独立持久状态、保存原发行预测、更新计数可追踪 |
| 三种native summary encoder | `model.py` | temporal/inverted/patch完整前向及训练路径 |
| 八种旧style编码器 | `backbones.py` | 原类真实调用、短/非整除长度、CPU/CUDA、梯度与保存加载；共同专家/LRP/controller不变 |
| 四类区域负荷数据与丽水九县 | `configs/paper_impl/validation_suite` | 准确列名、实际采样间隔、独立配置；统一有界流程入口 |
| 版本与证据记录 | `run.py:source_manifest`, audit目录 | 代码/旧依赖/数据哈希、实际配置、逐预测NPZ、逐次更新JSON、日志；禁止覆写结果 |

## 完整实现与实验验证的区别

已实现的是新版方法及上述对照的具体运算、训练和在线状态变化。`*_style`复用了原工程实际存在的自定义编码器；它们没有补造为官方原版模型。原稿中的shifted-stream实验、外部SOTA基线、大型统计比较仍属于历史协议和历史代码，未由新入口重新复现。相关结果继续保留原始证据，不能迁移为新版数字。

核心验收命令为 `python -m pytest tests/paper_impl -q`。真实数据短流程使用 `tools/paper_impl_validate_suite.py`，详见每次输出目录中的 `suite_report.json`。数据集范围相同不表示训练预算、输入列数或预测长度与历史论文实验相同，应以各运行配置为准。
