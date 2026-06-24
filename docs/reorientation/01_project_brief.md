# 第一阶段项目画像

生成时间：2026-06-24  
工作模式：只读重定向审计。除本 `docs/reorientation/` 审计输出外，不移动、不删除、不覆盖任何仓库文件。

## 1. 当前研究主线

当前仓库的论文主题是：在分布偏移与延迟反馈条件下的在线多变量区域负荷预测。活动稿 `bare_jrnl_new_sample4.tex` 的 abstract 和 introduction 已经围绕以下主线展开：

1. 离线阶段学习稳定预测结构；
2. 用 factor-mechanism / multi-expert forecasting backbone 承载负荷生成机制；
3. 在线阶段通过 Stage I / Stage II / Stage III 进行保守适配；
4. 在 public load benchmarks 与 Lishui-9 区域负荷数据族上验证 shifted-stream online forecasting。

需要注意：根目录主稿仍保留 IEEE 模板标题、作者和若干占位小节，因此它不是 journal-ready 稿件。`bare_jrnl_new_sample4_audit_annotated.tex`、`bare_jrnl_new_sample4_redline.tex` 和 `bare_jrnl_new_sample4_revision_markup.tex` 是审计/修订衍生稿，应先由作者确认哪一个会成为下一轮 canonical manuscript。

## 2. 当前代码框架

当前代码是配置驱动的 PyTorch 时间序列预测框架，主要路径如下：

| 层级 | 路径 | 当前作用 |
| --- | --- | --- |
| 数据读取与切分 | `data/readers.py`, `data/windowing.py`, `data/loaders.py`, `data/normalization.py` | 读取 CSV/Excel/XML，排序时间戳，构造滑动窗口，做 chronological / shifted split 和归一化。 |
| 离线不变学习 | `models/invariant_learning.py`, `train/offline_invariant.py` | MTIL/MTEI 风格环境推断与不变表示训练。 |
| 离线预测 backbone | `models/factor_mechanism.py`, `models/factor_mechanism_v*.py`, `train/offline_factor*.py` | Transformer/iTransformer/Crossformer 风格编码、VAE、专家机制、输出头与版本化损失。 |
| 旧版在线循环 | `train/run_experiment.py`, `train/online_loop.py`, `models/adaptation.py` | 单次训练和 Stage I/II/III 在线适配主入口，仍是 README 的基础入口。 |
| Stage III 审计链 | `train/run_stage3_v2_sensitivity_audit.py`, `train/online_loop_stage3_v2.py`, `models/adaptation_v2.py`, `models/aligned_representation_v2.py` | V2-V6 修复路径，强调 deterministic eval、zero-drift identity、rolling validation 和 counterfactual gates。 |
| Stage III teacher 链 | `models/stage3_alignment_v7.py`, `models/stage3_alignment_v9.py`, `models/stage3_residual_teacher_v9.py` 到 `v14.py` | V7-V15 的残差 teacher、alignment、history control、label availability 与 fluctuation diagnostic。 |
| 新探索主线 | `models/factor_mechanism_v16.py` 到 `v29.py`, `train/run_stage3_v16...v29*.py`, `configs/stage3_v16...v29*.yaml` | V16-V29 的 expert-local / release / observable-change / mode-gate 系列探索。 |
| 当前最接近新思路的入口 | `train/run_stage3_v29_expert_local_invariant_release_probe.py`, `models/factor_mechanism_v29.py`, `configs/stage3_v29_expert_local_invariant_release_probe.yaml` | V29 expert-local invariant-release 结构与 gate-aware tuning 入口。 |

代码框架的核心问题不是“缺少版本”，而是版本过多且每个版本都带有部分失败诊断、局部修复和不同协议约束。后续重构应优先建立新的 canonical protocol，而不是在现有多版本树中继续叠加入口。

## 3. 各研究组件角色

### Offline invariant learning

角色：建立稳定表示，试图将长期稳定预测结构与在线变化因素分开。旧主线使用 `models/invariant_learning.py`、`train/offline_invariant.py`。Stage III 后续版本中，尤其 V16-V29，将“全局不变表示”逐渐转向“专家局部不变 / release-state / servable-change”结构。

证据状态：代码存在，组件消融在 `PAPER_CLAIMS_MATRIX.md` 中被评为弱支持或有 caveat。`w/o Invariant` 并非在所有数据集都变差，因此不能写成无条件核心贡献。

### Factor-mechanism forecasting

角色：作为主预测模型，将多专家、VAE、RevIN、temporal encoder 和输出头组合起来，承载离线预测能力。`factor_mechanism.py` 是基础实现，`factor_mechanism_v16` 到 `v29` 是 Stage III 新机制逐步嵌入的版本族。

证据状态：比 Stage III 更接近论文可保留主线。RevIN 和多专家因子化有较强平均支持，但也存在 dataset-specific caveat。VAE gate 不应写成已验证核心贡献，因为 component ablation 中 `MLP Gate, no VAE` 表现更强。

### Stage I online adaptation

角色：在在线阶段做轻量门控/mixture recalibration。源码中主要表现为 `train/online_loop.py` 的 `gate_bias`、Stage I optimizer、rollback 与 decay。

证据状态：可作为在线控制器组成部分保留，但需要修正 delayed-feedback 叙述。当前实现有 batch-delay 和 current-label acceptance 风险，不能写成严格逐时间戳 delayed feedback。

### Stage II online adaptation

角色：基于 LRP/relevance map 与稳定性触发，对部分参数做稀疏更新。对应 `models/lrp.py`、`train/online_adapt.py`、`train/online_loop.py` 中 stability window、mask、EMA reference 和 rollback 逻辑。

证据状态：Stage II 相关表格存在，但 `PAPER_CLAIMS_MATRIX.md` 标注 Stage II / Full claim 有 leakage 和 provenance 风险。当前不宜把 Stage II 写成稳定有效的单组件，除非先重跑 leak-safe delayed-feedback 协议。

### Stage III online adaptation

角色：原始设计是基于 `phi_v - phi_i` 或 drift representation 生成瞬态参数/残差校正。版本演进显示 Stage III 的核心问题从“generator 是否存在”逐步转变为“drift signal 是否可归因、可服务、能超过 history-only 和 counterfactual controls”。

证据状态：Stage III 是仓库中最丰富但风险最大的证据链。V2-V15 多数证明机制修复和失败原因，V16-V29 继续探索 expert-local invariant-release，其中 V29 最接近新的结构设想，但仍未通过严格 teacher gate，未解锁 generator，不能作为性能贡献。

## 4. Canonical variants 与比较方式

`MAIN_PROGRAM.md` 定义了 8 个 canonical variants：

| Variant | 含义 | 当前审计建议 |
| --- | --- | --- |
| `baseline` | 冻结离线模型或无在线 stage | 必须保留为所有在线比较基准。 |
| `stage1` | 只启用 Stage I | 可作为轻量校准消融。 |
| `stage2` | 只启用 Stage II | 需先解决 current-label acceptance / delayed feedback 风险。 |
| `stage3` | 只启用 Stage III | 当前旧表可保留为历史证据；新论文不应直接沿用未过 gate 的 Stage III 性能 claim。 |
| `stage1_stage2` | Stage I + II | 需要与 delayed-feedback protocol 一起重跑。 |
| `stage1_stage3` | Stage I + III | 可作为组合消融，但 Stage III 必须先定义新可服务协议。 |
| `stage2_stage3` | Stage II + III | 高风险组合，需先锁定 Stage II/III 各自协议。 |
| `stage1_stage2_stage3` | 完整控制器 | 旧稿写成“应优于 baseline”，但现有证据是 dataset-dependent，不应写成 universal strongest。 |

主指标应是在线阶段相对 `baseline` 的 `MSE`，尤其 `adapted_online_mse` 或 `online_stage*_mse`。副指标包括 MAE、offline `val_mse`、offline `test_mse`、online drift diagnostics、stability trigger rate、reject rate、gate pass、rolling validation pass、counterfactual margin、generator unlock status。

## 5. 证据链与非证据链分类

### 可作为论文证据链候选

- `logs/load_benchmarks/*.csv` 与对应 report/figure scripts；
- `logs/lishui_benchmarks/*.csv` 与 Lishui family/county figures；
- `FIGURE_MAP.md` 中列出的图像和生成脚本；
- `docs/audit/PAPER_CLAIMS_MATRIX.md` 与 `docs/audit/EXPERIMENT_EVIDENCE_LEDGER.md`；
- `docs/audit/STAGE3_*_RECORD.md` 中记录的命令、日志、checkpoint 和 gate 结论；
- `configs/best_load_online.yaml`, `configs/best_lishui_online.yaml`, `configs/stage3_v*.yaml`；
- `ckpt/` 与 `checkpoints/` 中被 run manifest 或审计文档引用的权重。

### 探索、demo、失败诊断或历史遗留

- `logs/s3v*smoke*`、`logs/s3v*dry*`、单 seed pilot、teacher-only abort run；
- `configs/tuning/*` 与 V29 tuning variants；
- `artifacts/stage3_v29_*` 中的 selector、mode-gate、queue、preflight 工件；
- `archive/references/*` 中的论文 PDF/TXT：只能作为阅读材料，不等同于正式引用；
- `archive/external_sources/*` 外部项目副本；
- 旧 Stage III generator/alignment/teacher 版本：不能删除，因为它们记录失败原因，但也不能默认作为新主线贡献。

## 6. 当前最保守的项目定位

已验证事实：仓库包含完整的在线负荷预测代码、论文草稿、多版本 Stage III 实验、日志、checkpoint 和审计矩阵；现有 evidence ledger 已清楚标出很多 Stage III 版本未通过 gate。

可解释但需作者确认：后续新研究路线最可能从 V29 的 expert-local invariant-release 结构吸取经验，但需要新 protocol 和新证据链，而不是直接把 V29 写成成功结果。

缺失证据：缺少每个论文表格/图像的唯一 manifest；缺少正式引用链；缺少 delayed-feedback leak-safe 重跑；缺少 faithful baseline 证明；缺少 multi-seed uncertainty/statistical testing。

