# 文件盘点表

生成时间：2026-06-24  
盘点原则：不深度遍历 `archive/external_sources/`、`.venv/`、`__pycache__/`、`.pytest_cache/`、TensorBoard event 内容和大权重文件。只记录目录级用途、风险和建议去向。

## 1. 目录规模概览

| 路径 | 文件数 | 约大小 | 审计含义 |
| --- | ---: | ---: | --- |
| `data/` | 9 | <1 MB | 数据读取、窗口、归一化代码。 |
| `models/` | 42 | <1 MB | 基础模型和 Stage III 多版本模型。 |
| `train/` | 175 | ~2.7 MB | 训练、在线评估、Stage III 多版本实验、报告脚本。 |
| `utils/`, `losses/` | 15 | <1 MB | 通用工具和损失。 |
| `configs/` | 116 | <1 MB | 基础配置、Stage III 配置、tuning 配置。 |
| `tests/` | 107 | <1 MB | V2-V29 多版本协议/安全/回归测试。 |
| `docs/` | 117 | ~2 MB | 审计、项目政策、superpowers specs/plans。 |
| `logs/` | 9918 | ~1.5 GB | 原始实验日志、CSV、JSON、图片、TensorBoard。 |
| `artifacts/` | 448 | ~10 MB | 可再生成产物、LaTeX build、V29 selector/mode-gate 等。 |
| `ckpt/` | 2393 | ~3.8 GB | V7 后 Stage III checkpoint。 |
| `checkpoints/` | 970 | ~5.0 GB | 旧 Stage III V2-V6 和通用 checkpoint。 |
| `dataset/` | 18 | ~67 MB | 原始/benchmark 数据。 |
| `archive/` | 21（不含 external_sources 深遍历） | ~766 MB | 已归档参考、模板、外部压缩包、外部源码入口。 |

## 2. 分类盘点

| 类别 | 路径 | 当前作用 | 是否还在主线中 | 证据价值 | 是否可移动 | 移动风险 | 建议去向 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Active source code | `data/`, `models/`, `train/`, `utils/`, `losses/`, `main.py`, `run.sh`, `run.ps1` | 训练、评估、在线适配、报告生成。 | 是，但内部多版本混杂。 | 高。代码是所有实验和论文主张的上游。 | 不建议第一阶段移动。 | import 断裂、旧日志 manifest 无法解释、测试路径失效。 | 原位保留；后续重构另建新命名空间或兼容层。 |
| Active configs | `configs/base.yaml`, `configs/best_load_online.yaml`, `configs/best_lishui_online.yaml`, `configs/stage3_v29_expert_local_invariant_release_probe.yaml` | 基础实验、公有 benchmark、Lishui、V29 探索入口。 | 是。 | 高。对应协议、路径、超参。 | 不建议移动。 | 路径引用和 run manifest 断裂。 | 原位保留；新路线另建 `configs/redesign/` 或 `configs/v30_*`。 |
| Tuning configs | `configs/tuning/`, `configs/stage3_v*_tune*.yaml` | 参数筛选、候选矩阵、负结果定位。 | 非最终主线。 | 中高。说明哪些方向失败。 | 不建议直接移动；可登记索引。 | 与日志/审计记录按文件名对应。 | 原位保留，建立 manifest 标注 tuning-only。 |
| Paper/manuscript files | `bare_jrnl_new_sample4*.tex`, `bare_jrnl_new_sample4*.pdf`, `bare_jrnl_new_sample4.bbl`, `IEEEtran.cls`, `fig1.jpg`, `fig2.jpg` | 活动主稿、审计稿、红线稿、修订标注稿、模板类、静态图。 | 是，至少仍是论文工作入口。 | 高。 | 需要作者确认后才可移动衍生稿；主稿和当前引用图不动。 | LaTeX 编译路径、人工查阅入口、图像引用断裂。 | 根目录保留；确认 canonical manuscript 后再决定衍生稿是否登记或归档。 |
| Audit docs | `docs/audit/`, `docs/project/`, `docs/reorientation/` | 科研完整性、证据台账、路径政策、本次重定向审计。 | 是。 | 很高。 | 不移动。 | 证据链解释断裂。 | 原位保留，后续修改 claims 时同步更新。 |
| Superpowers specs/plans | `docs/superpowers/specs/`, `docs/superpowers/plans/` | V13-V29 设计和执行计划。 | 非执行主线，但解释版本动机。 | 高。 | 不移动。 | 版本谱系缺失。 | 原位保留。 |
| Experiment logs | `logs/load_benchmarks/`, `logs/lishui_benchmarks/`, `logs/s3v*/`, `logs/tensorboard/` | benchmark、Stage III 多版本、teacher gates、figures、TensorBoard。 | 部分主线、部分失败诊断。 | 极高。 | 原则上不可移动。 | 原始证据断裂、manifest 路径失效、GB 级移动风险。 | 原位保留；只建索引/manifest。 |
| Generated artifacts | `artifacts/autoresearch/`, `artifacts/latex/build/`, `artifacts/stage3_v29_*` | 可再生成 TSV、LaTeX build、V29 selector/mode-gate/preflight。 | 多为辅助和治理。 | 中高。 | 低风险可移动但无必要；V29 artifacts 已被 ledger 引用则不动。 | 文档链接失效；selector replay 证据不清。 | 原位保留；区分 reproducible artifact 与 raw evidence。 |
| Checkpoints/ckpt | `ckpt/`, `checkpoints/` | 模型权重、teacher/canonicalizer/generator artifacts。 | 部分主线，部分历史。 | 极高。 | 不移动。 | 5-9GB 级别，路径过长和引用断裂风险极高。 | 原位保留；建立 manifest，不搬。 |
| Data | `dataset/`, `dataset/load_benchmarks/`, `data/` | 原始 benchmark 数据和数据代码。 | 是。 | 极高。 | 不移动。 | split、path、encoding、reproducibility 断裂。 | 原位保留。 |
| Tests | `tests/` | V2-V29 协议、安全、回归测试。 | 是，尤其对重构有保护价值。 | 高。 | 不移动。 | 失去旧行为保护。 | 原位保留；重构时新增新测试，不删除旧测试。 |
| Tools | `tools/audit_project_paths.py`, `tools/audit_latex_assets.py`, `tools/audit_repository_layout.py` | 路径、LaTeX asset、layout 审计。 | 是。 | 中高。 | 不移动。 | 验证命令失效。 | 原位保留。 |
| Existing archive | `archive/references/`, `archive/latex_templates/`, `archive/external/`, `archive/external_sources/` | 已归档参考资料、IEEE how-to、外部压缩包、外部源码副本。 | 否。 | 参考/背景价值，不是实验主证据。 | 已是归档位置。 | 深遍历噪声高。 | 不重复创建同类归档；保留 README 规则。 |
| Unknown / needs author decision | `.claude/`, `.vscode/`, `.python-version`, `.gitignore`, `drawio2pptx.py`, `gen_flowcharts.py`, `gen_pptx_flowcharts.py`, `load_drift_analysis.py`, `pytest_tmp_stage3_review/` | 本地工具、编辑器状态、临时目录、图生成/分析脚本。 | 不明确。 | 低到中。 | 需要确认。 | 可能仍被作者工作流使用；`pytest_tmp_stage3_review/` 当前权限拒绝。 | 第二阶段前先问作者；不主动移动。 |

## 3. 主线与证据价值判定

### 主线中应保留

- 基础数据和训练路径：`data/`, `train/run_experiment.py`, `train/offline_factor.py`, `train/offline_invariant.py`, `train/online_loop.py`。
- 论文活动稿和当前引用图片：`bare_jrnl_new_sample4*.tex/pdf`, `fig1.jpg`, `fig2.jpg`。
- benchmark 与 Lishui 结果源：`logs/load_benchmarks/`, `logs/lishui_benchmarks/`。
- 审计矩阵和证据台账：`docs/audit/PAPER_CLAIMS_MATRIX.md`, `docs/audit/EXPERIMENT_EVIDENCE_LEDGER.md`。
- V29 结构参考：`models/factor_mechanism_v29.py`, `train/run_stage3_v29_expert_local_invariant_release_probe.py`, V29 config 和 docs。

### 只作探索/失败诊断

- Stage III V2-V15 的多版本 teacher/generator/alignment 运行。
- V16-V28 的逐步结构探索。
- V29 tuning queue、selector、mode-gate replay、generator-unlock preflight。
- smoke/dryrun/pilot directories。

### 不建议移动但建议索引

- 所有 `logs/**`；
- 所有 `ckpt/**` 和 `checkpoints/**`；
- 所有 `dataset/**`；
- 所有 `configs/stage3_v*.yaml`；
- 所有 `docs/audit/STAGE3_*.md`；
- 所有被 `FIGURE_MAP.md` 或 LaTeX 引用的 figures。

## 4. 需要作者决策的路径

1. `bare_jrnl_new_sample4_audit_annotated.*`、`bare_jrnl_new_sample4_redline.*`、`bare_jrnl_new_sample4_revision_markup.*` 是否继续视为活动论文材料。
2. `.claude/`、`.vscode/` 是否应保留为本地工作区配置。
3. `drawio2pptx.py`、`gen_flowcharts.py`、`gen_pptx_flowcharts.py` 是否仍用于当前论文图示。
4. `load_drift_analysis.py` 是否是当前分析脚本还是历史临时脚本。
5. `pytest_tmp_stage3_review/` 权限拒绝，第二阶段前不应触碰。

