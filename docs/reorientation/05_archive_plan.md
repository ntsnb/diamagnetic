# 归档方案（不执行）

生成时间：2026-06-24  
状态：方案阶段。未移动、未删除、未重命名任何文件。

## 1. 现有 archive 结构

当前 `archive/README.md` 已定义归档范围：保存不属于当前活动代码路径、活动论文主稿或原始实验资产的材料。

现有子目录：

| 路径 | 内容 | 处理建议 |
| --- | --- | --- |
| `archive/references/papers/` | 参考论文 PDF | 保留，不深度移动；引用前必须进入 LaTeX/BibTeX。 |
| `archive/references/extracted_text/` | 参考论文文本提取 | 保留，不能替代正式引用。 |
| `archive/latex_templates/ieee_howto/` | IEEE how-to 模板 | 已归档，勿重复建模板归档。 |
| `archive/external/` | 外部压缩包 | 保留。 |
| `archive/external_sources/` | `core/`, `CLI-Anything/`, `autoresearch-master/` | 不深遍历；保留为外部项目副本。 |

因此第二阶段不应盲目创建新的 `references`、`external_sources`、`latex_templates` 目录，也不应把实验日志或 checkpoint 移入 `archive/`。

## 2. 建议留在主线的文件

| 路径 | 理由 |
| --- | --- |
| `bare_jrnl_new_sample4.tex`, `bare_jrnl_new_sample4.pdf`, `IEEEtran.cls`, `fig1.jpg`, `fig2.jpg` | 活动稿和当前引用资源。 |
| `bare_jrnl_new_sample4_audit_annotated.*`, `bare_jrnl_new_sample4_redline.*`, `bare_jrnl_new_sample4_revision_markup.*` | 审稿/修订材料，是否归档需作者确认。 |
| `data/`, `models/`, `train/`, `utils/`, `losses/` | 活动代码和历史证据代码。 |
| `configs/` | 所有配置与日志证据链绑定。 |
| `tests/` | 重构保护和协议验证资产。 |
| `docs/audit/`, `docs/project/`, `docs/superpowers/`, `docs/reorientation/` | 科研证据链与设计记录。 |
| `logs/`, `ckpt/`, `checkpoints/`, `dataset/` | 原始证据资产，原则上不移动。 |
| `artifacts/stage3_v29_*` | 已被 V29 审计记录引用，建议保留并索引。 |

## 3. 建议归档但需作者确认的候选

| 候选 | 可能去向 | 是否建议移动 | 风险 |
| --- | --- | --- | --- |
| 非 canonical 的 LaTeX 修订衍生稿 | 可选：`archive/manuscript_reviews/<date>/` | 暂不执行，先确认 canonical manuscript | 根目录人工查阅入口可能改变；LaTeX/PDF 对照关系可能混乱。 |
| 过时的一次性流程图脚本或 drawio 工具 | 可选：`archive/tools_legacy/<date>/` | 需确认不再使用 | 可能仍用于论文图示。 |
| `load_drift_analysis.py` 等根目录分析脚本 | 可选：登记到 docs，不优先移动 | 需确认 | 可能仍是某个 figure/table 生成链。 |
| 本地 IDE/agent 配置 | 通常不归档，可能保留或加入 ignore | 需确认 | 可能是用户工作流配置。 |
| `pytest_tmp_stage3_review/` | 不处理 | 不建议动 | 当前权限拒绝，且可能是临时/锁定目录。 |

## 4. 只建议登记索引，不建议移动的内容

| 路径 | 方案 |
| --- | --- |
| `logs/**` | 建立 `docs/reorientation/manifests/log_manifest_stage3.md` 或 CSV 索引，记录 run id、版本、数据集、seed、gate status、用途。 |
| `ckpt/**`, `checkpoints/**` | 建立 checkpoint manifest，记录版本、run id、是否 generator served、是否 teacher-only、是否应引用。 |
| `configs/stage3_v*.yaml` | 建立 config lineage index，而不是移动 tuning configs。 |
| `docs/audit/STAGE3_*.md` | 在版本谱系中交叉引用，不移动。 |
| `artifacts/stage3_v29_*` | 标注为 governance/selector/mode-gate/preflight，不移动。 |
| `logs/tensorboard/**` | 只登记目录，不读取 event bodies。 |
| `dataset/**` | 只登记数据来源、编码、用途，不移动。 |

V29 manifest 应额外标注 `resolved_config.json` 中的 `online.stage3.drift_source`。对于仍显示 `delta_phi_v` 的早期快照，不应与修复后 `pooled_servable_change` 结果混列为同一证据口径。

## 5. 大文件和 checkpoint 处理原则

不移动：

- `ckpt/**`；
- `checkpoints/**`；
- raw `logs/**`；
- raw `dataset/**`；
- trained artifacts；
- seed-specific run directories；
- LaTeX 当前引用 figures。

原因：

1. 它们合计接近 10GB，移动成本高；
2. 许多 run manifest、审计文档和脚本直接引用原路径；
3. Windows 路径长度问题已经在 V29 审计中出现过；
4. 移动会破坏“代码/config/log/checkpoint/论文 claim”的可追溯链。

建议只生成 manifest：

```text
docs/reorientation/manifests/
  logs_manifest.md or .csv
  checkpoints_manifest.md or .csv
  stage3_gate_manifest.md or .csv
  paper_figures_manifest.md
```

这些 manifest 可在第二阶段作为低风险产物生成，但仍不搬大文件。

## 6. 必须由作者确认后才能动的路径

1. 任何 `bare_jrnl_new_sample4*.tex` 或 `.pdf`；
2. `fig1.jpg`, `fig2.jpg`；
3. `logs/`, `ckpt/`, `checkpoints/`, `dataset/`；
4. `configs/stage3_v*.yaml`；
5. `docs/audit/` 中任何已被 claims matrix 引用的审计记录；
6. `archive/external_sources/` 中外部项目副本；
7. `pytest_tmp_stage3_review/`；
8. 根目录工具脚本，除非确认不再被 figure/table generation 使用。

## 7. 第二阶段低风险执行建议

如果作者明确回复“开始归档”，建议执行顺序如下：

1. 运行 `git status --short`；
2. 再次确认 only-additive 输出范围；
3. 生成 manifest，不移动大文件；
4. 只对作者确认的低风险、非证据资产执行移动；
5. 移动后更新路径索引，不改实验结果；
6. 运行路径审计：
   - `python tools/audit_project_paths.py --strict`
   - `python tools/audit_latex_assets.py`
   - `python tools/audit_repository_layout.py --strict`
7. 若移动了脚本或 LaTeX 资源，再运行相关测试或 LaTeX build。

## 8. 归档后可能需要更新的引用

若未来真的移动文件，必须同步检查：

- Python import；
- shell/PowerShell run scripts；
- config path；
- `FIGURE_MAP.md`；
- `docs/audit/EXPERIMENT_EVIDENCE_LEDGER.md`；
- `docs/audit/PAPER_CLAIMS_MATRIX.md`；
- LaTeX `\includegraphics{...}`；
- report scripts 的默认 input/output path；
- tests 中 hard-coded path；
- run manifests 中是否需要兼容说明。

## 9. 当前方案摘要

第一阶段结论是不建议立即执行实质性搬家。最稳妥的归档策略是：

- 原始证据资产原位保留；
- 旧版本和失败实验不删除；
- 用 manifest 区分主线、探索、失败诊断、可再生成产物；
- 只在作者确认后处理非 canonical manuscript 衍生稿和少量根目录临时脚本；
- 先建立新研究路线的 clean protocol，再考虑是否做物理目录重构。
