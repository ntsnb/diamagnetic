# 论文路径 Manifest

生成时间：2026-06-24

## 论文文件

| 路径 | 当前作用 | 本次动作 | 路径依赖 |
| --- | --- | --- | --- |
| `bare_jrnl_new_sample4.tex` | 当前活动主稿。 | 保留原位 | 直接引用 `fig1.jpg`、`fig2.jpg`。 |
| `bare_jrnl_new_sample4.pdf` | 当前活动主稿 PDF。 | 保留原位 | 与主稿人工对照。 |
| `bare_jrnl_new_sample4.bbl` | 当前 bibliography 产物。 | 保留原位 | 虽然 bibliography 为空框架，但仍是 LaTeX 构建上下文。 |
| `bare_jrnl_new_sample4_audit_annotated.tex/.pdf` | 审计标注稿。 | 保留原位 | 被 `tools/audit_latex_assets.py` 检查。 |
| `bare_jrnl_new_sample4_redline.tex/.pdf` | 红线问题稿。 | 保留原位 | 被 `tools/audit_latex_assets.py` 检查。 |
| `bare_jrnl_new_sample4_revision_markup.tex/.pdf` | 修订建议稿。 | 保留原位 | `docs/audit/generate_revision_markup_tex.py` 输出目标；被 `tools/audit_latex_assets.py` 检查。 |
| `IEEEtran.cls` | IEEE LaTeX class。 | 保留原位 | 主稿编译依赖。 |
| `fig1.jpg`, `fig2.jpg` | 当前主稿和衍生稿图像。 | 保留原位 | 多个 `.tex` 直接 `\includegraphics`。 |

## 相关脚本和文档

| 路径 | 当前作用 | 本次动作 | 原因 |
| --- | --- | --- | --- |
| `FIGURE_MAP.md` | 图表来源索引。 | 保留原位 | 后续 claim tracing 需要更新，不移动。 |
| `docs/audit/generate_audit_annotated_tex.py` | 生成审计标注稿。 | 保留原位 | 输出根目录衍生稿。 |
| `docs/audit/generate_revision_markup_tex.py` | 生成 revision markup 稿。 | 保留原位 | 输出根目录衍生稿。 |
| `tools/audit_latex_assets.py` | 检查主稿和衍生稿图像路径。 | 保留原位 | 本轮验证命令依赖。 |
| `gen_flowcharts.py`, `gen_pptx_flowcharts.py` | 流程图/PPT 图示生成候选。 | 仅登记 | 未确认是否仍支撑论文图示。 |
| `load_drift_analysis.py` | drift 分析候选脚本。 | 仅登记 | 未确认是否支撑 case study 或图表。 |

## 结论

当前论文路径不适合物理归档。应先确认下一轮 canonical manuscript，再决定是否将审计稿、红线稿、revision markup 稿迁入 `archive/manuscript_reviews/<date>/`。若未来移动，必须同步更新：

- `tools/audit_latex_assets.py`
- `docs/audit/generate_*_tex.py`
- `FIGURE_MAP.md`
- 主稿和衍生稿中的 `\includegraphics`
- `docs/audit/PAPER_CLAIMS_MATRIX.md`
- `docs/audit/EXPERIMENT_EVIDENCE_LEDGER.md`
