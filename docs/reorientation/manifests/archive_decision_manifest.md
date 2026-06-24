# 归档决策 Manifest

生成时间：2026-06-24

## 本次实际移动

无。

原因：执行前引用扫描和现有路径策略显示，当前根目录剩余候选都存在配置、模型、论文、工具或本地工作流冲突风险。为保护证据链，本次归档执行改为新增 manifest 和执行报告，不做物理移动。

## 候选处理结果

| 候选路径 | 本次动作 | 决策理由 | 后续条件 |
| --- | --- | --- | --- |
| `bare_jrnl_new_sample4.tex`, `bare_jrnl_new_sample4.pdf` | 保留原位 | 当前活动主稿和 PDF。 | 不建议移动。 |
| `bare_jrnl_new_sample4_audit_annotated.*` | 保留原位 | `tools/audit_latex_assets.py` 会扫描；仍是审计材料。 | 确认 canonical manuscript 后再决定是否归档。 |
| `bare_jrnl_new_sample4_redline.*` | 保留原位 | `tools/audit_latex_assets.py` 会扫描；仍是修订对照材料。 | 确认不再用于审稿/对照后才移动。 |
| `bare_jrnl_new_sample4_revision_markup.*` | 保留原位 | `docs/audit/generate_revision_markup_tex.py` 输出到根目录；仍可能是下一轮修订入口。 | 确认下一轮主稿后再决定。 |
| `fig1.jpg`, `fig2.jpg` | 保留原位 | 主稿和三个衍生稿均直接 `\includegraphics` 引用。 | 只有同步改 LaTeX 并编译通过后才能移动。 |
| `IEEEtran.cls`, `bare_jrnl_new_sample4.bbl` | 保留原位 | LaTeX 编译依赖和当前 bibliography 产物。 | 不建议移动。 |
| `drawio2pptx.py` | 保留原位 | `.claude/skills/drawio2pptx/SKILL.md` 硬编码引用该路径。 | 先改本地 skill 路径并验证后才可移动。 |
| `gen_flowcharts.py`, `gen_pptx_flowcharts.py` | 仅登记 | 可能仍用于论文流程图或 PPT 图示生成。 | 作者确认不再使用后可归档到 `archive/tools_legacy/<date>/`。 |
| `load_drift_analysis.py` | 仅登记 | 可能仍是 drift/case-study 分析脚本。 | 确认不支撑论文图表后才移动。 |
| `.claude/`, `.vscode/`, `.python-version`, `.gitignore` | 保留原位 | 本地工作流/环境配置，非研究证据但移动会影响工具。 | 建议后续用 ignore/本地策略处理，不纳入科研 archive。 |
| `.pytest_cache/`, `__pycache__/` | 保留原位 | 缓存目录；用户要求不深遍历高噪声目录。 | 可后续清理，但不属于证据归档。 |
| `.venv/` | 保留原位 | 环境目录；不深遍历。 | 不建议归档。 |
| `pytest_tmp_stage3_review/` | 不处理 | `git status` 和 `rg` 均遇到权限拒绝。 | 需作者手动确认权限和用途。 |
| `logs/`, `ckpt/`, `checkpoints/`, `dataset/`, `data/` | 保留原位，只登记 | 原始证据资产和数据路径。 | 原则上不移动。 |
| `configs/stage3_v*.yaml` | 保留原位，只登记 | 与 run manifest、日志和 checkpoint 绑定。 | 不移动；新设计另建 clean configs。 |
| `models/factor_mechanism*.py`, `models/adaptation*.py`, `models/stage3_*.py` | 保留原位，只登记 | 版本谱系和失败实验证据。 | 不移动；后续通过新 namespace 重构。 |
| `archive/external_sources/` | 保留原位 | 已归档外部副本；本轮不深遍历。 | 不重复归档。 |

## 路径冲突结论

- 论文图像仍由主稿和修订衍生稿直接引用，不能移动。
- 修订衍生稿仍被 LaTeX 资产审计工具纳入检查，不能静默移动。
- `drawio2pptx.py` 被本地 skill 硬编码引用，不能移动。
- Stage III 配置、模型、日志、artifact、checkpoint 之间存在运行证据链绑定，只能登记，不能拆开移动。
