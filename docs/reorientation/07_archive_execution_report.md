# 归档执行报告

生成时间：2026-06-24

## 执行范围

本次执行第二阶段归档，但采用保守策略：只新增归档 manifest 和执行报告，不移动任何现有文件。这样做是为了避免破坏配置、模型、日志、checkpoint、LaTeX 图像和本地工具中的路径引用。

## 实际移动的文件

无。

## 新增的索引文件

| 文件 | 作用 |
| --- | --- |
| `docs/reorientation/manifests/README.md` | manifest 目录说明和执行原则。 |
| `docs/reorientation/manifests/archive_decision_manifest.md` | 本次候选路径、处理动作和冲突风险。 |
| `docs/reorientation/manifests/stage3_evidence_manifest.md` | Stage III 相关模型、配置、日志、artifact、checkpoint 的证据链索引。 |
| `docs/reorientation/manifests/paper_path_manifest.md` | 论文主稿、衍生稿、图像和生成/审计脚本的路径依赖索引。 |

## 未移动的文件及原因

| 路径 | 未移动原因 |
| --- | --- |
| `bare_jrnl_new_sample4*.tex/.pdf`, `bare_jrnl_new_sample4.bbl`, `IEEEtran.cls` | 当前论文主稿、衍生稿和 LaTeX 构建上下文；审计工具仍会扫描衍生稿。 |
| `fig1.jpg`, `fig2.jpg` | 主稿和衍生稿直接引用，移动会破坏 LaTeX。 |
| `models/factor_mechanism*.py`, `models/adaptation*.py`, `models/stage3_*.py` | Stage III 版本谱系和失败/成功诊断证据。 |
| `configs/stage3_v*.yaml` | 与 run manifest、日志和 checkpoint 绑定，移动会破坏证据链。 |
| `logs/`, `ckpt/`, `checkpoints/`, `dataset/`, `data/` | 原始日志、原始结果、checkpoint、数据和 split 资产；原则上不移动。 |
| `artifacts/stage3_v29_*` | V29 governance、selection、mode gate、preflight 证据，审计文档已引用。 |
| `drawio2pptx.py` | 被 `.claude/skills/drawio2pptx/SKILL.md` 硬编码引用。 |
| `gen_flowcharts.py`, `gen_pptx_flowcharts.py`, `load_drift_analysis.py` | 可能仍支撑图示或漂移分析，未确认前只登记。 |
| `.claude/`, `.vscode/`, `.venv/`, `.pytest_cache/`, `__pycache__/` | 本地工具/环境/缓存目录，不属于科研证据归档；高噪声目录不深遍历。 |
| `pytest_tmp_stage3_review/` | 当前权限拒绝，未处理。 |

## 被保留的证据链

- `source code/config -> logs/results -> artifacts/checkpoints -> audit docs -> manuscript claims` 的原路径未改变。
- V2.5 canonical checkpoint-selection 证据链原位保留。
- V10/V11 teacher matrix 负证据原位保留。
- V13-V15 failure diagnostics 原位保留。
- V18-V29 expert-local/release 线的 logs、artifacts、ckpt 原位保留。
- V29 teacher-only gate、served mode gate、generator unlock preflight 证据原位保留。

## import / config / docs 更新

| 类型 | 更新情况 |
| --- | --- |
| Python import | 未修改。 |
| YAML config path | 未修改。 |
| model/checkpoint path | 未修改。 |
| LaTeX figure path | 未修改。 |
| docs | 新增本报告和 3 个 manifest；未改写既有审计结论。 |

## 验证命令

已运行：

```powershell
git status --short
python tools\audit_project_paths.py --strict
python tools\audit_latex_assets.py
python tools\audit_repository_layout.py --strict
rg -n --hidden -g '!logs/**' -g '!ckpt/**' -g '!checkpoints/**' -g '!dataset/**' -g '!data/**' -g '!archive/external_sources/**' -g '!.venv/**' -g '!__pycache__/**' -g '!.pytest_cache/**' "bare_jrnl_new_sample4_(audit_annotated|redline|revision_markup)|drawio2pptx|gen_flowcharts|gen_pptx_flowcharts|load_drift_analysis|fig1\.jpg|fig2\.jpg"
```

验证结果：

- `audit_project_paths.py --strict`: `legacy_absolute_path_hits=0`, `unresolved_data_paths=0`。
- `audit_latex_assets.py`: `checked_tex_files=5`, `missing_graphics=0`。
- `audit_repository_layout.py --strict`: `layout_problems=0`。
- `rg` 引用扫描发现图像、衍生稿和工具脚本仍存在路径依赖；同时 `pytest_tmp_stage3_review/` 权限拒绝，因此该目录未处理。

## 未运行的检查

- 未运行完整训练或在线评估，因为本次没有改代码、配置、数据或模型。
- 未运行 LaTeX 完整编译，因为没有移动或修改 `.tex`、`.cls`、`.bbl` 或图像文件。
- 未深度扫描 `logs/`、`ckpt/`、`checkpoints/`、`archive/external_sources/`、`.venv/`、缓存目录。

## 残余风险

- 当前仓库几乎全部为未跟踪文件，后续任何移动都应继续视为高风险。
- `pytest_tmp_stage3_review/` 权限拒绝，暂未判断内容。
- Stage III V29 部分早期快照可能存在 `drift_source` 口径不一致；论文引用前应重新确认或重跑 canonical anchor。
- 论文表格/图仍缺逐项 machine-readable manifest；当前只建立了高层路径 manifest。
- 根目录工具脚本是否仍用于当前论文图示或 case study，需要作者确认。

## 后续重构优先级

1. 明确唯一 canonical manuscript。
2. 为每张论文表/图建立 `source_script/source_csv/source_log/source_checkpoint/claim_id` manifest。
3. 建立 Stage III run-level CSV manifest，区分 positive evidence、negative evidence、diagnostic、smoke、governance。
4. 在新 namespace 下重设 clean protocol，而不是继续移动旧版本代码。
5. 设计新实验路线后，再决定哪些衍生稿或工具脚本可以物理归档。
