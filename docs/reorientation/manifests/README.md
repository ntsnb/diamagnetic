# 归档执行 Manifest

生成时间：2026-06-24

本目录记录第二阶段归档执行的索引结果。当前执行策略是优先建立 manifest，避免移动会破坏路径、配置、模型、日志、checkpoint 或 LaTeX 引用的文件。

## 文件

| 文件 | 作用 |
| --- | --- |
| `archive_decision_manifest.md` | 记录本次归档候选、处理动作、未移动原因和冲突风险。 |
| `stage3_evidence_manifest.md` | 登记 Stage III 相关模型、配置、日志、artifact、checkpoint 的证据链位置。 |
| `paper_path_manifest.md` | 登记论文主稿、修订稿、图像和相关生成/审计脚本的路径依赖。 |

## 执行原则

- 不移动 `logs/`、`ckpt/`、`checkpoints/`、`dataset/`、`data/`。
- 不移动 `configs/stage3_v*.yaml` 或 `models/factor_mechanism*.py`、`models/adaptation*.py`、`models/stage3_*.py`。
- 不移动当前 LaTeX 主稿、PDF、`.bbl`、`IEEEtran.cls`、`fig1.jpg`、`fig2.jpg`。
- 对有硬编码引用、审计工具引用或潜在 figure/table 生成用途的脚本，仅登记，不移动。
- 对 `archive/external_sources/`、`.venv/`、`.pytest_cache/`、`__pycache__/`、`pytest_tmp_stage3_review/` 不做深度遍历。
