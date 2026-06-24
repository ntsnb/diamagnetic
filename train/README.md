# Train Directory Guide

本目录保存训练、在线评估、Stage III 多版本实验和报表生成脚本。

## 主入口

| 文件 | 用途 |
| --- | --- |
| `run_experiment.py` | 单次训练和可选在线适配主入口。 |
| `offline_factor.py` | 离线预测模型训练和评估。 |
| `offline_invariant.py` | 不变表示分支训练。 |
| `online_loop.py` | 旧版 Stage I/II/III 在线适配循环。 |
| `online_adapt.py` | 在线延迟反馈和稳定性工具。 |

## Stage III 多版本入口

| 版本族 | 主要脚本 |
| --- | --- |
| V2-V6 | `run_stage3_v2_sensitivity_audit.py`、`online_loop_stage3_v2.py`、`stage3_v2_*.py` |
| V10-V12 | `run_stage3_v10_*.py`、`run_stage3_v11_*.py`、`run_stage3_v12_*.py` |
| V12B-V12E | `run_stage3_v12b_*.py`、`run_stage3_v12c_*.py`、`run_stage3_v12d_*.py`、`stage3_v12e_*.py` |
| V13 | `run_stage3_v13_fluctuation_teacher_probe.py`、`run_stage3_v13_fluctuation_teacher_matrix.py` |
| V14 | `run_stage3_v14_stability_teacher_probe.py`、`run_stage3_v14_stability_teacher_matrix.py` |
| V15 | `run_stage3_v15_residual_target_predictability_probe.py`、`run_stage3_v15_residual_target_predictability_matrix.py` |

## 报告和分析脚本

- `report_*.py` 生成论文表格、图像或实验摘要。
- `analyze_*.py` 做案例分析、t-SNE、shift score 或失败诊断。
- `run_*benchmark*.py` 和 `run_*ablation*.py` 是批量实验入口。

移动本目录文件前，必须先确认对应 `configs/`、`tests/`、`docs/audit/` 和 `logs/` 路径引用。
