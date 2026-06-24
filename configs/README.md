# Config Directory Guide

本目录保存实验配置。新配置应使用仓库相对路径，不应写入用户机器绝对路径。

## 基础配置

| 文件 | 用途 |
| --- | --- |
| `base.yaml` | 单次实验基础配置。 |
| `online.yaml` | 在线学习相关基础配置。 |
| `model.yaml` | 模型相关配置片段。 |
| `data.yaml` | 数据相关配置片段。 |
| `best_online.yaml` | 自动搜索后的早期 best 配置。 |
| `best_load_online.yaml` | 负荷预测公开数据主配置。 |
| `best_lishui_online.yaml` | 丽水数据相关主配置。 |
| `autoresearch_main.yaml` | 自动搜索配置。 |

## Stage III 配置族

| 版本族 | 配置文件 |
| --- | --- |
| V2 | `stage3_v2_minimal.yaml`、`stage3_v2_strong.yaml` |
| V3 | `stage3_v3_strict.yaml` |
| V4 | `stage3_v4_counterfactual_hard_negative.yaml` |
| V5 | `stage3_v5_served_policy_counterfactual_gate.yaml` |
| V6 | `stage3_v6_whitened_procrustes_served_policy_gate.yaml` |
| V7 | `stage3_v7_residualized_rrcca_served_policy_gate.yaml` |
| V8 | `stage3_v8_hyperadapter_residualized_rrcca_served_policy_gate.yaml` |
| V9 | `stage3_v9_residual_teacher_rrcca_served_policy_gate.yaml` |
| V10 | `stage3_v10_crossfit_teacher_projection_served_policy_gate.yaml`、`stage3_v10_crossfit_teacher_projection_signed_alpha_gate.yaml` |
| V11 | `stage3_v11_block_calibrated_teacher_gate.yaml`、`stage3_v11_block_horizon_teacher_gate.yaml` |
| V12 | `stage3_v12_outcome_aligned_teacher_gate.yaml` |
| V12B | `stage3_v12b_oof_teacher_gate.yaml`、`stage3_v12b_oof_teacher_probe_loose_alpha.yaml` |
| V12C | `stage3_v12c_multi_policy_teacher_probe.yaml`、`stage3_v12c_multi_policy_teacher_probe_nonnegative.yaml` |
| V12D | `stage3_v12d_context_teacher_probe.yaml` |
| V13 | `stage3_v13_fluctuation_teacher_probe.yaml` |
| V14 | `stage3_v14_stability_teacher_probe.yaml` |
| V15 | `stage3_v15_residual_target_predictability_probe.yaml` |

更完整的版本-代码-日志-checkpoint 对照见 `docs/project/STAGE3_VERSION_INDEX.md`。

## Tuning

`tuning/` 保存候选配置。它们属于参数筛选和模型族对比，不应被当作最终论文协议，除非对应实验记录和证据台账明确引用。
