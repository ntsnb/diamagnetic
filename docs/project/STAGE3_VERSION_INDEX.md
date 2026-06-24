# Stage III 版本索引

本文档把 Stage III 多版本代码、配置、日志、checkpoint 和说明文档放到同一个索引中。它是查找入口，不替代原始日志和原始权重。

## 版本族

| 版本族 | 配置入口 | 代码入口 | 日志位置 | checkpoint 位置 | 文档记录 |
| --- | --- | --- | --- | --- | --- |
| V2 | `configs/stage3_v2_minimal.yaml`、`configs/stage3_v2_strong.yaml` | `train/run_stage3_v2_sensitivity_audit.py`、`train/online_loop_stage3_v2.py`、`train/stage3_v2_*.py` | `logs/load_benchmarks/stage3_v2/` | `checkpoints/stage3_v2/` | `docs/audit/STAGE3_V2_SENSITIVITY_LEDGER.md`、`docs/audit/STAGE3_V2_IMPLEMENTATION_LOG.md` |
| V3 | `configs/stage3_v3_strict.yaml` | `train/run_stage3_v2_sensitivity_audit.py` 及 V2 兼容模块 | `logs/load_benchmarks/stage3_v3/` | `checkpoints/stage3_v3/` | `docs/audit/STAGE3_V3_REPAIR_AND_SMOKE_RECORD.md` |
| V4 | `configs/stage3_v4_counterfactual_hard_negative.yaml` | `train/run_stage3_v2_sensitivity_audit.py` 及反事实 hard negative 配置路径 | `logs/load_benchmarks/stage3_v4/` | `checkpoints/stage3_v4/` | `docs/audit/STAGE3_V4_COUNTERFACTUAL_HARD_NEGATIVE_RECORD.md` |
| V5-V6 | `configs/stage3_v5_served_policy_counterfactual_gate.yaml`、`configs/stage3_v6_whitened_procrustes_served_policy_gate.yaml` | `train/run_stage3_v2_sensitivity_audit.py`、`train/stage3_sensitivity_gate_v2.py`、`train/stage3_risk_gate_v2.py` | `logs/load_benchmarks/stage3_v5/`、`logs/load_benchmarks/stage3_v6/` | `checkpoints/stage3_v5/`、`checkpoints/stage3_v6/` | `docs/audit/STAGE3_V5_SERVED_POLICY_COUNTERFACTUAL_GATE_RECORD.md`、`docs/audit/STAGE3_V5_V6_G5_CANARY_RECORD.md` |
| V7 | `configs/stage3_v7_residualized_rrcca_served_policy_gate.yaml` | `models/stage3_alignment_v7.py`、`tests/test_stage3_alignment_v7.py` | `logs/s3v7*` | `ckpt/s3v7*` | `docs/audit/STAGE3_V7_ALIGNMENT_RECORD.md` |
| V8 | `configs/stage3_v8_hyperadapter_residualized_rrcca_served_policy_gate.yaml` | `models/adaptation_v9.py`、`tests/test_hyperlora_generator_v2.py` 等相关模块 | `logs/s3v8*` | `ckpt/s3v8*` | `docs/audit/STAGE3_V8_HYPERADAPTER_RECORD.md` |
| V9 | `configs/stage3_v9_residual_teacher_rrcca_served_policy_gate.yaml` | `models/stage3_residual_teacher_v9.py`、`train/stage3_v9_teacher_diagnostics.py` | `logs/s3v9*` | `ckpt/s3v9*` | `docs/audit/STAGE3_V9_RESIDUAL_TEACHER_RECORD.md`、`docs/audit/STAGE3_V9_TEACHER_ONLY_DIAGNOSTIC_RECORD.md` |
| V10 | `configs/stage3_v10_crossfit_teacher_projection_served_policy_gate.yaml`、`configs/stage3_v10_crossfit_teacher_projection_signed_alpha_gate.yaml` | `models/stage3_residual_teacher_v10.py`、`models/stage3_alignment_v10.py`、`train/run_stage3_v10_*.py` | `logs/s3v10*` | `ckpt/s3v10*` | `docs/audit/STAGE3_V10_CROSSFIT_TEACHER_RECORD.md`、`docs/audit/STAGE3_V10_TEACHER_MATRIX_RECORD.md` |
| V11 | `configs/stage3_v11_block_calibrated_teacher_gate.yaml`、`configs/stage3_v11_block_horizon_teacher_gate.yaml` | `models/stage3_residual_teacher_v11.py`、`models/stage3_alignment_v11.py`、`train/run_stage3_v11_*.py` | `logs/s3v11*` | `ckpt/s3v11*` | `docs/audit/STAGE3_V11_SIGNAL_AND_HORIZON_ABLATION_RECORD.md`、`docs/audit/STAGE3_V11_TEACHER_MATRIX_RECORD.md` |
| V12 | `configs/stage3_v12_outcome_aligned_teacher_gate.yaml` | `models/stage3_residual_teacher_v12.py`、`models/stage3_alignment_v12.py`、`train/run_stage3_v12_*.py` | `logs/s3v12*` | `ckpt/s3v12*` | `docs/audit/STAGE3_V12_OUTCOME_ALIGNED_TRANSPORT_RECORD.md` |
| V12B-V12E | `configs/stage3_v12b_oof_teacher_gate.yaml`、`configs/stage3_v12c_multi_policy_teacher_probe*.yaml`、`configs/stage3_v12d_context_teacher_probe.yaml` | `models/stage3_residual_teacher_v12b.py`、`models/stage3_residual_teacher_v12c.py`、`models/stage3_residual_teacher_v12d.py`、`train/stage3_v12e_*.py` | `logs/s3v12b*`、`logs/s3v12c*`、`logs/s3v12d*` | `ckpt/s3v12b*`、`ckpt/s3v12c*`、`ckpt/s3v12d*` | `docs/audit/STAGE3_V12B...`、`docs/audit/STAGE3_V12C...`、`docs/audit/STAGE3_V12D_CONTEXT_TEACHER_RECORD.md`、`docs/audit/STAGE3_V12E_LABEL_AVAILABILITY_CONTEXT_TEACHER_RECORD.md` |
| V13 | `configs/stage3_v13_fluctuation_teacher_probe.yaml` | `models/stage3_residual_teacher_v13.py`、`train/run_stage3_v13_fluctuation_teacher_*.py` | `logs/s3v13fluct*` | `ckpt/s3v13fluct*` | `docs/audit/STAGE3_V13_FLUCTUATION_DESIGN_AND_EXPERIMENT_RECORD.md`、`docs/audit/stage3_v13_pilot_failure_diagnostics/` |
| V14 | `configs/stage3_v14_stability_teacher_probe.yaml` | `models/stage3_residual_teacher_v14.py`、`train/run_stage3_v14_stability_teacher_*.py` | `logs/s3v14stability*` | `ckpt/s3v14stability*` | `docs/audit/STAGE3_V14_STABILITY_FLUCTUATION_READOUT_RECORD.md`、`docs/audit/STAGE3_V14_ROOT_CAUSE_AND_NEXT_DIRECTION.md`、`docs/audit/stage3_v14_pilot_failure_diagnostics/` |
| V15 | `configs/stage3_v15_residual_target_predictability_probe.yaml` | `train/run_stage3_v15_residual_target_predictability_*.py`、`tests/test_stage3_v15_residual_target_predictability.py` | `logs/s3v15target*` | `ckpt/s3v15target*` | `docs/audit/STAGE3_V15_RESIDUAL_TARGET_PREDICTABILITY_RECORD.md` |
| V21/V23/V25 记录 | 主要继承 V2-V6 族实验结构 | 相关脚本需从对应记录反查 | `logs/load_benchmarks/stage3_v2/` 下可见多条 `codex_stage3_v21_*` 等 run | 主要在 `checkpoints/stage3_v2/` 或记录中指定 | `docs/audit/STAGE3_V21_SPLITSAFE_EXPERIMENT_RECORD.md`、`docs/audit/STAGE3_V23_SERVING_ALIGNED_EXPERIMENT_RECORD.md`、`docs/audit/STAGE3_V25_*.md` |

## 使用方式

查找某个实验时，按以下顺序走：

1. 先看本索引确定版本族。
2. 再看对应 `docs/audit/STAGE3_*.md` 记录结论和运行命令。
3. 再到 `logs/` 找原始 CSV、JSON、manifest。
4. 如需复现实验，再用对应 `configs/` 和 `train/run_stage3_*.py`。
5. 如需解释 checkpoint，再按同一版本族查 `ckpt/` 或 `checkpoints/`。

## 重要限制

本索引不声明任何实验结果是否支持论文创新点。实验结论仍以 `docs/audit/EXPERIMENT_EVIDENCE_LEDGER.md`、`docs/audit/PAPER_CLAIMS_MATRIX.md` 和具体 `STAGE3_*_RECORD.md` 为准。
