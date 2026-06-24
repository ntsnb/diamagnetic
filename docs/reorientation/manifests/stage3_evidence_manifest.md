# Stage III 证据链 Manifest

生成时间：2026-06-24

本 manifest 只登记位置和用途，不重述或强化实验结论。

## 不移动的核心证据资产

| 路径 | 当前作用 | 本次动作 | 备注 |
| --- | --- | --- | --- |
| `models/factor_mechanism.py` | 基础 factor-mechanism 模型。 | 保留原位 | 主线代码和历史兼容入口。 |
| `models/factor_mechanism_v16.py` 到 `models/factor_mechanism_v29.py` | Stage III 后期版本谱系。 | 保留原位 | V29 是当前最接近新设计参考线的版本，但不是已验证性能 claim。 |
| `models/adaptation.py`, `models/adaptation_v2.py`, `models/adaptation_v9.py`, `models/adaptation_v10.py` | 在线适配构造和历史适配器。 | 保留原位 | `adaptation_v2.py` 是多版 Stage III 构造关键入口。 |
| `models/stage3_alignment_v*.py` | V7-V12 左右的 alignment 探索组件。 | 保留原位 | 作为历史机制证据保留。 |
| `models/stage3_residual_teacher_v*.py` | residual teacher 探索组件。 | 保留原位 | 多数用于诊断/失败经验，不作为成功 claim。 |
| `configs/stage3_v*.yaml` | Stage III 版本配置。 | 保留原位 | 不移动，避免 run manifest 与配置路径断裂。 |
| `logs/load_benchmarks/stage3_v2/` | V2/V2.5 证据链。 | 保留原位 | 其中 V2.5 canonical ckpt-selection 可证明协议修复，不证明性能优势。 |
| `logs/s3v7*` 到 `logs/s3v29*` | Stage III V7-V29 日志谱系。 | 保留原位 | 多数为诊断、筛选、teacher gate、失败记录。 |
| `artifacts/stage3_v29_*` | V29 queue、selection、served mode gate、preflight 等治理产物。 | 保留原位 | 用于证明保守治理和 generator lock 状态，不等同预测性能结果。 |
| `ckpt/s3v7*` 到 `ckpt/s3v29*` | V7-V29 checkpoint 谱系。 | 保留原位 | 不移动大文件和 seed-specific artifact。 |
| `checkpoints/stage3_v2/` 到 `checkpoints/stage3_v6/` | 早期 Stage III checkpoint 谱系。 | 保留原位 | 与旧日志/manifest 绑定。 |

## 重要审计口径

| 版本/路线 | 证据价值 | 不可宣称内容 |
| --- | --- | --- |
| V2.5 canonical ckpt-selection | 配置、run manifest、checkpoint selection、gate 文件相对完整。 | 不宣称显著性能提升或可靠 drift capture。 |
| V10/V11 teacher matrix | 负证据和 gate failure 证据。 | 不宣称 teacher route 已成功。 |
| V13-V15 fluctuation/stability/target predictability | 失败诊断价值高。 | 不宣称 Stage III 已稳定改进。 |
| V18-V29 expert-local/release line | 记录了更系统的 gate、selector、counterfactual、mode governance 尝试。 | 不宣称 generator 已解锁或 Stage III 已验证成功。 |
| V29 official line | 当前新设计参考价值最高，尤其是 teacher-only gate、served mode gate、generator unlock preflight。 | 不把 `mode_selection_only_no_generator_unlock` 写成 forecasting result。 |

## V29 快照一致性注意事项

V29 后期意图使用 `pooled_servable_change` 语义。部分早期 V29 `resolved_config.json` 仍可能记录 `delta_phi_v`，不能与修复后的 `pooled_servable_change` 快照混列为同一证据口径。若论文引用 V29 anchor，应优先使用已修正快照，或重新跑一条 canonical anchor。

## 后续 manifest 建议

下一步可在不移动文件的前提下生成机器可读 CSV：

| 建议文件 | 字段 |
| --- | --- |
| `logs_manifest_stage3.csv` | `path,version,dataset,seed,run_type,gate_status,claim_scope` |
| `checkpoints_manifest_stage3.csv` | `path,version,dataset,seed,artifact_type,served_or_teacher_only,claim_scope` |
| `stage3_config_manifest.csv` | `config,version,drift_source,protocol_version,status` |
| `paper_table_figure_manifest.csv` | `paper_item,source_script,source_csv,source_log,source_checkpoint,claim_id` |
