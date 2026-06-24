# 模型版本谱系

生成时间：2026-06-24  
范围：只读梳理 `models/factor_mechanism*.py`、`models/adaptation*.py`、`models/stage3_alignment_v*.py`、`models/stage3_residual_teacher_v*.py`、`configs/stage3_v*.yaml`、`docs/superpowers/specs/`、`docs/superpowers/plans/`、`logs/`、`artifacts/`、`ckpt/`、`checkpoints/` 的版本关系。

## 1. 总体演化图

```text
legacy Stage III
  -> V2/V2.5: deterministic eval, zero-drift identity, split-safe protocol, rolling/counterfactual gate
  -> V3-V6: stricter counterfactual, served policy, risk gate, Procrustes/whitened variants
  -> V7-V8: residualized RRCCA and larger HyperAdapter capacity
  -> V9-V12: residual teacher, cross-fit calibration, block/horizon teacher, outcome-aligned transport
  -> V12b-V12e: OOF teacher, policy diagnostics, history controls, per-horizon label availability
  -> V13-V15: fluctuation-first, stability-first readout, residual-target predictability diagnostics
  -> V16-V28: expert-local / env / release / observable-change / multiscale / support-suppression attempts
  -> V29: expert-local invariant-release with native pooled_servable_change and gate-aware tuning governance
```

关键结论：版本演进不是线性性能提升，而是不断发现 Stage III 的可归因性、延迟反馈、history-only 对照、counterfactual safety 和 generator unlock 问题。V29 是结构上最接近新设计方向的版本，但现有记录仍判定为 gate-negative，不能晋升为论文性能证据。

## 2. 基础版本族

| 版本族 | 目标 | 关键机制 | 对应配置 | 对应日志/checkpoint | 成功点 | 失败点 | 新设计启发 | 建议状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Legacy | 在主在线循环中提供 Stage I/II/III 雏形 | `models/adaptation.py` 的 `AdaptationGenerator`；`train/online_loop.py` 中 transient update | `configs/base.yaml`, `configs/online.yaml` | `logs/offline_metrics.csv`, `logs/test_metrics.csv`, `logs/online_compare.csv`, `checkpoints/*.pt` | 建立了 8 variants 和基础在线评估入口 | 原始 Stage III generator 学习链不足；delayed feedback 叙述过强 | 保留作为历史基线，不再作为论文 Stage III 机制证明 | 主线保留但需降权 |
| V2/V2.5 | 修复 legacy Stage III 的协议与机制风险 | deterministic VAE eval、zero-drift identity、vector drift、split-safe preprocessing、rolling validation、counterfactual gate | `configs/stage3_v2_minimal.yaml`, `configs/stage3_v2_strong.yaml` | `logs/load_benchmarks/stage3_v2/`, `checkpoints/stage3_v2/`, `docs/audit/STAGE3_V2_*`, `STAGE3_V25_*` | 形成强审计协议；测试覆盖多 | public-3 / b240 / multi-seed 均未过 claim gate；report-only gain 不可推广 | 新设计必须保留 rolling/counterfactual gate 与 no-overwrite manifest | 归档参考 + 协议基线 |
| V3-V4 | 处理 V2.5 evidence problem | stricter repair、counterfactual hard negative | `configs/stage3_v3_strict.yaml`, `configs/stage3_v4_counterfactual_hard_negative.yaml` | `logs/load_benchmarks/stage3_v3/`, `stage3_v4/`, `checkpoints/stage3_v3/`, `stage3_v4/` | 机制更严；V4 降低 reject rate | rolling gate / counterfactual margin 仍失败 | 不要用单次 positive report-only 代替 prospective gate | 归档参考 |
| V5-V6 | served-policy 与 gating 修复 | served policy counterfactual gate、whitened Procrustes | `configs/stage3_v5_served_policy_counterfactual_gate.yaml`, `stage3_v6_whitened_procrustes_served_policy_gate.yaml` | `logs/load_benchmarks/stage3_v5/`, `stage3_v6/`, `checkpoints/stage3_v5/`, `stage3_v6/`, `ckpt/s3v5*`, `ckpt/s3v6*` | V5 有 near-miss | V6 退化；无可靠 served artifact | 复杂 alignment 不等于 forecasting correction | 归档参考 |

## 3. Alignment / teacher 版本族

| 版本族 | 目标 | 关键机制 | 对应配置 | 证据路径 | 成功点 | 失败点 | 新设计启发 | 建议状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V7 | 清理 `phi_v - phi_i` 中 invariant leakage | `models/stage3_alignment_v7.py` residualized RRCCA | `configs/stage3_v7_residualized_rrcca_served_policy_gate.yaml` | `logs/s3v7p*`, `ckpt/s3v7p*`, `docs/audit/STAGE3_V7_ALIGNMENT_RECORD.md` | alignment gate 和 active rank 诊断更严 | rolling validation `0/5` | alignment pass 不能替代 teacher/generator gate | 归档参考 |
| V8 | 增加 generator 容量 | HyperAdapter weight + bias update | `configs/stage3_v8_hyperadapter_residualized_rrcca_served_policy_gate.yaml` | `logs/s3v8*`, `ckpt/s3v8*`, `docs/audit/STAGE3_V8_HYPERADAPTER_RECORD.md` | 证明容量扩展能被训练和加载 | same-budget canary 比 V7 更差 | 不能靠增大 adapter 容量解决错误方向 | 失败诊断 |
| V9 | 引入 train-only residual teacher | `models/stage3_residual_teacher_v9.py`, `models/stage3_alignment_v9.py` | `configs/stage3_v9_residual_teacher_rrcca_served_policy_gate.yaml` | `logs/s3v9diag`, `logs/s3v9teacher_diag`, `ckpt/s3v9*`, `docs/audit/STAGE3_V9_*` | 定位失败上游到 drift-to-residual teacher | teacher-only 也失败；zero drift 多窗口最好 | 先做 teacher upper-bound，再谈 generator | 失败诊断 |
| V10 | cross-fitted teacher calibration | `stage3_alignment_v10.py`, `stage3_residual_teacher_v10.py` | `stage3_v10_crossfit_teacher_projection_*.yaml` | `logs/s3v10*`, `ckpt/s3v10*`, `docs/audit/STAGE3_V10_*` | Panama seed42 teacher-only 有局部改善 | public-3 `0/9` teacher gates | scalar shrinkage 有用但不足 | 归档参考 |
| V11 | block-calibrated residual-transfer teacher | `stage3_alignment_v11.py`, `stage3_residual_teacher_v11.py` | `stage3_v11_block_calibrated_teacher_gate.yaml`, `stage3_v11_block_horizon_teacher_gate.yaml` | `logs/s3v11*`, `ckpt/s3v11*`, `docs/audit/STAGE3_V11_*` | fail-closed 安全；alignment 全过 | teacher gates `0/9`, rolling `0/45`, alpha=0 | alignment 与 correction 必须分开证明 | 失败诊断 |
| V12 | outcome-aligned transport | `stage3_alignment_v12.py`, `stage3_residual_teacher_v12.py` | `stage3_v12_outcome_aligned_teacher_gate.yaml` | `logs/s3v12*`, `ckpt/s3v12*`, `docs/audit/STAGE3_V12_OUTCOME_*` | alignment audit 更强，public-3 alignment `9/9` | teacher gates `0/9`, rolling `0/45`, no generator | 坐标处理正确仍不代表 residual correction 有效 | 失败诊断 |
| V12b/V12c | OOF teacher 与 multi-policy diagnostic | `stage3_residual_teacher_v12b.py`, `stage3_residual_teacher_v12c.py` | `stage3_v12b_*`, `stage3_v12c_*` | `logs/s3v12b*`, `logs/s3v12c*`, `ckpt/s3v12b*`, `ckpt/s3v12c*`, `docs/audit/STAGE3_V12C_*` | 暴露 weak delta-vector signal | public-3 仍 `0/9`, `0/45` | 需要 history/control 和 sign calibration | 失败诊断 |
| V12d/V12e | drift vs history 对照和 per-horizon label availability | `stage3_residual_teacher_v12d.py`, `train/stage3_v12e_*.py` | `stage3_v12d_context_teacher_probe.yaml` + CLI overrides | `logs/s3v12dm`, `logs/s3v12emc`, `ckpt/s3v12dm`, `ckpt/s3v12emc`, `docs/audit/STAGE3_V12D_*`, `STAGE3_V12E_*` | 证明 history-only 是必要对照；label release 更严格 | `phi_v - phi_i` 没有稳定超过 history/counterfactual | 新设计必须显式处理标签可用性和 history baseline | 重要负证据 |
| V13-V15 | fluctuation signal 与 residual-target predictability | `stage3_residual_teacher_v13.py`, `stage3_residual_teacher_v14.py`, `run_stage3_v15_*` | `stage3_v13_*`, `stage3_v14_*`, `stage3_v15_*` | `logs/s3v13fluct_*`, `s3v14stability_b480`, `s3v15target_b480`, `ckpt/s3v13*`-`s3v15*` | 明确停止 unsupported fluctuation correction | primary gates 均失败，generator blocked | 不要再扩同一机制；改 representation objective | 失败诊断 |

## 4. V16-V29 expert-local / release 版本族

| 版本族 | 目标 | 关键机制 | 对应配置/计划 | 证据路径 | 成功点 | 失败点 | 新设计启发 | 建议状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V16 | expert-local dual stream | `factor_mechanism_v16.py`, `run_stage3_v16_expert_local_dual_stream_probe.py` | `configs/stage3_v16_expert_local_dual_stream_probe.yaml`, `docs/superpowers/specs/2026-06-10-stage3-v16-*` | `logs/s3v16expert_*`, `ckpt/s3v16expert_*`, `docs/audit/STAGE3_V16_*` | 将 Stage III 从全局 drift 转向专家局部结构 | 未形成 gate-passing result | 专家局部化是合理方向，但要绑定 evidence gates | 归档参考 |
| V17-V18 | causal env / structured env release | `factor_mechanism_v17.py`, `v18.py`, `run_stage3_v17*`, `run_stage3_v18*` | `configs/stage3_v17_*`, `stage3_v18_*`, V18/V17 explanation docs | `logs/s3v17*`, `logs/s3v18*`, `ckpt/s3v17*`, `ckpt/s3v18*` | 环境与 release 语义更明确 | 仍未过主 gate | environment inference 必须服务于可观测 online state | 归档参考 |
| V19-V23 | conditional release / release-driven / observable drift | `factor_mechanism_v19.py` 到 `v23.py`, `run_stage3_v19*` 到 `v23*` | `configs/stage3_v19_*` 到 `v23_*`, specs/plans | `logs/s3v19*` 到 `s3v23*`, `ckpt/s3v19*` 到 `s3v23*` | 逐步把 drift 从隐空间差转成可观测/释放状态 | 多为弱或负结果 | 可观测 drift package 比裸 `phi_v - phi_i` 更合理 | 归档参考 |
| V24-V26 | tail observable change / change quality / multiscale change | `factor_mechanism_v24.py` 到 `v26.py`, `stage3_generator_training_v24.py` 到 `v26.py` | `configs/stage3_v24_*`, `v25_*`, `v26_*` 和 tuning variants | `logs/s3v24*` 到 `s3v26*`, `artifacts/stage3_v29_*` 的前置选择逻辑 | 局部可调性增强，V26 有 raw-side tuning 信号 | 不能跨 seed/dataset/window 稳定 | 调参能移动指标，但不能替代 counterfactual gate | 归档参考 |
| V27-V28 | quality-conditioned dual stream / support suppression release | `factor_mechanism_v27.py`, `factor_mechanism_v28.py` | `configs/stage3_v27_*`, `stage3_v28_*`, specs/plans | `logs/s3v27*`, `logs/s3v28*`, `ckpt/s3v27*`, `ckpt/s3v28*`, `docs/audit/STAGE3_V27_*`, `STAGE3_V28_*` | 出现更明显 raw-side 或 support-push 局部信号 | gate/counterfactual/window 仍失败 | 需要把局部正信号变成可服务且反事实安全的协议 | 待确认参考 |
| V29 | expert-local invariant-release | `models/factor_mechanism_v29.py`, `train/run_stage3_v29_expert_local_invariant_release_probe.py`, `train/stage3_v29_*` | `configs/stage3_v29_expert_local_invariant_release_probe.yaml`, tuning configs, `docs/superpowers/plans/2026-06-12-stage3-v29-*`, `2026-06-14-stage3-v29-*` | `logs/s3v29expert_*`, `logs/s3v29rps`, `logs/s3v29hgr`, `ckpt/s3v29*`, `artifacts/stage3_v29_*`, `docs/audit/STAGE3_V29_*` | 结构上最接近专家局部不变释放假设；`pooled_servable_change` 比裸差分更合理；mode-gate 回放有治理价值 | 所有正式/筛选 run 主目标未过 strict teacher gate；generator locked；mode gate scope 是 `mode_selection_only_no_generator_unlock`；部分早期 V29 快照的 `resolved_config.json` 仍记录旧 `delta_phi_v`，引用时需优先使用已修正快照或重跑 anchor | 下一版应保留 V29 结构思想，但重新设计 protocol、teacher target 和 online-available features | 新设计参考，不是论文性能证据 |

## 5. 文件族归纳

### `models/factor_mechanism*.py`

- `factor_mechanism.py`：基础 factor-mechanism forecaster。
- `factor_mechanism_v2.py`：V2 deterministic VAE eval 包装。
- `factor_mechanism_v16.py` 到 `factor_mechanism_v29.py`：Stage III 后期结构探索链，逐步加入 expert-local、release、observable-change、support-suppression、expert-local invariant-release。

建议：保留全部文件。后续新主线可以参考 V29，但不要覆盖旧版本，因为旧版本与失败记录相互绑定。

### `models/adaptation*.py`

- `adaptation.py`：legacy Stage III / low-rank update。
- `adaptation_v2.py`：当前最关键的 identity-preserving / HyperLoRA / HyperAdapter delta 生成工具。
- `adaptation_v9.py`, `adaptation_v10.py`：历史扩展层，不是 V29 主入口核心。

建议：V2 adaptation 机制可作为“参数扰动安全层”参考；legacy adaptation 只作为历史入口。

### `models/stage3_alignment_v*.py`

V7/V9/V10/V11/V12 记录了从 residualized RRCCA 到 outcome-aligned transport 的 alignment 方向。多轮记录共同说明：alignment gate 通过不等于 residual correction 或 forecasting gain 通过。

建议：保留为 negative/diagnostic lineage；新设计若使用 alignment，应把 alignment diagnostics 与 forecasting target gate 分离。

### `models/stage3_residual_teacher_v*.py`

V9-V14 teacher 版本逐步加入 train-only residual teacher、calibration、OOF、多 policy、history controls、label availability、fluctuation readout。证据结论整体偏负。

建议：保留为 teacher 设计失败谱系；新设计应先定义 teacher-only prospective gate，再决定是否训练 generator。

### `configs/stage3_v*.yaml`

配置文件覆盖 V2-V29，且大量 tuning config 只对应一次局部筛选。它们是实验证据链的一部分，不应清理或重命名。新设计应另建清晰命名空间，例如 `configs/redesign/` 或 `configs/v30_protocol/`，避免复用旧 config 名。

### `docs/superpowers/specs/` 与 `docs/superpowers/plans/`

这些文件记录了 V13-V29 的设计意图和执行计划，属于“为什么这样改”的证据。它们不等同于实验结果，但对总结经验教训很重要。

建议：原位保留。新计划应新建而非覆盖旧计划。

## 6. 版本保留结论

- 主线保留：基础 data/model/train/utils、`run_experiment.py`、benchmark/report scripts、V29 结构参考。
- 归档参考：V2-V15 Stage III 修复与失败诊断、V16-V28 结构探索。
- 待确认：V29 是否作为新重构的直接起点，还是只提取思想后另起 V30/new-design。
- 不建议移动：所有 `logs/`、`ckpt/`、`checkpoints/`、run manifests、teacher_only_abort、gate CSV 和原始配置。
- 高价值证据链代表：V2.5 canonical checkpoint-selection 的 `run_manifest.json` + generator candidates + gate files；V29 official line 的 `run_manifest_v29.json` + `teacher_only_v29_target_gate.csv` + `teacher_only_abort.json` + generator-unlock preflight。这两类都应保留，但二者结论都是“协议/治理价值高，不支持当前性能 claim”。
