# Stage III V4 Counterfactual Hard-Negative Full Experiment Record

本记录对应 2026-05-26 的 Stage III V4 修复与 Panama seed42 full-run 审计。结论先行：

- V4 修复了 V3 的主要部署性失败：test reject rate 从 `24.67%` 降到 `0.80%`。
- V4 也把 amplified counterfactual 从“略优于 real”修到“real 略优于 amplified”，但 margin 只有 `0.000138`，低于预声明 `0.002`。
- V4 仍未通过 strict claim gate；rolling validation 仍是 `0/5`。
- 因此，V4 只能作为“机制修复后的部分正向证据”，不能写成论文中已经证明的 Stage III 创新点。

## 1. 修改内容

本轮没有覆盖原始 legacy Stage III 代码路径，而是在 V2/V3 审计路径上增量修改：

- `train/online_loop_stage3_v2.py`
  - 新增 `return_step_rows` 和 `trace_context`。
  - `step_metrics.csv` 逐步记录 `base_mse`、`candidate_mse`、`would_apply_blended_mse`、`final_mse`、`used`、`reject_reason`、`drift_norm`、`delta_norm_pre_clamp`、`pred_shift`、`risk_score` 等。
  - `would_apply_blended_mse/mae` 用于后续 common-accept-mask 或 gate 反事实分析。
- `train/stage3_generator_training.py`
  - `Stage3GeneratorLossConfig` 新增 `detach_negative_mse`。
  - 当 `detach_negative_mse=false` 时，intervention-compare loss 不再把 negative branch 的 MSE detach 掉，可以反向训练 generator 让 counterfactual negative drift 变差。
- `train/run_stage3_v2_sensitivity_audit.py`
  - 新增 `_build_stage3_generator_negative_drifts()`。
  - 支持 `amplified_negative_factors: [2.0, 5.0]`，把 amplified drift 纳入训练期 hard negatives。
- `configs/stage3_v4_counterfactual_hard_negative.yaml`
  - 新增 V4 配置。
  - 核心差异：`lambda_intervention_compare: 1.0`、`intervention_margin: 0.002`、`detach_negative_mse: false`、`amplified_negative_factors: [2.0, 5.0]`。

## 2. 验证

代码级验证：

- `C:\Users\admin\anaconda3\python.exe -m pytest tests -q`
- 结果：`97 passed in 3.67s`

语法验证：

- `C:\Users\admin\anaconda3\python.exe -m py_compile train\online_loop_stage3_v2.py train\stage3_generator_training.py train\run_stage3_v2_sensitivity_audit.py`
- 结果：exit code `0`

配置验证：

- `yaml.safe_load(configs/stage3_v4_counterfactual_hard_negative.yaml)`
- 结果：`yaml_ok`

## 3. V3 Full Panama 对照基线

输出目录：

- `logs/load_benchmarks/stage3_v3/full_20260526/codex_stage3_v3_full_panama_seed42_trace_20260526_01/panama/seed_42`

关键文件：

- `run_manifest.json`
- `sensitivity_summary.csv`
- `sensitivity_claim_gate.csv`
- `rolling_validation_claim_gate_aggregate.csv`
- `step_metrics.csv`

V3 test gate：

| Metric | Value |
| --- | ---: |
| claim_gate_pass | `False` |
| failure_reasons | `counterfactual;reject_rate` |
| real_mse_gain | `0.0418706` |
| real_mae_gain | `0.0310394` |
| best_counterfactual_mode | `amplified_5x` |
| best_counterfactual_mse_gain | `0.0425720` |
| real_vs_best_counterfactual_mse_gain_margin | `-0.0007014` |
| real_reject_rate | `0.246678` |
| real_shift_reject_rate | `0.232914` |
| real_risk_reject_rate | `0.013764` |

V3 逐步诊断：

- real test use rate: `0.753322`
- amplified_5x use rate: `0.765417`
- real drift_norm: `10.143215`
- amplified_5x drift_norm: `50.716075`
- real delta_norm_pre_clamp: `1.778308`
- amplified_5x delta_norm_pre_clamp: `1.778716`
- real pred_shift: `0.203955`
- amplified_5x pred_shift: `0.204067`

解释：5x drift 把输入范数放大约 5 倍，但 generator 输出的 delta 和 prediction shift 几乎不变，说明 V3 generator 对幅值不敏感或已饱和。amplified_5x 的 risk_score 比 real 高 `0.1970`，因此更容易通过 gate，这是 V3 counterfactual 失败的重要原因。

## 4. V4 Full Panama Run

命令：

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v2_sensitivity_audit.py --config configs\stage3_v4_counterfactual_hard_negative.yaml --dataset panama --run-id codex_stage3_v4_full_panama_seed42_trace_20260526_04 --seed 42 --device auto --modes real zero shuffled sign_flipped amplified_2x amplified_5x random_matched_norm --validation-windows 5 --drift-source delta_phi_v --drift-polarity 1.0 --signal-mode delta_vector --normalize-vector --out-root logs\load_benchmarks\stage3_v4\full_20260526 --checkpoint-root checkpoints\stage3_v4\full_20260526
```

输出目录：

- `logs/load_benchmarks/stage3_v4/full_20260526/codex_stage3_v4_full_panama_seed42_trace_20260526_04/panama/seed_42`

Manifest 要点：

- `stage3_protocol_version`: `v4_counterfactual_hard_negative`
- `stage3_generator_selected_candidate_epoch`: `1`
- `stage3_risk_gate_threshold`: `0.35`
- `stage3_step_metrics_enabled`: `true`
- `stage3_test_step_metric_rows`: `117481`
- `stage3_claim_gate_pass`: `false`
- `stage3_claim_gate_failure_reasons`: `counterfactual`

## 5. V4 Test Results

| Mode | MSE | MAE | Use Rate | Pred Shift | Risk Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| real | `0.1753267` | `0.2861148` | `0.991956` | `0.125448` | `0.559334` |
| zero | `0.1819699` | `0.2950983` | `0.000000` | `0.000000` | `0.271725` |
| shuffled | `0.1854411` | `0.3001891` | `0.983495` | `0.121829` | `0.565542` |
| sign_flipped | `0.1945471` | `0.3118227` | `0.967288` | `0.133504` | `0.571904` |
| amplified_2x | `0.1753519` | `0.2861462` | `0.991658` | `0.125436` | `0.703693` |
| amplified_5x | `0.1753660` | `0.2861501` | `0.991658` | `0.125414` | `0.939805` |
| random_matched_norm | `0.1854967` | `0.3002287` | `0.982125` | `0.122469` | `0.568619` |

V4 test gate：

| Metric | Value |
| --- | ---: |
| claim_gate_pass | `False` |
| failure_reasons | `counterfactual` |
| real_mse_gain | `0.0365069` |
| real_mae_gain | `0.0304425` |
| best_counterfactual_mode | `amplified_2x` |
| best_counterfactual_mse_gain | `0.0363687` |
| real_vs_best_counterfactual_mse_gain_margin | `0.0001382` |
| real_reject_rate | `0.0080439` |
| real_shift_reject_rate | `0.0079843` |
| real_risk_reject_rate | `0.0000596` |

## 6. V3 vs V4 结论

| Version | real MSE Gain | real MAE Gain | Best CF | CF Margin | Reject Rate | Failure |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| V3 | `0.0418706` | `0.0310394` | `amplified_5x` | `-0.0007014` | `0.246678` | `counterfactual;reject_rate` |
| V4 | `0.0365069` | `0.0304425` | `amplified_2x` | `+0.0001382` | `0.008044` | `counterfactual` |

V4 的确修复了两个具体问题：

- high reject rate 基本消失。
- amplified drift 不再超过 real drift，至少在 test aggregate 上 real 略优。

但 V4 也带来一个代价：

- real MSE gain 从 V3 的 `4.19%` 降到 `3.65%`。
- real-vs-amplified margin 只有 `0.000138`，远小于预声明 `0.002`，不能算强 counterfactual 证据。

## 7. Rolling Validation

V4 rolling validation aggregate：

| Metric | Value |
| --- | ---: |
| rolling_claim_gate_pass | `False` |
| rolling_failure_reasons | `window_pass_rate;counterfactual` |
| validation_window_count | `5` |
| validation_window_pass_count | `0` |
| validation_window_pass_rate | `0.0` |
| worst_window_real_mse_gain | `0.0471236` |
| worst_window_real_mae_gain | `0.0357425` |
| best_window_counterfactual_mse_gain | `0.0610193` |

这说明 V4 在 test aggregate 上更干净，但 validation windows 仍不满足预声明机制门槛。不能用 test aggregate 单独改写论文 claim。

## 8. 论文层面判断

不能写：

- Stage III 已经可靠学习 concept drift。
- Stage III 已经证明 real drift 优于 counterfactual drift。
- V4 是 SOTA 或显著优于所有 baseline 的核心贡献。

可以谨慎写成内部审计结论：

- V4 hard-negative training 明显改善了 V3 的 deployment gate 行为。
- V4 使 real drift 在 Panama seed42 test aggregate 上略优于 amplified counterfactuals。
- 该结果仍未通过 rolling validation 和预声明 counterfactual margin，因此应作为机制修复方向，而不是投稿级最终证据。

## 9. 下一步

最小安全下一步不是继续调论文措辞，而是继续模型/实验修复：

1. 做 V4 的 validation-window failure breakdown，定位每个窗口为什么 amplified 仍更强。
2. 利用 V4 `would_apply_blended_mse` 做 common-accept-mask 分析，隔离 gate 与 generator 本体贡献。
3. 若继续 V5，优先改 selection objective：不仅选 rolling gain，还要显式优化 real-vs-amplified margin。
4. 在 Panama seed42 以外，不要扩写论文 claim；至少需要 public-3 multi-seed 通过同一 gate。

## 10. Window-level common-mask diagnostics

2026-05-27 新增派生诊断脚本：

- `train/analyze_stage3_step_metrics.py`
- `tests/test_stage3_step_metrics_analysis.py`

该脚本只读取 V4 run directory，不改写原始 `step_metrics.csv`、训练日志或 checkpoint。它生成以下派生 CSV：

- `common_accept_mask_summary.csv`
- `paired_mode_differences.csv`
- `validation_window_failure_breakdown.csv`
- `rolling_common_accept_mask_summary.csv`
- `rolling_paired_mode_differences.csv`

Validation-window failure breakdown：

| Window | Failure | Real MSE gain | Real MAE gain | Best counterfactual | CF MSE gain | Margin |
| ---: | --- | ---: | ---: | --- | ---: | ---: |
| 0 | `counterfactual` | `0.047124` | `0.039042` | `amplified_5x` | `0.047586` | `-0.000463` |
| 1 | `counterfactual` | `0.059124` | `0.044083` | `amplified_5x` | `0.059595` | `-0.000471` |
| 2 | `counterfactual` | `0.055054` | `0.048428` | `amplified_2x` | `0.054831` | `+0.000223` |
| 3 | `counterfactual` | `0.061425` | `0.045721` | `amplified_2x` | `0.061019` | `+0.000405` |
| 4 | `counterfactual` | `0.047837` | `0.035743` | `amplified_2x` | `0.047782` | `+0.000055` |

解释：

- 5 个 rolling windows 仍全部因为 counterfactual margin 失败。
- Window 0 和 window 1 仍是负 margin，说明 amplified drift 仍强于 real drift。
- Window 2 到 window 4 虽然方向为正，但距离预声明 `0.002` margin 仍很远。
- 因此 V4 的主要剩余问题不是 reject rate，而是 real/amplified 的机制分离不足。

Common-accept-mask 结论：

- Test split 中，`amplified_2x` 的 `all_final_mse_minus_reference_mean = 2.515e-05`，`amplified_5x` 为 `3.925e-05`。这说明 real 在 aggregate 上略好于 amplified，但差距只有 `1e-5` 到 `1e-4` 量级。
- Rolling validation 中，在相同 accept mask 下 amplified 与 real 仍非常接近；window 0 和 window 1 仍偏向 amplified。
- 因此 V4 不能证明 Stage III 已形成强 drift-conditioned adaptation pathway。它只能证明 V3 的高拒绝率问题被明显缓解，并且 aggregate 方向被轻微修正。

V5 implication：

- 不应降低 claim gate。
- 下一轮修复应直接优化 window-level real-vs-amplified separation。
- Checkpoint selection 应在所有候选均失败时显式 abstain，而不是把“最不坏”的 checkpoint 当作已验证结果。
