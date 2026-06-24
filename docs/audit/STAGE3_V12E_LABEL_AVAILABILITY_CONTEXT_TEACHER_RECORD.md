# Stage III V12e Label-Availability Context Teacher 诊断记录

日期：2026-06-01

## 1. 本轮目的

V12d 已证明 `history_only` 能解释大部分 apparent correction，但 V12d 使用的是 sample-lagged H 维 residual 向量，并且默认 `feedback_delay=0`。对多步预测而言，上一行的高 horizon residual 可能尚未观测到。因此 V12d 只能称为 immediate previous-label diagnostic。

V12e 的目标是补上两个缺口：

1. 按 horizon 显式控制标签到达时间。
2. 固定 history features，只置换 drift signal，做 conditional drift null。

## 2. 协议定义

对 forecast origin `t` 和 horizon `h`：

```text
e(t,h) = y(t+h) - y_base_hat(t,h)
```

在当前 origin `k`，该 residual 只有满足以下条件才允许进入历史 buffer：

```text
t + h + label_delay <= k
```

等价地，如果使用样本行 lag `l`，则 horizon `h` 的 residual 合法条件为：

```text
l >= h + label_delay
```

无效 residual entry 必须置零，并保留 availability mask。

## 3. 新增代码

- `train/stage3_v12e_label_availability.py`
  - `PerHorizonLabelAvailabilityHistoryBuilderV12e`
  - 按 horizon offsets 和 label delay 构造 residual-history features。

- `train/stage3_v12e_conditional_permutation.py`
  - `build_history_norm_strata`
  - `conditional_permute_drift`
  - `conditional_random_matched_norm`

- `tests/test_stage3_v12e_label_availability.py`
- `tests/test_stage3_v12e_conditional_permutation.py`

同时扩展了 V12d runner/matrix 的 opt-in CLI：

- `--history-protocol v12e_per_horizon_label_availability`
- `--history-label-delay`
- `--history-horizon-offsets`
- `--diagnostic-include-offline-counterfactuals`

默认 V12d 路径未改；只有显式指定 V12e history protocol 时才启用。

## 4. 验证

TDD RED：

```text
ModuleNotFoundError: No module named 'train.stage3_v12e_label_availability'
ModuleNotFoundError: No module named 'train.stage3_v12e_conditional_permutation'
```

GREEN：

```powershell
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_stage3_v12d_context_features.py tests\test_stage3_v12d_teacher_modes.py tests\test_stage3_v12e_label_availability.py tests\test_stage3_v12e_conditional_permutation.py -q
```

结果：

```text
11 passed in 1.34s
```

编译：

```powershell
C:\Users\admin\anaconda3\python.exe -m py_compile train\stage3_v12e_label_availability.py train\stage3_v12e_conditional_permutation.py train\run_stage3_v12d_context_teacher_probe.py train\run_stage3_v12d_context_teacher_matrix.py train\stage3_v9_teacher_diagnostics.py
```

结果：通过。

## 5. Smoke

命令：

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v12d_context_teacher_probe.py --config configs\stage3_v12d_context_teacher_probe.yaml --dataset panama --run-id smoke_v12e_cond_panama_s42_20260601_01 --seed 42 --device auto --epochs 1 --invariant-epochs 1 --max-train-batches 2 --max-generator-batches 96 --max-eval-batches 32 --validation-windows 2 --modes real zero conditional_shuffled sign_flipped amplified_2x conditional_random_matched_norm --feature-modes drift_only history_only drift_plus_history --history-protocol v12e_per_horizon_label_availability --history-label-delay 0 --diagnostic-include-offline-counterfactuals --out-root logs\s3v12e_cond_smoke --checkpoint-root ckpt\s3v12e_cond_smoke
```

输出：

```text
logs/s3v12e_cond_smoke/smoke_v12e_cond_panama_s42_20260601_01/panama/seed_42
```

关键现象：

- history protocol 为 `v12e_per_horizon_label_availability_history`。
- horizon offsets 为 `1,2,3,4,5,6,7,8,9,10,11,12`。
- `conditional_shuffled` 出现在 `teacher_best_counterfactual_mode` 中，说明 fixed-history drift null 已进入诊断。
- 无 `generator*.pt` artifact。

## 6. Public-3 条件置换矩阵

命令：

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v12d_context_teacher_matrix.py --config configs\stage3_v12d_context_teacher_probe.yaml --datasets panama tetouan spain_energy_weather --seeds 42 43 44 --epochs 5 --invariant-epochs 5 --max-generator-batches 480 --max-eval-batches 480 --validation-windows 5 --device auto --signal-mode delta_vector --modes real zero conditional_shuffled sign_flipped amplified_2x conditional_random_matched_norm --history-protocol v12e_per_horizon_label_availability --history-label-delay 0 --diagnostic-include-offline-counterfactuals --out-root logs\s3v12emc --checkpoint-root ckpt\s3v12emc --run-prefix v12emc_delta
```

manifest：

```text
logs/s3v12emc/_manifests/v12emc_delta_20260601_214834_e5_ie5_b480_ew480
```

总结果：

| 项目 | 结果 |
| --- | ---: |
| dataset-seed runs | `9/9` completed |
| feature-mode rows | `27/27` completed |
| `drift_only` gates | `0/9` |
| `history_only` gates | `0/9` |
| `drift_plus_history` gates | `0/9` |
| `drift_only` rolling windows | `2/45` |
| `history_only` rolling windows | `0/45` |
| `drift_plus_history` rolling windows | `0/45` |
| generator artifacts | `0` |

Feature-mode 汇总：

| feature mode | gate pass | window pass | avg worst MSE gain | avg worst CF margin |
| --- | ---: | ---: | ---: | ---: |
| `drift_only` | `0/9` | `2/45` | `-0.004451` | `-0.010971` |
| `history_only` | `0/9` | `0/45` | `-0.000911` | `0.000000` |
| `drift_plus_history` | `0/9` | `0/45` | `0.001000` | `-0.005649` |

Partial gain over history:

| 指标 | 结果 |
| --- | ---: |
| rolling-min partial MSE positive runs | `2/9` |
| rolling-min partial MSE average | `-0.006893` |
| rolling-min partial MSE min/max | `-0.017344 / 0.005882` |
| test partial MSE positive runs | `5/9` |
| test partial MSE average | `0.000954` |
| test partial MSE min/max | `-0.009671 / 0.012513` |

Best counterfactual modes for `drift_plus_history` rolling windows:

| best counterfactual | count |
| --- | ---: |
| `amplified_2x` | `15` |
| `conditional_random_matched_norm` | `10` |
| `conditional_shuffled` | `9` |
| `zero` | `7` |
| `sign_flipped` | `4` |

`drift_plus_history` per run:

| dataset | seed | window pass | worst MSE | worst MAE | CF margin | sign | rolling partial | test partial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| panama | 42 | `0/5` | `0.002959` | `0.006783` | `-0.002098` | `0.527778` | `-0.010084` | `-0.006409` |
| panama | 43 | `0/5` | `-0.000526` | `0.006208` | `-0.009692` | `0.517361` | `0.002828` | `0.006578` |
| panama | 44 | `0/5` | `-0.042604` | `-0.016715` | `-0.008371` | `0.491319` | `-0.010946` | `0.012513` |
| tetouan | 42 | `0/5` | `0.001359` | `-0.000653` | `-0.001342` | `0.494792` | `-0.007242` | `0.000489` |
| tetouan | 43 | `0/5` | `0.001095` | `0.001595` | `-0.001998` | `0.464410` | `-0.017344` | `-0.009671` |
| tetouan | 44 | `0/5` | `-0.001636` | `-0.002191` | `-0.001923` | `0.344618` | `-0.007552` | `-0.005367` |
| spain_energy_weather | 42 | `0/5` | `-0.002819` | `0.003923` | `-0.017526` | `0.512153` | `-0.007200` | `0.007732` |
| spain_energy_weather | 43 | `0/5` | `0.034744` | `0.025392` | `-0.003218` | `0.594618` | `-0.010377` | `-0.007829` |
| spain_energy_weather | 44 | `0/5` | `0.016430` | `0.012178` | `-0.004670` | `0.550347` | `0.005882` | `0.010550` |

## 7. 科学解释

V12e 比 V12d 更严格，因为它不再把完整上一行 H 维 residual 当作已知历史，而是按 horizon release mask 构造历史 residual。结果显示：

1. `history_only` 的强收益在 V12e 下基本消失，说明 V12d 的部分 history benefit 很可能来自 immediate-label diagnostic 假设。
2. `drift_plus_history` 也没有稳定补回收益，rolling windows 为 `0/45`。
3. `drift_plus_history` 的平均 worst counterfactual margin 为负，说明 real drift 并不优于 fixed-history conditional drift controls。
4. test partial MSE 有 `5/9` 为正，但 rolling-min partial MSE 只有 `2/9` 为正，且平均为负，因此不能作为稳定证据。

结论：

```text
在 horizon-aware label-availability 和 fixed-history conditional drift controls 下，
当前 phi_v - phi_i drift signal 未提供稳定增量预测价值。
```

## 8. 对 Stage III 的影响

不能训练 generator。理由：

- V12e teacher gates 为 `0/27`。
- `drift_plus_history` rolling windows 为 `0/45`。
- real drift 没有战胜 conditional shuffled / conditional random matched norm / amplified controls。
- 无 served generator artifact 是正确结果，不是缺失。

可以写成论文限制：

> V12e shows that the apparent residual-history correction observed under immediate-feedback diagnostics does not survive a horizon-aware label-availability protocol. Moreover, the aligned drift representation does not provide robust incremental value beyond the delayed residual-history state under fixed-history conditional drift controls.

中文：

> V12e 表明，在显式逐 horizon 标签可用性约束下，V12d 中看到的历史残差修正收益大幅减弱；在固定 history 后，`phi_v - phi_i` drift signal 也没有稳定超过 conditional drift controls。因此 Stage III 仍应作为审计/诊断机制，而不是已验证的 drift-conditioned generator 贡献。

## 9. 下一步

如果继续追求 Stage III 机制，下一步不是加大训练，而是重构表示学习目标：

1. 让 `phi_v` 明确预测 future residual release events，而不是仅拟合当前 residual。
2. 加入 per-horizon residual sign calibration。
3. 将 drift representation 训练为在 delayed label availability 下对 residual correction 有增量信息。
4. 只有当 teacher-only public-3 达到 `9/9` gates 且 `drift_plus_history` 稳定超过 `history_only` 和 conditional controls，才允许 generator canary。
