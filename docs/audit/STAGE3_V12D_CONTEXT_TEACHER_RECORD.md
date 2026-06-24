# Stage III V12d Context Teacher 诊断记录

日期：2026-06-01

## 1. 本轮问题

V12/V12b/V12c 均未证明 `phi_v - phi_i` 形式的 drift signal 能稳定产生可部署的 Stage III 改进。V12c public-3 best-delta 的结果为 `9/9` completed、`0/9` teacher gates、`0/45` rolling windows，说明不能继续直接训练 generator。

V12d 的目的不是提高论文数字，而是回答一个更基础的问题：

```text
在已有 base model residual 存在时间自相关时，
phi_v - phi_i 是否提供了超过历史残差的增量预测信息？
```

因此 V12d 只做 teacher-only 诊断，对比三种输入：

- `drift_only`：只使用 V12c `delta_vector` drift signal。
- `history_only`：只使用因果历史预测残差特征。
- `drift_plus_history`：拼接 drift signal 和历史残差特征。

## 2. Subagent 结论

三个 5.5 xhigh subagents 均为只读审计，未修改文件。

- 协议/泄漏审计结论：`residual-history-only` 是必要对照，但它使用标签反馈信息。若没有显式 label availability schedule，只能称为 teacher-only diagnostic / oracle-assisted diagnostic，不能称为 deployment-ready delayed-feedback evidence。
- 代码接口审计结论：V12d 应新增独立文件，不要把逻辑塞回 V12c；可复用 V12c 的 loader、aligned stream、OOF teacher policy、rolling diagnostics，但要避免 V12c 的 checkpoint/signal-builder/rolling-delta 边界风险。
- 统计实验审计结论：核心指标应是 `drift_plus_history` 相对 `history_only` 的 partial gain，而不是 `drift_plus_history` 的绝对提升。若 `history_only` 已解释大部分提升，而 `drift_plus_history` 无稳定增量，则不能声称 `phi_v - phi_i` 学到了独立 concept drift。

## 3. 新增代码

- `train/stage3_v12d_context_features.py`
  - `CausalResidualHistoryFeatureBuilderV12d`
  - 构造只使用过去 residual 的 lag 特征。
  - 支持 `previous_residual_tail`，避免 rolling window 起点人为重置。
  - 支持 zero padding + availability mask。

- `models/stage3_residual_teacher_v12d.py`
  - `TrainOnlyContextOOFResidualTeacherV12d`
  - 继承 V12c chronological OOF teacher policy，但记录 V12d context feature mode。
  - 独立 checkpoint save/load，避免污染 V12c 证据链。

- `train/run_stage3_v12d_context_teacher_probe.py`
  - V12d teacher-only runner。
  - 不进入 generator/risk-gate 训练路径。
  - 输出三种 feature mode 的 validation/test/rolling/partial-gain diagnostics。

- `train/run_stage3_v12d_context_teacher_matrix.py`
  - public-3 matrix runner。
  - 每个 dataset-seed 产生 3 行 summary：`drift_only`、`history_only`、`drift_plus_history`。

- `configs/stage3_v12d_context_teacher_probe.yaml`
  - 从 V12c nonnegative probe 配置机械派生。
  - 默认 `delta_vector`。
  - 默认 history lags：`1,2,3,6,12,24`。
  - `feedback_delay=0`，因此当前结果只能解释为 immediate previous-label diagnostic。

- `tests/test_stage3_v12d_context_features.py`
- `tests/test_stage3_v12d_teacher_modes.py`

## 4. TDD 与基础验证

先写失败测试，确认新模块不存在时 RED：

```powershell
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_stage3_v12d_context_features.py tests\test_stage3_v12d_teacher_modes.py -q
```

RED 失败原因为：

```text
ModuleNotFoundError: No module named 'train.stage3_v12d_context_features'
ModuleNotFoundError: No module named 'models.stage3_residual_teacher_v12d'
```

实现后 GREEN：

```text
4 passed in 1.33s
```

编译检查：

```powershell
C:\Users\admin\anaconda3\python.exe -m py_compile train\stage3_v12d_context_features.py models\stage3_residual_teacher_v12d.py train\run_stage3_v12d_context_teacher_probe.py train\run_stage3_v12d_context_teacher_matrix.py
```

结果：通过。

## 5. Smoke 结果

命令：

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v12d_context_teacher_probe.py --config configs\stage3_v12d_context_teacher_probe.yaml --dataset panama --run-id smoke_v12dctx_panama_s42_20260601_01 --seed 42 --device auto --epochs 1 --invariant-epochs 1 --max-train-batches 2 --max-generator-batches 96 --max-eval-batches 32 --validation-windows 2 --modes real zero sign_flipped amplified_2x --out-root logs\s3v12dctx_smoke --checkpoint-root ckpt\s3v12dctx_smoke
```

输出：

```text
logs/s3v12dctx_smoke/smoke_v12dctx_panama_s42_20260601_01/panama/seed_42
```

结果：

- `teacher_only_abort.json` 已写入。
- `generator*.pt` 数量为 `0`。
- `history_only` validation MSE gain 高于 `drift_plus_history`。
- validation partial MSE gain vs history 为负：`-0.172009`。
- test partial MSE gain vs history 为正：`0.046608`，但 MAE 与 sign 均变差，因此不能视为稳定改进。

## 6. Public-3 运行

第一次 public-3 使用较长路径：

```text
logs/s3v12dctxm/_manifests/v12dctxm_delta_nonneg_20260601_200112_e5_ie5_b480_ew480
```

结果中 Panama/Tetouan 完成，但 Spain 三个 seed 失败。失败根因不是模型失败，而是 Windows 路径过长：长 dataset 名称 + 长 run id + 长 alpha-block 文件名导致 `FileNotFoundError`。已将 V12d per-feature 诊断文件名缩短，例如 `v12d_alpha_blocks_d.csv`、`v12d_alpha_blocks_h.csv`、`v12d_alpha_blocks_dh.csv`。

第二次 public-3 使用短路径：

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v12d_context_teacher_matrix.py --config configs\stage3_v12d_context_teacher_probe.yaml --datasets panama tetouan spain_energy_weather --seeds 42 43 44 --epochs 5 --invariant-epochs 5 --max-generator-batches 480 --max-eval-batches 480 --validation-windows 5 --device auto --signal-mode delta_vector --out-root logs\s3v12dm --checkpoint-root ckpt\s3v12dm --run-prefix v12dm_delta
```

manifest：

```text
logs/s3v12dm/_manifests/v12dm_delta_20260601_203114_e5_ie5_b480_ew480
```

总结果：

| 项目 | 结果 |
| --- | ---: |
| dataset-seed runs | `9/9` completed |
| summary rows | `27/27` completed |
| `drift_only` teacher gates | `0/9` |
| `history_only` teacher gates | `0/9` |
| `drift_plus_history` teacher gates | `0/9` |
| `drift_only` rolling windows | `3/45` |
| `history_only` rolling windows | `0/45` |
| `drift_plus_history` rolling windows | `6/45` |
| generator artifacts | `0` |

注意：`history_only` 的 rolling gate 为 `0/45`，主要是因为它不使用 drift，所以 real/counterfactual drift 的 margin 按定义接近 `0`。因此 `history_only` 的绝对 residual correction 只能作为对照，不应按 drift-counterfactual gate 解释。

## 7. Public-3 关键结果

`drift_plus_history` 的逐 run 结果：

| dataset | seed | window pass | worst MSE gain | worst MAE gain | worst CF margin | sign | rolling partial MSE vs history | test partial MSE vs history |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| panama | 42 | `2/5` | `0.674088` | `0.431302` | `-0.007036` | `0.830729` | `-0.030336` | `-0.015386` |
| panama | 43 | `0/5` | `0.659706` | `0.460085` | `-0.015116` | `0.839410` | `0.005554` | `0.018320` |
| panama | 44 | `0/5` | `0.767179` | `0.533077` | `-0.003631` | `0.827257` | `-0.013439` | `-0.002338` |
| tetouan | 42 | `0/5` | `0.394956` | `0.228014` | `-0.001342` | `0.790799` | `-0.138562` | `-0.098515` |
| tetouan | 43 | `0/5` | `0.355655` | `0.173498` | `-0.006030` | `0.758681` | `-0.138212` | `-0.113435` |
| tetouan | 44 | `0/5` | `0.462361` | `0.231597` | `-0.005435` | `0.775174` | `-0.091659` | `-0.080980` |
| spain_energy_weather | 42 | `1/5` | `0.393873` | `0.283005` | `-0.023727` | `0.778646` | `-0.043727` | `0.024603` |
| spain_energy_weather | 43 | `0/5` | `0.639773` | `0.402242` | `-0.013539` | `0.804688` | `0.034356` | `0.054644` |
| spain_energy_weather | 44 | `3/5` | `0.487250` | `0.333342` | `-0.010966` | `0.815104` | `0.027787` | `0.041691` |

Partial gain 汇总：

| 指标 | 结果 |
| --- | ---: |
| rolling min partial MSE positive runs | `3/9` |
| rolling min partial MSE average | `-0.043138` |
| rolling min partial MSE min/max | `-0.138562 / 0.034356` |
| test partial MSE positive runs | `4/9` |
| test partial MSE average | `-0.019044` |
| test partial MSE min/max | `-0.113435 / 0.054644` |

## 8. 科学解释

V12d 明确显示：

1. `drift_only` 仍然很弱，平均 worst rolling MSE gain 为负：`-0.004451`。
2. `history_only` 的绝对 MSE gain 很高，说明 base residual 存在强历史自相关/可预测结构。
3. `drift_plus_history` 的绝对 gain 也很高，但它没有稳定超过 `history_only`。
4. `drift_plus_history` 的 real-vs-counterfactual margin 在全部 9 个 dataset-seed 上均未通过，平均 worst CF margin 为 `-0.009647`。
5. `drift_plus_history` 对 `history_only` 的 rolling minimum partial MSE gain 平均为负，只有 `3/9` 个 run 为正。

因此，本轮结果不能证明 `phi_v - phi_i` 在历史残差之外提供稳定增量信息。更合理的结论是：

```text
当前 Stage III 的 apparent correction signal 主要可由历史残差自相关解释；
phi_v - phi_i 作为 drift-conditioned correction source 尚未通过 public-3 prospective gates。
```

## 9. 对论文 claim 的影响

不能写：

- Stage III learns concept drift.
- The proposed invariant/variant embedding difference provides robust online adaptation.
- The generator should be trained because the teacher signal is validated.

可以写，但必须标为限制/诊断：

- V12d teacher-only ablation shows that causal residual-history baselines explain most of the apparent correction signal.
- Under the current chronological rolling protocol, `phi_v - phi_i` does not provide robust incremental predictive value beyond residual autocorrelation.
- Generator training remains scientifically blocked until `drift_plus_history` beats `history_only` under the same prospective gates and real-drift counterfactual tests.

## 10. 下一步

最小安全下一步不是训练 generator，而是：

1. 引入 explicit label availability schedule，将 `feedback_delay=0` 扩展为可配置 delayed-feedback protocol。
2. 增加 per-horizon residual history，避免把多步 horizon 的尚未到达标签放入历史 residual buffer。
3. 对 `drift_plus_history` 做 conditional permutation：固定 history，只打乱 drift，验证性能是否下降。
4. 如果 conditional drift 仍无增量，应停止把 `phi_v - phi_i` 作为核心创新，改写论文贡献为负结果/诊断框架/在线协议审计。
