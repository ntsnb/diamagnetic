# Stage III V2.1 Split-Safe Experiment Record

Date: 2026-05-25

本记录汇总本轮针对 Stage III 的实质性修正、验证命令和较大规模 bounded retraining 结果。它不是论文结论，目的是判断当前修复是否足以支持“Stage III 学到了可靠的 drift-conditioned online adaptation”。

## 1. 本轮修正范围

本轮仍然遵守“不覆盖原模型/原训练代码”的原则，所有实现放在 Stage III V2 路径或新增文件中。

### 1.1 Split-safe preprocessing

新增：

```text
data/readers_stage3_v2.py
tests/test_stage3_v2_split_safe_preprocessing.py
```

接入：

```text
train/run_stage3_v2_sensitivity_audit.py
configs/stage3_v2_strong.yaml
configs/stage3_v2_minimal.yaml
```

核心变化：

```text
旧路径：全序列 time_interpolate / ffill / bfill -> raw split
新路径：raw 时间轴逻辑先确定 train/val/test -> 只用已可见历史做 ffill -> 段首缺失用 train-only median
```

manifest 已记录：

```text
split_safe_imputation = true
preprocessing_uses_future_values = false
imputation_uses_bfill = false
preprocessing_row_count_before = preprocessing_row_count_after
```

### 1.2 Projected Residual Alignment V2.1

修改：

```text
models/aligned_representation_v2.py
train/offline_aligned_representation_v2.py
train/run_stage3_v2_sensitivity_audit.py
configs/stage3_v2_strong.yaml
```

新增能力：

1. `delta_raw -> stop-gradient anchored orthogonal projection -> delta_phi_v`。
2. `lambda_inv_forecast`：让 invariant branch 必须独立承担预测能力。
3. `lambda_barlow`：Barlow-style cross-correlation suppression。
4. `lambda_temporal_ranking`：让 `||delta_phi_v||` 对 residual/adaptation need 有排序约束。
5. `lambda_phi_vicreg_*`：同时监控 `phi_i` 分支 variance/covariance。
6. 训练和验证输出 `parallel_ratio`、`inv_forecast_mse`、`variant_residual_mse` 等指标。

## 2. 验证命令

完整 Stage III V2 测试集：

```powershell
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_factor_mechanism_v2.py tests\test_adaptation_v2.py tests\test_hyperlora_generator_v2.py tests\test_aligned_representation_v2.py tests\test_offline_aligned_representation_v2.py tests\test_stage3_functional_v2.py tests\test_stage3_generator_training_v2.py tests\test_stage3_sensitivity_v2.py tests\test_stage3_sensitivity_gate_v2.py tests\test_online_loop_stage3_v2.py tests\test_stage3_v2_protocol_safety.py tests\test_stage3_v2_split_safe_preprocessing.py tests\test_stage3_signals.py tests\test_stage3_ablation_benchmark.py tests\test_shapes.py -q
```

结果：

```text
63 passed in 3.40s
```

V2.1 smoke：

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v2_sensitivity_audit.py --config configs\stage3_v2_strong.yaml --dataset panama --run-id codex_stage3_v21_splitsafe_smoke_20260525 --seed 42 --epochs 1 --invariant-epochs 1 --generator-epochs 1 --device cpu --modes real zero --max-train-batches 2 --max-generator-batches 2 --max-eval-batches 2 --validation-windows 2 --drift-source delta_phi_v
```

结果：runner 成功生成 split-safe manifest、rolling validation CSV 和新增 alignment metrics。

## 3. Bounded Larger Retraining

运行配置：

```text
run_id: codex_stage3_v21_splitsafe_panama_b60_20260525
dataset: panama
seeds: 42, 43
base epochs: 2
aligned representation epochs: 2
generator epochs: 4
max_train_batches: 60
max_generator_batches: 60
max_eval_batches: 60
validation_windows: 3
drift_source: delta_phi_v
signal_mode: delta_vector
normalize_vector: true
modes: real / zero / shuffled / sign_flipped / random_matched_norm
```

输出目录：

```text
logs/load_benchmarks/stage3_v2/codex_stage3_v21_splitsafe_panama_b60_20260525/panama/seed_42
logs/load_benchmarks/stage3_v2/codex_stage3_v21_splitsafe_panama_b60_20260525/panama/seed_43
```

## 4. 表示对齐是否生效

V2.1 的 hard projection 在数值上生效。

| seed | epoch | val forecast MSE | val inv forecast MSE | val parallel_ratio |
| --- | ---: | ---: | ---: | ---: |
| 42 | 1 | 0.327478 | 0.321264 | 3.84e-08 |
| 43 | 1 | 0.339742 | 0.333704 | 2.84e-08 |

解释：

```text
parallel_ratio 接近 0，说明 projected residual 的样本级正交约束确实生效。
```

但这只证明表示几何约束生效，不等于 Stage III online adaptation 已经有效。

## 5. Rolling Validation 结果

| seed | rolling gate | pass rate | failed windows | worst real MSE gain | worst real MAE gain | best counterfactual MSE gain |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| 42 | fail | 0.6667 | 0 | -0.4716% | -0.7792% | 0.2972% |
| 43 | fail | 0.6667 | 0 | 0.0069% | 0.1157% | 0.0501% |

解释：

1. 两个 seed 都只有 2/3 validation windows 通过。
2. 按当前 `rolling_min_window_pass_rate=1.0`，两者都失败。
3. seed 42 的 validation 总体 real gain 为负。
4. seed 43 的 validation real gain 虽为正，但低于 claim gate 阈值，并且 `shuffled` 比 real 更好。

这说明：

```text
V2.1 没有解决 validation 稳定性问题。
```

## 6. Test Report-Only 结果

| seed | test gate | real MSE gain | real MAE gain | best counterfactual | margin vs best counterfactual |
| --- | --- | ---: | ---: | --- | ---: |
| 42 | fail | 0.4785% | -0.2272% | zero | 0.4785% |
| 43 | fail | 0.2498% | 0.1117% | zero | 0.2498% |

与旧结果对照：

| run | setting | test/online real MSE gain | interpretation |
| --- | --- | ---: | --- |
| old Panama b30 | validation pass, test report-only | -3.5996% | held-out test 反转失败 |
| old Panama b50 | advanced V2 | -0.5096% | real drift 仍变差 |
| new V2.1 seed 42 b60 | split-safe + projected residual | 0.4785% | MSE 由负转正，但 MAE 仍负，validation 失败 |
| new V2.1 seed 43 b60 | split-safe + projected residual | 0.2498% | MSE/MAE 小幅正，但低于 gate，validation 失败 |

## 7. 科学判断

本轮修复带来了两个确定进展：

1. preprocessing 由“已知有泄漏风险”升级为 split-safe V2 路径。
2. `delta_phi_v` 的几何约束明显生效，`parallel_ratio` 接近 0。

但预测证据没有达到“显著提升”标准：

1. rolling validation 两个 seed 都失败。
2. validation 中仍出现 counterfactual 优于 real 的情况。
3. test MSE 有改善，但 test 是 report-only，不能用来回头证明配置选择正确。
4. seed 42 test MAE 仍为负，seed 43 虽为正但幅度小。

因此当前不能写：

```text
Stage III V2.1 significantly improves online forecasting.
Stage III V2.1 reliably learns concept drift.
Projected residual alignment validates drift-conditioned adaptation.
```

当前可以写：

```text
Projected Residual Alignment V2.1 improves the auditability and geometry of the drift representation, and split-safe preprocessing removes a major protocol risk. However, bounded retraining on Panama still fails rolling validation, so the current Stage III path remains an unverified adaptation candidate rather than a confirmed contribution.
```

## 8. 下一步建议

不要继续用 test 调参。下一步应在 validation 内解决失败原因：

1. 分析 failed validation window 0 的样本、季节、负荷水平和 drift norm。
2. 增加 generator 的 validation-aware early stopping 或 risk-aware selection，但只能基于 validation。
3. 将 `shuffled/random_matched_norm` 保持为 offline diagnostic，不作为可部署 online baseline。
4. 如果继续扩展实验，应按 `public-3 x seeds` 运行固定配置；但只有 rolling validation 先稳定后，才值得跑完整 test。

如果后续仍失败，论文应把 Stage III 降级为 negative diagnostic ablation，把主要贡献转向：

```text
split-safe online evaluation protocol
counterfactual sensitivity audit
drift-conditioned adaptation 的可证伪性分析
```

## 9. Failed validation window 0 初步定位

两个 seed 都失败在 validation window 0：

| seed | window 0 real MSE gain | window 0 real MAE gain | best counterfactual | interpretation |
| --- | ---: | ---: | --- | --- |
| 42 | -0.4716% | -0.7792% | zero | real drift 直接使预测变差 |
| 43 | 0.0069% | 0.1157% | shuffled | real drift 极弱，且不如离线打乱 drift |

窗口 1 和窗口 2 的情况明显不同：

| seed | window 1 real MSE gain | window 2 real MSE gain |
| --- | ---: | ---: |
| 42 | 2.0537% | 2.4985% |
| 43 | 0.7863% | 0.6642% |

这说明当前失败不是全局完全无效，而是：

```text
Stage III V2.1 对 validation 开头 regime 不稳定。
```

可能原因包括：

1. validation window 0 紧贴 train/validation 边界，drift standardizer 和 generator 在边界后的最初 regime 上方向不稳。
2. generator 当前始终使用 `use_rate=1.0`，没有学会“这个窗口不该更新”。
3. prediction shift 不大且没有触发 reject，说明问题不是 clamp 或 shift budget，而是 delta 方向错误。
4. seed 43 中 `shuffled` 优于 real，说明 drift 的时间对齐语义仍未被稳定利用。

因此下一步如果继续修模型，不应继续加大 test 规模，而应先在 validation 内加入：

```text
update/no-update policy
risk-aware gating
window-aware generator early stopping
real-vs-zero/shuffled validation selection
```

只有 validation window 0 的失败被解决后，才值得跑 public-3 x multi-seed 的更大评估。
