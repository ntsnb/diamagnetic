# Stage III V18 到 V29 实验结果与 Baseline 对比摘要

日期：2026-06-24

本文档目的：汇总当前仓库中 `V18` 到 `V29` 的主要实验测试结果，并解释它们相对几个 baseline 的表现。本文档只汇总已有日志、审计记录和工件，不新增训练结果，不改动原始实验文件。

结论先行：从 `V18` 到 `V29`，所有正式或筛选 run 的主目标都没有通过严格 `teacher gate`，所有生成器训练仍然应该保持锁定。局部版本在某些数据集、某些 seed、某些目标上出现正向信号，但这些信号没有稳定跨 seed、跨数据集、跨窗口，也没有通过反事实归因门槛。

---

## 1. 这里比较的 baseline 是什么

下表中的核心列含义如下。

| 名称 | 中文解释 | 如何解读 |
| --- | --- | --- |
| `history_only` | 只用历史残差或历史模式的基线 | 这是很强的保守 baseline。新漂移信号必须超过它，才说明当前 drift signal 有额外价值。 |
| `fluctuation_plus_history` | 历史模式加当前版本的漂移或波动信号 | 如果它比 `history_only` 好，说明当前版本的漂移信号有增量价值。 |
| `Delta MSE vs history` | `fluctuation_plus_history` 的 MSE gain 减去 `history_only` 的 MSE gain | 正数表示漂移信号超过历史基线；负数表示漂移信号不如只用历史。 |
| `Delta MAE vs history` | 同上，但用 MAE 衡量 | 正数表示 MAE 上超过历史基线。 |
| `pos/win` | 滚动验证窗口中有多少窗口是正收益 | 比 aggregate 更严格。一个版本如果只在平均值上正，但很多窗口伤害预测，不应推广。 |
| `CF margin` | 真实漂移信号相对反事实对照的安全边际 | 负数表示真实信号没有赢过反事实或安全边际不足。 |
| `gate` | 严格教师门控是否通过 | 只有主目标 gate 通过，才有资格考虑生成器训练。当前表中均为 `False`。 |

注意：`history_only` 和 `fluctuation_plus_history` 列是 teacher-only 诊断中的“收益值”，不是最终在线预测 MSE 本身。收益越高越好；`Delta MSE vs history` 越高越好。

---

## 2. 总体趋势

`V18` 到 `V25`：主要在结构语义上不断修正，但主目标表现总体偏弱。少数 run 的 `residual_change_vector` 或 `raw_residual_vector` 有小正数，但反事实 margin 和窗口稳定性不过关。

`V26`：多尺度变化读出带来一些 raw-side tuning 信号，例如 `v26hyb_weather_s42_b240_01` 的 raw `Delta MSE vs history = 0.010245`，但 seed43 和 panama 转移失败。

`V27`：是 `V18` 之后最明显的正向阶段之一。`weather b60` 在 seed42 和 seed43 上 raw-side 都为正，panama seed42 的 `residual_change_vector` 也为正。但这些信号仍然无法通过反事实和窗口门控。

`V28`：support/suppression release 让 panama seed42 的 change target 有改善，尤其 support-push 下 `residual_change_vector Delta MSE vs history = 0.041691`，但同一方向损伤 weather raw 稳定性。

`V29`：结构上最接近 expert-local invariant-release 思路。`stability_release_deconstrained` 让 weather raw aggregate 明显转正，但反事实 margin 很差；后续 gate-aware selector 和 mode gate 只能作为保守模式选择治理，不是 generator-unlocked online forecasting 证据。

---

## 3. 主要版本数值表

表中 `F+H` 表示 `fluctuation_plus_history`。为了控制长度，只列主要正式 run、筛选 run 和调参中最关键的代表 run。完整原始 CSV 位于 `logs/**/teacher_only_validation_partial_gain_vs_history_v*.csv` 和 `logs/**/teacher_only_*_target_gate.csv`。

| V | run | data | seed | target | history MSE | F+H MSE | Delta MSE | Delta MAE | pos/win | min Delta MSE | CF margin | gate |
| -: | --- | --- | ---: | --- | -: | -: | -: | -: | --- | -: | -: | --- |
| V18 | `v18_weather_s42_b480_01` | weather | 42 | raw | 0.176281 | 0.101616 | -0.074665 | -0.077379 | 1/5 | -0.275217 | -0.046464 | False |
| V18 | `v18_weather_s42_b480_01` | weather | 42 | change | 0.000000 | 0.002438 | 0.002438 | 0.003024 | 4/5 | -0.000963 | -0.002059 | False |
| V18 | `v18_weather_s42_b480_02` | weather | 42 | raw | 0.292427 | 0.131395 | -0.161032 | -0.163009 | 0/5 | -0.440456 | -0.023176 | False |
| V18 | `v18_weather_s42_b480_02` | weather | 42 | change | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0/5 | 0.000000 | 0.000000 | False |
| V18 | `v18bal_weather_s42_b240_01` | weather | 42 | raw | 0.011024 | 0.018097 | 0.007073 | 0.003087 | 2/5 | -0.019211 | -0.007589 | False |
| V18 | `v18bal_weather_s42_b480_01` | weather | 42 | raw | 0.227268 | 0.091978 | -0.135290 | -0.134556 | 0/5 | -0.409908 | -0.032775 | False |
| V18 | `v18bal_weather_s42_b480_01` | weather | 42 | change | 0.000000 | 0.001395 | 0.001395 | 0.001596 | 4/5 | -0.000017 | -0.000428 | False |
| V19 | `v19_weather_s42_b240_01` | weather | 42 | raw | 0.244660 | 0.255476 | 0.010816 | 0.006465 | 1/5 | -0.123140 | -0.064644 | False |
| V19 | `v19_weather_s42_b240_01` | weather | 42 | change | 0.022770 | 0.000000 | -0.022770 | -0.027663 | 0/5 | -0.069235 | 0.000000 | False |
| V20 | `v20rel_weather_s42_b240_01` | weather | 42 | raw | 0.192547 | 0.185956 | -0.006591 | 0.011870 | 2/5 | -0.097270 | -0.091555 | False |
| V20 | `v20rel_weather_s42_b240_01` | weather | 42 | change | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0/5 | 0.000000 | 0.000000 | False |
| V21 | `v21_weather_s42_b240_01` | weather | 42 | raw | 0.241845 | 0.236485 | -0.005361 | 0.062561 | 2/5 | -0.095819 | -0.049228 | False |
| V21 | `v21_weather_s42_b240_01` | weather | 42 | change | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0/5 | 0.000000 | 0.000000 | False |
| V22 | `v22_weather_s42_b240_01` | weather | 42 | raw | 0.222829 | 0.183609 | -0.039220 | 0.012321 | 1/5 | -0.251358 | -0.202345 | False |
| V23 | `v23_weather_s42_b240_01` | weather | 42 | raw | 0.244778 | 0.222601 | -0.022177 | 0.009619 | 2/5 | -0.144693 | -0.071006 | False |
| V23 | `v23_weather_s42_b240_01` | weather | 42 | change | 0.000000 | 0.000417 | 0.000417 | -0.000283 | 3/5 | -0.004449 | -0.004449 | False |
| V23 | `v23_weather_s43_b240_01` | weather | 43 | raw | 0.201233 | 0.175794 | -0.025439 | -0.022056 | 1/5 | -0.137374 | -0.146421 | False |
| V23 | `v23_weather_s43_b240_01` | weather | 43 | change | 0.056420 | 0.000000 | -0.056420 | -0.045433 | 0/5 | -0.133194 | 0.000000 | False |
| V24 | `v24_weather_s42_b240_01` | weather | 42 | raw | 0.230659 | 0.207785 | -0.022874 | -0.020319 | 1/5 | -0.206192 | -0.049756 | False |
| V24 | `v24_weather_s42_b240_01` | weather | 42 | change | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0/5 | 0.000000 | 0.000000 | False |
| V25 | `v25_weather_s42_b240_01` | weather | 42 | raw | 0.213306 | 0.144924 | -0.068382 | -0.144357 | 1/5 | -0.373853 | -0.050604 | False |
| V25 | `v25_weather_s42_b240_01` | weather | 42 | change | 0.089490 | 0.000000 | -0.089490 | -0.066251 | 0/5 | -0.144102 | 0.000000 | False |
| V26 | `v26_weather_s42_b240_01` | weather | 42 | raw | 0.237700 | 0.229302 | -0.008398 | -0.047519 | 1/5 | -0.115048 | -0.101337 | False |
| V26 | `v26hyb_weather_s42_b240_01` | weather | 42 | raw | 0.233565 | 0.243810 | 0.010245 | 0.020707 | 2/5 | -0.120923 | -0.008465 | False |
| V26 | `v26hyb_weather_s43_b240_01` | weather | 43 | raw | 0.230449 | 0.221178 | -0.009271 | 0.002019 | 1/5 | -0.077121 | -0.080373 | False |
| V26 | `v26hyb_panama_s42_b240_01` | panama | 42 | raw | 0.379670 | 0.314234 | -0.065435 | -0.092409 | 1/5 | -0.266842 | -0.076274 | False |
| V26 | `v26hyb_panama_s42_b240_01` | panama | 42 | change | 0.304497 | 0.026512 | -0.277985 | -0.225582 | 0/5 | -0.511337 | -0.026612 | False |
| V27 | `v27_weather_s42_b60_02` | weather | 42 | raw | 0.644507 | 0.839662 | 0.195155 | 0.143591 | 2/5 | -0.222272 | -0.071075 | False |
| V27 | `v27_weather_s43_b60_01` | weather | 43 | raw | 0.647706 | 0.758064 | 0.110359 | 0.080191 | 2/5 | -0.297580 | -0.136993 | False |
| V27 | `v27_panama_s42_b60_01` | panama | 42 | change | 0.012794 | 0.064182 | 0.051388 | 0.027674 | 4/5 | 0.014264 | -0.042183 | False |
| V27 | `v27_panama_s43_b60_01` | panama | 43 | raw | 0.052452 | 0.168280 | 0.115828 | 0.072393 | 4/5 | -0.017011 | -0.024853 | False |
| V28 | `v28_weather_s42_b60_01` | weather | 42 | raw | 0.375743 | 0.224744 | -0.150998 | -0.028576 | 3/5 | -0.628605 | -1.491706 | False |
| V28 | `v28_weather_s43_b60_01` | weather | 43 | raw | 0.323875 | 0.412522 | 0.088647 | 0.163085 | 3/5 | -0.294888 | -1.171826 | False |
| V28 | `v28_panama_s42_b60_01` | panama | 42 | change | 0.018290 | 0.022931 | 0.004641 | -0.002969 | 2/5 | -0.098116 | -0.032167 | False |
| V28 | `v28sup_panama_s42_b60_01` | panama | 42 | raw | 0.078729 | 0.095621 | 0.016892 | 0.005373 | 2/5 | -0.174305 | -0.028133 | False |
| V28 | `v28sup_panama_s42_b60_01` | panama | 42 | change | 0.002612 | 0.044304 | 0.041691 | 0.025303 | 3/5 | -0.029099 | -0.041682 | False |
| V28 | `v28sup_weather_s43_b60_01` | weather | 43 | raw | 0.414947 | 0.341286 | -0.073661 | 0.021715 | 4/5 | -0.509984 | -0.279963 | False |
| V29 | `v29_weather_s42_b60_01` | weather | 42 | raw | 0.265971 | -0.062749 | -0.328720 | -0.233114 | 2/5 | -0.664736 | -0.298687 | False |
| V29 | `v29_panama_s42_b60_01` | panama | 42 | raw | 0.107443 | 0.057318 | -0.050125 | -0.047135 | 0/5 | -0.223377 | -0.014771 | False |
| V29 | `v29stable_weather_s42_b60_01` | weather | 42 | raw | 0.132637 | 0.480089 | 0.347452 | 0.261412 | 4/5 | -0.146506 | -0.724609 | False |
| V29 | `v29stable_weather_s42_b60_01` | weather | 42 | change | -0.002388 | 0.000000 | 0.002388 | 0.006411 | 2/5 | -0.001605 | 0.000000 | False |
| V29 | `v29stable_panama_s42_b60_01` | panama | 42 | raw | 0.060009 | 0.026431 | -0.033579 | -0.018689 | 2/5 | -0.144569 | -0.051453 | False |
| V29 | `v29stable_weather_s42_b60g60e1_epochdiag01` | weather | 42 | raw | 0.663458 | 0.736526 | 0.073067 | -0.023886 | 2/5 | -0.346788 | -0.011714 | False |
| V29 | `v29stable_weather_s42_b60g60e1_epochdiag01` | weather | 42 | change | 0.048210 | 0.000000 | -0.048210 | -0.213653 | 1/5 | -0.618583 | 0.000000 | False |

---

## 4. V29 mode-gate 与固定 baseline 的对比

`V29` 后续做了 complete-queue online mode-gate replay。这个实验不是生成器解锁后的真实在线预测，而是基于已有 teacher-outcome 的模式选择回放。

| 策略 | Mean MSE gain | harmful windows | 说明 |
| --- | -: | -: | --- |
| fixed `history_only` | 0.168302 | 6 | 固定只用历史模式。 |
| fixed `fluctuation_plus_history` | 0.201850 | 18 | 固定总是使用波动加历史模式，平均更高但伤害窗口很多。 |
| selected mode gate, threshold 0.15 | 0.255461 | 0 | 通过保守门控选择模式，大量 fallback，避免有害窗口。 |
| oracle cf-safe | 0.327554 | 未作为可部署策略 | 理想上界，不可直接作为部署结果。 |

对应工件：

- `artifacts/stage3_v29_tuning_selection_full_queue/v29_queue_complete_mode_gate_threshold_sweep.md`
- `artifacts/stage3_v29_tuning_selection_full_queue/v29_queue_complete_online_mode_gate_conf015_outcome_replay_summary.json`

解释：mode gate 说明“保守选择机制”有治理价值，但它的 scope 是 `mode_selection_only_no_generator_unlock`。它没有改变 `teacher gate` 未通过的事实，也不能证明 Stage III generator 在线适应已经有效。

---

## 5. 当前最重要的科学判断

1. `V18` 到 `V26` 没有形成稳定优于 `history_only` 的主目标证据。
2. `V27` 开始出现更明显 raw-side 正信号，尤其 weather seed42/43 和 panama seed43 的 raw target，但窗口门控和反事实归因仍失败。
3. `V28` 的 support-push 能提升 panama seed42 的 change target，但不能迁移到 weather，说明它不是稳健通用改进。
4. `V29` 的稳定化配置能在 weather 上产生很大的 aggregate raw gain，但反事实 margin 很差，说明它可能学到了有用信号，也可能被不安全或不可归因因素驱动。
5. V29 complete mode-gate 可以减少 harmful windows，并在回放中超过固定 `history_only` 和固定 `fluctuation_plus_history`，但它是模式选择治理证据，不是 generator-unlocked online forecasting 证据。
6. 到目前为止，所有 `V18` 到 `V29` 的主线实验都仍然应该写成：`do_not_train_generator`。

---

## 6. 可以对论文怎么说，不能怎么说

可以谨慎说：

> Later Stage III versions progressively improve the structure of drift representation, release-state modeling, and mode selection. Some versions show localized positive teacher-only signals relative to history-only baselines, especially V27 raw-side targets, V28 panama support-push, and V29 conservative mode-gate replay.

中文含义：

> 后续 Stage III 版本逐步改进了漂移表示、残差释放状态和模式选择结构。部分版本相对 history-only baseline 显示局部正向 teacher-only 信号，尤其是 V27 的 raw-side target、V28 的 panama support-push，以及 V29 的保守 mode-gate 回放。

不能说：

> 不能把 V18 到 V29 解读为 Stage III 在线学习已经获得可投稿级优势。

也不能说：

> V29 已经解锁生成器并形成真实在线预测优势。

当前证据只支持“局部信号和治理机制正在变清楚”，不支持“最终在线学习方法已经成功”。

---

## 7. 证据来源

主要来源：

- `docs/audit/STAGE3_V18_STRUCTURED_ENV_RELEASE_IMPLEMENTATION_RECORD.md`
- `docs/audit/STAGE3_V19_CONDITIONAL_RELEASE_IMPLEMENTATION_RECORD.md`
- `docs/audit/STAGE3_V20_CORRECTIVE_BUDGETED_DUAL_STREAM_IMPLEMENTATION_RECORD.md`
- `docs/audit/STAGE3_V21_RELEASE_FIRST_BUDGETED_IMPLEMENTATION_RECORD.md`
- `docs/audit/STAGE3_V22_RELEASE_DRIVEN_IMPLEMENTATION_RECORD.md`
- `docs/audit/STAGE3_V23_OBSERVABLE_DRIFT_IMPLEMENTATION_RECORD.md`
- `docs/audit/STAGE3_V24_TAIL_OBSERVABLE_CHANGE_IMPLEMENTATION_RECORD.md`
- `docs/audit/STAGE3_V25_CHANGE_QUALITY_IMPLEMENTATION_RECORD.md`
- `docs/audit/STAGE3_V26_MULTISCALE_CHANGE_IMPLEMENTATION_RECORD.md`
- `docs/audit/STAGE3_V27_QUALITY_CONDITIONED_DUAL_STREAM_IMPLEMENTATION_RECORD.md`
- `docs/audit/STAGE3_V28_SUPPORT_SUPPRESSION_RELEASE_IMPLEMENTATION_RECORD.md`
- `docs/audit/STAGE3_V29_EXPERT_LOCAL_INVARIANT_RELEASE_IMPLEMENTATION_RECORD.md`
- `docs/audit/STAGE3_V29_TUNING_PRIORITY_CORRECTION_REPORT.md`
- `logs/**/teacher_only_validation_partial_gain_vs_history_v*.csv`
- `logs/**/teacher_only_*_target_gate.csv`
- `artifacts/stage3_v29_tuning_selection_full_queue/*`
- `artifacts/stage3_v29_generator_unlock_preflight/*`
