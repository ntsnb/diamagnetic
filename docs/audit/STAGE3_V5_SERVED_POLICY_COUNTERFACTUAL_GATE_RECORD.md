# Stage III V5 Served-policy Counterfactual-gate Record

本记录对应 2026-05-27 的 V5/V5.1 审计修复。结论先行：

- V5 的目标是修复 V4 的两个剩余问题：checkpoint selection 会选择失败候选，以及 risk gate 没有把 counterfactual drift 当作负例。
- V5 首个 bounded probe 证明 counterfactual-aware risk gate 会过度保守：最终选择 threshold `0.8`，test real reject rate 为 `1.0`，所有 Stage III 更新被拒绝。
- V5.1 加入 positive class weight 和低阈值网格后，过度拒绝明显缓解：test real reject rate 降为 `0.0625`，test real-vs-best-CF margin 达 `0.005633`。
- 但是 V5.1 rolling validation 仍只有 `2/5` 通过，失败原因包括 `window_pass_rate;real_mse_gain;real_mae_gain;counterfactual`。因此 V5.1 仍不能支撑论文 claim，只能作为下一步修复方向。

## 1. Code changes

新增或修改的主要文件：

- `train/stage3_generator_selection_v2.py`
  - 新增 `score_served_policy_candidate()`。
  - selection score 显式加入 `worst_window_real_vs_best_counterfactual_mse_gain_margin`。
  - 对高 `max_window_real_reject_rate` 加强惩罚。
  - `select_best_generator_candidate(..., abstain_if_no_pass=True)` 可在所有候选失败时记录 abstain。

- `train/stage3_sensitivity_gate_v2.py`
  - rolling aggregate 新增：
    - `worst_window_real_vs_best_counterfactual_mse_gain_margin`
    - `max_window_real_reject_rate`

- `train/run_stage3_v2_sensitivity_audit.py`
  - risk gate training 可加入 counterfactual examples。
  - counterfactual risk-gate examples 默认 label 为 reject。
  - manifest 记录 `stage3_generator_selection_abstained` 和 abstain reason。

- `train/stage3_risk_gate_v2.py`
  - risk gate train metrics 新增 positive/negative recall、false positive/negative rate、predicted positive rate。

- `configs/stage3_v5_served_policy_counterfactual_gate.yaml`
  - V5 配置副本，不覆盖 V4。
  - risk gate 使用 counterfactual negative examples。
  - V5.1 更新加入 `positive_weight: 8.0` 和 threshold grid `[0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8]`。

## 2. Verification

已运行：

- `python -m pytest tests\test_stage3_generator_selection_v2.py tests\test_stage3_risk_gate_v2.py tests\test_stage3_sensitivity_gate_v2.py tests\test_stage3_v2_protocol_safety.py -q`
- 结果：`34 passed`

已运行：

- `python -m py_compile train\stage3_generator_selection_v2.py train\stage3_risk_gate_v2.py train\run_stage3_v2_sensitivity_audit.py`
- 结果：exit code `0`

YAML 解析：

- `configs/stage3_v5_served_policy_counterfactual_gate.yaml`
- 结果：可解析，threshold grid 和 positive weight 正确读入。

## 3. Aborted full-run note

尝试启动过 V5 20-epoch full generator run：

- run id: `codex_stage3_v5_full_panama_seed42_20260527_01`
- log root: `logs/load_benchmarks/stage3_v5/full_20260527`
- checkpoint root: `checkpoints/stage3_v5/full_20260527`

该进程运行 1 小时后工具超时，仍只写出 `generator_epoch_000.pt`，按当前速度完整 20 epoch 会超过十几个小时。为避免不可控半成品进程继续占用资源，已停止该 orphan process。该目录只作为 aborted artifact，不作为实验结果。

## 4. V5 bounded probe

Run：

- run id: `codex_stage3_v5_probe_panama_seed42_g3_b240_20260527_01`
- generator epochs: `3`
- max generator batches: `240`
- max eval batches: `240`
- output: `logs/load_benchmarks/stage3_v5/probe_20260527/codex_stage3_v5_probe_panama_seed42_g3_b240_20260527_01/panama/seed_42`

Key result：

| Metric | Value |
| --- | ---: |
| test claim gate | `False` |
| test failure | `real_mse_gain;real_mae_gain;counterfactual;reject_rate` |
| test real MSE gain | `0.0` |
| test real MAE gain | `0.0` |
| test real reject rate | `1.0` |
| selected threshold | `0.8` |
| rolling pass rate | `0/5` |

Diagnosis：

- counterfactual examples were correctly included, but the training set became highly imbalanced.
- risk gate training positive rate was only `0.1072`.
- The model learned a very conservative policy and rejected every test step.
- This is a risk-gate calibration failure, not evidence that generator adaptation is invalid.

## 5. V5.1 balanced bounded probe

Run：

- run id: `codex_stage3_v5_probe_panama_seed42_g3_b240_balanced_20260527_01`
- generator epochs: `3`
- max generator batches: `240`
- max eval batches: `240`
- output: `logs/load_benchmarks/stage3_v5/probe_20260527/codex_stage3_v5_probe_panama_seed42_g3_b240_balanced_20260527_01/panama/seed_42`

Test gate：

| Metric | Value |
| --- | ---: |
| claim_gate_pass | `False` |
| failure_reasons | `real_mae_gain` |
| real_mse_gain | `0.007531` |
| real_mae_gain | `0.001228` |
| best_counterfactual_mode | `amplified_2x` |
| best_counterfactual_mse_gain | `0.001898` |
| real_vs_best_counterfactual_mse_gain_margin | `0.005633` |
| real_reject_rate | `0.0625` |
| selected risk threshold | `0.05` |

Rolling validation：

| Metric | Value |
| --- | ---: |
| rolling_claim_gate_pass | `False` |
| rolling_failure_reasons | `window_pass_rate;real_mse_gain;real_mae_gain;counterfactual` |
| validation_window_pass_count | `2/5` |
| validation_window_pass_rate | `0.4` |
| worst_window_real_mse_gain | `0.001905` |
| worst_window_real_mae_gain | `-0.001290` |
| worst real-vs-best-CF margin | `-0.001715` |
| max real reject rate | `0.108333` |

Window breakdown：

| Window | Pass | Real MSE gain | Real MAE gain | Best CF | CF gain | Margin | Reject |
| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |
| 0 | `True` | `0.008154` | `0.009610` | `amplified_2x` | `0.001042` | `0.007112` | `0.108333` |
| 1 | `True` | `0.015205` | `0.012116` | `amplified_2x` | `0.001882` | `0.013323` | `0.087500` |
| 2 | `False` | `0.003991` | `0.001589` | `amplified_2x` | `0.005706` | `-0.001715` | `0.066667` |
| 3 | `False` | `0.004433` | `0.004719` | `amplified_2x` | `0.001535` | `0.002898` | `0.075000` |
| 4 | `False` | `0.001905` | `-0.001290` | `zero` | `0.0` | `0.001905` | `0.050000` |

Risk gate training：

| Metric | Value |
| --- | ---: |
| positive_rate | `0.107207` |
| predicted_positive_rate | `0.381179` |
| positive_recall | `0.95` |
| negative_recall | `0.687125` |
| false_positive_rate | `0.312875` |
| false_negative_rate | `0.05` |

## 6. Scientific interpretation

V5.1 is an improvement over the first V5 probe and over the V4 failure mode in one important sense:

- V4 had low reject rate but weak real/amplified separation.
- V5 first probe had strong counterfactual rejection but rejected real too.
- V5.1 partially balances both: low test reject rate and strong test real-vs-amplified margin.

However, V5.1 is still not publishable evidence:

- The run is bounded (`g3_b240`), not full-scale final evidence.
- Rolling validation pass rate is only `0.4`, below the pre-declared `1.0`.
- Window 2 still has amplified `2x` better than real.
- Window 4 has negative MAE gain.
- Generator selection still abstains because no pre-risk-gate generator candidate passed the claim gate.

## 7. Next step

The next model-side fix should not be manuscript rewriting. The safest next experiment is:

1. Keep V5.1 risk gate settings.
2. Add served-policy generator checkpoint selection, not only pre-risk-gate selection.
3. Select candidate by rolling validation after applying the calibrated risk gate.
4. Increase generator budget gradually, for example `g5_b480`, before attempting another very long full run.
5. Only if rolling validation passes should we run public-3 multi-seed full evaluation.

## 8. Follow-up: served-policy g5_b480 result

The served-policy repair was implemented and rerun with a non-downgraded
Panama seed42 `g5_b480` canary:

- run id: `v5g5s42r1`
- output: `logs/s3v5g5/v5g5s42r1/panama/seed_42`
- checkpoint root: `checkpoints/s3v5g5/panama/v5g5s42r1/seed_42`
- record: `docs/audit/STAGE3_V5_V6_G5_CANARY_RECORD.md`

Result:

- All served-policy candidates still abstained.
- Best candidate: epoch `4`.
- Rolling pass: `4/5`.
- Failure reasons: `window_pass_rate;real_mse_gain`.
- Worst-window MSE gain: `0.004620`, below the pre-declared `0.005`.
- Worst-window MAE gain: `0.004778`.
- Worst real-vs-best-counterfactual margin: `0.004575`.
- Max real reject rate: `0.095833`.

Interpretation:

- V5 g5_b480 is a near-miss repair signal, not validated Stage III evidence.
- The abstention behavior is correct and must remain visible.
- Do not update the manuscript to claim Stage III validity from this run.

## 9. Follow-up: larger served-policy g8_b960 result

The user requested no downgraded compute, so the same served-policy gate was
rerun with a larger generator budget:

- run id: `v5g8s42r1`
- output: `logs/s3v5g8/v5g8s42r1/panama/seed_42`
- checkpoint root: `checkpoints/s3v5g8/panama/v5g8s42r1/seed_42`
- budget: `generator_epochs=8`, `max_generator_batches=960`,
  `max_eval_batches=960`
- record: `docs/audit/STAGE3_V5_V6_G5_CANARY_RECORD.md`

Result:

- All served-policy candidates still abstained.
- Best rolling pass rate: `3/5`.
- Epoch `2` had the best MSE-positive profile among the candidates, but still
  failed with worst MSE gain `0.003612`, worst MAE gain `-0.002499`, and
  rolling pass `3/5`.
- Epochs `3` through `7` repeatedly failed windows `3;4`; most also failed the
  counterfactual margin.
- No `generator_served_policy.pt` was saved.

Interpretation:

- More generator training alone did not validate Stage III.
- The g5 near miss should be treated as a diagnostic clue, not evidence that
  the mechanism is ready for manuscript claims.
- The next useful step is failed-window mechanism diagnosis rather than public
  multi-seed evaluation.
