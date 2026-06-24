# Stage III V9 Residual-Teacher 实验记录

日期：2026-05-28

本记录说明 opt-in V9 分支的设计、实现、测试和 Panama seed42 同预算 canary 结果。它是科研完整性审计记录，不是论文正向性能证据。

## 1. 为什么要做 V9

V7 和 V8 之后，Stage III 的问题已经不再只是“drift 表示是否干净”或“generator 容量是否足够”。

已有证据如下：

- V7 使用 residualized RRCCA 后，`phi_v - phi_i` 的 invariant leakage 风险被更严肃地处理，但 rolling validation 仍然是 `0/5`。
- V8 允许 generator 同时更新 `output_head` 的 weight 和 bias，容量更大，但同预算 Panama seed42 canary 比 V7 更差，真实 drift 的 MSE/MAE gain 变成负数。

因此 V9 测试一个更深的假设：

```text
Stage III 失败的核心原因可能不是 generator 不够大，
而是 generator 没有一个明确的 residual-correction teacher，
不知道 drift 应该引导预测往哪个方向修正。
```

V9 的目标不是降低 claim gate，也不是把失败结果包装成成功结果，而是在不覆盖 legacy 代码的前提下，建立一条更可辩护的路径：

```text
train-only residual teacher
  -> teacher-aware drift canonicalization
  -> residual-basis hyper generator
  -> served-policy rolling validation
  -> 失败则显式 abstain
```

## 2. 实现范围

V9 是新增 opt-in 分支，没有替换 legacy Stage III。

新增或扩展的文件：

- `models/stage3_residual_teacher_v9.py`
  - 新增 `TrainOnlyResidualTeacherV9`。
  - 只允许在 `split="train"` 上 `fit()` 和 `update()`。
  - 使用 bias-free drift feature map：`[z, |z|, z^2, ||z||]`。
  - 目标是从 drift 特征预测 frozen-base residual `y - y_base`。
  - `zero drift` 的 residual correction 结构上为零。
  - 支持 checkpoint 保存、加载和 hash。
- `models/stage3_alignment_v9.py`
  - 新增 `TrainSetTeacherResidualizedReducedRankCCAAligner`。
  - 在 V7 residualized RRCCA 基础上加入 teacher residual 信息。
  - 初版 high-dimensional residualizer 在 smoke 中把 drift 压塌到 active rank `0`，随后改为 train-only low-rank PCA invariant residualizer。
  - 保留物理 zero-drift 保护：`transform_phi_pair(phi_i, phi_i)` 必须输出精确零 drift。
- `models/adaptation_v9.py`
  - 新增 `ResidualBasisHyperGeneratorV9`。
  - 继承 V8 的 weight+bias adapter 路径。
  - 增加 bias-free residual correction head。
  - `residual_correction(drift, reference_prediction)` 在 zero drift 时输出零修正。
- `train/stage3_generator_training.py`
  - 增加 V9 teacher/residual/cosine 损失项。
  - 支持 generator 的 residual correction head 参与训练。
  - 默认配置不启用 V9 项，因此不改变 V7/V8 行为。
- `train/online_loop_stage3_v2.py`
  - serving replay 时如果 generator 有 residual correction head，则在 parameter delta 之外叠加 residual correction。
- `train/run_stage3_v2_sensitivity_audit.py`
  - 支持 `canonicalizer_method: teacher_residual_rrcca_v9`。
  - 支持 `generator.type: residual_basis_hypergenerator_v9`。
  - 支持训练并保存 `residual_teacher_v9.pt`。
  - 风险门控和 served-policy selection 继续沿用严格 pre-declared gate。
- `train/stage3_v2_artifacts.py`
  - artifact bundle 可加载 V9 generator、V9 canonicalizer 和 V9 residual teacher。
- `configs/stage3_v9_residual_teacher_rrcca_served_policy_gate.yaml`
  - 新增 V9 配置。
  - 输出命名空间为 `logs/s3v9*`、`ckpt/s3v9*`、`results/s3v9*`。
  - strict gate 没有放松。
- `tests/test_stage3_v9_teacher_protocol.py`
  - 新增 V9 协议测试。

未覆盖的 legacy 文件：

- `models/adaptation.py`
- `models/factor_mechanism.py`
- `train/online_loop.py`
- 原始日志、checkpoint、表格和图像

## 3. V9 修复了哪些旧问题

V9 修复或缓解了这些旧问题：

1. **AdaptationGenerator 未训练的问题**
   - V9 generator 通过 `stage3_generator_training.py` 显式训练。
   - 训练目标包含预测误差、teacher correction、residual prediction 和 teacher cosine 项。

2. **zero drift 非零扰动的问题**
   - V9 residual teacher 和 residual correction head 都是 bias-free。
   - zero drift 在结构上不会生成 residual correction。
   - V9 测试覆盖 zero-drift identity。

3. **只改 weight 太弱的问题**
   - V9 沿用 V8 的 weight+bias adapter。
   - 但 V9 的实验结果说明，仅仅增加 action space 不够，update 方向仍然可能错。

4. **`phi_v - phi_i` 空间未对齐的问题**
   - V9 不直接相信 raw `phi_v - phi_i`。
   - 它先移除 invariant-explainable 部分，再用 teacher residual 方向做校准。
   - 同时保留 active-rank gate 和 zero-drift 保护。

5. **失败候选被误发布的问题**
   - V9 仍使用 served-policy selection。
   - 若所有候选都不通过 claim gate，则不保存 `generator_served_policy.pt`。

## 4. 测试

新增 V9 targeted tests：

```text
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_stage3_v9_teacher_protocol.py -q
7 passed
```

覆盖内容：

- teacher 只能用 train split 拟合；
- teacher 拒绝 validation/test update；
- toy residual teacher 能降低训练残差；
- teacher checkpoint round-trip；
- V9 teacher-aware aligner 的 physical zero drift identity；
- V9 generator/online zero-drift identity；
- artifact bundle 可加载 V9 generator、teacher、canonicalizer。

完整测试套件在 V9 实验前通过：

```text
C:\Users\admin\anaconda3\python.exe -m pytest tests -q
134 passed
```

## 5. Smoke 过程

V9 先做了多次 Panama seed42 smoke，命名空间为：

- logs: `logs/s3v9smoke/...`
- checkpoints: `ckpt/s3v9smoke/...`

### 5.1 第一次 smoke：high-dimensional residualizer 压塌 drift

第一次 smoke 的结果显示：

- active rank 为 `0`；
- canonicalizer 把 drift 几乎压塌；
- teacher residual MSE 没有改善；
- 所有 delta / shift 接近零。

诊断：

```text
teacher-aware residualization 太高维，在小样本 smoke 下过拟合并吸收了 drift。
```

修复：

- 改为 train-only low-rank PCA invariant residualizer；
- rank 受 `max_invariant_residualizer_rank=16` 和样本数限制；
- 只在样本足够时启用 quadratic residualizer。

### 5.2 第二次 smoke：teacher 可学习，但 correction 过大

修复 low-rank 后：

- post-canonical active rank 为 `5`；
- teacher train residual MSE 从 `0.355760` 降到 `0.257751`；
- teacher residual R2 为 `0.22794`；
- generator correction norm 和 candidate shift 过大；
- rolling validation 仍 `0/5`。

诊断：

```text
V9 teacher 路径开始工作，但 residual correction 太强，online 验证窗口受损。
```

修复：

- 降低 `max_scale`；
- 降低 `residual_scale`；
- 加强 shift budget。

### 5.3 第三、四次 smoke：结构更稳，但仍不满足 claim gate

降尺度后，V9 smoke 的 MSE 部分有所改善，但 MAE 和 counterfactual attribution 仍不稳定。

在第四次 smoke 中：

- selected epoch `0`；
- worst MSE gain 为 `+0.000978`；
- worst MAE gain 为 `-0.000937`；
- worst margin 为 `-0.000216`；
- max reject rate 为 `0.0`。

结论：

```text
V9 smoke 说明结构性 bug 被部分修复，
但还不能证明真实 drift 是最优或稳定有效的修正信号。
```

## 6. Same-Budget Canary

V9 同预算 canary 使用与 V7/V8 可比的 Panama seed42 设置：

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v2_sensitivity_audit.py --config configs\stage3_v9_residual_teacher_rrcca_served_policy_gate.yaml --dataset panama --run-id v9rt_s42_g5b480_01 --seed 42 --device auto --modes real zero shuffled sign_flipped amplified_2x amplified_5x random_matched_norm --validation-windows 5 --drift-source delta_phi_v --drift-polarity 1.0 --signal-mode delta_vector --normalize-vector --epochs 5 --invariant-epochs 5 --generator-epochs 5 --max-generator-batches 480 --max-eval-batches 480 --out-root logs\s3v9diag --checkpoint-root ckpt\s3v9diag
```

输出：

- logs: `logs/s3v9diag/v9rt_s42_g5b480_01/panama/seed_42`
- checkpoints: `ckpt/s3v9diag/panama/v9rt_s42_g5b480_01/seed_42`
- exit status: strict abstain
- final served artifact: `generator_served_policy.pt` 未保存

关键文件：

- `generator_checkpoint_selection_served_policy.csv`
- `served_policy_selected_candidate_window_breakdown.csv`
- `residual_teacher_v9_diagnostics.csv`
- `alignment_diagnostics.csv`
- `generator_train_metrics.csv`

## 7. Same-Budget 结果

### 7.1 Teacher 结果

`residual_teacher_v9_diagnostics.csv` 显示：

| 指标 | 数值 |
| --- | ---: |
| fit split | `train` |
| fit sample count | `480` |
| drift dim | `64` |
| feature dim | `193` |
| output dim | `12` |
| teacher residual MSE before | `0.1608959436` |
| teacher residual MSE after | `0.1450631768` |
| teacher residual R2 | `0.0926146507` |

解释：

- teacher 确实只在 train split 上学习；
- teacher 能解释一小部分 train residual；
- 但 R2 只有约 `9.26%`，说明当前 drift 对 residual 的可预测性很弱，或者 teacher 特征仍不够合适。

### 7.2 Alignment 结果

`alignment_diagnostics.csv` 显示：

| split | stage | active rank | top abs transition residual corr |
| --- | --- | ---: | ---: |
| train | pre | - | `0.438055` |
| train | post | `10` | `0.373576` |
| validation | pre | - | `0.280047` |
| validation | post | `10` | `0.356291` |
| test | pre | - | `0.344814` |
| test | post | `10` | `0.250567` |

解释：

- V9 canonicalizer 没有再压塌到 rank `0`；
- train/test 的 top correlation 下降，validation 的 top correlation 反而上升；
- 这说明 alignment 在不同 split 上不稳定，不足以作为性能 claim。

### 7.3 Generator train 结果

`generator_train_metrics.csv` 的 epoch mean 显示：

| epoch | rows | train base MSE | train blended MSE | teacher loss | residual loss | correction norm | delta norm | candidate shift |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 480 | `0.160896` | `0.160682` | `0.010854` | `0.081002` | `0.013751` | `0.000160` | `0.104609` |
| 1 | 480 | `0.160896` | `0.160413` | `0.011558` | `0.081911` | `0.018263` | `0.000294` | `0.116547` |
| 2 | 480 | `0.160896` | `0.159442` | `0.011097` | `0.080357` | `0.017799` | `0.000420` | `0.115935` |
| 3 | 480 | `0.160896` | `0.158339` | `0.010682` | `0.078726` | `0.017937` | `0.000568` | `0.114284` |
| 4 | 480 | `0.160896` | `0.157462` | `0.010062` | `0.077431` | `0.018329` | `0.000638` | `0.114343` |

解释：

- train blended MSE 随 epoch 下降；
- 但真实 rolling validation 更差；
- 因此 generator 的训练目标和 online validation 目标仍然错位。

### 7.4 Served-policy candidate selection

所有候选都失败：

| epoch | rolling pass | worst real MSE gain | worst real MAE gain | worst real-vs-best-CF margin | max reject rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `0/5` | `-0.007082` | `-0.005109` | `-0.007082` | `0.0` |
| 1 | `0/5` | `-0.010964` | `-0.008656` | `-0.010964` | `0.0` |
| 2 | `0/5` | `-0.010355` | `-0.009026` | `-0.010355` | `0.0` |
| 3 | `0/5` | `-0.012125` | `-0.011283` | `-0.012125` | `0.0` |
| 4 | `0/5` | `-0.014675` | `-0.012813` | `-0.014675` | `0.0` |

selected candidate 是 epoch `0`，因为它是最不差的候选，但它也没有通过 gate。

失败原因一致为：

```text
window_pass_rate;real_mse_gain;real_mae_gain;counterfactual
```

### 7.5 Window-level 诊断

selected epoch `0` 的五个 validation windows 中，real drift 全部为负：

| window | real MSE gain | real MAE gain | best counterfactual | best CF MSE gain | margin | delta norm pre-clamp | pred shift |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 0 | `-0.002918` | `-0.004959` | zero | `0.0000000175` | `-0.002918` | `0.316493` | `0.097266` |
| 1 | `-0.001387` | `-0.004059` | zero | `0.0000000013` | `-0.001387` | `0.315217` | `0.097493` |
| 2 | `-0.003993` | `-0.001566` | zero | `0.0000000068` | `-0.003993` | `0.316670` | `0.095934` |
| 3 | `-0.007082` | `-0.003191` | zero | `0.0000000053` | `-0.007082` | `0.318243` | `0.095585` |
| 4 | `-0.003996` | `-0.005109` | zero | `0.0000000014` | `-0.003996` | `0.315246` | `0.097284` |

这张表非常重要：

- zero drift 几乎不改变预测，这是结构修复的成功点；
- real drift 产生了稳定的 prediction shift；
- 但是这个 shift 在所有 validation windows 上都伤害 MSE/MAE；
- best counterfactual 是 zero，说明当前最安全策略是不使用 V9 real-drift correction。

## 8. 与 V7/V8 对比

| Run | selected epoch | rolling pass | worst MSE gain | worst MAE gain | worst margin | served artifact |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| V7 residualized RRCCA | 4 | `0/5` | `+0.000490` | `+0.000534` | `+0.000267` | no |
| V8 hyperadapter | 3 | `0/5` | `-0.008390` | `-0.005176` | `-0.008390` | no |
| V9 residual teacher | 0 | `0/5` | `-0.007082` | `-0.005109` | `-0.007082` | no |

解释：

- V9 比 V8 稍好，但仍明显差于 V7。
- V9 修复了 zero-drift identity 和 teacher supervision 这类结构问题。
- 但 V9 没有把结构修复转化成可验证的在线预测收益。
- 因此不能把 V9 写成“Stage III 终于验证成功”。

## 9. 科学结论

V9 的结论应当分成两层。

第一层：工程和协议上，V9 是进步。

- generator 被训练；
- teacher 只用 train split；
- zero drift identity 被保护；
- residual teacher 可以被 checkpoint；
- failed candidates 不会被发布成 served artifact；
- 每个候选都有 rolling-window 和 counterfactual 证据。

第二层：论文 claim 上，V9 仍然失败。

- 真实 drift 在同预算 Panama seed42 canary 的五个 validation windows 全部产生负 MSE/MAE gain；
- zero drift 是最佳 counterfactual；
- teacher train R2 很低；
- generator train loss 下降但 validation 变差；
- 没有保存 `generator_served_policy.pt`。

因此：

```text
V9 不能支持 Stage III learns concept drift。
V9 不能支持 Stage III reliably improves online load forecasting。
V9 不能支持 SOTA 或显著优势叙事。
```

当前最可辩护的表述是：

```text
Stage III is retained as an audited transient-adaptation research module.
The V9 repair introduces train-only residual supervision and zero-drift identity,
but validation-gated evidence still shows that converting representation drift
into reliable online residual correction remains unsolved.
```

中文可写为：

```text
Stage III 可作为一个经过严格审计的瞬时在线适应候选机制保留。
V9 引入了仅基于训练集的残差教师、teacher-aware 表示校准和 zero-drift identity 约束，
但当前验证门控结果仍显示：把表示漂移稳定转换为有益的在线残差修正尚未被证明。
```

## 10. 下一步建议

后续已经补做 teacher-only 诊断，记录见：

- `docs/audit/STAGE3_V9_TEACHER_ONLY_DIAGNOSTIC_RECORD.md`

该诊断显示：即使完全绕过 generator，只用 `y_base + teacher(z)`，Panama seed42 同预算 rolling validation 仍失败 `0/5`，worst MSE gain 为 `-0.197784`，worst MAE gain 为 `-0.118167`，每个窗口都是 zero drift 最好。因此 V9 的失败已经被定位到更上游的 drift-to-residual teacher 迁移问题。

不建议把 V9 直接扩展到 public-3 作为性能实验。这不是因为计算成本，而是因为 pre-declared Panama seed42 canary 和 teacher-only 诊断都已经失败。继续扩大同一失败机制，只会产生更多不可用于 claim 的负结果。

如果继续研究 Stage III，建议下一步改为诊断性实验，而不是性能扩展：

1. **先做 teacher 上界诊断**
   - 在 validation windows 上只评估 teacher residual correction 的可迁移性。
   - 不经过 generator，不更新参数。
   - 判断 teacher 本身是否能离开 train split 后改善 residual。

2. **做 oracle residual sign diagnostic**
   - 检查真实 residual 与 teacher correction 的符号一致率。
   - 如果符号经常相反，说明 drift-to-residual 映射在验证流上不稳定。

3. **把 generator 训练目标改成 rolling-window meta-objective**
   - 目前 train mini-batch loss 下降但 validation 恶化。
   - 下一版不能只优化训练窗口上的 blended MSE。

4. **先报告 negative result，再决定是否继续重构**
   - 当前 Stage III 最强的论文价值可能不是性能提升，而是揭示 drift-conditioned online adaptation 在负荷预测中的失败模式和审计方法。

## 11. 审计决策

V9 不进入论文主结果表。

V9 可以进入内部审计或附录式负结果说明，条件是明确写出：

- 它是 opt-in repair branch；
- 它没有替代 legacy paper table；
- 它没有通过 rolling validation；
- 它没有保存 served artifact；
- 它不能证明 Stage III 学会了 concept drift。
