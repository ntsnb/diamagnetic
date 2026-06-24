# Stage III V2 实施记录

日期：2026-05-25

本文记录 Stage III V2 的代码修复进展、已验证行为、尚未验证内容和下一步实验计划。本文不声称模型在真实负荷数据上已经提升；真实提升必须由后续隔离输出目录下的重新训练结果证明。

---

## 1. 本轮目标

基于前期审计和 `docs/superpowers/plans/2026-05-24-stage3-generator-repair-plan.md`，本轮开始将 Stage III 从“未训练、随机、scalar drift 的 transient perturbation”重构为 opt-in 的 V2 路径。

本轮遵循三条约束：

1. 不覆盖旧模型、旧训练循环、旧实验日志、旧 checkpoint。
2. 新增 V2 文件和测试，旧 Stage III 仍可复现。
3. 先用单元测试验证机制，再进入真实数据训练和论文表述更新。

---

## 2. 已新增文件

### 2.1 VAE deterministic eval V2

文件：

- `models/factor_mechanism_v2.py`
- `tests/test_factor_mechanism_v2.py`

作用：

- 新增 `FactorMechanismV2`。
- 训练阶段仍使用 VAE reparameterization sampling。
- `eval()` 阶段默认使用 `mu_z`，保证同一 checkpoint 和同一输入输出一致。
- 如果显式设置 `vae_eval_mode="sample"`，才保留 eval 随机采样。

科学意义：

> 后续 MSE/MAE、Stage III candidate prediction、real/zero/shuffled drift 对比不再被 VAE eval 随机噪声污染。

### 2.2 Identity-preserving Stage III generator V2

文件：

- `models/adaptation_v2.py`
- `tests/test_adaptation_v2.py`
- `tests/test_hyperlora_generator_v2.py`

作用：

- 新增 `IdentitySignedSeparableAdaptationGenerator`。
- 新增 `LayerAwareHyperLoRAGenerator`。
- `drift = 0` 时严格输出 zero delta。
- 使用 bias-free signed conditioner，避免旧 generator 中 `sigmoid + 1.0` 导致的 zero drift 非零扰动。
- 支持 delta norm、clamp fraction 等诊断统计。
- HyperLoRA 版本会为每个目标 2D weight 注册独立 LoRA adapter，并共享一个 centered drift conditioner。
- HyperLoRA 的 `lora_b` 零初始化，因此初始状态是 identity；非零 drift 仍能给 `lora_b` 产生梯度。

科学意义：

> 新 generator 不再把“没有 drift”解释为“应该修改参数”，这是 drift-conditioned adaptation 的必要条件。

注意：

当前 V2 runner 默认已切换到 layer-aware HyperLoRA。signed separable adapter 保留为轻量对照和单元测试用。

### 2.3 对齐表示模块

文件：

- `models/aligned_representation_v2.py`
- `tests/test_aligned_representation_v2.py`

作用：

- 新增 `SharedResidualFactorEncoder`：

```text
x -> shared trunk -> phi_i
                 -> delta_phi_v
phi_v = phi_i + delta_phi_v
```

- 因此：

```text
phi_v - phi_i = delta_phi_v
```

- 新增 `TrainSetAffineAligner` 作为 legacy encoder 的 train-only fallback alignment。
- 新增 aligned representation losses，包括 forecast、residual、orthogonal、smooth、norm 项。

科学意义：

> V2 不再默认把两个互不对齐的独立 embedding 生硬相减，而是让 variant residual 与 invariant embedding 共享坐标基础。

### 2.4 Differentiable transient forward

文件：

- `train/stage3_functional.py`
- `tests/test_stage3_functional_v2.py`

作用：

- 使用 `torch.func.functional_call` 实现带 delta 的 temporary forward。
- 不使用 `.data`。
- 不在 `no_grad` 中执行 adapted forward。
- 保证 generator 训练时梯度可以从 adapted prediction 回传到 generator。
- 保证 base model 参数不会被 transient delta 污染。

科学意义：

> 旧 `temporary_params()` 适合推理时临时扰动，但不适合训练 generator。V2 训练必须使用 differentiable functional forward。

### 2.5 Generator training step

文件：

- `train/stage3_generator_training.py`
- `tests/test_stage3_generator_training_v2.py`

作用：

- 新增 `stage3_generator_train_step()`。
- 冻结 base model。
- 只更新 generator 参数。
- 记录 `base_mse`、`adapted_mse`、`delta_norm`、`zero_delta_norm`、`candidate_shift`。
- 提供 checkpoint 保存函数 `save_stage3_generator_checkpoint()`。

科学意义：

> 当前 V2 已经具备“训练 generator”的基础机制，不再依赖随机初始化 generator 直接上线。

### 2.6 Drift sensitivity interventions

文件：

- `train/stage3_sensitivity_v2.py`
- `tests/test_stage3_sensitivity_v2.py`

作用：

生成确定性的 drift 干预：

- `real`
- `zero`
- `shuffled`
- `sign_flipped`
- `amplified_2x`
- `amplified_5x`
- `random_matched_norm`

科学意义：

> 后续可以直接检验 Stage III V2 是否真正使用 drift 内容，而不是只依赖 clamp、blend 或 generator bias。

### 2.7 Opt-in online loop V2

文件：

- `train/online_loop_stage3_v2.py`
- `tests/test_online_loop_stage3_v2.py`

作用：

- 新增独立 V2 online loop。
- 不替换旧 `train/online_loop.py`。
- 没有 trained generator 时默认拒绝运行。
- 只有 `allow_untrained_generator: true` 时才允许诊断性未训练运行。
- 输出 Stage III V2 诊断：
  - use rate；
  - recon error；
  - prediction shift；
  - blend；
  - delta norm；
  - clamp fraction；
  - reject count。

科学意义：

> V2 online path 已经具备最基本的“拒绝未训练 generator”和“输出可审计诊断”的安全行为。

### 2.8 V2 minimal config

文件：

- `configs/stage3_v2_minimal.yaml`

作用：

- 提供 opt-in V2 配置草案。
- 默认 `vae_eval_mode: mean`。
- 默认 `stage3.version: v2`。
- 默认 `require_aligned_embeddings: true`。
- 输出根目录规划为：
  - `logs/load_benchmarks/stage3_v2/`
  - `checkpoints/stage3_v2/`
  - `results/stage3_v2/`

---

## 3. 已运行验证

命令：

```powershell
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_factor_mechanism_v2.py tests\test_adaptation_v2.py tests\test_hyperlora_generator_v2.py tests\test_aligned_representation_v2.py tests\test_offline_aligned_representation_v2.py tests\test_stage3_functional_v2.py tests\test_stage3_generator_training_v2.py tests\test_stage3_sensitivity_v2.py tests\test_online_loop_stage3_v2.py tests\test_stage3_v2_protocol_safety.py tests\test_stage3_signals.py tests\test_stage3_ablation_benchmark.py tests\test_shapes.py -q
```

结果：

```text
44 passed in 3.18s
```

已验证事实：

1. V2 eval 默认确定性。
2. V2 generator 在 zero drift 下不产生参数扰动。
3. V2 generator 可以通过 adapted prediction loss 得到梯度。
4. base model 在 generator training step 中保持冻结。
5. `phi_v = phi_i + delta_phi_v` 的 aligned residual 表示成立。
6. drift sensitivity interventions 可复现。
7. V2 online loop 缺少 trained generator 时会拒绝运行。
8. V2 online loop 的 zero drift identity 行为成立。
9. `drift_source=delta_phi_v` 可以直接使用 aligned residual，不再依赖外层重新解释 `phi_v - phi_i`。
10. `drift_polarity` 可以显式进入训练和 online sensitivity path，便于诊断符号方向问题。

---

## 4. 尚未完成

以下内容还没有完成，因此不能声称 Stage III V2 在真实负荷预测上已经提升：

1. 尚未完成 full `offline_aligned_representation` 训练脚本。
2. 尚未完成真实数据上的 generator 多 epoch 训练脚本。
3. 尚未完成 `run_stage3_sensitivity_audit.py` 对真实数据的 CSV 输出。
4. 尚未完成 public 3 dataset、多 seed 的重新训练。
5. 尚未完成 Lishui family 或 shifted protocol 的大规模验证。
6. 尚未完成 paired statistical comparison。
7. 尚未更新论文正文 claim。

---

## 5. 真实数据受限 smoke 记录

### 5.1 第一次未受限 smoke

命令：

```powershell
C:\Users\admin\anaconda3\python.exe -m train.run_stage3_v2_sensitivity_audit --config configs\stage3_v2_minimal.yaml --dataset panama --run-id codex_stage3_v2_smoke_20260525 --seed 42 --epochs 1 --invariant-epochs 1 --generator-epochs 1 --modes real zero --device cpu
```

结果：

```text
command timed out after 604026 milliseconds
```

结论：

- CPU 上即使 1 epoch 仍然偏慢。
- 超时前只创建了空输出目录，没有生成 checkpoint、manifest 或结果 CSV。
- 该运行没有产生可用实验结果。

### 5.2 batch-limited smoke

为验证真实数据入口和隔离输出流程，增加了 batch 限制参数：

- `--max-train-batches`
- `--max-generator-batches`
- `--max-eval-batches`

命令：

```powershell
C:\Users\admin\anaconda3\python.exe -m train.run_stage3_v2_sensitivity_audit --config configs\stage3_v2_minimal.yaml --dataset panama --run-id codex_stage3_v2_batch1_smoke_20260525 --seed 42 --epochs 1 --invariant-epochs 1 --generator-epochs 1 --modes real zero --device cpu --max-train-batches 1 --max-generator-batches 1 --max-eval-batches 1
```

结果：

```text
[stage3_v2] wrote audit outputs to logs\load_benchmarks\stage3_v2\codex_stage3_v2_batch1_smoke_20260525\panama\seed_42
```

生成文件：

- `logs/load_benchmarks/stage3_v2/codex_stage3_v2_batch1_smoke_20260525/panama/seed_42/offline_metrics.csv`
- `logs/load_benchmarks/stage3_v2/codex_stage3_v2_batch1_smoke_20260525/panama/seed_42/aligned_representation_metrics.csv`
- `logs/load_benchmarks/stage3_v2/codex_stage3_v2_batch1_smoke_20260525/panama/seed_42/generator_train_metrics.csv`
- `logs/load_benchmarks/stage3_v2/codex_stage3_v2_batch1_smoke_20260525/panama/seed_42/sensitivity_summary.csv`
- `logs/load_benchmarks/stage3_v2/codex_stage3_v2_batch1_smoke_20260525/panama/seed_42/run_manifest.json`
- `checkpoints/stage3_v2/panama/codex_stage3_v2_batch1_smoke_20260525/seed_42/base_model.pt`
- `checkpoints/stage3_v2/panama/codex_stage3_v2_batch1_smoke_20260525/seed_42/aligned_representation.pt`
- `checkpoints/stage3_v2/panama/codex_stage3_v2_batch1_smoke_20260525/seed_42/generator.pt`

核心 smoke 结果：

| mode | online_base_mse | online_stage3_mse | online_base_mae | online_stage3_mae | delta_norm_pre_clamp | pred_shift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| real | 0.4775509834 | 0.4775510728 | 0.5510783792 | 0.5510783195 | 0.0045360276 | 0.0000111030 |
| zero | 0.4775509834 | 0.4775509834 | 0.5510783792 | 0.5510783792 | 0.0000000000 | 0.0000000000 |

解释：

- 该 smoke 只使用了 1 个 train batch、1 个 generator batch、1 个 eval batch。
- 它证明 V2 runner 能在真实 `panama` 数据入口上完成隔离训练、checkpoint 保存、manifest 记录和 real/zero sensitivity 输出。
- 它不证明模型性能提升。
- `real` 相比 base 的 MSE 没有提升，且变化量极小；这不能作为论文结果。
- `zero` delta norm 为 0，符合 zero-drift identity 预期。

---

## 6. HyperLoRA strong bounded public-3 记录

### 6.1 配置

新增配置：

- `configs/stage3_v2_strong.yaml`

关键差异：

- `generator.type: layer_aware_hyperlora`
- `generator.rank: 8`
- `generator.max_scale: 0.5`
- `generator.lr: 0.003`
- `online.stage3.target_modules: [output_head]`
- `online.stage3.max_delta: 0.05`
- `online.stage3.blend_alpha: 0.35`

### 6.2 命令

`panama`：

```powershell
C:\Users\admin\anaconda3\python.exe -m train.run_stage3_v2_sensitivity_audit --config configs\stage3_v2_strong.yaml --dataset panama --run-id codex_stage3_v2_hyperlora_strong_panama_b100_20260525 --seed 42 --epochs 1 --invariant-epochs 1 --generator-epochs 5 --modes real zero shuffled sign_flipped amplified_2x random_matched_norm --max-train-batches 100 --max-generator-batches 100 --max-eval-batches 100
```

`tetouan`：

```powershell
C:\Users\admin\anaconda3\python.exe -m train.run_stage3_v2_sensitivity_audit --config configs\stage3_v2_strong.yaml --dataset tetouan --run-id codex_stage3_v2_hyperlora_strong_public_b100_20260525 --seed 42 --epochs 1 --invariant-epochs 1 --generator-epochs 5 --modes real zero shuffled sign_flipped amplified_2x random_matched_norm --max-train-batches 100 --max-generator-batches 100 --max-eval-batches 100
```

`spain_energy_weather`：

```powershell
C:\Users\admin\anaconda3\python.exe -m train.run_stage3_v2_sensitivity_audit --config configs\stage3_v2_strong.yaml --dataset spain_energy_weather --run-id codex_stage3_v2_hyperlora_strong_public_b100_20260525 --seed 42 --epochs 1 --invariant-epochs 1 --generator-epochs 5 --modes real zero shuffled sign_flipped amplified_2x random_matched_norm --max-train-batches 100 --max-generator-batches 100 --max-eval-batches 100
```

### 6.3 结果摘要

| dataset | mode | base MSE | V2 MSE | MSE gain | base MAE | V2 MAE | MAE gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| panama | real | 0.358965547 | 0.346360209 | 3.5116% | 0.422300885 | 0.413814158 | 2.0096% |
| panama | zero | 0.358965547 | 0.358965549 | -0.0000% | 0.422300885 | 0.422300884 | 0.0000% |
| panama | shuffled | 0.358965547 | 0.358415662 | 0.1532% | 0.422300885 | 0.421934804 | 0.0867% |
| panama | sign_flipped | 0.358965547 | 0.372164616 | -3.6770% | 0.422300885 | 0.431314158 | -2.1343% |
| tetouan | real | 0.277275934 | 0.267678418 | 3.4614% | 0.378744558 | 0.369416756 | 2.4628% |
| tetouan | zero | 0.277275934 | 0.277275931 | 0.0000% | 0.378744558 | 0.378744555 | 0.0000% |
| tetouan | shuffled | 0.277275934 | 0.276991531 | 0.1026% | 0.378744558 | 0.378453796 | 0.0768% |
| tetouan | sign_flipped | 0.277275934 | 0.273197021 | 1.4711% | 0.378744558 | 0.375581305 | 0.8352% |
| spain_energy_weather | real | 0.598551363 | 0.587709204 | 1.8114% | 0.605040031 | 0.599084869 | 0.9843% |
| spain_energy_weather | zero | 0.598551363 | 0.598551360 | 0.0000% | 0.605040031 | 0.605040033 | -0.0000% |
| spain_energy_weather | shuffled | 0.598551363 | 0.594954998 | 0.6008% | 0.605040031 | 0.602911072 | 0.3519% |
| spain_energy_weather | sign_flipped | 0.598551363 | 0.604750322 | -1.0357% | 0.605040031 | 0.607567134 | -0.4177% |

### 6.4 初步分析

已验证事实：

1. `real` 在三个 public 数据集的 bounded run 中均优于 base/zero。
2. `zero` 基本保持 identity，MSE/MAE 不变。
3. `shuffled` 的收益明显小于 `real`：
   - panama: `real 3.5116%` vs `shuffled 0.1532%`
   - tetouan: `real 3.4614%` vs `shuffled 0.1026%`
   - spain: `real 1.8114%` vs `shuffled 0.6008%`
4. `sign_flipped` 在 panama 和 spain 上变差，说明方向信息确实影响预测。
5. `amplified_2x` 与 `real` 几乎相同，说明当前 drift conditioner 或 downstream clamp/blend 可能已经饱和。

不能过度解释的点：

1. 这只是 seed 42。
2. 每个阶段只使用 100 batch，不是完整训练。
3. `tetouan` 的 `sign_flipped` 仍有正收益，但低于 `real`，说明方向干预不是所有数据集都严格反向。
4. `real` 优于 `zero/shuffled` 是积极信号，但还不能称统计显著。
5. 由于旧 Stage III / Full 在同一 protocol 下尚未重跑，当前还不能声称 V2 已经显著超过旧方案。

当前科学判断：

> Stage III V2 HyperLoRA 在 bounded public-3 实验中显示了比 zero/shuffled 更强的 real-drift adaptation 信号，并且 zero-drift identity 成立。这是一个值得扩展到多 seed 和完整训练的正向证据，但不是最终论文结论。

---

## 7. 协议审查后的修复

代码审查指出，上述 HyperLoRA strong bounded public-3 结果仍存在协议风险，因此该结果被降级为：

```text
修复前探索性结果，不可用于论文 claim。
```

已确认并修复的问题：

1. **窗口切分泄漏风险**  
   旧路径先从完整 series 构造所有 sliding windows，再切 train/val/test。对于 `horizon > 1`，边界附近的目标时间可能跨 split。  
   修复：新增 `train/stage3_v2_data.py`，V2 runner 默认使用 raw timeline split first，然后分别对 train/val/test 构造窗口。manifest 中记录：

```text
split_mode: chronological_raw_first
leakage_guard: raw_split_before_windowing
```

2. **online batch 内未来样本泄漏风险**  
   旧 V2 runner 用 batch size 64 生成 online drift，vector mode 会对 batch 维度求均值，导致当前样本使用同 batch 后续样本 embedding。  
   修复：V2 默认 `online_batch_size: 1`，并在 `run_online_adaptation_v2` 中当 `causal_online: true` 且 batch size 不为 1 时直接报错。

3. **aligned representation smoothness 批次顺序风险**  
   smoothness loss 使用相邻 row 差分，但原 train loader shuffle=True。  
   修复：V2 aligned representation 训练使用 ordered loader。

4. **config contract 不完整**  
   `require_aligned_embeddings: true` 以前没有强制执行。  
   修复：缺少 aligned embeddings 时 fail-fast。

5. **HyperLoRA checkpoint 不可重载风险**  
   旧 checkpoint 只保存 state_dict。  
   修复：checkpoint 现在保存 generator class、config、registered param shapes，并新增 round-trip load 测试。

已新增测试：

- `tests/test_stage3_v2_protocol_safety.py`

验证命令：

```powershell
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_stage3_v2_protocol_safety.py -q
```

结果：

```text
5 passed
```

完整轻量测试：

```text
40 passed
```

---

## 8. Causal-safe bounded public-3 记录

### 8.1 设置

配置：

- `configs/stage3_v2_strong.yaml`

协议修复：

- raw split before windowing；
- online batch size 1；
- ordered aligned representation training；
- aligned embeddings required；
- HyperLoRA checkpoint manifest。

运行规模：

- seed: `42`
- train batches: `20`
- generator batches: `20`
- eval batches: `20`
- base epochs: `1`
- aligned epochs: `1`
- generator epochs: `2`

### 8.2 结果摘要

| dataset | mode | base MSE | V2 MSE | MSE gain | base MAE | V2 MAE | MAE gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| panama | real | 0.412460954 | 0.445298255 | -7.9613% | 0.531847960 | 0.552319153 | -3.8491% |
| panama | zero | 0.412460954 | 0.412460946 | 0.0000% | 0.531847960 | 0.531847963 | -0.0000% |
| panama | shuffled | 0.412460954 | 0.410871636 | 0.3853% | 0.531847960 | 0.529111312 | 0.5146% |
| panama | sign_flipped | 0.412460954 | 0.379930006 | 7.8870% | 0.531847960 | 0.506987363 | 4.6744% |
| tetouan | real | 1.204716940 | 1.245623824 | -3.3956% | 1.002269991 | 1.017973123 | -1.5668% |
| tetouan | zero | 1.204716940 | 1.204716988 | -0.0000% | 1.002269991 | 1.002270013 | -0.0000% |
| tetouan | shuffled | 1.204716940 | 1.210122469 | -0.4487% | 1.002269991 | 1.004361461 | -0.2087% |
| tetouan | sign_flipped | 1.204716940 | 1.180129485 | 2.0409% | 1.002269991 | 0.994516894 | 0.7736% |
| spain_energy_weather | real | 0.504287973 | 0.502081996 | 0.4374% | 0.630403464 | 0.628779261 | 0.2576% |
| spain_energy_weather | zero | 0.504287973 | 0.504287969 | 0.0000% | 0.630403464 | 0.630403467 | -0.0000% |
| spain_energy_weather | shuffled | 0.504287973 | 0.501847041 | 0.4840% | 0.630403464 | 0.628635137 | 0.2805% |
| spain_energy_weather | sign_flipped | 0.504287973 | 0.501900744 | 0.4734% | 0.630403464 | 0.629116856 | 0.2041% |

### 8.3 解释

已验证事实：

1. `zero` 仍然保持 identity。
2. causal-safe 协议下，`real` 没有稳定优于 zero/shuffled/sign-flipped。
3. panama 和 tetouan 上 `real` 变差，而 `sign_flipped` 反而变好。
4. spain 上多个非真实干预都接近或超过 real。

科学含义：

> 当前 V2 HyperLoRA 机制虽然工程上更规范，但在 causal-safe bounded public-3 实验中没有证明 real drift 被正确利用。相反，结果提示 drift 符号、训练目标或 temporal drift 定义仍可能存在语义方向不稳定问题。

当前不应声称：

- Stage III V2 显著提升；
- Stage III V2 已学到正确 concept drift；
- `phi_v - phi_i` temporal delta 当前已经可靠。

下一步应优先做：

1. 检查 `delta_vector = (phi_v - phi_i)_t - (phi_v - phi_i)_{t-1}` 的符号定义是否和 generator 训练目标一致。
2. 在 generator training 中加入 validation-stream early stopping，而不是只用 train stream。
3. 对比 `real` 与 `sign_flipped` 作为正式 sensitivity pair。
4. 尝试 `variant_residual_vector = delta_phi_v(t)` 和 `variant_residual_delta = delta_phi_v(t)-delta_phi_v(t-1)` 两个更干净的 aligned residual drift。
5. 只有当 causal-safe、多 seed、多数据集结果支持 real drift，才更新论文 claim。

---

## 9. delta_phi_v 与 polarity 诊断记录

### 9.1 代码变更

本轮新增两个显式配置项：

```yaml
online:
  stage3:
    drift_source: delta_phi_v
    drift_polarity: 1.0
```

含义：

- `drift_source: phi_gap`：保留旧 V2 口径，stream 发送 `phi_i, phi_v`，online loop 内部计算 `phi_v - phi_i`。
- `drift_source: delta_phi_v`：直接发送 aligned representation 中的 `delta_phi_v`，stream 中设置 `phi_i = 0, phi_v = delta_phi_v`。
- `drift_polarity: 1.0`：正常使用 drift。
- `drift_polarity: -1.0`：训练和评估时同时使用反向 drift，用于诊断符号约定是否可能反了。

新增或更新文件：

- `train/run_stage3_v2_sensitivity_audit.py`
- `train/online_loop_stage3_v2.py`
- `train/stage3_signals.py`
- `configs/stage3_v2_strong.yaml`
- `tests/test_stage3_signals.py`
- `tests/test_online_loop_stage3_v2.py`
- `tests/test_stage3_v2_protocol_safety.py`

### 9.2 Panama b30 polarity 诊断

运行设置：

- dataset: `panama`
- seed: `42`
- base epochs: `2`
- aligned epochs: `2`
- generator epochs: `4`
- train/generator/eval batches: `30`
- split: raw timeline first
- online batch size: `1`
- drift source: `delta_phi_v`
- modes: `real, zero, shuffled, sign_flipped`

正向 polarity：

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v2_sensitivity_audit.py --config configs\stage3_v2_strong.yaml --dataset panama --run-id codex_stage3_v2_polarity_pos_panama_b30_20260525 --seed 42 --epochs 2 --invariant-epochs 2 --generator-epochs 4 --device cpu --modes real zero shuffled sign_flipped --max-train-batches 30 --max-generator-batches 30 --max-eval-batches 30 --drift-source delta_phi_v --drift-polarity 1.0
```

反向 polarity：

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v2_sensitivity_audit.py --config configs\stage3_v2_strong.yaml --dataset panama --run-id codex_stage3_v2_polarity_neg_panama_b30_20260525 --seed 42 --epochs 2 --invariant-epochs 2 --generator-epochs 4 --device cpu --modes real zero shuffled sign_flipped --max-train-batches 30 --max-generator-batches 30 --max-eval-batches 30 --drift-source delta_phi_v --drift-polarity -1.0
```

结果摘要：

| polarity | mode | MSE gain | MAE gain | use rate | reject count |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1.0 | real | -3.5689% | -1.9644% | 0.900 | 3 |
| 1.0 | zero | 0.0000% | 0.0000% | 1.000 | 0 |
| 1.0 | shuffled | 0.2302% | 0.3962% | 1.000 | 0 |
| 1.0 | sign_flipped | 2.2552% | 1.3576% | 0.900 | 3 |
| -1.0 | real | -3.5689% | -1.9644% | 0.900 | 3 |
| -1.0 | zero | 0.0000% | 0.0000% | 1.000 | 0 |
| -1.0 | shuffled | 0.2302% | 0.3962% | 1.000 | 0 |
| -1.0 | sign_flipped | 2.2552% | 1.3576% | 0.900 | 3 |

解释：

1. `delta_phi_v` 直接作为 drift source 后，`real` 仍未优于 `zero/shuffled/sign_flipped`。
2. `sign_flipped` 仍明显优于 `real`，说明 drift 方向或训练目标仍存在未解决的语义问题。
3. 训练和评估同时整体翻转 polarity 后，结果逐项相同。这说明全局符号翻转会被 HyperLoRA 参数学习吸收，不能作为真正修复。
4. 该诊断进一步支持当前判断：问题不只是 `phi_v - phi_i` 写法，而是 generator objective、drift target semantics 和 online validation 之间没有形成可靠闭环。

当前决策：

> 不更新论文主张，不声称 Stage III 已学到有效 concept drift。下一步应改 generator 训练目标，而不是继续调全局符号。

---

## 10. 下一步实验安全要求

在运行真实训练前，必须保证：

1. 所有输出写入新目录：
   - `logs/load_benchmarks/stage3_v2/<run_id>/`
   - `logs/load_benchmarks/stage3_v2_sensitivity/<run_id>/`
   - `checkpoints/stage3_v2/<dataset>/<run_id>/`
   - `results/stage3_v2/<run_id>/`
2. 不使用旧的全局 checkpoint 名称：
   - 不写 `checkpoints/{variant}_best.pt`。
3. 不追加污染旧全局日志：
   - 不写 `logs/offline_metrics.csv`。
   - 不写 `logs/test_metrics.csv`。
4. 每次实验保存 manifest：
   - command；
   - config hash；
   - git status summary；
   - seed；
   - dataset；
   - split；
   - Python / PyTorch / CUDA 版本。

---

## 11. 当前科学结论

当前可以说：

> Stage III V2 的机制底座已经建立，并通过轻量单元测试验证了 deterministic eval、zero-drift identity、aligned residual representation、differentiable generator training、drift sensitivity intervention、missing-generator rejection、raw-split-before-windowing、causal online batch 和 checkpoint round-trip。

当前不能说：

> Stage III V2 在真实负荷数据上显著优于旧 Stage III 或 SOTA。

原因：

> causal-safe bounded run 尚未支持 real-drift 优势；尚未完成多 seed、完整训练、旧 Stage III 同协议复跑和 paired statistical comparison。

---

## 12. Advanced V2 修复与 b50 诊断

日期：2026-05-25

### 12.1 本轮新增机制

根据 5.5/xhigh 子代理专家复核，本轮将 Stage III V2 从“只优化 adapted prediction loss”升级为更严格的 advanced path：

1. **Generator objective 加入 advantage / counterfactual 约束**
   - 文件：`train/stage3_generator_training.py`
   - 新增：
     - `lambda_improvement`
     - `improvement_margin`
     - `lambda_intervention_compare`
     - `intervention_margin`
     - `lambda_shift_budget`
     - `shift_budget`
   - 训练时可同时比较：
     - real drift；
     - zero drift；
     - sign-flipped drift；
     - previous/shuffled drift；
     - random matched-norm drift。

2. **Aligned representation 加入非线性独立性和防 collapse 约束**
   - 文件：`models/aligned_representation_v2.py`
   - 新增：
     - linear HSIC penalty；
     - VICReg-style variance penalty；
     - VICReg-style covariance penalty。

3. **Variant branch 支持 zero initialization**
   - 文件：`train/offline_aligned_representation_v2.py`
   - 目的：避免训练初期 `delta_phi_v` 直接变成任意 residual 垃圾桶。

4. **Train-only delta calibration**
   - 文件：`models/aligned_representation_v2.py`
   - 新增 `TrainSetDeltaStandardizer`。
   - runner 只用 train stream fit mean/std，然后用于 generator training 和 test sensitivity。
   - manifest 记录：

```text
stage3_delta_calibrated: true
stage3_delta_calibration_fit_sample_count: 50
```

5. **Sensitivity claim gate**
   - 文件：`train/stage3_sensitivity_gate_v2.py`
   - 新增输出：
     - `sensitivity_claim_gate.csv`
   - gate 只有在 `real` 优于 base 且优于 zero/shuffled/sign_flipped/random 等 counterfactual 时才通过。

### 12.2 验证

命令：

```powershell
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_factor_mechanism_v2.py tests\test_adaptation_v2.py tests\test_hyperlora_generator_v2.py tests\test_aligned_representation_v2.py tests\test_offline_aligned_representation_v2.py tests\test_stage3_functional_v2.py tests\test_stage3_generator_training_v2.py tests\test_stage3_sensitivity_v2.py tests\test_stage3_sensitivity_gate_v2.py tests\test_online_loop_stage3_v2.py tests\test_stage3_v2_protocol_safety.py tests\test_stage3_signals.py tests\test_stage3_ablation_benchmark.py tests\test_shapes.py -q
```

结果：

```text
50 passed in 3.30s
```

### 12.3 Panama b50 causal-safe 结果

运行：

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v2_sensitivity_audit.py --config configs\stage3_v2_strong.yaml --dataset panama --run-id codex_stage3_v2_adv_panama_b50_20260525 --seed 42 --epochs 2 --invariant-epochs 2 --generator-epochs 4 --device cpu --modes real zero shuffled sign_flipped amplified_2x random_matched_norm --max-train-batches 50 --max-generator-batches 50 --max-eval-batches 50 --drift-source delta_phi_v --drift-polarity 1.0
```

输出目录：

```text
logs/load_benchmarks/stage3_v2/codex_stage3_v2_adv_panama_b50_20260525/panama/seed_42/
```

结果：

| mode | base MSE | V2 MSE | MSE gain | base MAE | V2 MAE | MAE gain | delta norm | pred shift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| real | 0.326349954 | 0.328013048 | -0.5096% | 0.486295304 | 0.488819465 | -0.5191% | 0.290135 | 0.067722 |
| zero | 0.326349954 | 0.326349953 | 0.0000% | 0.486295304 | 0.486295300 | 0.0000% | 0.000000 | 0.000000 |
| shuffled | 0.326349954 | 0.325846718 | 0.1542% | 0.486295304 | 0.486183077 | 0.0231% | 0.286658 | 0.076613 |
| sign_flipped | 0.326349954 | 0.325328412 | 0.3130% | 0.486295304 | 0.482983772 | 0.6810% | 0.290135 | 0.075366 |
| amplified_2x | 0.326349954 | 0.328013047 | -0.5096% | 0.486295304 | 0.488819465 | -0.5191% | 0.290135 | 0.067722 |
| random_matched_norm | 0.326349954 | 0.328217084 | -0.5721% | 0.486295304 | 0.487551312 | -0.2583% | 0.246417 | 0.059458 |

claim gate 结果：

```text
claim_gate_pass: False
failure_reasons: real_mse_gain;counterfactual
real_mse_gain: -0.005096
best_counterfactual_mode: sign_flipped
best_counterfactual_mse_gain: 0.003130
real_vs_best_counterfactual_mse_gain_margin: -0.008226
```

### 12.4 解释

已验证事实：

1. 新 V2 advanced path 可以完成 causal-safe 训练、保存 generator、保存 delta standardizer、输出 sensitivity CSV。
2. zero drift 仍保持 identity。
3. real drift 产生了非零 delta 和 prediction shift，说明 generator 确实响应了 drift 输入。
4. 但 real drift 没有改善预测，且弱于 shuffled 和 sign-flipped。
5. `amplified_2x` 与 real 几乎完全相同，因为当前 `normalize_vector=True` 会抵消幅值放大；因此该干预不能作为幅值敏感性证据。

科学结论：

> 即使加入 improvement-aware objective、counterfactual loss、HSIC/VICReg、zero-init variant 和 train-only delta calibration，当前 Stage III V2 仍未通过 drift-conditioned claim gate。

当前仍不能说：

- Stage III V2 学到了有效 concept drift；
- `delta_phi_v` 已经是可靠 drift 表示；
- 该机制相较 zero/shuffled/sign-flipped 有显著优势；
- 可以据此更新论文创新点。

### 12.5 下一步建议

不要继续只调 polarity 或 max_scale。更合理的下一步是：

1. 对比 `signal_mode=level_vector` 与 `delta_vector`，因为当前 temporal delta 可能把有效 level residual 转成错误方向。
2. 关闭或改写 `normalize_vector` 后重跑幅值干预，否则 amplified sensitivity 没有解释力。
3. 将 generator target 从 output_head-only calibration 扩展为受控的 feature adapter / residual head adapter，但必须保留 claim gate。
4. 在 validation stream 上选择 checkpoint，不允许用 test sensitivity 反复调参。
5. 若下一轮仍失败，论文应把 Stage III 降级为 negative/diagnostic ablation，而不是核心贡献。

### 12.6 level_vector + no-normalization 对照

为了排查 `delta_vector` 和 `normalize_vector=True` 的影响，又运行了：

```text
codex_stage3_v2_adv_level_nonorm_panama_b30_20260525
```

关键结果：

| mode | MSE gain | MAE gain | use rate | reject count |
| --- | ---: | ---: | ---: | ---: |
| real | -0.9948% | -0.4538% | 0.1667 | 25 |
| zero | 0.0000% | 0.0000% | 1.0000 | 0 |
| shuffled | 0.2371% | 0.4306% | 0.6333 | 11 |
| sign_flipped | -0.5010% | -0.3470% | 0.3333 | 20 |
| random_matched_norm | 0.3769% | 0.0394% | 0.3000 | 21 |

claim gate：

```text
claim_gate_pass: False
failure_reasons: real_mse_gain;counterfactual
best_counterfactual_mode: random_matched_norm
```

结论：

> `level_vector` 和关闭 normalization 没有修复 Stage III；相反，未归一化 drift 导致 prediction shift 过大和大量 reject。这说明后续如果要研究幅值敏感性，需要先加入 drift norm clipping / validation-gated adapter selection，而不是直接使用原始幅值。

---

## 13. Validation gate 与在线协议修复

日期：2026-05-25

### 13.1 协议修复

根据 5.5/xhigh 子代理协议复核，本轮继续修复两个实验协议风险：

1. `run_online_adaptation_v2` 不再预先 `list(stream)`，改为逐 batch 消费 stream。
   - 文件：`train/online_loop_stage3_v2.py`
   - 新增测试：`test_v2_online_loop_consumes_stream_incrementally_without_prefetching_all_batches`
   - 输出新增：
     - `online_batch_count`
     - `online_sample_count`

2. runner 新增 validation sensitivity 和 validation gate。
   - validation split：`gate_role=selection`
   - test split：`gate_role=report_only`
   - 输出文件：
     - `validation_sensitivity_summary.csv`
     - `validation_sensitivity_claim_gate.csv`
     - `sensitivity_summary.csv`
     - `sensitivity_claim_gate.csv`

3. manifest 记录更多可追溯字段：
   - base/aligned/generator/delta standardizer checkpoint sha256；
   - `runner_stream_materialized: false`；
   - `stage3_validation_gate_role: selection`；
   - `stage3_test_gate_role: report_only`；
   - train/generator/eval batch caps。

4. claim gate 加严：
   - `min_real_mse_gain: 0.005`
   - `min_real_mae_gain: 0.002`
   - `counterfactual_margin: 0.002`
   - `max_real_reject_rate: 0.2`

### 13.2 验证

命令：

```powershell
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_factor_mechanism_v2.py tests\test_adaptation_v2.py tests\test_hyperlora_generator_v2.py tests\test_aligned_representation_v2.py tests\test_offline_aligned_representation_v2.py tests\test_stage3_functional_v2.py tests\test_stage3_generator_training_v2.py tests\test_stage3_sensitivity_v2.py tests\test_stage3_sensitivity_gate_v2.py tests\test_online_loop_stage3_v2.py tests\test_stage3_v2_protocol_safety.py tests\test_stage3_signals.py tests\test_stage3_ablation_benchmark.py tests\test_shapes.py -q
```

结果：

```text
54 passed in 3.32s
```

### 13.3 validation-gated b30 结果

运行：

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v2_sensitivity_audit.py --config configs\stage3_v2_strong.yaml --dataset panama --run-id codex_stage3_v2_valgate_panama_b30_20260525 --seed 42 --epochs 2 --invariant-epochs 2 --generator-epochs 4 --device cpu --modes real zero shuffled sign_flipped --max-train-batches 30 --max-generator-batches 30 --max-eval-batches 30 --drift-source delta_phi_v --drift-polarity 1.0
```

Validation gate：

| split | gate | real MSE gain | real MAE gain | best counterfactual | reject rate |
| --- | --- | ---: | ---: | --- | ---: |
| validation | pass | 1.6523% | 1.5913% | zero | 0.1000 |

Test report-only gate：

| split | gate | failure reasons | real MSE gain | real MAE gain | best counterfactual |
| --- | --- | --- | ---: | ---: | --- |
| test | fail | `real_mse_gain;real_mae_gain;counterfactual` | -3.5996% | -1.4569% | zero |

### 13.4 解释

这次结果非常重要：

1. 在 validation split 上，advanced Stage III V2 可以通过严格 gate。
2. 但冻结同一 checkpoint 和配置后，在 test split 上失败。
3. 这说明当前机制存在明显泛化不稳或 validation 过拟合风险。
4. 因为 test gate 是 `report_only`，不能回头用 test 失败结果继续调参后再把同一 test 当最终结果。

科学结论：

> 当前 advanced Stage III V2 还不能作为论文正向贡献。即使 validation gate 通过，test report-only gate 失败，说明 drift-conditioned adaptation 的泛化证据不足。

下一步如果继续修：

1. 必须增加多 validation windows 或 rolling validation，而不是单一 validation split。
2. 需要把 selection criterion 从单次 gate 改成跨 validation windows 的稳定性。
3. 需要解决数据读取阶段 split 前 time interpolation 可能跨边界使用未来信息的问题。
4. 若 rolling validation 仍无法泛化，论文应明确报告 Stage III 为 negative diagnostic ablation。

---

## 14. Rolling validation 与协议可解释性补丁

日期：2026-05-25

### 14.1 新增内容

根据二次代码仓审计，本轮补了两个轻量但关键的协议层输出：

1. rolling validation gate。
   - 文件：`train/stage3_sensitivity_gate_v2.py`
   - 文件：`train/run_stage3_v2_sensitivity_audit.py`
   - 新增输出：
     - `rolling_validation_sensitivity_summary.csv`
     - `rolling_validation_claim_gate.csv`
     - `rolling_validation_claim_gate_aggregate.csv`

2. sensitivity intervention 协议标记。
   - 每行新增：
     - `intervention_online_causal`
     - `intervention_protocol`
   - `real/zero/sign_flipped/amplified_*` 标记为 `streaming_causal`。
   - `shuffled/random_matched_norm` 标记为 `offline_diagnostic`。

3. manifest 新增风险说明。
   - `counterfactual_protocol_note`
   - `preprocessing_leakage_note`

4. strong 配置新增：
   - `validation_windows: 3`
   - `rolling_min_window_pass_rate: 1.0`

### 14.2 科学含义

rolling validation 的目的不是制造正结果，而是防止把单个 validation slice 的偶然成功误写成机制已经稳定。

新的 selection 逻辑是：

```text
只有多个 chronological validation windows 稳定通过 gate，
才允许冻结配置后进入 test report-only。
```

如果 rolling validation 或 test report-only 失败，论文不能声称 Stage III V2 已经学到稳定 concept drift adaptation。

### 14.3 仍未修复的最高风险

缺失值插补仍然发生在全量时间轴上：

```text
data/readers.py -> time_interpolate / ffill / bfill -> 后续才 raw split
```

这意味着当前仍不能声称完整 preprocessing pipeline 严格因果无泄漏。下一步必须实现 split-safe imputation，并补边界缺失值测试。

### 14.4 验证

新增/更新的单元测试：

```powershell
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_stage3_sensitivity_gate_v2.py -q
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_stage3_v2_protocol_safety.py -q
```

结果：

```text
4 passed
10 passed
```

Smoke run：

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v2_sensitivity_audit.py --config configs\stage3_v2_strong.yaml --dataset panama --run-id codex_stage3_v2_rolling_smoke_20260525 --seed 42 --epochs 1 --invariant-epochs 1 --generator-epochs 1 --device cpu --modes real zero --max-train-batches 2 --max-generator-batches 2 --max-eval-batches 2 --validation-windows 2 --drift-source delta_phi_v
```

输出目录：

```text
logs/load_benchmarks/stage3_v2/codex_stage3_v2_rolling_smoke_20260525/panama/seed_42
```

关键验证：

```text
rolling_validation_claim_gate_aggregate.csv 已生成
rolling_validation_sensitivity_summary.csv 已生成
run_manifest.json 已记录 rolling validation 与 preprocessing leakage note
```

该 smoke run 不是性能证据，只证明 runner 协议输出链路可用。

---

## 15. Stage III V2.1 split-safe b60 结果

日期：2026-05-25

详细记录见：

```text
docs/audit/STAGE3_V21_SPLITSAFE_EXPERIMENT_RECORD.md
```

### 15.1 修正

本轮新增：

- `data/readers_stage3_v2.py`：split-safe preprocessing；
- `Projected Residual Alignment V2.1`：
  - hard orthogonal projection；
  - invariant-only forecast loss；
  - Barlow cross-correlation suppression；
  - temporal ranking；
  - `phi_i` 与 `delta_phi_v` 双分支 VICReg 监控；
- strong/minimal V2 配置默认启用 split-safe imputation。

### 15.2 验证

命令：

```powershell
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_factor_mechanism_v2.py tests\test_adaptation_v2.py tests\test_hyperlora_generator_v2.py tests\test_aligned_representation_v2.py tests\test_offline_aligned_representation_v2.py tests\test_stage3_functional_v2.py tests\test_stage3_generator_training_v2.py tests\test_stage3_sensitivity_v2.py tests\test_stage3_sensitivity_gate_v2.py tests\test_online_loop_stage3_v2.py tests\test_stage3_v2_protocol_safety.py tests\test_stage3_v2_split_safe_preprocessing.py tests\test_stage3_signals.py tests\test_stage3_ablation_benchmark.py tests\test_shapes.py -q
```

结果：

```text
63 passed in 3.40s
```

### 15.3 Bounded retraining

运行：

```text
dataset: panama
seeds: 42, 43
epochs: base 2 / aligned 2 / generator 4
batch caps: train 60 / generator 60 / eval 60
validation_windows: 3
modes: real, zero, shuffled, sign_flipped, random_matched_norm
```

关键结果：

| seed | rolling pass rate | validation gate | test real MSE gain | test real MAE gain | test gate |
| --- | ---: | --- | ---: | ---: | --- |
| 42 | 0.6667 | fail | 0.4785% | -0.2272% | fail |
| 43 | 0.6667 | fail | 0.2498% | 0.1117% | fail |

表示几何：

| seed | val parallel_ratio |
| --- | ---: |
| 42 | 3.84e-08 |
| 43 | 2.84e-08 |

### 15.4 结论

V2.1 修复了两个重要工程/科研风险：

1. split-safe preprocessing 消除了全序列插补导致的未来信息泄漏风险；
2. hard projection 让 `delta_phi_v` 与 `phi_i` 的几何正交约束真正生效。

但它没有通过预测证据门槛：

1. 两个 seed 的 rolling validation 都失败；
2. validation 中仍有 counterfactual 优于 real；
3. test MSE 从历史负值改善为小幅正值，但 test 是 report-only，且 MAE 不稳定；
4. 当前不能写“显著提升”或“稳定 drift-conditioned adaptation”。

科学结论：

> Stage III V2.1 improves protocol rigor and representation auditability, but it is still not a verified positive contribution for the paper.

---

## 16. V2.3 Serving-Aligned 修复与 bounded retraining

日期：2026-05-25

详细记录见：

```text
docs/audit/STAGE3_V23_SERVING_ALIGNED_EXPERIMENT_RECORD.md
```

### 16.1 修正

本轮新增或修改：

- deterministic eval guard：Stage III V2 online/generator 入口拒绝 stochastic VAE eval；
- generator training：增加 served-output-aligned 的 candidate/blended metrics、soft reject proxy、L1 pred shift 口径；
- artifact reload：保存 generator/risk gate/delta standardizer 后重新加载再做 validation/test；
- risk gate：新增 logistic gate 与 checkpoint 兼容；
- aligned representation：新增 deployed temporal ranking 和 temporal contrastive loss；
- strong 配置：启用 logistic gate、soft reject 与部署时序 loss。

### 16.2 验证

目标测试：

```text
27 passed in 2.66s
```

Stage III V2 回归套件：

```text
75 passed in 3.90s
```

Smoke run 确认：

```text
stage3_artifacts_reloaded_for_eval = true
risk_gate model_type = logistic
test role = report_only
validation role = selection
```

### 16.3 Panama b60 结果

| seed | test MSE gain | test MAE gain | test gate | rolling pass rate | rolling gate |
| --- | ---: | ---: | --- | ---: | --- |
| 42 | -0.2431% | -0.4971% | fail | 0.6667 | fail |
| 43 | +0.7797% | +0.2071% | pass | 0.3333 | fail |

均值：

| metric | value |
| --- | ---: |
| mean test MSE gain | +0.2683% |
| mean test MAE gain | -0.1450% |
| mean rolling pass rate | 0.5000 |

### 16.4 结论

V2.3 通过了工程协议验证，但没有通过科研证据门槛：

1. seed42 退化；
2. seed43 test report-only 通过，但 rolling validation 失败；
3. 两个 seed 的 validation/rolling 证据不支持“稳定 drift-conditioned adaptation”；
4. 当前不应更新论文中的性能提升 claim。

安全表述：

> Stage III V2.3 improves protocol rigor by enforcing deterministic evaluation, artifact reload before online replay, and serving-aligned generator diagnostics. However, Panama b60 evidence remains seed-dependent and fails rolling validation, so it does not yet support a strong accuracy or SOTA claim.

---

## 17. V2.5 Canonical Drift + Generator Checkpoint Selection

日期：2026-05-25

详细记录见：

```text
docs/audit/STAGE3_V25_CANONICAL_CKPT_SELECTION_EXPERIMENT_RECORD.md
```

### 17.1 修正

本轮新增：

- generator checkpoint selection on rolling validation；
- `TrainSetResidualDriftCanonicalizer`；
- artifact bundle reload residual canonicalizer；
- claim gate 的 reject rate 改为 shift/risk/missing 三类拒绝总和；
- strong 配置启用 `canonicalize_delta` 和 generator checkpoint selection。

### 17.2 验证

完整 Stage III V2 回归套件：

```text
82 passed in 3.63s
```

Smoke run 确认：

```text
stage3_residual_drift_canonicalized = true
stage3_artifacts_reloaded_for_eval = true
stage3_generator_checkpoint_selection_enabled = true
```

### 17.3 Panama seed42/43 b60/b120

| seed | budget | test MSE gain | test MAE gain | rolling pass rate | test gate |
| ---: | --- | ---: | ---: | ---: | --- |
| 42 | b60 | -0.0938% | -0.3505% | 0.3333 | fail |
| 42 | b120 | +1.2656% | -0.4907% | 0.3333 | fail |
| 43 | b60 | -0.3250% | +0.1074% | 0.3333 | fail |
| 43 | b120 | +1.6361% | +1.1517% | 0.3333 | fail |

配对变化：

| seed | b120-b60 MSE gain | b120-b60 MAE gain | b120-b60 rolling pass rate |
| ---: | ---: | ---: | ---: |
| 42 | +1.3594 pp | -0.1401 pp | 0.0000 |
| 43 | +1.9611 pp | +1.0443 pp | 0.0000 |
| mean | +1.6602 pp | +0.4521 pp | 0.0000 |

结论：

> V2.5 的 b120 在 seed42/43 上改善了 report-only test MSE，并在 seed43 上也改善 MAE；但所有 b60/b120 rolling validation 都只有 1/3，通过协议失败。seed42 b120 的 MAE 仍退化，seed43 b120 的 reject rate 达到 37.5%。因此当前结果仍不能支持显著提升或 SOTA 叙事。
