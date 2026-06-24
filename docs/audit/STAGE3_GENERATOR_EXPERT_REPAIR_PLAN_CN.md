# Stage III Generator 专家诊断与详细修改方案

日期：2026-05-24

本文是 `STAGE3_GENERATOR_BEGINNER_EXPLANATION.md` 的专家版补充，专门回答四个问题：

1. 当前 `AdaptationGenerator` 是否真的接入了训练？
2. 当前 generator 的网络结构设计是否合理？
3. `phi_v - phi_i` 表示空间未严格对齐应该如何解决？
4. `eval()` 阶段 VAE 继续随机采样到底对不对？

本文只做诊断和修改方案，不直接修改模型训练代码，不编造实验结果。

---

## 0A. 2026-05-25 实施状态更新

本文最初是修改方案。后续已经在不覆盖原始模型和原始训练循环的前提下，新增了 Stage III V2 路径。当前状态如下：

已完成的工程修复：

1. 新增 `models/factor_mechanism_v2.py`：训练阶段仍采样 VAE，`eval()` 阶段默认使用 `mu_z`。
2. 新增 `models/adaptation_v2.py`：包含 zero-drift identity 的 generator 和 layer-aware HyperLoRA generator。
3. 新增 `models/aligned_representation_v2.py`：使用 shared residual 表示，满足 `phi_v = phi_i + delta_phi_v`。
4. 新增 `train/stage3_functional.py`：使用 differentiable functional forward 训练 transient delta。
5. 新增 `train/stage3_generator_training.py`：冻结 base model，只训练 generator。
6. 新增 `train/online_loop_stage3_v2.py`：缺少 trained generator 时默认拒绝运行。
7. 新增 `train/stage3_v2_data.py`：raw timeline split first，降低滑窗切分泄漏风险。
8. 新增 `drift_source: delta_phi_v`：可直接使用 aligned residual drift。
9. 新增 `drift_polarity`：可显式诊断 drift 符号方向。

已验证命令：

```powershell
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_factor_mechanism_v2.py tests\test_adaptation_v2.py tests\test_hyperlora_generator_v2.py tests\test_aligned_representation_v2.py tests\test_offline_aligned_representation_v2.py tests\test_stage3_functional_v2.py tests\test_stage3_generator_training_v2.py tests\test_stage3_sensitivity_v2.py tests\test_online_loop_stage3_v2.py tests\test_stage3_v2_protocol_safety.py tests\test_stage3_signals.py tests\test_stage3_ablation_benchmark.py tests\test_shapes.py -q
```

结果：

```text
44 passed in 3.18s
```

最新 Panama b30 causal-safe 诊断显示：

| setting | mode | MSE gain | MAE gain |
| --- | --- | ---: | ---: |
| `delta_phi_v`, polarity 1.0 | real | -3.5689% | -1.9644% |
| `delta_phi_v`, polarity 1.0 | shuffled | 0.2302% | 0.3962% |
| `delta_phi_v`, polarity 1.0 | sign_flipped | 2.2552% | 1.3576% |
| `delta_phi_v`, polarity -1.0 | real | -3.5689% | -1.9644% |
| `delta_phi_v`, polarity -1.0 | sign_flipped | 2.2552% | 1.3576% |

当前专家判断更新为：

> V2 已解决“未训练 generator、zero drift 非零扰动、eval 随机、表示空间未对齐、online batch 泄漏”等工程问题的一部分，但 causal-safe 结果仍未证明 real drift 被正确利用。下一轮核心不应继续调 drift 符号，而应重构 generator 的训练目标和模型选择标准。

下一轮应新增的目标：

```text
L = adapted_loss
  + lambda_improve * max(0, adapted_loss - base_loss + margin)
  + lambda_zero_compare * max(0, adapted_loss_real - adapted_loss_zero + margin)
  + lambda_shuffle_compare * max(0, adapted_loss_real - adapted_loss_shuffled + margin)
  + lambda_shift_budget * max(0, pred_shift - shift_budget)^2
  + regularization
```

也就是说，generator 不能只学“让 adapted prediction 拟合 y”，还要学“真实 drift 必须比不使用 drift 或乱配 drift 更值得使用”。

---

## 0. 总结性判断

当前 Stage III 还不能被称为一个已经被证据支持的“drift-conditioned adaptation”机制。

它现在更接近下面这条链：

```text
可能未对齐的 embedding 差
  -> 在原始在线循环中被压成一个标量
  -> 输入一个未训练的随机 generator
  -> 即使 drift = 0 也产生非零参数扰动
  -> 经过 clamp / blend 后形成很小的临时预测变化
```

而论文真正需要证明的是这条链：

```text
有语义的 concept drift 表示
  -> 训练过的 generator 读取 drift
  -> zero drift 时不更新或几乎不更新
  -> drift 的方向、大小、时间对应关系会影响生成的参数更新
  -> 这种更新在预测指标上优于 zero / shuffled / sign-flipped drift
```

目前代码还没有完成第二条链。

因此，最安全的科学表述是：

> 当前代码实现包含一个 Stage III transient adaptation 机制，但它尚未形成充分可辩护的 drift-conditioned generator 证据链。必须先修复 deterministic evaluation、generator training、zero-drift identity、vector drift、representation alignment，再重新跑 sensitivity diagnostics。

---

## 1. 问题一：`AdaptationGenerator` 当前是否接入训练？

### 1.1 核查结论

没有发现它被正常训练。

在 `train/online_loop.py` 中，generator 是在在线适应函数内部直接新建的：

```python
gen = AdaptationGenerator(
    drift_dim=online_cfg["stage2"]["lrp_bottleneck"],
    rank=online_cfg["stage2"]["lrp_bottleneck"],
    max_dim=512,
).to(device)
```

对应位置：

- `train/online_loop.py:97-102`

但全仓库搜索没有发现类似下面的训练逻辑：

```python
optimizer = torch.optim.Adam(gen.parameters(), ...)
loss.backward()
optimizer.step()
```

也没有发现单独的 generator checkpoint 训练、保存、加载流程。

### 1.2 这意味着什么

这意味着 Stage III 里的 `AdaptationGenerator` 当前不是一个学到映射关系的模块。

一个合理的 generator 应该学会：

```text
如果 drift 是这种方向和大小
  -> 哪些层需要调整
  -> 参数应该往哪个方向调整
  -> 调整幅度应该多大
  -> 调整之后预测误差应该降低
```

但当前代码中 generator 的权重来自随机初始化。它没有经过目标函数约束，也没有看过训练数据中的 drift 和误差之间的关系。

因此，如果论文说：

> generator learns to transform concept drift into adaptive parameter updates

这在当前代码证据下属于过度声明。

更准确的说法只能是：

> The current implementation applies a generator-shaped transient perturbation conditioned on a drift input, but the generator itself is not trained in the present pipeline.

中文含义是：

> 当前实现把 drift 输入进了一个 generator 形状的模块，但这个模块没有被训练，所以不能声称它学会了 drift 到参数更新的映射。

### 1.3 应该如何修

必须增加一个显式的 Stage III generator 训练阶段。

推荐流程：

```text
离线训练 base forecasting model
  -> 离线训练或校准 phi_i / phi_v 表示
  -> 冻结 base model 和表示模块
  -> 在 train stream 上回放在线序列
  -> 每个时间步计算 drift
  -> generator 生成临时 delta
  -> 用 adapted prediction 和 y_true 计算 loss
  -> 只更新 generator.parameters()
  -> 保存 generator checkpoint
  -> test 阶段加载该 checkpoint
```

推荐 generator loss：

```text
L_gen = MSE(y_adapted, y_true)
      + lambda_delta * ||delta||_2
      + lambda_zero * ||G(0)||_2
      + lambda_shift * ||y_adapted - y_base||_2
      + lambda_smooth * ||delta(t) - delta(t-1)||_2
```

各项含义：

- `MSE(y_adapted, y_true)`：要求 generator 真正降低预测误差。
- `||delta||_2`：防止参数扰动过大。
- `||G(0)||_2`：要求没有 drift 时不要乱改参数。
- `||y_adapted - y_base||_2`：防止预测突然跳变。
- `||delta(t) - delta(t-1)||_2`：防止在线更新过度抖动。

---

## 2. 问题二：当前 generator 网络结构设计如何？

### 2.1 当前结构

当前 `models/adaptation.py` 中的结构大致是：

```python
class AdaptationGenerator(nn.Module):
    def __init__(self, drift_dim, rank, max_dim=512):
        self.w1 = nn.Linear(drift_dim, rank)
        self.w2_alpha = nn.Linear(rank, max_dim)
        self.w2_beta = nn.Linear(rank, max_dim)

    def forward(self, drift, out_dim, in_dim):
        h = torch.sigmoid(self.w1(drift))
        alpha_full = torch.sigmoid(self.w2_alpha(h)) + 1.0
        beta_full = torch.sigmoid(self.w2_beta(h)) + 1.0
        return alpha_full[:out_dim], beta_full[:in_dim]
```

然后用：

```python
scale = alpha.unsqueeze(1) * beta.unsqueeze(0)
delta = (scale - 1.0) * p
```

给原始参数 `p` 构造扰动。

### 2.2 主要设计问题

#### 问题 A：zero drift 不等于 zero update

因为 Linear 层有 bias：

```python
self.w1(drift) = W * drift + bias
```

当 `drift = 0` 时，输出仍然可能是 `bias`。

再经过：

```python
sigmoid(...) + 1.0
```

`alpha` 和 `beta` 大约落在 `(1, 2)`。因此：

```text
scale = alpha * beta
scale - 1.0 通常不是 0
delta 通常不是 0
```

这和 drift adaptation 的直觉冲突。没有漂移时，模型不应该凭空大幅改变参数。

#### 问题 B：输出方向表达能力弱

`sigmoid(...) + 1.0` 让 `alpha` 和 `beta` 总是正数且大于 1。这样 generator 更像是在放大或缩放原始参数，而不是生成真正有符号的更新。

深度学习中的参数更新通常需要正负方向：

```text
某些权重应该增大
某些权重应该减小
某些权重应该保持不变
```

当前结构没有很好地表达这种 signed correction。

#### 问题 C：未训练时所有输出都是随机偏置

因为 generator 没训练，`w1`、`w2_alpha`、`w2_beta` 的随机初始值会主导结果。

这会造成一个危险现象：

```text
real drift
zero drift
shuffled drift
sign-flipped drift
amplified drift
```

都可能产生非常接近的预测指标，因为真正起作用的是随机 bias、clamp、blend，而不是 drift 内容。

#### 问题 D：原始在线循环把向量压成标量

在原始 `train/online_loop.py` 中：

```python
d_v = phi_v_cur - phi_i_cur
drift_scalar = d_v.mean()
drift_vec = drift_scalar.repeat(lrp_bottleneck)
```

这会丢掉所有 embedding 维度方向信息。

如果论文说的是“两个向量之间插值”或“vector drift”，代码就不应该把它压成一个均值标量。

### 2.3 推荐的新结构

建议新增 V2 generator，不要直接覆盖旧类：

```python
class IdentityLowRankAdaptationGenerator(nn.Module):
    def __init__(self, drift_dim, hidden_dim, max_dim, max_scale=0.05):
        super().__init__()
        self.max_dim = max_dim
        self.max_scale = max_scale
        self.net = nn.Sequential(
            nn.Linear(drift_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * max_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, drift, out_dim, in_dim):
        raw = self.net(drift)
        alpha_raw = raw[:out_dim]
        beta_raw = raw[self.max_dim:self.max_dim + in_dim]
        alpha = self.max_scale * torch.tanh(alpha_raw)
        beta = self.max_scale * torch.tanh(beta_raw)
        return alpha, beta
```

这个结构的优点：

- 最后一层零初始化，刚开始 `G(0) = 0`。
- `tanh` 允许正负方向。
- `max_scale` 限制扰动幅度。
- 训练前不会凭空修改参数。
- 如果训练后仍然没有提升，就能清楚说明 Stage III 不成立，而不是被随机扰动掩盖。

---

## 3. 问题三：`phi_v - phi_i` 表示空间未严格对齐怎么办？

### 3.1 为什么这是严重问题

向量相减只有在两个向量处于同一坐标系时才有明确语义。

例如：

```text
phi_i[3] 表示长期日周期强度
phi_v[3] 表示天气残差响应
```

这时：

```text
phi_v[3] - phi_i[3]
```

并不是一个清晰的 concept drift。

当前代码中：

- `encoder_i` 是一个 invariant encoder。
- `encoder_v` 是另一个 MLP encoder。
- `encoder_v` 用 residual target 训练。
- 两者没有共享 trunk。
- 没有对齐投影。
- 没有 contrastive calibration。
- 没有 orthogonality / disentanglement 约束。

所以 `phi_v - phi_i` 目前只能说是“两个向量的数值差”，不能强说是“variant unobserved factor driven fluctuation”。

### 3.2 最小修复：train-set calibration

如果你希望尽量少改代码，可以先做校准。

步骤：

1. 只用 train split 计算 `phi_i` 和 `phi_v`。
2. 分别估计均值和标准差：

```python
phi_i_norm = (phi_i - mean_i) / std_i
phi_v_norm = (phi_v - mean_v) / std_v
drift = phi_v_norm - phi_i_norm
```

3. 或者学习一个小投影，把 `phi_v` 投影到 `phi_i` 空间：

```python
phi_v_aligned = W_align(phi_v)
drift = phi_v_aligned - phi_i
```

这个方法成本较低，但只能算“工程校准”，还不是最强理论方案。

### 3.3 更强修复：shared trunk + residual variant head

更合理的结构是：

```text
x
  -> shared_encoder(x) = h
  -> invariant_head(h) = phi_i
  -> variant_residual_head(h) = delta_phi_v
  -> phi_v = phi_i + delta_phi_v
  -> drift = delta_phi_v
```

这样做的好处是：

```text
phi_v - phi_i = delta_phi_v
```

也就是说，差值本身就是模型明确学出来的 variant residual，而不是两个无关空间的生硬相减。

### 3.4 推荐损失函数

```text
L = L_forecast
  + lambda_inv * L_invariance
  + lambda_var * L_residual
  + lambda_orth * ||corr(phi_i, delta_phi_v)||_F
  + lambda_smooth * ||delta_phi_v(t) - delta_phi_v(t-1)||_2
  + lambda_norm * ||delta_phi_v||_2
```

解释：

- `L_forecast`：保证整体预测有效。
- `L_invariance`：让 `phi_i` 保持稳定因素。
- `L_residual`：让 `delta_phi_v` 解释残差变化。
- `L_orth`：减少 invariant 和 variant 混在一起。
- `L_smooth`：避免把高频噪声当漂移。
- `L_norm`：避免 variant embedding 任意变大。

### 3.5 和你提出的 temporal drift 定义的关系

你提出的想法是：

```text
[phi_v(t) - phi_i(t)] - [phi_v(t-1) - phi_i(t-1)]
```

这个想法是合理的，因为它试图去掉长期稳定偏移，只保留相邻时刻的变化。

但前提是：

```text
phi_v(t) - phi_i(t)
```

本身要有意义。

因此推荐顺序是：

1. 先解决 `phi_i` / `phi_v` 对齐。
2. 再使用 temporal delta。
3. 再训练 generator。
4. 最后做 zero / shuffled / sign-flipped / amplified sensitivity audit。

---

## 4. 问题四：eval 阶段 VAE 仍然随机，到底对不对？

### 4.1 训练阶段随机是对的

VAE 的训练依赖 reparameterization trick：

```python
z = mu + eps * std
eps ~ N(0, I)
```

这样模型才能学习一个潜变量分布，而不是只学习一个固定点。

所以：

```text
model.train() 时随机采样是正确的
```

### 4.2 普通验证/测试阶段随机通常是不对的

论文表格里的 MSE/MAE 通常表示：

```text
给定同一个模型 checkpoint
给定同一个 test set
指标应该可复现
```

如果 `model.eval()` 仍然采样，那么：

```text
第一次预测: y_hat_1
第二次预测: y_hat_2
y_hat_1 != y_hat_2
```

这会污染：

- offline test metrics；
- online base metrics；
- Stage III candidate prediction；
- drift sensitivity audit；
- real vs zero drift 对比；
- 表格中的 MSE/MAE。

### 4.3 什么时候 eval 随机是合理的

只有在你明确做 Monte Carlo evaluation 时才合理，例如：

```text
每个样本采样 30 次
报告均值和方差
用固定 seed
论文说明这是 uncertainty-aware prediction
```

如果论文没有这样写，那么默认 test/eval 应该用：

```python
z = mu_z
```

### 4.4 推荐代码策略

新增配置：

```yaml
model:
  use_vae: true
  vae_eval_mode: mean
```

模型逻辑：

```python
if self.training:
    z = self._reparameterize(mu_z, log_var_z)
elif self.vae_eval_mode == "mean":
    z = mu_z
elif self.vae_eval_mode == "sample":
    z = self._reparameterize(mu_z, log_var_z)
else:
    raise ValueError(...)
```

这样：

- 训练阶段保留 VAE 正确训练方式。
- 测试阶段默认确定性。
- 如果将来要做不确定性实验，可以显式打开 `sample`。

---

## 5. 详细修改路线图

### Tier 1：必须先修

#### 任务 1：修复 VAE eval 确定性

文件：

- `models/factor_mechanism.py`
- `train/run_experiment.py`
- `configs/*.yaml`
- `tests/test_factor_mechanism_eval.py`

验收：

- 同一个 eval 模型、同一个输入，两次输出完全一致。
- train 模式仍然随机。
- 如果显式设置 `vae_eval_mode: sample`，才允许 eval 随机。

#### 任务 2：新增 Stage III sensitivity audit

文件：

- `train/run_stage3_sensitivity_audit.py`
- `train/stage3_signals.py`
- `tests/test_stage3_signals.py`

必须比较：

- real drift；
- zero drift；
- shuffled drift；
- sign-flipped drift；
- amplified drift；
- random matched-norm drift。

验收：

- 输出 `delta_norm`、`candidate_shift`、`blend_weight`、`MSE`、`MAE`。
- 如果 real 和 zero/shuffled 几乎一样，论文不得声称 drift-conditioned adaptation。

#### 任务 3：禁止 V2 使用未训练 generator

文件：

- `train/online_loop_stage3_v2.py`
- `models/adaptation.py`

验收：

- `stage3.version: v2` 时必须提供 `generator_checkpoint`。
- 只有 `allow_untrained_generator: true` 时才允许诊断性运行未训练 generator。

### Tier 2：增强论文可信度

#### 任务 4：新增 identity-preserving generator

文件：

- `models/adaptation.py`
- `tests/test_stage3_generator_v2.py`

验收：

- zero drift -> zero delta。
- 非零 drift 训练后能改变 delta。
- sign-flipped drift 训练后能改变方向。

#### 任务 5：训练 generator

文件：

- `train/stage3_generator_training.py`
- `configs/stage3_v2.yaml`
- `tests/test_stage3_generator_training.py`

验收：

- 有明确的 `optimizer = Adam(gen.parameters(), ...)`。
- base model 冻结。
- generator loss 在 toy / small stream 上下降。
- 保存独立 generator checkpoint。

#### 任务 6：对齐 `phi_i` 和 `phi_v`

最小方案：

- train-set normalization；
- train-set affine projection。

强方案：

- shared trunk；
- invariant head；
- variant residual head；
- `phi_v = phi_i + delta_phi_v`。

验收：

- 不再默认使用未校准的 `phi_v - phi_i`。
- temporal drift 建立在已校准或 residual variant 表示上。

### Tier 3：论文增强项

#### 任务 7：重构创新点表述

只有当证据支持时，才使用：

> drift-conditioned online adaptation

如果证据不足，应改成：

> efficiency-aware transient online adaptation under temporal distribution shift

或者：

> a diagnosed and constrained online adaptation mechanism with explicit sensitivity checks

#### 任务 8：补充消融表格

至少包含：

- base；
- Stage I；
- Stage II；
- legacy Stage III；
- Stage III V2 zero drift；
- Stage III V2 real drift；
- Stage III V2 shuffled drift；
- Stage III V2 sign-flipped drift。

每项必须同时报告 MSE 和 MAE。

---

## 6. 推荐最小安全下一步

不要立刻重写整套 Stage III。

先做这一步：

```text
只修复 VAE eval 确定性
```

原因：

1. 它是最小改动。
2. 它不改变训练阶段的 VAE 学习机制。
3. 它能让所有后续指标可复现。
4. 如果不先修它，Stage III 的 real / zero / shuffled drift 对比可能全被随机采样噪声污染。

完成后再做：

```text
identity generator -> generator training -> aligned phi -> full sensitivity audit
```

---

## 7. 论文层面的暂定建议

在修复和重新实验之前，不建议论文继续强声称：

```text
Stage III learns concept drift and transforms it into adaptive parameter updates.
```

更安全的临时写法是：

```text
Stage III is designed as a transient adaptation module that conditions parameter perturbations on estimated representation shifts. However, its effectiveness must be validated through explicit generator training and drift-sensitivity diagnostics.
```

中文含义：

```text
Stage III 的设计目标是根据表示漂移生成临时参数扰动，但它是否真正利用 drift，需要通过 generator 训练和 drift sensitivity 诊断来验证。
```

等修复并重新实验后，再根据证据决定是否升级创新点表述。
