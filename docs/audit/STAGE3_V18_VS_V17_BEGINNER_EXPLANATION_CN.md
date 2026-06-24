# Stage III V18 相对 V17 的改进与数学原理说明

日期：2026-06-18

本文档目的：用尽量可读的中文解释 `V18` 相对 `V17` 到底改进了什么，为什么这些改动在数学语义上更清晰，以及为什么这种结构改进目前还不能直接等同于实验性能更强。

本文档边界：本文档是机制说明和科研审计说明，不是论文结论。根据当前仓库证据，`V18` 在结构上比 `V17` 更符合“稳定锚点、环境抑制、残差释放、置信度控制分离”的思想，但是现有正式实验没有证明 `V18` 比 `V17` 更强，也没有通过严格 `teacher gate`（教师门控）来解锁生成器训练。

---

## 1. 先给一页版结论

`V17` 已经比更早版本更合理，因为它把环境推断做成了一个有时间记忆的因果滤波器。它会先估计环境先验，再根据当前专家状态得到环境后验，并用“后验减先验”的差值表示环境变化。

但是，`V17` 仍然有一个概念问题：环境后验同时参与了不变表示 `phi_i` 的构造和漂移表示 `delta_phi_v` 的构造。这样会让几个概念混在一起：

- 哪一部分是真正应该长期稳定保留的基础表示；
- 哪一部分只是环境可以解释的扰动，应该从稳定表示中扣掉；
- 哪一部分是不变分支解释不了、需要释放给短期变化分支的残差；
- 模型对这个释放动作到底有多自信。

`V18` 的主要改进就是把这些东西拆开。

可以把 `V18` 的核心结构概括成：

```text
专家状态
-> 共享后处理表示
-> 稳定锚点 anchor
-> 环境抑制 suppression
-> 不变表示 phi_i
-> 残差释放状态 release state
-> 可变漂移 delta_phi_v
-> 释放置信度 release confidence
-> 专家预测与专家路由
```

一句话总结：

> `V18` 的结构比 `V17` 更像“先建立稳定基础，再扣除环境可解释扰动，再把剩余残差交给变化分支”，而不是让环境后验直接混入所有表示。

但是请注意：

> 结构更清楚，不等于实验已经更好。当前审计记录明确指出，`V18` 是更原则化的架构，但在当前 `weather seed42 b480` 等正式记录上没有强于 `V17`，也没有通过教师门控。

---

## 2. 先复习 V17 在做什么

为了理解 `V18`，必须先看 `V17` 的核心公式。

对第 \(k\) 个专家和第 \(t\) 个时间步，模型先得到专家内部状态：

\[
h_{k,t}
=
\mathrm{expert}_k(x_{\mathrm{model}})_t
\]

这里的 \(x_{\mathrm{model}}\) 是模型实际使用的输入序列。它可能已经经过 `RevIN` 归一化。

然后专家状态经过 `shared_post`，得到整理后的专家内部表示：

\[
s_{k,t}
=
\mathrm{shared\_post}(h_{k,t})
\]

你可以把 \(h_{k,t}\) 理解为“第 \(k\) 个专家的原始观察”，把 \(s_{k,t}\) 理解为“整理后的专家观察”。后面的环境推断、不变表示和漂移表示，主要都从 \(s_{k,t}\) 出发。

### 2.1 V17 的环境先验、后验和环境转移

`V17` 的环境推断会维护一个环境记忆状态 \(m_{k,t}\)。在第 \(t\) 个时间步，先根据过去记忆得到先验：

\[
p^{-}_{k,t}
=
\mathrm{softmax}
(
W_{\mathrm{prior}}m_{k,t-1}
+
b_{\mathrm{prior}}
)
\]

然后从当前专家表示 \(s_{k,t}\) 提取当前证据：

\[
e_{k,t}
=
W_{\mathrm{evidence}}s_{k,t}
+
b_{\mathrm{evidence}}
\]

再把先验分数和当前证据相加，得到后验：

\[
p^{+}_{k,t}
=
\mathrm{softmax}
(
W_{\mathrm{prior}}m_{k,t-1}
+
b_{\mathrm{prior}}
+
e_{k,t}
)
\]

最后计算环境转移量：

\[
\Delta p_{k,t}
=
p^{+}_{k,t}
-
p^{-}_{k,t}
\]

这里：

- \(p^{-}_{k,t}\) 是环境先验，表示模型看当前证据之前的判断。
- \(p^{+}_{k,t}\) 是环境后验，表示模型看当前证据之后的判断。
- \(\Delta p_{k,t}\) 是环境转移，表示当前证据让环境判断发生了多大变化。

这套环境滤波机制在 `V18` 中仍然保留。也就是说，`V18` 不是推翻 `V17` 的环境推断，而是在它上面重新组织不变表示和漂移表示。

---

## 3. V17 的主要概念弱点

在 `V17` 中，不变表示大致是这样构造的：

\[
u_{k,t}
=
W_{\mathrm{anchor}}s_{k,t}
+
b_{\mathrm{anchor}}
\]

\[
c^{i}_{k,t}
=
\sum_{j=1}^{E}
p^{+}_{k,t,j}
B^{i}_{j}
\]

\[
g^{i}_{k,t}
=
\sigma(W_{g_i}s_{k,t}+b_{g_i})
\]

\[
\phi^{i}_{k,t}
=
u_{k,t}
-
g^{i}_{k,t}
\odot
c^{i}_{k,t}
\]

这里 \(u_{k,t}\) 是不变锚点，\(c^i_{k,t}\) 是由环境后验加权得到的环境相关修正项，\(g^i_{k,t}\) 是门控向量。

这个公式本身不是错误的。它的问题在于语义不够干净。

因为 \(p^{+}_{k,t}\) 直接参与了 \(\phi^i_{k,t}\) 的定义，所以 \(\phi_i\) 其实是“被环境后验调制过的稳定表示”。它不应该被过度解释为严格意义上的真实不变因子。

`V17` 的漂移分支大致是：

\[
\Delta\phi^{v}_{k,t}
=
d^{\mathrm{base}}_{k,t}
+
d^{\mathrm{env}}_{k,t}
+
d^{\mathrm{trans}}_{k,t}
\]

其中：

\[
d^{\mathrm{env}}_{k,t}
\text{ 使用 }
p^{+}_{k,t}
\]

\[
d^{\mathrm{trans}}_{k,t}
\text{ 使用 }
\Delta p_{k,t}
\]

这比旧版本更好，因为它不仅看当前环境，还看环境变化。但是它仍然把“漂移向量本身”和“模型对漂移是否可靠的判断”混在了一起，没有显式的释放状态和置信度。

所以，`V17` 的问题不是“完全没有环境推断”，而是：

> 环境推断已经有了，但环境抑制、残差释放和释放置信度没有被明确拆开。

---

## 4. V18 的第一项改进：把 anchor 和 suppression 分开

`V18` 中，`phi_i` 不再写成“环境后验直接生成不变修正项”的语义，而是改成：

\[
\phi^{i}_{k,t}
=
a_{k,t}
-
r^{\mathrm{sup}}_{k,t}
\]

其中 \(a_{k,t}\) 是 `anchor`（稳定锚点），\(r^{\mathrm{sup}}_{k,t}\) 是 `suppression`（环境抑制量）。

### 4.1 anchor 从哪里来

`anchor` 仍然从整理后的专家表示 \(s_{k,t}\) 来：

\[
a_{k,t}
=
W_{\mathrm{anchor}}s_{k,t}
+
b_{\mathrm{anchor}}
\]

代码中对应：

```python
anchor = self.anchor_head(shared)
```

这里的 `shared` 就是 \(s_{k,t}\)。

直观解释：

> `anchor` 是专家从当前序列中提取出的稳定基础表示。它是后面不变表示的起点。

它被叫作“稳定锚点”，不是因为它天然已经稳定，而是因为模型结构和训练损失希望它承担更稳定、更长期、更基础的预测信息。

### 4.2 suppression 从哪里来

`suppression` 来自环境后验和一个门控网络。

先用环境后验 \(p^{+}_{k,t}\) 对环境抑制基向量加权求和：

\[
b^{\mathrm{sup}}_{k,t}
=
\sum_{j=1}^{E}
p^{+}_{k,t,j}
B^{\mathrm{sup}}_{j}
\]

其中：

- \(E\) 是环境数量。
- \(B^{\mathrm{sup}}_{j}\) 是第 \(j\) 个环境对应的抑制基向量。
- \(b^{\mathrm{sup}}_{k,t}\) 表示“当前环境可能带来的可解释扰动方向”。

然后模型生成抑制门控：

\[
g^{\mathrm{sup}}_{k,t}
=
\sigma(W_{\mathrm{sup}}s_{k,t}+b_{\mathrm{sup}})
\]

最后：

\[
r^{\mathrm{sup}}_{k,t}
=
g^{\mathrm{sup}}_{k,t}
\odot
b^{\mathrm{sup}}_{k,t}
\]

代码中对应：

```python
suppression_basis = torch.einsum("ble,er->blr", env_posteriors, self.env_basis_suppression)
suppression = self.suppression_gate(shared) * suppression_basis
phi_i = anchor - suppression
```

### 4.3 这比 V17 好在哪里

从公式表面看，`V18` 的 `anchor - suppression` 和 `V17` 的 `anchor - environment correction` 有相似形状。

但是语义不同。

`V17` 更像是在说：

> 环境后验参与定义不变表示。

`V18` 更像是在说：

> 先有稳定锚点，再从中扣掉环境可以解释的扰动。

这个区别很重要。

因为 `V18` 把 `anchor_seq` 和 `suppression_seq` 都显式输出出来，后续训练也能单独约束它们。这样我们可以更清楚地检查：

- `anchor` 是否保留了稳定预测信息；
- `suppression` 是否过大；
- `suppression` 是否变成了新的万能修正器；
- `phi_i` 到底是稳定锚点，还是被环境过度改写后的表示。

---

## 5. V18 的第二项改进：显式 residual release state

`V17` 里的 `delta_phi_v` 是直接由当前专家表示、环境后验和环境转移组合出来的。它没有一个单独的“释放状态”。

`V18` 改成维护一个循环状态，叫作 `release_hidden`。你可以把它理解为：

> 模型内部专门记录“当前还有哪些不变分支解释不了、应该释放给变化分支的残差信息”的记忆状态。

### 5.1 release state 的输入

在每个时间步，`V18` 把三类信息拼在一起：

\[
[s_{k,t}, \phi^{i}_{k,t}, \Delta p_{k,t}]
\]

也就是：

- \(s_{k,t}\)：整理后的专家状态；
- \(\phi^{i}_{k,t}\)：已经经过环境抑制后的不变表示；
- \(\Delta p_{k,t}\)：环境转移量。

然后通过 `GRUCell` 更新释放状态：

\[
q_{k,t}
=
\mathrm{GRUCell}
(
[s_{k,t}, \phi^{i}_{k,t}, \Delta p_{k,t}],
q_{k,t-1}
)
\]

这里 \(q_{k,t}\) 就是 release hidden state，也就是残差释放隐藏状态。

代码中对应：

```python
release_input = torch.cat([shared_t, phi_i_t, env_shift_t], dim=-1)
release_hidden = self.release_update(release_input, release_hidden)
```

### 5.2 release state 产生基础释放量

释放状态再经过一个线性层：

\[
r^{\mathrm{base}}_{k,t}
=
W_{\mathrm{release}}q_{k,t}
+
b_{\mathrm{release}}
\]

代码中对应：

```python
release_base = self.release_head(release_hidden)
```

这表示模型从释放记忆里读出一个基础残差释放向量。

### 5.3 加上环境转移项

`V18` 仍然使用环境转移量：

\[
r^{\mathrm{trans}}_{k,t}
=
g^{T}_{k,t}
\odot
\sum_{j=1}^{E}
\Delta p_{k,t,j}
B^{T}_{j}
\]

代码中对应：

```python
transition = self.transition_gate(shared_t) * torch.matmul(env_shift_t, self.env_basis_transition)
```

这表示：

> 如果当前环境判断相对先验发生变化，释放分支可以根据这种变化调整残差释放方向。

### 5.4 加上上下文项

`V18` 还加入了一个 context 项：

\[
r^{\mathrm{ctx}}_{k,t}
=
g^{C}_{k,t}
\odot
\phi^{i}_{k,t}
\]

代码中对应：

```python
context = self.context_gate(shared_t) * phi_i_t
```

这个项的直觉是：

> 残差释放不是完全独立于不变表示的。变化分支需要知道当前稳定分支已经解释了什么，才能决定还要释放什么。

### 5.5 得到 raw_release

三部分相加：

\[
r^{\mathrm{raw}}_{k,t}
=
r^{\mathrm{base}}_{k,t}
+
r^{\mathrm{trans}}_{k,t}
+
r^{\mathrm{ctx}}_{k,t}
\]

代码中对应：

```python
raw_release = release_base + transition + context
```

然后如果启用正交投影，模型会把 `raw_release` 中和 \(\phi_i\) 平行的部分去掉：

\[
\Delta\phi^{v}_{k,t}
=
\mathrm{ProjectOrthogonal}
(
r^{\mathrm{raw}}_{k,t},
\phi^{i}_{k,t}
)
\]

当前代码中，投影后的 `raw_release` 就是：

\[
\Delta\phi^{v}_{k,t}
=
r^{\mathrm{raw}}_{k,t}
\]

更准确地说，如果启用投影，`raw_release` 会先被正交化，然后赋值给 `delta_phi_v`。

代码中对应：

```python
if self.use_projected_residual:
    raw_release = _project_delta_orthogonal_to_anchor(raw_release, phi_i_t, self.projection_eps)
delta = raw_release
```

### 5.6 这比 V17 好在哪里

`V17` 的可变分支更像是一个“当前时刻公式组合”：

\[
\Delta\phi_v
=
\text{base drift}
+
\text{environment drift}
+
\text{transition drift}
\]

`V18` 的可变分支更像是一个“残差释放过程”：

\[
\text{当前释放}
=
\text{释放记忆}
+
\text{环境转移}
+
\text{不变表示上下文}
\]

这个变化更符合你的原始思想：`Stage III` 不应该只是拿一个漂亮的向量差，而应该拿一个更明确的“哪些残差需要被释放给变化分支”的状态。

---

## 6. V18 的第三项改进：release confidence 被单独拿出来

`V18` 还输出：

\[
\rho_{k,t}
=
\sigma
(
W_{\mathrm{conf}}
[q_{k,t}, s_{k,t}, p^{+}_{k,t}]
+
b_{\mathrm{conf}}
)
\]

这里 \(\rho_{k,t}\) 是 `release_confidence`（释放置信度）。

代码中对应：

```python
release_confidence = torch.sigmoid(
    self.release_confidence_head(torch.cat([release_hidden, shared_t, env_post_t], dim=-1))
)
```

它的含义是：

> 模型判断当前这个残差释放动作有多可信。

这个量很重要，因为不是所有残差都应该被释放给变化分支。如果模型不确定，那么在线更新时应该更保守。

### 6.1 注意：当前 V18 代码中 confidence 不直接定义 delta_phi_v

这一点非常关键。

早期 `V18` 的一次实现曾经把 `release_confidence` 直接混进 `delta_phi_v` 的定义。后来审计认为这样会把“漂移状态”和“模型对漂移的置信度”混在一起，所以做了语义修正。

当前仓库中的 `V18` 代码是：

\[
\Delta\phi^{v}_{k,t}
=
\mathrm{raw\_release}_{k,t}
\]

而不是：

\[
\Delta\phi^{v}_{k,t}
=
\rho_{k,t}
\cdot
\mathrm{raw\_release}_{k,t}
\]

`release_confidence` 当前主要用于：

- 局部可变分支读出；
- 专家路由上下文；
- 训练中的置信度对齐损失；
- 后续风险控制或在线策略设计。

这比把置信度直接乘进漂移向量更清楚。因为：

- `delta_phi_v` 表示“释放出来的变化状态”；
- `release_confidence` 表示“模型对这个释放状态有多相信”。

这两个东西不应该合成一个向量，否则后续很难判断到底是漂移变小了，还是模型只是没信心。

---

## 7. V18 的第四项改进：路由器使用更丰富的结构上下文

`V17` 的专家路由使用：

\[
\alpha
=
\mathrm{softmax}
(
\mathrm{alpha\_head}(z)
+
\mathrm{env\_router\_bias}(\mathrm{env\_shift})
)
\]

也就是说，`V17` 的环境路由主要看环境转移。

`V18` 的路由输入更丰富。它把三类信息拼在一起：

\[
\mathrm{router\_features}
=
[
\mathrm{pooled\_env\_shift},
\mathrm{pooled\_release\_confidence},
\|\mathrm{pooled\_suppression}\|
]
\]

代码中对应：

```python
router_features = torch.cat([pooled_env_shift, pooled_release_confidence, pooled_suppression_norm], dim=-1)
env_router_bias = self.router_context_head(router_features.reshape(router_features.shape[0], -1))
alpha_logits = self.alpha_head(z) + env_router_bias
alpha = torch.softmax(alpha_logits, dim=-1)
```

这表示专家路由不仅看“环境有没有变”，还看：

- 模型当前对释放动作有多自信；
- 当前不变表示被环境抑制了多少；
- 哪个专家在当前结构状态下更值得信任。

这比 `V17` 的路由上下文更细。

---

## 8. V18 的第五项改进：新增训练损失来约束中间变量

`V18` 不是只改模型结构，还新增了几个损失函数来约束新拆出来的变量。

### 8.1 anchor preservation loss

\[
\mathcal{L}_{anchor}
=
\|a_{k}-\phi^{i}_{k}\|^2
\]

因为：

\[
\phi^{i}_{k}
=
a_k
-
r^{\mathrm{sup}}_k
\]

所以这个损失本质上会惩罚过大的 suppression。

直觉是：

> 不要让环境抑制把稳定锚点改得面目全非。

### 8.2 suppression sparsity loss

\[
\mathcal{L}_{sparse}
=
\|r^{\mathrm{sup}}\|_1
\]

直觉是：

> 环境抑制应该尽量少用，只有在确实需要扣除环境扰动时才用。

这可以防止 suppression 成为新的万能修正器。

### 8.3 release confidence alignment loss

`V18` 会比较：

- 只用不变分支的误差；
- 使用完整预测后的误差。

如果完整预测比不变分支好，说明 release 分支可能有帮助，置信度应该更高。

数学上，代码先计算：

\[
\mathrm{improvement}
=
\mathrm{error}_{inv}
-
\mathrm{error}_{full}
\]

然后把它映射成目标置信度：

\[
\rho^{target}
=
\sigma
(
\mathrm{improvement}/0.1
)
\]

再训练模型的 `release_confidence` 去接近这个目标。

直觉是：

> 如果可变释放分支确实让预测变好，模型应该更自信；如果没有变好，模型应该降低自信。

### 8.4 release temporal loss

`V18` 还约束 `delta_phi_v` 的时间变化：

\[
\Delta r_t
=
\Delta\phi^{v}_{t}
-
\Delta\phi^{v}_{t-1}
\]

当环境转移很小的时候，释放状态不应该剧烈跳变；当环境转移很大时，释放状态可以更自由地变化。

代码中使用：

\[
\mathrm{stability\_weight}
=
\exp(-\|\Delta p_t\|^2)
\]

所以如果环境变化小，权重大，模型会更强地惩罚释放状态跳动；如果环境变化大，权重小，模型允许释放状态变化。

直觉是：

> 没有环境变化时，漂移状态应该稳定；有环境变化时，漂移状态可以改变。

---

## 9. V17 和 V18 的核心差异表

| 维度 | V17 | V18 |
| --- | --- | --- |
| 环境推断 | 因果先验/后验滤波 | 沿用 V17 的因果环境滤波 |
| `phi_i` 构造 | 环境后验直接参与不变修正 | 先生成 anchor，再减去 suppression |
| 环境作用 | 同时调制不变表示和漂移表示 | 主要作为 suppression、release 和 router 的条件变量 |
| 可变分支 | 当前公式组合得到 `delta_phi_v` | 通过 recurrent release state 产生残差释放 |
| 置信度 | 没有显式 release confidence | 显式输出 `release_confidence` |
| 路由上下文 | 主要看 `env_shift` | 看 `env_shift`、`release_confidence`、`suppression_norm` |
| 新增监督 | 主要沿用表示损失和局部预测损失 | 增加 anchor preservation、suppression sparsity、release confidence、release temporal 等 |
| 实验状态 | 当前 weather benchmark 上经验线更强 | 结构更原则化，但当前正式证据弱于 V17 |

---

## 10. 为什么说 V18 更符合数学语义

`V18` 的核心价值是语义分离。

在一个理想的在线负荷预测模型里，我们希望不同变量各司其职：

- `anchor` 负责长期稳定的信息；
- `suppression` 负责扣掉环境可以解释的扰动；
- `phi_i` 负责稳定预测；
- `release state` 负责释放不变分支解释不了的残差；
- `delta_phi_v` 负责短期变化或漂移相关状态；
- `release_confidence` 负责告诉下游这个释放动作是否可靠；
- `router` 负责根据这些结构状态选择专家权重。

`V17` 已经有不变分支和可变分支，但很多东西还是绑在一起。

`V18` 把这些量拆开后，审计和调参会更有方向。例如：

- 如果 `suppression` 太大，就说明环境抑制可能过度；
- 如果 `release_confidence` 高但预测没变好，就说明置信度校准失败；
- 如果 `delta_phi_v` 在没有环境转移时仍剧烈跳动，就说明 release state 不稳定；
- 如果 `anchor` 本身没有预测能力，就说明稳定基础没有学好。

这就是为什么说 `V18` 在数学语义上更清楚。

---

## 11. 为什么 V18 结构更合理，却实验不一定更好

这是你必须特别理解的一点。

深度学习里，结构更符合人类直觉，不代表训练一定更容易。

`V18` 把问题拆得更细，但也增加了训练难度。模型现在必须同时学会：

- 什么是稳定 anchor；
- 什么应该被 suppression 扣除；
- 哪些 residual 应该 release；
- release state 什么时候该变化；
- release confidence 怎么校准；
- router 怎么利用这些新信号。

如果这些信号没有被训练目标很好地协调起来，模型可能会变得更难训练。

当前审计记录显示：

- `V18` 没有通过严格 `teacher gate`；
- 当前 `V18` 正式结果没有强于 `V17`；
- 更干净的语义拆分没有自动带来更强 residual-target evidence；
- `release-friendly` 和 `balanced` 调参没有把 V18 转换成更强的正式结果；
- V18 的跨随机种子和跨数据集表现仍未充分验证。

所以最准确的判断是：

> `V18` 是结构上更合理的下一代设计，但还不是经验上最强的 Stage III 方案。

---

## 12. 用一个简单类比理解 V17 到 V18

可以把模型想象成一个做负荷预测的团队。

`V17` 像是一个团队已经有了天气观察员。天气观察员会告诉大家：当前环境是什么，环境有没有变化。然后稳定预测员和变化预测员都会直接参考这个天气观察结果。

这已经比没有天气观察员更好。

但是问题是：天气观察员的信息同时被很多地方使用，团队很难分清楚：

- 哪些是长期规律；
- 哪些是天气造成的短期扰动；
- 哪些扰动应该从长期规律中扣掉；
- 哪些误差应该交给变化预测员；
- 团队对这次变化预测到底有多自信。

`V18` 像是给团队重新分工。

先由稳定预测员写出基础判断，这就是 `anchor`。

再由环境抑制员指出：这里有些东西可能只是环境扰动，应该从基础判断中扣掉，这就是 `suppression`。

扣完之后得到稳定判断，这就是 `phi_i`。

然后残差释放员专门看：稳定判断解释不了的那部分，要不要交给短期变化分支，这就是 `release state` 和 `delta_phi_v`。

最后风险管理员给出：我对这次释放有多自信，这就是 `release_confidence`。

这个分工更清楚，但如果每个人还没训练好，团队整体成绩未必立刻更好。这正是当前 V18 的状态。

---

## 13. 如果写论文，应该如何谨慎表述

可以谨慎写成：

> Compared with V17, V18 further decomposes the expert-local representation into an invariant anchor, an environment-conditioned suppression component, and a recurrent residual-release state. This design separates the drift state from release confidence and provides richer routing context.

中文含义是：

> 相比 V17，V18 进一步将专家内部表示分解为不变锚点、环境条件抑制项和循环残差释放状态。该设计将漂移状态与释放置信度分离，并为专家路由提供更丰富的结构上下文。

不应该写成：

> V18 has proven superior to V17.

也不应该写成：

> V18 solves Stage III online adaptation.

因为当前仓库证据并不支持这些强结论。

---

## 14. 最后总结

`V18` 相对 `V17` 的关键改进不是“多加几个网络层”，而是“重新定义每个中间变量应该承担的科学含义”。

`V17` 的重点是：

> 让环境推断从无记忆变成有时间记忆，并让环境变化参与漂移分支。

`V18` 的重点是：

> 在保留 V17 环境滤波的基础上，把稳定锚点、环境抑制、残差释放和释放置信度拆开。

因此，从数学建模角度看，`V18` 更清楚、更可审计、更接近“稳定规律与短期扰动分离”的研究设想。

但是，从实验结论角度看，`V18` 还没有证明自己比 `V17` 更强。当前最安全的科研判断是：

> V18 是更原则化的结构候选，但不是已经验证成功的最终在线学习方法。

---

## 15. 证据来源

本文档依据以下仓库文件整理：

- `models/factor_mechanism_v17.py`：V17 的环境滤波、不变表示、漂移表示和专家路由实现。
- `models/factor_mechanism_v18.py`：V18 的 `anchor`、`suppression`、`release_hidden`、`raw_release`、`release_confidence` 和结构化路由实现。
- `train/offline_factor_v17.py`：V17 的环境均衡、环境时间平滑、局部不变预测、局部残差预测和路由稳定损失。
- `train/offline_factor_v18.py`：V18 新增的 anchor preservation、suppression sparsity、release confidence alignment、anchor diversity 和 release temporal loss。
- `docs/audit/STAGE3_V17_CAUSAL_ENV_DUAL_STREAM_IMPLEMENTATION_RECORD.md`：V17 的设计目的、实验证据和限制。
- `docs/audit/STAGE3_V18_STRUCTURED_ENV_RELEASE_IMPLEMENTATION_RECORD.md`：V18 的设计目的、当前语义修正、实验记录和相对 V17 的审计结论。

