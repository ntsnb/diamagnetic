# 2026-10-07 论文完整实现契约

## 范围与证据

- 后续范围更正：本契约已验收的是主方法核心闭环。全文实验所需外部基线、部分组件消融和原选段协议尚有代码缺项；不能将本契约的完成状态扩展到全部实验。详见 `EXPERIMENT_READINESS_20261007.md`。

- 当前论文：`C:/Users/admin/Documents/杂活/论文前三部分修订_20260921/Methodology图件紧凑新罗马版_20261004/stagewise_adaptation_review_20261006/manuscript_stagewise_review.tex`。
- 原工程仅作只读来源，完整当前源码导入见 `source_import_manifest.json`；原始数据、日志和 checkpoint 不覆盖。
- 当前分支：`codex/paper-complete-20261007`。不提交 Git。
- 四月负荷结果与七月 Weather V30SA 结果均是历史证据，不能验证十月新方法。
- 新包 `paper_impl` 为可独立训练、验证、流式预测的实现；legacy 包保留。拒绝把简化编码器当作官方骨干复现。

## 参数与张量接口

1. 输入 `X[B,L,N]`；预测 `Y[B,H,T]`，T为明确的目标列数。数据归一化统计只从训练前缀拟合。
2. B节的 invariant/variant 分支各有自己的 encoder 与 output head，与 C 节预测器不共享参数。环境标签在全训练序列按 sample ID 保存。MEI/MIL 交替且冻结另一参数组；空环境显式修复。两分支输出均拟合同一归一化目标。训练结束后在线冻结。
3. C 节 router 使用含位置编码的 temporal Transformer → VAE posterior → linear softmax；专家独立初始化，按变量注意力处理带变量/滞后位置编码的标量 token，输出 `[B,L,D]`，加权融合后由共享 output head 输出预测。
4. 专家相关性单独解释 `output_head(mean_time(expert_m(X)))`，不经过 gating weights。每一目标、每一预测步单独播种，先绝对值再对预测步平均。共享 output-head 参数在 specialization 路径固定，但输入导数保留。
5. 手写可微 LRP：linear ε-rule，双输入矩阵乘法均分规则，softmax Taylor 规则，逐点激活/除标准差 identity rule，均值中心化 linear rule，residual addition ε-rule。位置/变量 embedding 的 relevance 与 bias/stabilizer residual 分开记录。禁止以完整模型的 gradient×input 冒充专家 LRP。
6. `PaperForecaster.forward(x, stochastic=False, modulations=None)` 返回含 `prediction/gates/mu/log_var/reconstruction` 的对象。`modulations` 将专家 weight 名称映射至 `[B,Cout,Cin]` 乘法增益，PyTorch weight 使用 out×in，不能把输出通道当列索引。
7. `PaperForecaster.attribute(x)` 返回 `.maps[B,T,K,N,L]`、`.channels`（expert weight全名 → `[B,T,Cout]` 绝对通道 relevance）、`.residual[B,T,K]`、`.conservation_error[B,T,K]`。此接口的 maps 不作总量归一化。
8. `model.expert_weights()` 返回 `{parameter_name: Parameter}`，只含专家 linear weight；router输出bias为 `model.router.bias`。`model.config` 含 horizon、targets、lookback、n_variables、n_experts。

## 训练与流式协议

- 训练顺序：FOIL分支 → 主预测器（prediction+VAE+specialization）→ generator因果回放；验证只用于选模/阈值，测试一次冻结评估。
- 发布整段预测的起点为 t，目标为 t+1..t+H。整段标签最早在 t+H+extra_delay 释放。发行的预测永远保存，不用更新后重算值替代指标。
- 回放/在线每一步：释放已有标签 → 以当前窗口选择通道并进行 Stages I/II → 生成临时增益 → 发出预测 → 入队。没有已释放标签就没有参数监督更新。
- 回放 generator 优化在释放事件的持久更新前，用该事件起始持久状态的 detached 参数计算损失；然后以新的 generator 冻结参数执行 Stage I/II。episode开始恢复初始持久预测器、reference、队列和输入缓存。W1随机初始化、W2零初始化，避免全零死亡。
- Stage I仅更新router.bias。Stage II固定mask和临时增益，以确定性完整预测损失做裁剪梯度和Armijo回溯；gradient须通过增益到持久专家权重。误差由发行时保存的预测计算。
- Stage II参考模型与历史输入集在一个间隔内都冻结；EMA仅包含持久专家参数，不包含临时增益。相关性无效时不静默改分母；监督误差触发可用同预算梯度通道。
- Stage III在预测空间比较两分支（坐标一致），中心化有界乘法调制；预测完成不写回专家权重。
- 完整可运行对照：三阶段全部8种开关组合；选择器 relevance+error、gradient-only；generator centered-bounded 与 identity（消融）。不把对照结果当作实现前已有结论。

## 首轮验证计划

- run_id：`20261007_paper_contract_validation_v1`
- 问题：论文规定是否完整实现，数值/梯度/标签时序是否正确？
- 基线：原工程只作语义对照；独立数学参考值、因果事件人工时间线、未更新参数快照为验收依据。
- 变化：修复实现缺项，无性能优越性预设。
- 数据：先小规模合成可验证系统，再真实 Panama 数据的短流程；完整负荷实验另记独立run。
- 命令：`python -m pytest tests/paper_impl -q`；训练CLI完成后登记精确命令再运行。
- 输出：测试日志、逐步更新记录、归因守恒账本、训练与验证曲线、冻结测试指标、独立ZCode审查与裁决记录。
- 初始状态：实现中；当时本契约只列验收要求。2026-10-07收尾状态：实现和功能验证已完成，逐项证据见 `REQUIREMENT_MATRIX.md` 与 `VALIDATION_REPORT.md`；全量性能研究不在这些验证结果内。

## 外部原始来源

- AttnLRP（ICML2024）：https://arxiv.org/abs/2402.05602
- FOIL（ICML2024）：https://proceedings.mlr.press/v235/liu24ae.html
- PROCEED（KDD2025）：https://arxiv.org/abs/2412.08435
- ADAPT-Z（ICLR2026）：https://proceedings.iclr.cc/paper_files/paper/2026/hash/f418594e90047a10f4c158f70d6701cc-Abstract-Conference.html
- DynaME（WWW2026）：https://arxiv.org/abs/2603.09062
