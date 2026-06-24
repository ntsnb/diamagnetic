# 新代码结构与实验路线建议

生成时间：2026-06-24  
定位：为下一阶段“重新按新的研究思路设计代码和论文实验路线”提供结构建议，不改动当前仓库。

## 1. 总体原则

新路线不应继续在 Stage III V2-V29 树上无限叠加版本号。更安全的做法是建立一个新的 canonical protocol 层，把旧代码当作 evidence archive 和设计教训来源。

建议新主线遵循以下原则：

1. 先定义协议，再写模型；
2. 先 teacher-only / diagnostic gate，再 generator / online adaptation；
3. 每个结果必须有 manifest；
4. 每个论文 claim 必须能回到 config、run、log、checkpoint 和 source commit；
5. 不用一个表格同时承载“搜索结果”“最终结果”“失败诊断”三种语义；
6. 保留旧版本，但新论文只引用经过冻结的新 evidence chain。

## 2. 建议的新目录逻辑

可选的新命名空间：

```text
configs/redesign/
  datasets/
  offline/
  online_protocols/
  ablations/

train/redesign/
  run_offline.py
  run_online_protocol.py
  run_ablation_suite.py
  run_baseline_suite.py

models/redesign/
  backbone.py
  invariant.py
  adaptation.py
  teacher.py

docs/redesign/
  PROTOCOL.md
  CLAIM_TRACE.md
  TABLE_FIGURE_MANIFEST.md
```

如果短期不想移动代码，也可以不建 `models/redesign/`，而是在现有 `train/` 和 `models/` 中新增 `*_redesign.py` 或 `v30_*` 文件。关键是不要覆盖 V2-V29 文件。

## 3. 数据加载与时间切分

必须固化为一个可审计协议：

- 明确 dataset id、原始文件路径、时间列、目标列、协变量列；
- 缺失值处理必须 split-safe，不能用未来 test 信息填补 train；
- normalization 只 fit train prefix；
- 先 raw timeline split，再构造 train/val/test windows，或明确说明 window-first 的后果；
- online test stream 必须 chronological 且 non-shuffled；
- shifted-stream 如果仍使用 selected shifted block，应明确写成 candidate-pool selection，而不是 full suffix；
- 每个 split 输出 manifest：起止时间、样本数、窗口数、horizon、lookback、seed、shift score。

建议最小产物：

```text
logs/<run>/data_manifest.json
logs/<run>/split_manifest.csv
logs/<run>/normalization_manifest.json
```

## 4. Offline training

离线训练应拆成两个层级：

1. Backbone training：只训练预测模型，输出 offline checkpoint 和 offline metrics；
2. Representation/diagnostic fitting：只在 train/validation 允许范围内拟合 invariant/release/teacher 所需对象。

需要避免的问题：

- 不能在最终 test 上选择 checkpoint；
- 不能混用不同 offline checkpoint 比较 online variants；
- 不能用同一 CSV 同时记录搜索和最终评估；
- best-validation checkpoint 与最后 epoch 模型必须明确区分。

建议每个 offline run 输出：

```text
offline_metrics.csv
offline_checkpoint_manifest.json
resolved_config.json
environment_manifest.json
```

## 5. Online update protocol

这是当前论文最需要重设的部分。建议定义严格事件流：

```text
predict at t
store pending prediction and metadata
label y_{t+h} becomes available after delay
update allowed modules using only released labels
evaluate next prediction without future labels
```

Stage I/II/III 的更新边界必须写进代码：

| Stage | 可用信息 | 允许动作 | 禁止动作 |
| --- | --- | --- | --- |
| Stage I | 已释放标签、当前输入、历史状态 | 轻量门控/偏置校准 | 用当前未释放标签做 acceptance。 |
| Stage II | 已释放标签、LRP/relevance/history state | 稀疏参数更新或 mask 更新 | 用 test future loss 选择是否接受。 |
| Stage III | online-available drift/release package、已释放 residual history | transient correction 或 teacher-gated update | 在 teacher gate 未过时训练/发布 generator。 |

建议实现在线协议测试：

- delayed label availability test；
- no-current-label acceptance test；
- per-horizon release mask test；
- online stream batch size 1 或 causality-safe batch test；
- run manifest test。

## 6. Drift/adaptation mechanism

从现有经验看，直接 `phi_v - phi_i` 风险很高。新机制应优先考虑 V29 的教训：

- 使用 online-available 的 servable change package；
- 将 history-only baseline 作为硬对照；
- 将 counterfactual drift controls 作为硬门；
- 分离 representation alignment、teacher predictability、generator deployment 三个层级；
- 先证明 teacher-only correction prospectively 有效，再训练 generator；
- 每个 target 分清 raw residual、residual change、residual sign，不要让 secondary target 解锁 generator。

建议新的 Stage III admission 流程：

```text
1. Fit teacher only on train/inner validation.
2. Evaluate rolling validation windows.
3. Require real drift > history-only and real drift > counterfactual controls.
4. Require no harmful early window or predeclared tolerance.
5. Only then allow bounded generator canary.
6. Generator canary passes only if served artifact improves online test without changing protocol.
```

## 7. Baseline fairness

现有 baseline 风险很高，尤其 OneNet/SOLID/SOLID++/PROCEED 可能是 proxy。新论文必须二选一：

1. 明确称为 “implemented proxy baselines inspired by ...”，并解释统一 backbone、公平 tuning、限制；
2. 或补 faithful reproduction：官方代码/论文配置、同 split、同 horizon、同 normalization、同 tuning budget。

每个 baseline 必须记录：

- code path；
- source/citation；
- hyperparameter grid；
- tuning split；
- final checkpoint；
- metric rows；
- failure or timeout policy。

## 8. Ablation suite

建议把 ablation 分三类，不再混在一个大表中：

### Protocol ablations

- strict delayed feedback vs oracle-assisted；
- full suffix vs selected shifted block；
- history-only vs drift-plus-history；
- counterfactual modes。

### Model ablations

- no invariant；
- no RevIN；
- single expert；
- no VAE；
- no release state；
- no expert-local environment。

### Online controller ablations

- baseline；
- stage1；
- stage2；
- stage3；
- all combinations。

每个 ablation 必须共享同一个 offline checkpoint 或清楚声明重新训练策略。

## 9. Metric logging

主指标：

- online MSE vs frozen baseline；
- online MAE vs frozen baseline。

副指标：

- offline val/test MSE/MAE；
- rolling window pass rate；
- worst-window gain；
- counterfactual margin；
- reject/abstain rate；
- generator unlock status；
- update latency；
- memory/update cost；
- seed mean/std。

所有 metrics 建议拆成：

```text
metrics_offline.csv
metrics_online_summary.csv
metrics_online_steps.csv
metrics_rolling_windows.csv
metrics_gate_decisions.csv
```

不要把 debug print 当最终证据。

## 10. Reproducibility manifest

每个 run 至少写：

```json
{
  "run_id": "...",
  "script": "...",
  "command": "...",
  "resolved_config": "...",
  "dataset": "...",
  "split_manifest": "...",
  "seed": 42,
  "offline_checkpoint": "...",
  "online_protocol_version": "...",
  "git_status": "recorded or unavailable",
  "output_files": ["..."],
  "claim_scope": "search|diagnostic|final_candidate|paper_table"
}
```

特别重要：`claim_scope` 必须强制区分 search、diagnostic、final_candidate、paper_table。没有 `paper_table` scope 的结果不得进入 abstract/introduction/conclusion。

## 11. Figure/table generation

建议新增一层 table/figure manifest：

```text
docs/redesign/TABLE_FIGURE_MANIFEST.md
```

每个表/图记录：

- LaTeX label；
- generator script；
- input CSV；
- input checkpoint；
- command；
- output file；
- hash；
- claim ids；
- limitations。

case study、relevance map、t-SNE 必须标注为 illustrative/diagnostic，不得支撑平均性能或机制因果 claim。

## 12. Manuscript claim tracing

建议建立新的 claim trace：

```text
docs/redesign/CLAIM_TRACE.md
```

列：

- claim id；
- manuscript sentence；
- evidence file；
- metric；
- scope；
- limitation；
- wording allowed；
- wording forbidden。

写作时执行规则：

- 没有 manifest 的数字不得进入论文；
- 没有 uncertainty 的结果不得写 “significant”；
- proxy baseline 不得写 “faithful reproduction”；
- teacher gate negative 的 Stage III 不得写 “validated online adaptation”；
- t-SNE 不得写 “causal proof”；
- Lishui 上不超过 SOLID++ 时必须显式说明 remaining gap。

## 13. 推荐的下一步路线

1. 先冻结当前仓库为 evidence archive，不做破坏性整理；
2. 作者确认 canonical manuscript 与新研究问题；
3. 写 `docs/redesign/PROTOCOL.md`，先定义 split、delayed feedback、baseline fairness、metrics；
4. 新增最小 offline + online protocol runner；
5. 先重跑 baseline/stage variants 的 leak-safe 小规模 smoke；
6. 再做 public-3 + Lishui 的 canonical table；
7. 最后才决定是否把 Stage III 新机制作为主贡献，还是把论文主线转为 factor-mechanism + conservative online controller。

## 14. 从旧仓库吸取的核心教训

- 失败实验不能删，它们证明哪些 claim 不成立；
- Stage III 的关键不是 generator 更大，而是 teacher target 和可服务 drift signal 是否通过 prospective gates；
- alignment 指标、t-SNE、aggregate gain 都不能替代 rolling/counterfactual evidence；
- selected shifted block 必须老实写成 selected protocol；
- Lishui-9 是有价值的新区域数据族，但不能包装成 SOTA 胜利；
- 后续论文更稳的主线是“证据可追溯、保守在线适配、分布偏移诊断与部署约束”，而不是“无条件显著提升”。

