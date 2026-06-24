# 论文主张与证据缺口检查

生成时间：2026-06-24  
对照来源：`docs/audit/PAPER_CLAIMS_MATRIX.md`、`docs/audit/EXPERIMENT_EVIDENCE_LEDGER.md`、`docs/audit/PROJECT_MAP.md`、`FIGURE_MAP.md`、活动 LaTeX 稿。

## 1. 已有证据支持但必须限定范围的主张

| 主张 | 当前证据 | 可接受 wording | 限制 |
| --- | --- | --- | --- |
| 框架由 invariant representation、factor-mechanism forecasting、Stage I/II/III controller 组成 | `models/`, `train/offline_invariant.py`, `train/online_loop.py`, `README.md`, `MAIN_PROGRAM.md` | “proposes a plug-and-play stage-wise online adaptation framework” | 不应把每个 stage 都写成已验证有效。 |
| 使用训练前缀统计量做标准化 | `train/run_experiment.py`, `data/normalization.py`，claims matrix C4 | “standardized with training-prefix statistics” | 需说明 missing-value interpolation/imputation 发生在 normalization 前。 |
| 相对 frozen deployment 在 shifted public benchmarks 上有改善 | `logs/load_benchmarks/shifted_sota_methods_with_mae.csv` 等，claims matrix C6 | “improves over frozen deployment under the reported shifted-stream protocol” | 不等于所有 baselines 或所有县域都胜出。 |
| 可组合多个 transformer-family backbones | backbone screening CSV/figure，claims matrix C9 | “tested across multiple transformer-based backbones” | provenance 中等置信；需冻结唯一 CSV/manifest。 |
| RevIN 与 multi-expert factorization 平均上有价值 | component ablation，claims matrix C14 | “important on average in the reported suite” | RevIN 在个别数据集上例外。 |
| Lishui-9 提供区域数据族迁移验证 | Lishui family summary，claims matrix C18 | “validates transfer setting and reveals remaining gap to SOLID++” | 不能写成超过 SOLID++。 |
| V2/V2.5 后的 Stage III 审计路径更可审计 | V2 docs/tests/logs，claims matrix C23 | “improves auditability of Stage III protocol” | 工程/协议 claim，不是性能 claim。 |

## 2. 证据不足或过度宣称的主张

| 风险主张 | 问题 | 必要处理 |
| --- | --- | --- |
| 严格 delayed feedback online learning | `LabelBuffer` 存在，但当前审计指出 batch-delay 和 Stage II/Full current `y_t` acceptance 风险。 | 要么修实现并重跑，要么写成 oracle-assisted / batch-delayed diagnostic。 |
| shifted-stream 等于完整后缀流 | 实际是 later candidate pool 中按 shift score 选 contiguous block。 | 改写为 selected shifted block protocol。 |
| 时间轴完全无随机打乱 | offline train DataLoader 有 shuffle；在线/test 才是 chronological。 | 改写为 splits and online evaluation preserve chronological order。 |
| 优于 representative online baselines / broad superiority / SOTA | Lishui 上 SOLID++ 更强；基线多为内部 proxy。 | 改为 “competitive against implemented proxy baselines; remains behind SOLID++ on Lishui-9”。 |
| Stage III 是最可靠单组件 | all-8 table/prose 有冲突；Stage II/III rows 可能不同 checkpoint/config。 | 冻结单一可比表前删除该 claim。 |
| Full controller 最强 | 数据集依赖，Panama/Lishui 等不一定 full 最好。 | 改为 dataset-dependent。 |
| Stage II 有效 | current-label acceptance leakage 影响 Stage II/Full。 | 先做 leak-safe 重跑。 |
| VAE gate 是核心提升来源 | component ablation 中 `MLP Gate, no VAE` 更强。 | 不作为贡献点。 |
| t-SNE 证明机制因果有效 | t-SNE 是 diagnostic/visualization。 | 用 “consistent with intended separation”，不能写 “proves”。 |
| Stage III 学到 concept drift 并带来稳定性能提升 | V2-V29 多数 gate-negative、generator locked、teacher-only abort。 | 只写成审计/负结果/探索方向。 |
| manuscript citation-backed / journal-ready | 主稿无 `\cite{}`、无 `.bib`，`.bbl` 基本空。 | 补真实引用链并更新 claims matrix。 |

## 3. Stage III 证据结论

### V2-V6

已有证据支持：机制更可审计，修复 deterministic eval、zero-drift identity、split-safe preprocessing、rolling validation 等问题。

缺口：public-3、b240/full-eval、多 seed、rolling validation 与 counterfactual gates 未支持稳定性能 claim。

结论：只能作为协议修复和负/部分证据，不能写入主贡献性能表。

### V7-V12e

已有证据支持：alignment、teacher、history control、label availability 的审计路径逐渐严谨。

缺口：alignment gate 与 teacher gate 分离后，teacher/generator claim 基本失败；V12e 是更严格 delayed-feedback diagnostic，但仍 `0/27` teacher gates、`0/45` drift-plus-history rolling windows。

结论：这条链最重要的教训是 history-only 和 conditional counterfactual 必须作为 Stage III 新协议的硬对照。

### V13-V15

已有证据支持：fluctuation-first / stability-first / residual-target predictability 三种解释被系统检查。

缺口：primary residual targets 未稳定超过 history-only，teacher_only_abort 阻断 generator。

结论：不应继续扩大同一机制；下一版要重设 representation objective。

### V16-V29

已有证据支持：从 expert-local dual stream 到 V29 expert-local invariant-release，结构逐步接近“可服务变化包”而不是裸 `phi_v - phi_i`。V29 的 `pooled_servable_change` 方向比直接差分更可辩护，mode-gate replay 有治理价值。

缺口：`docs/audit/STAGE3_V18_TO_V29_RESULTS_BASELINE_SUMMARY_CN.md` 明确指出 V18-V29 主目标均未通过 strict teacher gate，generator 仍应锁定。V29 mode gate scope 是 `mode_selection_only_no_generator_unlock`，不能当成真实 online generator performance。

额外一致性风险：V29 实现记录要求修复后快照应写入 `pooled_servable_change`，但已抽查到部分早期 V29 tuning run 的 `resolved_config.json` 仍保留旧 `delta_phi_v` 口径。若未来论文或设计文档引用 V29 anchor，应优先引用已修正快照，或重跑 anchor 以避免“代码意图”和“日志快照”不一致。

结论：V29 是新设计参考，不是当前论文性能证据。

## 4. 需要重新跑或补齐的实验

| 缺口 | 建议实验/产物 | 目的 |
| --- | --- | --- |
| 表格/图像 provenance 不冻结 | 每张表/图建立 manifest：CSV row、command、config、seed、checkpoint、code version、生成脚本、输出文件 hash | 防止后续改稿引用错版本。 |
| delayed feedback 不严格 | 重新实现并测试 per-horizon label availability + no-current-label acceptance 的 Stage I/II/Full online loop | 支撑 delayed feedback claim。 |
| Stage II/Full 表格可比性不足 | 同一 offline checkpoint、同一 split、同一 seed、同一 online stream 下重跑 8 variants | 支撑 stage ranking。 |
| proxy baselines 风险 | 明确 proxy baseline 或补 faithful reproduction / citation / implementation parity | 支撑 baseline comparison。 |
| 缺少 multi-seed uncertainty | 对 canonical benchmark 做多 seed mean/std、paired tests 或 block bootstrap | 删除或支撑 robust/significant wording。 |
| Stage III 新主线 | 先 teacher-only prospective public-3 gate，再 generator canary | 防止再次把坏 teacher 放大成 generator 结果。 |
| V29 快照一致性 | 核对或重跑被引用的 V29 anchor，确认 `resolved_config.json`、`run_manifest_v29.json`、online drift source 和审计记录一致 | 防止把旧 `delta_phi_v` 快照误当成修复后 `pooled_servable_change` 结果。 |
| Lishui 图表 | 建立县级 CSV/运行命令/checkpoint manifest | 支撑 regional-family claim。 |
| citation chain | 新增 `.bib` 和真实 `\cite{}`，同步 claims matrix | 让稿件具备基本学术可追溯性。 |

## 5. 无法完全追溯或置信不足的表格/图

| 项目 | 当前证据 | 风险 |
| --- | --- | --- |
| `tab:impl_details` | 多个 config 文件 | 混合 public/Lishui/shifted 设置，缺单一 protocol manifest。 |
| `tab:shifted_online_sota` | 多个 public/Lishui summary CSV | baseline 是 proxy，Lishui 不支持 superiority。 |
| `tab:backbone_screening` | `backbone_suite_all8_consistent.csv`, `backbone_variants_summary.csv` | 多版本 CSV，缺 command/config/checkpoint manifest。 |
| `tab:all8_online_variants` | `backbone_variants_summary.csv` | Stage II/III ranking 不 decision-safe。 |
| case study figures | checkpoint + trace CSV + selected informative windows | illustrative，不代表平均效果。 |
| relevance heatmaps | generated figures and traces | 解释性图，不能支撑因果机制。 |
| t-SNE figures/tables | t-SNE scripts and CSV | diagnostic visualization，不能支撑 causal proof。 |
| shift-score quantitative | sampled shifted blocks | 每 public dataset 只有少量 sampled blocks，无 uncertainty。 |
| Lishui county figures | county CSV/figures | 需补县级 manifest。 |

## 6. 结果只能作为探索性或失败诊断的内容

- V2.5 public-3 / b240 / full-eval Stage III runs；
- V7-V15 teacher/generator/alignment probes；
- V16-V28 expert-local / observable-change / support-suppression probes；
- V29 tuning selector、mode gate replay、generator-unlock preflight；
- smoke/dryrun/pilot logs；
- `artifacts/autoresearch/main_results.tsv` 与 `main_search_results.tsv`，除非被明确定义为搜索过程而非最终表格。

## 7. 建议改写边界

可写：本仓库支持一个“evidence-aware online load forecasting framework”，其 strongest current evidence 是相对 frozen deployment 的 shifted-stream 改善、tested backbone portability、Lishui-9 的区域迁移评估以及 Stage III 失败诊断带来的新协议设计依据。

不可写：SOTA、显著提升、全面优于 baselines、Stage III 已验证 concept drift learning、V29 已解锁 generator、t-SNE 证明机制、strict delayed feedback 已完全实现。
