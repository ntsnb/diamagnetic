# Revision Roadmap

This roadmap is evidence-first. Do not rewrite the manuscript narrative until the Tier 1 evidence and protocol issues are resolved or explicitly relabeled.

## Tier 1: Must Fix Before Submission

1. Resolve delayed-feedback protocol validity.
   - Decide whether the online unit is per window/sample or per batch.
   - Remove current-label dependence from Stage II/Full acceptance, or label affected results as oracle-assisted ablations.
   - Re-run affected Stage II/Full comparisons after the protocol fix if those claims remain central.

2. Reconcile method prose with implementation.
   - For the legacy path, either train the Stage II/III `AdaptationGenerator` as described, or rewrite the method as a bounded heuristic parameter-perturbation controller.
   - For Stage III V2.5, state that generator training, deterministic eval, zero-drift identity, vector drift, and train-only canonicalization are engineering repairs, not proof of accuracy improvement.
   - Do not describe Stage III V2.5 as a reliable concept-drift learner unless future rolling-validation and counterfactual gates pass.
   - Align all "two-stage" references to the actual three-stage design.

3. Freeze canonical experiment evidence.
   - For every manuscript table and figure, choose one canonical source CSV/report/script.
   - Add command, config, seed, checkpoint, data path, and code version to a manifest.
   - Rebuild or remove the all-backbone Stage II/Stage III ranking claim until rows are comparable.

4. Fix baseline framing.
   - If OneNet/SOLID/SOLID++/PROCEED are internal proxies, label them as proxy implementations inspired by those methods.
   - If they are meant to be faithful baselines, add citations, implementation notes, and validation against the original papers/code.

5. Reframe abstract, introduction, and conclusion claims.
   - Remove any implied universal SOTA/superiority claim.
   - Use evidence-safe wording: improvement over Frozen on aggregate shifted benchmarks; competitive against online baselines; remaining Lishui gap to `SOLID++`.

6. Repair journal-readiness defects.
   - Replace template title/author/header/acknowledgments.
   - Rename placeholder sections.
   - Add real, verified citations and a bibliography.
   - Fix notation for `T`, sample count, lookback `L`, horizon `H`, and feedback delay.

7. Correct shifted-stream protocol text.
   - Distinguish full chronological suffix from the searched shifted block.
   - State that the reported shifted protocol selects a 20 percent online block from a later 35 percent candidate pool when that config applies.
   - Discuss selection bias as a stress-test design rather than ordinary deployment sampling.

8. Downgrade Stage III performance claims after V2.5 evidence.
   - Record Panama seed42/43 b60/b120 and public-3 seed42/43/44 b240/full-eval as audit evidence, not final performance evidence.
   - Explicitly state that all completed V2.5 claim-gated runs failed rolling/test claim gates, including 9/9 b240/full-eval public runs.
   - Remove or rewrite any sentence implying Stage III V2.5 significantly improves forecasting accuracy, reliably captures concept drift, or supports SOTA superiority.
   - Separate legacy Stage III result tables from V2.5 repair experiments so readers do not infer that the repaired path produced the original paper metrics.
   - Do not interpret b120-vs-b60 as a clean paired retraining-scale improvement, because the current runs also change online evaluation sample count.
   - Note the nuanced positive seeds separately: Panama seed43, Tetouan seed44, and Spain seed42/44 show report-only gains or direction signals, but rolling validation still fails, rejection remains high, and amplified drift remains nearly tied with real drift.

9. Keep Stage III V3 as repair/probe evidence only.
   - V3 adds temporal residual CCA alignment, generator-input counterfactual interventions, no-overwrite safeguards, resolved-config manifests, and alignment diagnostics.
   - Panama seed42 b120/full-eval shows positive report-only MSE/MAE gains and lower reject rate, but still fails because `amplified_2x` beats real drift and rolling validation pass rate is `0/5`.
   - Do not use V3 to revise the manuscript performance claim until amplified-real attribution and rolling validation are fixed across multiple seeds/datasets.

10. Keep Stage III V4 as partial mechanism repair evidence only.
   - V4 adds amplified `2x/5x` hard negatives and non-detached intervention-compare training.
   - Panama seed42 full run fixes the reject-rate failure (`24.67%` in V3 to `0.80%` in V4) and makes real drift slightly better than amplified on test aggregate.
   - The real-vs-best-counterfactual margin is only `0.000138`, below the pre-declared `0.002`, and rolling validation still fails with pass rate `0/5`.
   - Do not claim Stage III reliable concept-drift learning from V4 until validation windows and multi-seed/multi-dataset gates pass.

11. Keep Stage III V5/V5.1 as bounded repair evidence only.
   - V5 adds served-policy selection scoring, explicit abstention when no generator candidate passes, and counterfactual-aware risk-gate training.
   - The first V5 bounded probe over-rejected all real test steps.
   - The V5.1 balanced bounded probe fixes most over-rejection and reaches positive test real-vs-best-counterfactual margin, but rolling validation still passes only `2/5` windows.
   - Do not use V5/V5.1 in manuscript claims until served-policy generator selection and rolling validation pass on full, multi-seed, multi-dataset runs.

12. Keep Stage III V5 g5_b480 and V6 g5_b480 as audit evidence only.
   - V5 g5_b480 with candidate-specific served-policy risk gates improved Panama seed42 rolling validation to a best candidate of `4/5`, but still abstained because the worst-window MSE gain was `0.004620`, below the pre-declared `0.005` threshold.
   - A larger V5 g8_b960 run did not improve the claim gate; the best rolling pass rate was only `3/5`, with repeated MSE/MAE/counterfactual failures.
   - V6 whitened-Procrustes + temporal residual CCA was implemented as an opt-in train-only phi-space alignment path, but the first Panama seed42 g5_b480 canary was worse than V5 (`1/5` best rolling pass and negative worst-window MSE gain).
   - Do not update the manuscript to claim Stage III, longer-generator-budget, or V6 validity from these canaries.

13. Keep Stage III V7 residualized RRCCA as diagnostic evidence only.
   - V7 directly addressed the `phi_v - phi_i` alignment concern by removing invariant-explainable drift and retaining only train-calibrated active CCA dimensions.
   - The V7 alignment gate passed on Panama seed42 with active rank `8`, but served-policy rolling validation was `0/5` and all generator candidates abstained.
   - The non-downgraded diagnostic rerun `v7ps42g5b480diag1` preserved `125` candidate/window/mode rows. Selected epoch `4` still failed all five windows: real drift was used often and reject rates were low, but MSE/MAE gains remained below the pre-declared thresholds.
   - This supports the diagnosis that raw residual drift contains invariant leakage, but it does not support a forecasting-performance or concept-drift-learning claim.

14. Keep Stage III V8 hyperadapter as negative repair evidence only.
   - V8 added an opt-in `LayerAwareHyperAdapterGenerator` that can update both `output_head` weights and biases.
   - The implementation passed the full test suite (`127 passed`) and preserves zero-drift identity, checkpoint round-trip, artifact reload, and online replay behavior.
   - The same-budget Panama seed42 canary was worse than V7: selected epoch `3` had rolling validation `0/5`, worst MSE gain `-0.008390`, worst MAE gain `-0.005176`, and no served generator was saved.
   - Do not update the manuscript to claim that bias adapters or larger generator capacity validate Stage III.

15. Keep Stage III V9 residual-teacher repair as negative repair evidence only.
   - V9 added `TrainOnlyResidualTeacherV9`, teacher-aware RRCCA alignment, and `ResidualBasisHyperGeneratorV9` with a residual correction head.
   - V9 repaired important protocol issues: train-only teacher fitting, zero-drift identity, checkpointable teacher/generator artifacts, and explicit abstention when no served candidate passes.
   - The same-budget Panama seed42 canary still failed: selected epoch `0`, rolling validation `0/5`, worst MSE gain `-0.007082`, worst MAE gain `-0.005109`, worst real-vs-best-counterfactual margin `-0.007082`, and no served generator was saved.
   - Zero drift was the best counterfactual in every selected validation window, so V9 cannot be used to claim that real drift drives beneficial online adaptation.
   - Do not expand V9 to public-3 as a performance claim unless a future pre-declared single-dataset canary first passes the strict served-policy gate.

16. Keep Stage III V9 teacher-only diagnostics as negative mechanism evidence.
   - The new teacher-only path bypasses the generator and evaluates `y_base + teacher(z)` directly.
   - Panama seed42 same-budget teacher-only rolling validation failed `0/5`, with worst MSE gain `-0.197784`, worst MAE gain `-0.118167`, and zero drift best in every window.
   - Test also failed with MSE gain `-0.097074`, MAE gain `-0.052800`, sign agreement `0.513368`, and residual-correction cosine `0.057671`.
   - This shows the current train-only residual teacher does not transfer; do not train larger generators against this teacher direction.

17. Keep Stage III V10 as partial mechanism-repair evidence only.
   - V10 adds cross-fitted zero-preserving leakage projection and train-calibrated residual-teacher shrinkage.
   - The scalar same-budget Panama seed42 teacher-only canary improves over V9: rolling validation rises from `0/5` to `3/5`, worst MSE gain becomes positive (`0.005937`), and test MSE/MAE gains are positive (`0.026775`/`0.015189`).
   - The pre-declared teacher gate still fails: worst rolling MAE gain is `-0.000710` and worst sign agreement is `0.529688 < 0.55`.
   - The signed per-output alpha variant is worse (`2/5`, worst MSE gain `-0.001860`).
   - The expanded public-3 teacher-only matrix fails `0/9` gates and passes only `8/45` rolling windows, so the issue is not a single Panama seed42 artifact.
   - Do not train or report a V10 generator until teacher-only rolling validation passes `5/5`.

18. Keep Stage III V11 as safety-abstention evidence only.
   - V11 adds residual-transfer PLS alignment, worst-block teacher calibration, and a hard teacher-only guard before generator training.
   - The public-3 teacher-only matrix completed `9/9` runs but failed `0/9` teacher gates and `0/45` rolling validation windows.
   - Every run selected `teacher_alpha_source=fallback_block_gate` with `teacher_shrinkage_alpha=0.0`; no generator artifact was produced.
   - Alignment diagnostics passed in all rows, but this did not translate into a nonzero useful residual teacher. Do not use alignment gate pass as evidence of online adaptation.
   - Follow-up `level_vector` and `block_horizon` teacher-only ablations also failed the gate. `level_vector` stayed at `0/9` gates and `0/45` windows; `block_horizon` produced nonzero alpha in `4/9` runs but still failed `0/9` gates and `0/45` windows.
   - Do not train or report a V11 generator until a nonzero teacher first passes prospective rolling teacher-only validation.

19. Keep Stage III V12 as negative teacher-only evidence only.
   - V12 adds a train-only shared latent anchor, zero-preserving leakage removal, and outcome-aligned transport to address the `phi_v - phi_i` coordinate problem.
   - The V12 public-3 teacher-only matrix completed `9/9` runs and passed `9/9` alignment gates, with active rank `2-3`.
   - The same matrix failed `0/9` teacher gates and `0/45` rolling validation windows. Every served teacher selected `teacher_alpha_source=fallback_block_gate` with `teacher_shrinkage_alpha=0.0`.
   - Every run wrote `teacher_only_abort.json`; no `generator*.pt` artifact exists under `ckpt/s3v12m`.
   - This is evidence that V12 improves representation auditability, not evidence that Stage III now performs useful drift-conditioned prediction correction.
   - Do not train or report a V12 generator. The next implementation target is V12b teacher redesign with pre-gate alpha/block diagnostics, per-horizon/per-output correction, and a unified aligner-teacher signal contract.

20. Keep Stage III V12b/V12c as diagnostic-only evidence.
   - V12b implemented train-only signal scaling, purged forward OOF teacher selection, and per-output alpha diagnostics, but public-3 still failed `0/9` gates and `0/45` rolling windows.
   - V12b loose-alpha Panama seed42 showed that forced nonzero correction still failed rolling validation `0/5`; `sign_flipped` was the best validation counterfactual, so this was not merely a too-strict-gate issue.
   - V12c split alpha into `ungated`, `relaxed`, and `strict` policies and preserved signal-builder checkpoint metadata.
   - V12c Panama seed42 8-combination probe found `delta_vector` better than `level_vector`, but every combination still failed the teacher gate.
   - V12c best-delta public-3 completed `9/9` runs but passed `0/9` teacher gates and `0/45` rolling windows.
   - Do not train or report a V12b/V12c generator. Use these results to motivate a residual-history/context teacher redesign, not manuscript claim upgrade.

21. Keep Stage III V13/V14 as negative fluctuation-signal evidence.
   - V13 corrected the signal semantics to raw `delta_phi_v(t)-delta_phi_v(t-1)` and added the full counterfactual gate, but both corrected Panama b120 and b480 probes wrote `teacher_only_abort.json`.
   - V13 b480 failed rolling validation (`2/5`) and had negative validation partial gains versus `history_only` (`MSE -0.004330`, `MAE -0.000359`).
   - V14 then replaced the inherited V12b quadratic readout with a linear readout and made `selected_policy=strict` the served prediction path.
   - V14 Panama b480 still wrote `teacher_only_abort.json`: `fluctuation_plus_history` accepted `0/12` outputs, failed `0/5` rolling windows, and had negative validation/test partial gains versus `history_only`.
   - Do not use V13 or V14 in manuscript claims, tables, figures, or generator-training justification. Treat them as evidence that the current fluctuation representation/readout does not yet support a defensible Stage III mechanism.

## Tier 2: Strengthen Top-Journal Competitiveness

1. Add uncertainty evidence.
   - Run multi-seed evaluations for the core tables or reduce "robust/consistent" wording.
   - Report mean and standard deviation for public datasets and Lishui family means.

2. Add a held-out tuning/evaluation separation.
   - Separate hyperparameter selection from final reported benchmark datasets or disclose tuning scope clearly.

3. Add deployment-cost evidence.
   - Measure online update latency, parameter update counts, memory, and frequency for Stage I/II/III.
   - This is a defensible innovation angle if current SOTA superiority is not strong.

4. Strengthen ablation interpretation.
   - Emphasize evidence-supported components: multi-expert factorization and normalization on average.
   - Treat invariant learning as dataset-dependent.
   - Treat VAE gate as optional unless new evidence supports it.

5. Improve Lishui positioning.
   - Present Lishui as a realistic noisy regional family where Ours improves over Frozen but does not beat `SOLID++`.
   - Use county heterogeneity to support deployment-adaptation discussion, not SOTA claims.

6. Make figures reproducible.
   - Add figure manifests for case studies, relevance maps, and t-SNE plots.
   - Include selected window indices, checkpoint paths, and exact commands.

7. Add low-cost V2.5 replay diagnostics before any further expensive training.
   - Re-evaluate existing b60/b120 checkpoints on matched first-60 and first-120 prefixes where feasible.
   - Replay seed43 b120 with risk gate disabled, reject rate capped at 20 percent, and matched rejection masks.
   - Export per-sample MSE/MAE loss for paired uncertainty analysis or block bootstrap.
   - Increase rolling-window granularity by replaying existing checkpoints, not by tuning on the report-only test set.

8. Add V3 generator amplitude calibration diagnostics.
   - Export per-step rows for real, amplified, zero, shuffled, sign-flipped, and random matched-norm modes.
   - Add a generator loss or selection criterion that rewards real drift over amplified counterfactuals when validation labels support it.
   - Keep generator-input counterfactuals enabled so amplification is not normalized away.

9. Redesign V9 teacher/alignment after the negative teacher-only diagnostics.
   - Teacher-only diagnostics have now been run and are negative on Panama seed42.
   - Add train-calibrated shrinkage for teacher correction and test whether the failure is mainly magnitude over-correction.
   - Add residual-sign-aware diagnostics and losses; sign agreement around `0.51-0.54` is too weak for a reliable correction.
   - Replace additive `D - f(phi_i)` style residualization with cross-fitted leakage-subspace projection that preserves zero identity.
   - Only after teacher-only rolling validation is non-harmful should generator training be expanded.

9. Add V4 common-mask and selection-objective diagnostics.
   - Completed first-pass derived diagnostics for Panama seed42 V4: `common_accept_mask_summary.csv`, `paired_mode_differences.csv`, `validation_window_failure_breakdown.csv`, `rolling_common_accept_mask_summary.csv`, and `rolling_paired_mode_differences.csv`.
   - The diagnosis shows V4 is no longer mainly blocked by rejection rate; it is blocked by weak window-level real-vs-amplified separation.
   - If a V5 run is attempted, select generator checkpoints by rolling real-vs-amplified margin as well as MSE/MAE gain, and explicitly abstain if no checkpoint passes the pre-declared gate.
   - Add counterfactual-aware risk-gate calibration before using V5 as manuscript evidence, because the V4 gate still tends to score amplified drift as safer rather than less plausible.

10. Inspect the failed V5 served-policy windows before expanding.
   - The best V5 g5_b480 candidate was close to the pre-declared gate, but the larger V5 g8_b960 run dropped to a best rolling pass of `3/5`.
   - The next useful analysis is window-level diagnosis of the g5 epoch-4 near miss and the recurring g8 windows 3-4 failures, not manuscript rewriting.
   - If the failure is explainable and fixable without relaxing thresholds, rerun a single-seed validation-gated experiment; otherwise stop Stage III manuscript promotion.

11. Use V7 only to analyze representation leakage.
   - V7 residualizer R2 mean was `0.931512`, showing that most raw `delta_phi_v` variance was explainable from `phi_i`.
   - The post-canonical active subspace became much lower rank and did not improve served-policy forecasting.
   - The added V7 breakdown files should be used to analyze why the risk gate accepts real drift but rejects most zero/amplified drift; they should not be converted into a positive accuracy claim.
   - Future model work should inspect why `delta_phi_v` is so invariant-explainable before adding more generator capacity.

12. Build V10 only after redesigning the teacher target.
   - The V9 teacher-only diagnostic proves the current teacher itself fails on rolling validation.
   - First-pass V10 has now implemented train-calibrated shrinkage and cross-fitted zero-preserving leakage projection.
   - This substantially improves teacher-only evidence but still fails by late-window MAE and sign agreement.
   - Expanded V10 public-3 teacher-only matrix fails `0/9` gates, so the issue is not a single Panama seed42 late-window artifact.
   - Next work should target V11 horizon-wise robust residual fitting, worst-block calibration, selective reliability gating, and representation training against residual-transfer rather than post-hoc cleanup alone.
   - Keep the same strict served-policy gate; do not relax thresholds to make a failed generator appear successful.

13. Diagnose V11 signal mismatch before any generator work.
   - Current V11 public-3 evidence shows `alignment_gate_pass=True` but `teacher_alpha_source=fallback_block_gate` in every run.
   - The completed `level_vector` ablation shows the failure is not caused only by temporal differencing in `delta_vector`: effective public-3 result remained `0/9` gates, `0/45` windows, and `0/9` nonzero alpha.
   - The completed `block_horizon` ablation shows horizon-wise alpha can become nonzero (`4/9` runs), but the corrections are not validation-stable: `0/9` gates, `0/45` windows, and mean worst MSE/MAE gains of `-0.000762`/`-0.000592`.
   - Add diagnostics for train correction gain (`1 - after_mse / before_mse`) and per-block candidate-alpha metrics, because centered teacher R2 can be misleading when the served correction is exactly zero.

14. Build V12b teacher before any generator work.
   - V12 shows that alignment can pass without a usable residual teacher. The next competitive route is not larger generator capacity.
   - Use one consistent `y_base` definition across aligner, teacher fit, teacher-only diagnostics, and served inference.
   - Add a shared signal contract so aligner output, temporal delta/level choice, train-only scaling, counterfactual interventions, and teacher features are identical across fit/eval.
   - Replace global all-or-nothing `block_scalar` shrinkage with per-horizon/per-output or grouped shrinkage, so one bad block/dimension does not force every correction to zero.
   - Save pre-gate diagnostics for every candidate alpha/block/horizon before applying the final claim gate.
   - Preserve strict final gates; do not lower MSE/MAE/counterfactual/sign thresholds to make V12 appear positive.

15. Redesign after V12c around residual-history controls.
   - V12c proves that a drift-only polynomial residual teacher can expose weak local signal, but not stable cross-dataset correction.
   - Add residual-history-only and drift+history teacher baselines so reviewers can see whether `phi_v - phi_i` contributes beyond recent residual autocorrelation.
   - Add per-horizon sign calibration because V12c's closest public-3 case failed only by sign agreement.
   - Keep V12c's `ungated/relaxed/strict` diagnostic split, but require strict public-3 gates before any generator training.

16. Treat V12d as a blocking negative diagnostic before generator training.
   - V12d completed public-3 with `9/9` dataset-seed runs and `27/27` feature-mode rows.
   - All feature modes failed teacher gates (`0/27`), and no generator artifact was produced.
   - `drift_plus_history` passed only `6/45` rolling windows and did not robustly beat `history_only`.
   - Rolling-min partial MSE gain over history was positive in only `3/9` runs and averaged `-0.043138`.
   - Next model work must add explicit label-availability schedules, per-horizon residual buffers, and conditional drift permutation tests before any generator training.

17. Treat V12e as the current strictest blocker for Stage III performance claims.
   - V12e implements per-horizon label release: a residual for horizon `h` is visible only after the corresponding future label has arrived.
   - It also fixes history features while applying conditional drift nulls, so the test asks whether `phi_v - phi_i` adds information beyond delayed residual-history state.
   - The public-3 conditional matrix completed `9/9` dataset-seed runs and `27/27` feature-mode rows, but all teacher gates failed.
   - `drift_plus_history` passed `0/45` rolling windows; rolling-min partial MSE gain over history was positive in only `2/9` runs and averaged `-0.006893`.
   - Do not train a generator from the current V12e teacher. The next model target must be a new representation/teacher objective for delayed residual-release prediction and per-horizon residual correction.

## Tier 3: Optional Polish and Enhancements

1. Add a notation table or glossary.
2. Consolidate repeated discussion paragraphs.
3. Improve English fluency and reduce redundancy after evidence-safe claims are settled.
4. Clean LaTeX layout warnings and overfull tables.
5. Improve figure captions so they state source protocol and whether the evidence is illustrative or quantitative.
6. Add a limitations subsection covering proxy baselines, single-seed status if unresolved, selected shifted-block protocol, and dataset-dependent controller ranking.

## Recommended Minimal Safe Next Step

The smallest safe next modification is now an evidence-safe manuscript decision, not another tuning run:

1. Patch or relabel Stage II/Full online adaptation so it no longer claims delayed-feedback validity while using current labels.
2. Freeze one canonical evidence source for `tab:shifted_online_sota`, `tab:all8_online_variants`, and `tab:ablation_components`.
3. Downgrade Stage III wording using the V2.5 audit result: engineering repairs exist, but rolling validation failed.
4. Then make a narrow evidence-first manuscript patch to the abstract, conclusion, and protocol subsection, replacing universal superiority language with scoped claims.
5. Add the V11/V12 result to any Stage III limitation: strict block calibration currently abstains on public-3 (`0/9` gates, `0/45` windows), and V12 alignment success (`9/9`) still fails teacher validation (`0/9`, `0/45`). Stage III remains a candidate diagnostic rather than a validated performance module.
6. Add the V12b/V12c result to the same limitation: V12c exposes weak delta-vector teacher signal on Panama, but public-3 best-delta remains `0/9` teacher gates and `0/45` windows, so it cannot support a performance claim.
7. Add the V12d result to the Stage III limitation: historical residual autocorrelation explains most apparent correction, and `phi_v - phi_i` has not shown stable incremental value beyond `history_only`.
8. Add the V12e result to the Stage III limitation: under per-horizon label availability and fixed-history conditional drift controls, `drift_plus_history` still passes `0/45` rolling windows and cannot support deployable drift-conditioned adaptation.
9. Add the V13/V14/V15 result chain to the Stage III limitation: the corrected `fluctuation_delta` definition, the stability-first linear readout, and the residual-target predictability diagnostic all fail to show stable incremental value beyond `history_only`; generator training remains blocked.

## Evidence-Safe Contribution Statement Draft

Use this only after the matrix is accepted by the author:

> The paper presents a modular online load-forecasting framework for shifted deployment streams, combining factorized expert forecasting, invariant/variant representation diagnostics, and conservative stage-wise adaptation. In the current evidence, the framework improves over frozen deployment on aggregate shifted benchmarks and remains competitive with internal online adaptation baselines, while the relative benefit of individual stages is dataset- and backbone-dependent.

Stage III-specific addendum if it is retained:

> Stage III is retained as an audited transient-adaptation diagnostic. The V2.5 path fixes several implementation and protocol weaknesses of the legacy generator, but current Panama/Tetouan/Spain claim-gated evidence does not yet support claiming reliable drift-conditioned accuracy gains.

V3-specific addendum:

> A V3 probe further improves auditability through train-only temporal residual CCA and generator-input counterfactuals. The first Panama b120/full-eval probe shows positive report-only gains and lower rejection, but it still fails real-vs-amplified attribution and rolling validation, so it remains exploratory.

V4-specific addendum:

> A V4 hard-negative variant reduces rejection and improves real-vs-amplified direction on Panama seed42, but it still fails the pre-declared counterfactual margin and rolling validation. It should be reported, if at all, as a partial repair and diagnostic result rather than a validated performance contribution.

V5/V5.1-specific addendum:

> A V5 served-policy audit adds explicit failed-checkpoint abstention and counterfactual-aware risk-gate training. The balanced bounded probe improves test attribution and rejection behavior, but rolling validation still fails, so it remains a repair direction rather than validated evidence.

V8-specific addendum:

> A V8 hyperadapter variant adds bias updates to the residualized RRCCA Stage III path, but the same-budget Panama seed42 canary produces negative rolling-validation gains. This result should be reported, if at all, as evidence that additional generator capacity is insufficient without residual-teacher supervision.

V9-specific addendum:

> A V9 residual-teacher variant adds train-only residual supervision, teacher-aware alignment, and zero-drift identity. The same-budget Panama seed42 canary still fails rolling validation, and zero drift is the best validation counterfactual, so V9 should be reported only as negative repair evidence.

V9 teacher-only addendum:

> A teacher-only diagnostic shows that the train-only residual teacher itself is harmful on Panama seed42 rolling validation; applying `y_base + teacher(z)` fails all five validation windows and zero drift is best in every window. This localizes the current failure upstream of generator capacity.

V11 teacher-only addendum:

> A V11 block-calibrated teacher-only matrix over Panama, Tetouan, and Spain Energy--Weather completed all nine runs but passed no teacher gates and no rolling validation windows. The V11 safety gate abstained in every run (`alpha=0`), so it should be reported only as evidence that the stricter audit protocol blocks unsupported Stage III corrections.

V12 teacher-only addendum:

> A V12 outcome-aligned transport matrix over Panama, Tetouan, and Spain Energy--Weather completed all nine runs and passed all alignment gates, but passed no teacher gates and no rolling validation windows. V12 therefore improves the auditability of the `phi_v - phi_i` representation but does not validate Stage III forecasting correction or generator training.

V12d context-teacher addendum:

> A V12d context-teacher matrix compares drift-only, residual-history-only, and drift-plus-history teachers. The public-3 result completes all nine dataset-seed runs but passes no teacher gates. The drift-plus-history teacher does not robustly improve over residual-history-only, indicating that the current apparent correction signal is largely explained by residual autocorrelation rather than stable incremental information in `phi_v - phi_i`.

V12e label-availability context-teacher addendum:

> A V12e matrix strengthens the V12d diagnostic by enforcing per-horizon label availability and adding fixed-history conditional drift controls. Across Panama, Tetouan, and Spain Energy--Weather with three seeds, all 27 feature-mode rows fail the teacher gate and drift-plus-history passes no rolling validation windows. This indicates that the current aligned drift signal does not provide robust incremental correction under realistic delayed-feedback timing.

V15 residual-target predictability addendum:

> A V15 diagnostic directly tests whether the corrected fluctuation signal predicts pre-declared residual targets beyond causal residual history. On Panama seed42 b480, both primary targets fail: `raw_residual_vector` and `residual_change_vector` each have `0/5` positive rolling partial windows and negative validation/test partial MSE and MAE gains versus `history_only`. V15 therefore blocks generator training from the current fluctuation representation and should be reported, if at all, as negative target-predictability evidence.

Do not use this stronger version unless new evidence supports it:

> The method achieves state-of-the-art performance and consistently outperforms representative online forecasting baselines.
