# Project Map

Audit date: 2026-05-23

Scope: first research-integrity audit for the online deep-learning load-forecasting manuscript and codebase. This file maps the repository evidence chain from code/configs/data protocol to logs/results/figures/tables/manuscript claims. No raw data, logs, checkpoints, generated figures, or manuscript source files were modified during this audit.

## Workspace State

- `git status --short` was run before substantive inspection.
- The repository currently has many untracked files, including `AGENTS.md`, manuscript artifacts, code, data, logs, checkpoints, and result CSVs.
- `AGENTS.md` already exists and defines research-integrity rules; it was not recreated.
- Audit documents are the only intended write target for this first run: `docs/audit/*`.

## Main Manuscript

- Primary LaTeX draft: `bare_jrnl_new_sample4.tex`.
- Compiled PDF: `bare_jrnl_new_sample4.pdf`.
- LaTeX build intermediates were later organized under `artifacts/latex/build/`; `bare_jrnl_new_sample4.bbl` remains at the root because the audit found no `.bib` file and an effectively empty bibliography chain.
- The manuscript is still structurally mixed with IEEE template material:
  - Template title/author/header: `bare_jrnl_new_sample4.tex:42-53`.
  - Placeholder subsection headings: `bare_jrnl_new_sample4.tex:130`, `211`, `327`, `407`.
  - Placeholder acknowledgments: `bare_jrnl_new_sample4.tex:823-824`.
- No in-text citations, no `.bib` file, and an effectively empty `.bbl` were found. This is a top-priority manuscript-defensibility gap.

## Core Code Structure

| Path | Role | Audit Notes |
| --- | --- | --- |
| `README.md` | Minimal project entry point | Points to `python train/run_experiment.py --config configs/base.yaml`. |
| `main.py` | Small entry/wrapper script | Not the main benchmark orchestrator. |
| `train/run_experiment.py` | Core single-experiment driver | Loads config, data, offline training, optional online adaptation, metrics, checkpoints. |
| `train/online_loop.py` | Stage I/II/III online adaptation loop | Critical protocol risks: batch-based delay, current-label Stage II acceptance, untrained generator. |
| `train/run_stage3_v2_sensitivity_audit.py` | Opt-in Stage III V2/V2.5 audit runner | Uses deterministic V2 components, trained generator, validation/test separation, rolling validation, and counterfactual drift modes. V2.5 evidence currently fails performance claim gates. |
| `train/stage3_generator_selection_v2.py` | V2.5 generator checkpoint selection | Selects generator checkpoint using rolling validation rather than final epoch or test feedback. |
| `train/stage3_sensitivity_gate_v2.py` | V2/V2.5 claim gate utilities | Encodes real-vs-counterfactual, MSE/MAE, nonzero shift, and reject-rate criteria for Stage III claim safety. |
| `train/online_adapt.py` | Online delay/stability utilities | Defines `LabelBuffer` and relevance-map stability tracker. |
| `train/offline_factor.py` | Offline factor-mechanism training/evaluation | Metric path for offline MSE/MAE. |
| `train/offline_invariant.py` | Offline invariant representation pretraining | Supports invariant/variant branch narrative. |
| `train/run_shifted_sota_methods.py` | Shifted-stream baseline/SOTA comparison wrapper | Implements internal proxies for named online baselines. |
| `train/run_shifted_online_baselines.py` | Simpler shifted online baseline wrapper | Frozen/GD/head-style comparison. |
| `train/run_*benchmark*.py`, `train/run_*ablation*.py` | Sweep/benchmark wrappers | Multiple wrappers create provenance risk if table sources are not frozen. |
| `train/report_*.py`, `train/analyze_*.py` | Report and figure generators | Feed paper figures/tables from logs and checkpoints. |
| `data/readers.py` | Dataset loading and timestamp sorting | Supports chronological ordering and imputation. |
| `data/windowing.py` | Sliding-window construction | Builds lookback/horizon windows before split. |
| `data/loaders.py` | Split and DataLoader construction | Implements `chronological`, `chronological_shifted`, and `random`. |
| `data/normalization.py` | Standard scaler | Training-prefix statistics are used in main path. |
| `models/factor_mechanism.py` | Main forecasting architecture | Factor/VAE/expert/backbone components. |
| `models/adaptation.py` | Online parameter delta utilities | Defines `AdaptationGenerator`; audit did not find optimizer updates for it. |
| `models/factor_mechanism_v2.py` | Deterministic VAE eval wrapper | Keeps training-time VAE sampling but defaults eval to `mu_z`; used by Stage III V2 audit path. |
| `models/adaptation_v2.py` | Identity-preserving Stage III V2 generator utilities | Bias-free / zero-drift identity design and HyperLoRA-style parameter deltas for the repaired audit path. |
| `models/aligned_representation_v2.py` | Aligned residual representation utilities | Provides residual `phi_v = phi_i + delta_phi_v`, temporal losses, standardization, and train-only residual drift canonicalizer. |
| `models/stage3_alignment_v7.py`, `models/stage3_alignment_v9.py` | Opt-in Stage III alignment repairs | V7 adds residualized RRCCA; V9 adds teacher-residual-aware calibration. Both are audit/repair paths, not validated manuscript performance evidence. |
| `models/adaptation_v9.py`, `models/stage3_residual_teacher_v9.py` | Opt-in V9 residual-teacher repair | Adds train-only residual teacher and residual-basis hypergenerator. Panama seed42 same-budget canary failed, so these files remain negative repair evidence. |
| `train/stage3_v9_teacher_diagnostics.py` | V9 teacher-only mechanism diagnostics | Evaluates `y_base + teacher(z)` without generator parameter deltas; outputs residual-sign/cosine/gain diagnostics. Panama seed42 same-budget diagnostic shows the teacher itself is harmful on rolling validation. |
| `models/stage3_alignment_v10.py`, `models/stage3_residual_teacher_v10.py`, `models/adaptation_v10.py` | Opt-in V10 teacher/alignment repair | Adds zero-preserving cross-fitted leakage-subspace projection and train-calibrated residual-teacher shrinkage. Panama seed42 same-budget teacher-only improves over V9 (`3/5` rolling windows) but still fails the pre-declared gate, so no V10 generator is promoted. |
| `train/run_stage3_v10_sensitivity_audit.py`, `train/stage3_generator_training_v10.py`, `train/stage3_v10_artifacts.py` | V10 isolated audit runner and artifact loaders | Copies the V2/V9 audit path into a V10 opt-in entry point, with teacher-only abort before generator training when rolling teacher diagnostics fail. The runner now also supports `--teacher-only` to stop after teacher diagnostics even if a teacher passes. |
| `train/run_stage3_v10_teacher_audit_matrix.py` | V10 public-3 teacher-only matrix runner | Non-overwriting batch runner for dataset/seed matrices. It writes matrix manifests and summaries, defaults to short Windows-safe paths, and always calls the V10 runner with `--teacher-only`. |
| `models/stage3_alignment_v11.py`, `models/stage3_residual_teacher_v11.py` | Opt-in V11 teacher/alignment repair | Adds residual-transfer PLS alignment and block-calibrated residual teacher with fail-closed shrinkage. The public-3 teacher-only matrix failed `0/9` gates and `0/45` rolling windows; every teacher abstained with `alpha=0`. |
| `train/run_stage3_v11_sensitivity_audit.py`, `train/stage3_v11_artifacts.py` | V11 isolated audit runner and artifact loader | Uses explicit V9/V10/V11 teacher dispatch, writes V11 teacher diagnostics, loads V11 artifacts, and treats `--teacher-only` as a hard guard before generator training. |
| `train/run_stage3_v11_teacher_audit_matrix.py` | V11 public-3 teacher-only matrix runner | Non-overwriting batch runner for public-3/seeds `42/43/44`; writes manifests under isolated `logs/s3v11*` roots and always calls the V11 runner with `--teacher-only`. It was also used for `level_vector` and `block_horizon` teacher-only ablations. |
| `models/stage3_alignment_v12.py`, `models/stage3_residual_teacher_v12.py` | Opt-in V12 outcome-aligned transport repair | V12 adds a train-only shared latent anchor, zero-preserving leakage removal, and chronological residual-transfer active-dimension calibration. The current teacher is a V12 protocol wrapper around the V11 strict block/reliability teacher. The public-3 matrix passed `9/9` alignment gates but failed `0/9` teacher gates and `0/45` windows; every served alpha was zero. |
| `train/run_stage3_v12_sensitivity_audit.py`, `train/stage3_v12_artifacts.py` | V12 isolated audit runner and artifact loader | Copies the V11 runner boundary into an opt-in V12 path, writes V12 alignment/teacher diagnostics, and keeps `--teacher-only` as a hard guard before generator training. |
| `train/run_stage3_v12_teacher_audit_matrix.py` | V12 public-3 teacher-only matrix runner | Non-overwriting batch runner for public-3/seeds `42/43/44`; writes manifests under isolated `logs/s3v12m` roots and always calls the V12 runner with `--teacher-only`. Current result is negative for teacher/generator claims despite positive alignment gates. |
| `models/stage3_residual_teacher_v12b.py`, `models/stage3_residual_teacher_v12c.py` | Opt-in V12b/V12c residual-teacher redesigns | V12b adds train-only chronological OOF teacher diagnostics; V12c separates `ungated`, `relaxed`, and `strict` alpha policies so weak signal is visible before final claim gates. V12c also preserves policy and signal-builder metadata in checkpoints. Public-3 remains negative (`0/9` gates, `0/45` windows). |
| `train/stage3_v12b_signal_contract.py`, `train/stage3_v12c_signal_contract.py` | Train-only Stage III teacher signal contracts | Fit robust scale on train split only, reuse it at eval/test, and preserve zero-drift identity. V12c runner also syncs CLI `--signal-mode` into the residual-teacher config to avoid silent signal-mode splits. |
| `train/run_stage3_v12c_sensitivity_audit.py`, `train/run_stage3_v12c_teacher_audit_matrix.py` | V12c isolated teacher-only runner and matrix runner | Non-overwriting V12c audit path. Used for Panama seed42 8-combination probe and best-delta public-3 matrix. It kept generator training blocked because teacher gates failed. |
| `train/stage3_v12d_context_features.py`, `models/stage3_residual_teacher_v12d.py` | V12d context teacher diagnostic | Adds causal residual-history features and a V12d context OOF teacher to compare `drift_only`, `history_only`, and `drift_plus_history`. Public-3 shows history explains most apparent correction and `phi_v - phi_i` does not robustly add incremental value. |
| `train/run_stage3_v12d_context_teacher_probe.py`, `train/run_stage3_v12d_context_teacher_matrix.py` | V12d teacher-only runner and matrix runner | Non-overwriting diagnostic path under short roots such as `logs/s3v12dm`; never trains generator. Public-3 completed `9/9` dataset-seed runs and `27/27` feature-mode rows with `0/27` teacher gates. |
| `train/stage3_v12e_label_availability.py` | V12e per-horizon residual-history builder | Enforces horizon-aware label release before residuals enter history features. This prevents higher-horizon future labels from appearing in the online history buffer too early. |
| `train/stage3_v12e_conditional_permutation.py` | V12e conditional drift controls | Adds history-norm stratified drift permutation and matched-norm random drift controls so `phi_v - phi_i` is tested against fixed-history nulls. |
| V12e opt-in extensions in `train/run_stage3_v12d_context_teacher_probe.py` and `train/run_stage3_v12d_context_teacher_matrix.py` | V12e label-availability matrix path | Defaults remain V12d-compatible; V12e is enabled only by explicit `--history-protocol v12e_per_horizon_label_availability`. Final public-3 conditional matrix still failed `0/27` gates and `0/45` `drift_plus_history` windows. |
| `models/lrp.py` | Relevance-map support | Used by Stage II gating and interpretability figures. |
| `utils/*.py` | Config, logging, paths, seed, plotting | CSV append logging and plotting helpers. |
| `tests/test_shapes.py` | Minimal smoke test | Only covers model output shapes. |

## Config Map

| Path | Role | Evidence Notes |
| --- | --- | --- |
| `configs/base.yaml` | Default single-run config | Chronological split, batch size 32, epochs 10, target defaults. |
| `configs/best_online.yaml` | Tuned default-like online config | Similar to `base.yaml`; not necessarily the shifted-stream paper config. |
| `configs/best_load_online.yaml` | Public load benchmark config | Batch size 64, epochs 5, lr `5e-4`, RevIN enabled; supports implementation-details table. |
| `configs/best_lishui_online.yaml` | Lishui shifted config | `split_mode: chronological_shifted`, `online_test_ratio: 0.2`, `shift_pool_ratio: 0.35`. |
| `configs/tuning/load_candidate_i_shifted.yaml` | Shifted tuning config | Same 20 percent selected online block inside later candidate pool. |
| `configs/model.yaml`, `configs/data.yaml`, `configs/online.yaml` | Component configs | Need schema documentation before final reproducibility claim. |
| `configs/stage3_v11_block_calibrated_teacher_gate.yaml` | V11 teacher-only repair config | Uses `residual_teacher.version: v11`, `cross_fitted_residual_transfer_pls_v11`, and the strict teacher-only claim gate. Current public-3 result is complete abstention, not performance evidence. |
| `configs/stage3_v11_block_horizon_teacher_gate.yaml` | V11 horizon-alpha ablation config | Uses the same strict teacher-only claim gate with `residual_teacher.alpha_mode: block_horizon`. The ablation produced nonzero alpha in `4/9` runs but still failed `0/9` gates and `0/45` windows. |
| `configs/stage3_v12_outcome_aligned_teacher_gate.yaml` | V12 teacher-only repair config | Uses `residual_teacher.version: v12`, `canonicalizer_method: outcome_aligned_transport_v12`, strict teacher-only abort gates, and V12 output roots. Current public-3 result supports alignment audit only, not forecasting improvement. |
| `configs/stage3_v12c_multi_policy_teacher_probe.yaml`, `configs/stage3_v12c_multi_policy_teacher_probe_nonnegative.yaml` | V12c multi-policy teacher probes | Signed and nonnegative alpha configs used for the Panama 8-combination probe and best-delta public-3 matrix. Current evidence supports diagnostic improvement only, not Stage III performance validation. |
| `configs/stage3_v12d_context_teacher_probe.yaml` | V12d drift-versus-history context teacher config | Defaults to `delta_vector`, `drift_only/history_only/drift_plus_history`, history lags `1,2,3,6,12,24`, and `feedback_delay=0`. Current results are diagnostic/immediate-feedback evidence, not deployment-ready delayed-feedback proof. |
| V12e CLI overrides on `configs/stage3_v12d_context_teacher_probe.yaml` | V12e label-availability context teacher config path | Uses the V12d config with explicit `--history-protocol v12e_per_horizon_label_availability`, `--history-label-delay`, conditional drift modes, and `--diagnostic-include-offline-counterfactuals`; this keeps V12e isolated without overwriting V12d defaults. |

## Data and Artifact Map

| Path | Role | Risk |
| --- | --- | --- |
| `dataset/weather.csv` | Default config data source | Not necessarily a paper benchmark. |
| `dataset/丽水全社会负荷.csv` | Local Lishui raw/load data | Encoding/path handling needs care. |
| `dataset/load_benchmarks/*.csv` | Public and Lishui benchmark files | Raw evidence; must not be overwritten. |
| `checkpoints/` | Saved model states | Paper figures may depend on these; no deletion or regeneration in audit. |
| `logs/offline_metrics.csv`, `logs/test_metrics.csv`, `logs/online_compare.csv` | Single-run metrics | Append-mode logs; not final table sources by default. |
| `logs/load_benchmarks/` | Main public benchmark results, figures, reports | Multiple versions exist; canonical source must be frozen per table. |
| `logs/load_benchmarks/stage3_v2/` | Isolated Stage III V2/V2.5 audit outputs | New namespace used to avoid overwriting original logs. Latest V2.5 Panama seed42/43 b60/b120 runs are diagnostic evidence only because claim gates fail. |
| `logs/s3v7p*`, `logs/s3v8*`, `logs/s3v9*`, `logs/s3v10*`, `logs/s3v10m*`, `logs/s3v11m*`, `logs/s3v11lv*`, `logs/s3v11bh*`, `logs/s3v12m*`, `logs/s3v12b*`, `logs/s3v12c*`, `logs/s3v12dm*`, `logs/s3v12em*`, `logs/s3v12emc*` | Later Stage III served-policy/teacher repair canaries and matrices | V7/V8/V9/V10/V11/V12/V12b/V12c/V12d/V12e diagnostics and matrices. V10 improves teacher-only behavior on Panama seed42 but fails public-3 (`0/9` gates). V11/V12 public-3 fail `0/9` and `0/45`. V12c exposes weak delta-vector signal on Panama but best-delta public-3 still fails. V12d shows residual-history explains most apparent correction and still fails `0/27` gates. V12e adds per-horizon label availability and conditional drift controls but still fails `0/27` gates and `0/45` drift-plus-history rolling windows. No served V10/V11/V12/V12b/V12c/V12d/V12e generator artifact should be promoted as manuscript performance evidence. |
| `ckpt/s3v12m/` | V12 public-3 teacher-only checkpoint root | Contains V12 teacher/alignment artifacts where runs completed, but no `generator*.pt` served artifact was found. This is expected because teacher-only gates failed and generator training remains blocked. |
| `ckpt/s3v12c_probe8/`, `ckpt/s3v12cm/` | V12c teacher-only checkpoint roots | Contain V12c teacher/alignment artifacts from completed probes and public-3 matrix. No `generator*.pt` artifact was found; this is expected because V12c remained teacher-only and failed gates. |
| `ckpt/s3v12dm/` | V12d context-teacher checkpoint root | Contains V12d teacher/context artifacts from the short-path public-3 matrix. No `generator*.pt` artifact was found; this is expected because V12d is teacher-only and all gates failed. |
| `ckpt/s3v12emc/` | V12e label-availability context-teacher checkpoint root | Contains V12e teacher/context artifacts from the conditional public-3 matrix. No `generator*.pt` artifact was found; this is expected because V12e remained teacher-only and all gates failed. |
| `logs/lishui_benchmarks/` | Lishui family results, figures, reports | Strong evidence for Lishui transfer and SOLID++ gap. |
| `logs/tensorboard/` | Training traces | Not audited deeply in first pass. |
| `artifacts/autoresearch/main_results.tsv`, `artifacts/autoresearch/main_search_results.tsv` | Search/tuning trial summaries | Not direct paper-table evidence; separate from benchmark ledger. |

## Data and Evaluation Flow

Verified core path:

1. YAML config is loaded by `train/run_experiment.py`.
2. Tabular time series is loaded and sorted in `data/readers.py`.
3. Missing values are dropped or imputed before tensor conversion.
4. Standardization in `train/run_experiment.py:155-158` fits on the first `train_ratio` of the raw series, then transforms the full series.
5. Sliding windows are built in `data/windowing.py`.
6. Window indices are split in `data/loaders.py`.
7. Offline factor model is trained and evaluated in `train/run_experiment.py`.
8. Online evaluation streams the test subset through a non-shuffled DataLoader.
9. Stage I/II/III online adaptation is executed in `train/online_loop.py`.
10. Metrics are appended to CSV logs and later summarized by report scripts.

Important protocol finding:

- Chronological splitting exists, but training batches are shuffled (`data/loaders.py:120`). The manuscript should say splits and online evaluation are chronological, not that no train-batch shuffling occurs anywhere.
- `chronological_shifted` selects a high-shift contiguous block from a later candidate pool (`data/loaders.py:97-110`), not necessarily the full 35 percent suffix.

## Result and Figure Map

`FIGURE_MAP.md` provides a useful first-generation map. Key paper evidence families:

- `tab:shifted_online_sota`: `logs/load_benchmarks/shifted_sota_methods_with_mae.csv`, `logs/load_benchmarks/shifted_sota_methods_with_lishui.csv`, `logs/lishui_benchmarks/shifted_sota_methods_with_mae.csv`.
- `tab:backbone_screening` and `tab:all8_online_variants`: `logs/load_benchmarks/backbone_suite_all8_consistent.csv`, `logs/load_benchmarks/backbone_variants_summary.csv`.
- `tab:ablation_stage_combo`: `logs/load_benchmarks/ablation_stage_combo_by_dataset.csv`, `logs/lishui_benchmarks/family_summary.csv`.
- `tab:ablation_components`: `logs/load_benchmarks/ablation_component_by_dataset.csv`, `logs/load_benchmarks/ablation_component_with_lishui.csv`.
- `tab:tsne_summary`: `logs/load_benchmarks/tsne/tsne_summary.csv`.
- `tab:shift_score_quantitative`: `logs/load_benchmarks/shift_score_quantitative_table.csv`.
- `tab:lishui_family`: `logs/lishui_benchmarks/family_summary.csv`, `logs/lishui_benchmarks/final_shifted_tuned_summary.csv`.
- Stage III V2.5 audit evidence: `docs/audit/STAGE3_V25_CANONICAL_CKPT_SELECTION_EXPERIMENT_RECORD.md` plus `logs/load_benchmarks/stage3_v2/codex_stage3_v25_canonical_ckptselect_panama_b60_20260525/` and `logs/load_benchmarks/stage3_v2/codex_stage3_v25_canonical_ckptselect_panama_b120_20260525/`. These are not manuscript performance-table evidence unless future validation gates pass.

## Immediate Reproducibility Gaps

- No per-table manifest with command, config, git hash, data file, seed, checkpoint, and output CSV.
- Multiple overlapping result versions exist for backbone and SOTA comparisons.
- No multi-seed uncertainty/statistical test evidence was found.
- The main single-run driver saves best-validation checkpoints but evaluates the last epoch model unless a wrapper reloads separately.
- Named baselines in `run_shifted_sota_methods.py` appear to be internal proxy implementations, not verified faithful reproductions of external methods.
- Current audit did not run expensive training or regenerate tables.
- Stage III V2.5 has mechanism/protocol repair evidence but not reliable performance evidence: all current Panama seed42/43 b60/b120 rolling validation pass rates are `0.3333` and all test claim gates fail.
- Stage III V7/V8/V9 later repairs improve auditability, but their same-budget Panama seed42 served-policy canaries all fail; V9 specifically fixes train-only residual teacher and zero-drift identity, yet real drift is harmful and zero drift is the best validation counterfactual.
- V9 teacher-only diagnostics now localize the failure upstream of generator capacity: the train-only residual teacher alone fails Panama seed42 rolling validation `0/5`.
- V11 public-3 teacher-only diagnostics now show complete safety abstention: `0/9` teacher gates, `0/45` rolling windows, `teacher_alpha_source=fallback_block_gate`, and no generator artifact. Passing alignment diagnostics is not enough to support a Stage III forecasting claim.
- V12 public-3 teacher-only diagnostics show stronger alignment auditability but not a usable correction: `9/9` alignment gates, `0/9` teacher gates, `0/45` rolling windows, all `teacher_alpha_source=fallback_block_gate`, and no generator artifact. The next code work should be V12b teacher redesign and pre-gate alpha/block diagnostics, not generator training.
- V12d public-3 context-teacher diagnostics show that residual-history explains most apparent correction. `drift_plus_history` does not robustly beat `history_only`, all `27/27` feature-mode rows fail teacher gates, and no generator artifact exists.
- V12e public-3 label-availability context-teacher diagnostics are now the strictest Stage III evidence: per-horizon release masks and fixed-history conditional drift controls still produce `0/27` teacher gates, `0/45` `drift_plus_history` rolling windows, negative average rolling partial gain over history, and no generator artifact.
