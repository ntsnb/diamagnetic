# AGENTS.md

## Project mission

This repository contains a deep-learning codebase, experiment records, and a LaTeX manuscript for a top-journal paper on online learning for energy load forecasting.

Codex must treat this as a research-integrity and manuscript-improvement project, not as ordinary copyediting. The central goal is to maintain a verifiable chain:

`source code/configs/data protocol -> experiment logs/results -> figures/tables -> manuscript claims -> contribution statements`.

## Non-negotiable scientific rules

- Never invent, extrapolate, or silently “improve” experimental results, baselines, statistical tests, citations, or claims.
- Do not claim “state-of-the-art”, “significant improvement”, “substantial superiority”, or “large margin” unless the repository contains evidence supporting that wording.
- If results only slightly exceed or do not exceed SOTA, reframe the contribution around online-learning protocol, adaptation under distribution shift, update efficiency, robustness, latency/memory trade-offs, ablation evidence, or practical deployment value, but only when supported by evidence.
- Separate three categories clearly:
  1. **Verified facts** from code, logs, tables, or compiled artifacts.
  2. **Plausible interpretations** that need author confirmation.
  3. **Missing evidence** that blocks a strong top-journal claim.
- Preserve raw logs, raw result files, data splits, trained model artifacts, and seeds. Do not overwrite, delete, or rename them unless explicitly instructed.
- Prefer precise, modest scientific language over inflated novelty claims.
- Every major manuscript claim must be traceable to at least one source file, log, table, figure, or cited reference.

## Repository discovery rules

Before making substantive edits, inspect the repository and identify the actual structure. Do not assume directory names. Look for:

- LaTeX entry points: `main.tex`, `paper.tex`, `manuscript.tex`, `*.bib`, `sections/`, `figures/`, `tables/`.
- Training/evaluation code: `src/`, `models/`, `train*.py`, `eval*.py`, `scripts/`, notebooks, shell scripts.
- Experiment evidence: `results/`, `logs/`, `runs/`, `wandb/`, `tensorboard/`, `checkpoints/`, `csv`, `json`, `yaml`, `xlsx`.
- Configuration: `configs/`, `*.yaml`, `*.json`, `requirements.txt`, `environment.yml`, `pyproject.toml`, `Dockerfile`.
- Documentation: `README*`, `docs/`, previous rebuttals, reviewer notes, TODO files.

Always begin with `git status --short` and protect uncommitted user work.

## Standard work modes

### 1. Audit mode

Use this mode when asked to inspect the whole project or paper.

Deliver or update:

- `docs/audit/PROJECT_MAP.md`
- `docs/audit/PAPER_CLAIMS_MATRIX.md`
- `docs/audit/EXPERIMENT_EVIDENCE_LEDGER.md`
- `docs/audit/RISK_REGISTER.md`
- `docs/audit/REVISION_ROADMAP.md`

Do not rewrite the manuscript during the first audit unless the user explicitly asks for immediate edits.

### 2. Evidence-first revision mode

Use this mode before editing scientific prose.

For each proposed manuscript change, record:

- original claim or sentence
- issue type: language, logic, inconsistency, missing evidence, overclaim, experimental ambiguity, citation gap
- evidence source
- proposed revision
- residual risk

### 3. Experiment verification mode

Use this mode when touching experimental claims, tables, or figures.

Check:

- chronological train/validation/test split
- online-learning update protocol and whether future information leaks into training
- baseline fairness: same data split, horizon, metrics, preprocessing, tuning budget
- metrics: MAE, RMSE, MAPE/sMAPE, normalized metrics, runtime, memory, update cost where available
- random seeds and number of runs
- whether reported averages, standard deviations, and best values match logs
- whether ablations prove the stated innovation
- whether tables/figures can be regenerated from scripts or documented source files

### 4. Manuscript editing mode

Only after evidence mapping, improve:

- abstract: claims match verified results
- introduction: problem, gap, novelty, and contribution are aligned
- related work: positioning is fair and not straw-manned
- method: online-learning mechanism is reproducible and mathematically clear
- experiments: protocol, datasets, baselines, metrics, and tables are unambiguous
- discussion: limitations and failure cases are explicit
- conclusion: no stronger than the evidence

## Subagent policy

For broad audits, explicitly use subagents when available. Spawn one specialized subagent per bounded axis, then wait for all results and synthesize them.

Recommended subagents:

1. **Codebase cartographer**  
   Map training/evaluation scripts, model definitions, configs, data flow, environment, and reproducibility risks.

2. **Experiment auditor**  
   Trace all reported numbers back to logs/results; identify missing seeds, inconsistent metrics, or non-reproducible tables.

3. **LaTeX manuscript auditor**  
   Extract the paper structure, contribution claims, tables, figures, captions, labels, equations, and cross-reference issues.

4. **Claims-and-novelty auditor**  
   Build a claim-evidence matrix. Mark overclaims and suggest defensible reframing.

5. **Online-learning protocol auditor**  
   Check streaming/chronological assumptions, update timing, leakage risks, deployment realism, and adaptation/evaluation mismatch.

6. **Language-and-logic reviewer**  
   Review English, redundancy, paragraph flow, terminology consistency, and top-journal tone only after evidence risks are identified.

Do not flood the main thread with raw logs. Each subagent should return concise findings, file paths, line references when available, and actionable recommendations.

## Hooks and deterministic checks

If project hooks are configured, use them only for deterministic validation, not for inventing scientific judgments.

Good hook candidates:

- block destructive commands affecting raw data, logs, checkpoints, or result archives
- warn before editing `.bib`, result tables, or generated figures
- run lightweight LaTeX syntax checks after `.tex` edits
- run unit tests or smoke tests after Python changes
- verify that changed manuscript claims are represented in `PAPER_CLAIMS_MATRIX.md`

Hooks must be transparent. If a hook blocks or warns, explain what happened and how to proceed.

## LaTeX rules

- Preserve labels, citations, equations, table numbering, figure paths, and bibliography keys unless deliberately fixing a verified issue.
- Do not remove citations without checking whether the claim still needs support.
- Prefer minimal diffs for manuscript edits.
- After substantive `.tex` changes, run the discovered LaTeX build command if available. If no command exists, try to infer it from repository files and document the uncertainty.
- Check for unresolved references, missing bibliography entries, overfull/underfull warnings when relevant, and broken figure/table paths.

## Python/deep-learning rules

- Do not modify data loading, preprocessing, or evaluation code to make results look better.
- Do not change seeds, splits, metric definitions, or baseline settings without explicitly documenting the scientific consequence.
- If adding tests, prefer small smoke tests that validate shape, metric calculation, data leakage guards, or config parsing.
- If running full training is expensive, propose a minimal reproducibility smoke test and document what remains unverified.
- Do not add new dependencies unless necessary; explain why and prefer existing project tooling.

## Load forecasting and online-learning checklist

When evaluating the research contribution, check whether the manuscript and code clearly define:

- forecasting horizon and granularity
- datasets, time range, missing-value handling, normalization, and exogenous variables
- chronological split and validation protocol
- online update cadence and what information is available at each update
- whether the model updates per sample, batch, window, day, or drift event
- whether hyperparameters are tuned using future test information
- whether baselines receive comparable online updates or are intentionally offline baselines
- how concept drift, seasonality, holidays, weather, and anomalous demand are handled
- computational cost of updates: latency, memory, parameter updates, retraining frequency
- robustness across datasets, horizons, seasons, or drift regimes
- ablation evidence for the claimed innovation

## Verification commands

Discover and use the repository’s real commands. Common candidates:

- `pytest`, `python -m pytest`, or project-specific test command
- `python train*.py --help` and `python eval*.py --help` for interface discovery
- `latexmk -pdf -interaction=nonstopmode <main>.tex`
- `make`, `make test`, `make paper`, or scripts in `scripts/`

Do not run long training jobs unless requested. For expensive commands, first describe the expected cost and propose a smoke-test alternative.

## Change discipline

- Start each task by summarizing what files may be touched.
- Prefer small, reviewable patches.
- Maintain or update audit documents when changing scientific claims.
- Do not commit unless explicitly asked.
- Before finishing, report:
  - files changed
  - checks run
  - checks not run and why
  - unresolved scientific risks
  - next recommended action

## Definition of done

A task is done only when:

- manuscript wording matches available evidence
- all modified claims are recorded in the claims matrix
- relevant tests/builds/checks have been run or explicitly marked unavailable
- limitations and residual risks are not hidden
- the final response distinguishes verified fixes from recommendations
