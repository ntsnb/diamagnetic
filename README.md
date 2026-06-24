# Time-Series OOD Forecasting Framework

This repository contains a modular framework for:
- Invariant learning (FOIL-style MTIL/MTEI)
- Factor-mechanism forecasting (Transformer + VAE + MoE experts)
- Online adaptation (Stage I/II/III with LRP-driven updates)

## Data Format
Your dataset should have:
- First row: column names
- One timestamp column
- Multiple numeric feature columns
- File type: `.xlsx/.xls` or `.xml`

Example row (per timestamp):
```
timestamp, var1, var2, var3, ...
2020-06-01 16:15:00, 22.486267, 239.35866, 263.36392, ...
```

## Config
Edit `configs/base.yaml`:
```
data:
  source: file
  path: "PATH_TO_YOUR_FILE.xlsx"
  timestamp_col: "timestamp"
  feature_cols: []   # empty = use all columns except timestamp
```

## Run
```
python train/run_experiment.py --config configs/base.yaml
```

### Ablation options
```
--no_stage2
--no_stage3
--no_lrp
--no_invariant
```

### Save checkpoint
```
python train/run_experiment.py --save_ckpt
```

## Notes
- If you don't know `timestamp_col` or `feature_cols`, leave them as-is for now.
- XML loading depends on the XML schema; Excel is recommended.

## Project Organization

This repository now keeps the active manuscript, source code, configs, audit records, logs, and checkpoints in separate documented categories.

- Repository map: `docs/project/REPOSITORY_ORGANIZATION.md`
- Path policy: `docs/project/PATH_POLICY.md`
- Stage III version index: `docs/project/STAGE3_VERSION_INDEX.md`
- Reorganization log: `docs/project/REORGANIZATION_LOG.md`
- Archive index: `archive/README.md`
- Build artifact index: `artifacts/README.md`
- Config guide: `configs/README.md`
- Training script guide: `train/README.md`
- Model guide: `models/README.md`
- Research audit map: `docs/audit/PROJECT_MAP.md`
- Claims matrix: `docs/audit/PAPER_CLAIMS_MATRIX.md`
- Experiment evidence ledger: `docs/audit/EXPERIMENT_EVIDENCE_LEDGER.md`

Use repository-relative paths in configs, for example `dataset/weather.csv`. Historical absolute paths are tolerated at runtime through `utils.paths.resolve_project_path`, but new configs should not contain user-machine-specific paths.

Path audit:
```
python tools/audit_project_paths.py --strict
```

LaTeX asset audit:
```
python tools/audit_latex_assets.py
```

Repository layout audit:
```
python tools/audit_repository_layout.py --strict
```
