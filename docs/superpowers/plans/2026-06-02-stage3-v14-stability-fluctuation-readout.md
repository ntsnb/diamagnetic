# Stage III V14 Stability-First Fluctuation Readout Implementation Plan

> For implementation agents: use test-first development. Do not modify V2-V13 behavior in place. Do not overwrite raw data, logs, checkpoints, generated tables, or generated figures. No commits unless the user explicitly asks.

## Goal

Create a non-overwriting V14 Stage III path that keeps the corrected V13 `fluctuation_delta` signal but replaces the inherited high-dimensional V12b teacher readout with a lower-risk stability-first readout. Restart experiments only after tests pass, and stop immediately if the pilot shows the same direction-level failure.

## Files

Create:

- `models/stage3_residual_teacher_v14.py`
- `train/run_stage3_v14_stability_teacher_probe.py`
- `train/run_stage3_v14_stability_teacher_matrix.py`
- `configs/stage3_v14_stability_teacher_probe.yaml`
- `tests/test_stage3_v14_teacher_protocol.py`
- `docs/audit/STAGE3_V14_STABILITY_FLUCTUATION_READOUT_RECORD.md`

Update after actual evidence exists:

- `docs/audit/EXPERIMENT_EVIDENCE_LEDGER.md`
- `docs/audit/RISK_REGISTER.md`
- `docs/audit/PAPER_CLAIMS_MATRIX.md`

## Implementation Steps

- [ ] Write V14 teacher protocol tests first.
- [ ] Implement `TrainOnlyStabilityFluctuationResidualTeacherV14`.
- [ ] Implement V14 checkpoint save/load helpers with no-overwrite protection.
- [ ] Copy the V13 probe runner into a V14 runner and switch it to the V14 teacher/readout.
- [ ] Copy the V13 matrix runner into a V14 matrix runner and keep public-3 blocked until the pilot passes.
- [ ] Create the V14 config with `linear` readout, strict policy, full counterfactual gate, and V12e history.
- [ ] Run targeted tests and `py_compile`.
- [ ] Run the Panama b480 V14 pilot only after tests pass.
- [ ] If the pilot writes `teacher_only_abort.json`, stop experiments and update audit docs.
- [ ] If the pilot writes `teacher_only_complete.json`, inspect the evidence before deciding whether public-3 is justified.

## Test Requirements

`tests/test_stage3_v14_teacher_protocol.py` must cover:

- V14 teacher defaults to `linear` readout.
- V14 `linear` readout does not create V12b-style `[x, abs(x), x^2, norm(x)]` features.
- V14 uses train-only standardization and refuses non-train fitting.
- Zero context remains zero after standardization and prediction.
- Strict policy is the default served/claim policy.
- Checkpoint save refuses overwrite.
- Loaded checkpoints preserve readout mode, standardizer state, and V14 protocol metadata.

Existing V13/V12 tests must still pass:

- `tests/test_stage3_v13_signal_contract.py`
- `tests/test_stage3_v13_teacher_protocol.py`
- `tests/test_stage3_v13_failure_diagnostics.py`
- `tests/test_stage3_v12e_label_availability.py`
- `tests/test_stage3_v12d_context_features.py`
- `tests/test_stage3_v12b_signal_contract.py`

## Verification Commands

Run before any experiment:

```powershell
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_stage3_v14_teacher_protocol.py -q
```

```powershell
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_stage3_v13_failure_diagnostics.py tests\test_stage3_v13_signal_contract.py tests\test_stage3_v13_teacher_protocol.py tests\test_stage3_v12e_label_availability.py tests\test_stage3_v12d_context_features.py tests\test_stage3_v12b_signal_contract.py -q
```

```powershell
C:\Users\admin\anaconda3\python.exe -m py_compile models\stage3_residual_teacher_v14.py train\run_stage3_v14_stability_teacher_probe.py train\run_stage3_v14_stability_teacher_matrix.py
```

First allowed experiment:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v14_stability_teacher_probe.py --config configs\stage3_v14_stability_teacher_probe.yaml --dataset panama --run-id v14stable_panama_s42_b480_01 --seed 42 --epochs 5 --invariant-epochs 5 --device auto --max-train-batches 480 --max-generator-batches 480 --max-eval-batches 480 --validation-windows 5 --out-root logs\s3v14stability_b480 --checkpoint-root ckpt\s3v14stability_b480
```

## Stop Rules

Stop before any larger experiment if the Panama b480 pilot shows:

- rolling validation gate failure;
- non-positive partial MSE or MAE gain over `history_only`;
- `real` tied or beaten by any required counterfactual;
- strict alpha collapses to mostly zero while relaxed diagnostics appear better;
- non-finite features, collapsed signal, or a mismatch in raw V13 signal semantics.

If a stop rule fires, write/update:

- `docs/audit/STAGE3_V14_STABILITY_FLUCTUATION_READOUT_RECORD.md`
- `docs/audit/EXPERIMENT_EVIDENCE_LEDGER.md`
- `docs/audit/RISK_REGISTER.md`
- `docs/audit/PAPER_CLAIMS_MATRIX.md`

and do not run public-3 or generator canaries.

