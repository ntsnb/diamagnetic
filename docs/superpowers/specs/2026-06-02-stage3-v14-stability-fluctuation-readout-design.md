# Stage III V14 Stability-First Fluctuation Readout Design

## 1. Purpose

V14 is a non-overwriting successor to V13. It keeps the user's corrected Stage III theory, but changes the residual-teacher readout after two corrected V13 Panama probes failed.

The corrected Stage III signal remains:

```text
fluctuation_delta(t) = delta_phi_v(t) - delta_phi_v(t-1)
```

where:

```text
delta_phi_v(t) = phi_v(t) - phi_i(t)
```

V14 does not reinterpret `phi_v - phi_i` as the short-term fluctuation itself. It treats `phi_v - phi_i` as a variable-state offset and uses the temporal change of that offset as the Stage III fluctuation candidate.

## 2. Why V13 Must Not Be Expanded Directly

The V13 implementation fixed the most important signal-contract problems, but the teacher/readout path did not become stable.

Verified V13 evidence:

- `logs/s3v13fluct_pilot/v13fluct_pilot_panama_s42_01/panama/seed_42/teacher_only_abort.json`
- `logs/s3v13fluct_b480/v13fluct_panama_s42_b480_01/panama/seed_42/teacher_only_abort.json`
- `docs/audit/STAGE3_V13_FLUCTUATION_DESIGN_AND_EXPERIMENT_RECORD.md`

Both corrected V13 runs stopped with:

- `fluctuation_plus_history_teacher_gate_failed`
- `partial_mae_gain_vs_history_nonpositive`
- `partial_mse_gain_vs_history_nonpositive`

The larger b480 run is especially important because it weakens the hypothesis that the b120 failure was only a small-sample artifact. V13 b480 increased OOF-covered samples to `416`, but `fluctuation_plus_history` still failed rolling validation and had negative validation partial gains over `history_only`.

Therefore, V14 must not simply increase the experiment size under the same relaxed V13 readout. If the V14 pilot also shows the same direction-level failure, experiments must stop again before public-3 or generator training.

## 3. Non-Overwrite Boundary

V14 must be added as a new version. It must not overwrite V2-V13 code, configs, logs, checkpoints, generated tables, or generated figures.

V14 will create new files:

- `models/stage3_residual_teacher_v14.py`
- `train/run_stage3_v14_stability_teacher_probe.py`
- `train/run_stage3_v14_stability_teacher_matrix.py`
- `configs/stage3_v14_stability_teacher_probe.yaml`
- `tests/test_stage3_v14_teacher_protocol.py`
- `docs/audit/STAGE3_V14_STABILITY_FLUCTUATION_READOUT_RECORD.md`

V14 may reuse these existing files without modifying their behavior:

- `train/stage3_v13_signal_contract.py`
- `train/stage3_v12e_label_availability.py`
- `train/stage3_v12e_conditional_permutation.py`
- `train/stage3_v12d_context_features.py`
- selected loader, training, and evaluation helpers from existing V12/V13 runners

V14 outputs must use new roots such as:

- `logs/s3v14stability/...`
- `ckpt/s3v14stability/...`

Normal V14 runs must refuse to overwrite existing run directories unless an explicit `--overwrite` flag is passed.

## 4. What V14 Selectively Inherits

V14 inherits these mechanisms:

- V13 raw `fluctuation_delta` signal contract.
- V13 full counterfactual set, including conditional controls.
- V12e per-horizon residual label availability for causal history features.
- V12d-style comparison between `fluctuation_only`, `history_only`, and `fluctuation_plus_history`.
- V12b/V12c chronological out-of-fold teacher calibration and per-output alpha diagnostics.
- Fail-closed behavior: failed teacher candidates must not produce deployable generator artifacts.

V14 must not inherit these assumptions:

- default V12b quadratic feature expansion, namely `[x, abs(x), x^2, norm(x)]`;
- relaxed correction as the served or acceptance policy;
- public-3 expansion before the single-dataset pilot passes;
- generator training before teacher-only evidence passes;
- any manuscript claim that Stage III has learned concept drift before counterfactual gates support it.

## 5. Core Algorithm Change

The key V14 change is the readout used by the train-only residual teacher.

V13 inherited a high-dimensional V12b feature map. For a context vector `x`, V12b creates:

```text
[x, abs(x), x^2, norm(x)]
```

This is useful for exploratory capacity, but it creates three risks in the current setting:

1. It expands the feature dimension without proving that the original signal is stable.
2. It can fit magnitude artifacts rather than direction-sensitive fluctuation information.
3. It makes relaxed alpha policies more likely to over-apply small unstable corrections.

V14 replaces the default teacher readout with a stability-first readout:

```text
readout_features = train_only_standardize(context_features)
```

No quadratic or absolute-value expansion is allowed by default.

Allowed readout modes:

- `linear`: raw standardized context features only.
- `blockwise_linear`: raw standardized features with recorded feature-group metadata. It must not add polynomial or norm features.

The default V14 pilot must use `linear`.

## 6. Feature Modes

The V14 teacher must evaluate the same scientific comparison that matters for Stage III:

1. `fluctuation_only`
2. `history_only`
3. `fluctuation_plus_history`

The main gate remains:

```text
fluctuation_plus_history > history_only
```

This comparison must be evaluated on rolling validation windows before test metrics are used as report-only diagnostics.

V14 may keep `level_offset_only` and `level_offset_plus_history` as optional diagnostics, but they must not be used to claim that Stage III is responding to short-term fluctuation.

## 7. Teacher Calibration And Policy

The V14 teacher must be fit only on the training split.

It must use chronological out-of-fold calibration:

- prefix-only training folds;
- optional purge gap;
- no validation or test labels during teacher fitting;
- no test-based alpha selection;
- per-output alpha selection from OOF evidence.

The served policy for V14 pilot acceptance must be strict:

- outputs that fail OOF block stability receive alpha `0`;
- outputs that fail sign-agreement criteria receive alpha `0`;
- if the strict policy zeros most outputs, the run should be treated as mechanistic negative evidence, not as a reason to use relaxed correction.

Relaxed or ungated policies may be logged as diagnostics only. They must not be used to pass V14.

## 8. Counterfactual Gate

V14 must keep the full V13 counterfactual set:

- `real`
- `zero`
- `shuffled`
- `sign_flipped`
- `amplified_2x`
- `amplified_5x`
- `random_matched_norm`
- `conditional_shuffled`
- `conditional_random_matched_norm`

The pilot can pass only if `real` beats the strongest counterfactual under the pre-declared margin. A result where `zero`, `shuffled`, or a conditional counterfactual ties or beats `real` must stop the expansion.

## 9. Stop Rules

V14 experiments must stop immediately if any of these happen in the first real-data pilot:

- `fluctuation_plus_history` fails the rolling validation gate.
- rolling partial MSE gain over `history_only` is non-positive.
- rolling partial MAE gain over `history_only` is non-positive.
- `real` does not beat the strongest required counterfactual.
- strict alpha zeros nearly all outputs while relaxed alpha appears positive.
- signal diagnostics show fluctuation collapse, non-finite features, or train/test semantic mismatch.

If any stop rule fires, V14 must write an abort file and the audit record must state that public-3 and generator training remain blocked.

## 10. First Experiment

The first meaningful V14 experiment is a single Panama real-data pilot:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v14_stability_teacher_probe.py --config configs\stage3_v14_stability_teacher_probe.yaml --dataset panama --run-id v14stable_panama_s42_b480_01 --seed 42 --epochs 5 --invariant-epochs 5 --device auto --max-train-batches 480 --max-generator-batches 480 --max-eval-batches 480 --validation-windows 5 --out-root logs\s3v14stability_b480 --checkpoint-root ckpt\s3v14stability_b480
```

This command is allowed only after V14 tests and import checks pass.

Public-3 and generator canaries are blocked until this pilot prospectively passes.

## 11. Manuscript Policy

V14 must not be added to manuscript tables or claims unless it passes the pre-declared gate and its evidence is added to the audit ledger.

If V14 fails, the scientifically correct manuscript action is to state a limitation or remove the Stage III performance claim. It is not acceptable to promote relaxed or report-only test gains as validated evidence.

