# Stage III Generator Expert Diagnosis and Repair Strategy

Date: 2026-05-24

This document is a deeper expert-level follow-up to:

- `docs/audit/STAGE3_GENERATOR_BEGINNER_EXPLANATION.md`
- `docs/audit/STAGE3_ABLATION_EXPANDED_RESULTS.md`

It answers four concrete questions:

1. Is the current `AdaptationGenerator` connected to training?
2. Is the current generator architecture well designed?
3. Why is `phi_v - phi_i` not yet a rigorous drift representation, and how should it be fixed?
4. Is VAE stochastic sampling during `eval()` correct or incorrect?

No model code is changed in this document. It is a diagnosis and repair strategy.

---

## 2026-06-01 Expert Update: V12e Label-Availability Context Teacher

V12e tightens V12d in the direction requested by the online-learning protocol
auditors. V12d compared `drift_only`, `history_only`, and
`drift_plus_history`, but its residual-history contract still allowed a
sample-lagged full horizon vector to enter history. For multi-step forecasting,
that is too optimistic: a residual for horizon `h` is observable only after the
corresponding future label has arrived.

Implemented V12e code:

- `train/stage3_v12e_label_availability.py`
- `train/stage3_v12e_conditional_permutation.py`
- opt-in extensions in `train/run_stage3_v12d_context_teacher_probe.py`
- opt-in passthrough in `train/run_stage3_v12d_context_teacher_matrix.py`
- `tests/test_stage3_v12e_label_availability.py`
- `tests/test_stage3_v12e_conditional_permutation.py`

The V12e residual-history rule is:

```text
e(t,h) = y(t+h) - y_base_hat(t,h)
available at current origin k only if t + h + label_delay <= k
```

Equivalently, for a sample row lag `l`, horizon `h` is legal only when:

```text
l >= h + label_delay
```

This is a materially stricter online contract than V12d. V12e also adds
fixed-history conditional drift controls:

- `conditional_shuffled`: permute drift rows only within history-norm strata.
- `conditional_random_matched_norm`: replace drift direction with random vectors
  while preserving row norms and history strata.

Final V12e public-3 evidence:

- Manifest:
  `logs/s3v12emc/_manifests/v12emc_delta_20260601_214834_e5_ie5_b480_ew480`
- Detailed record:
  `docs/audit/STAGE3_V12E_LABEL_AVAILABILITY_CONTEXT_TEACHER_RECORD.md`
- `9/9` dataset-seed runs completed.
- `27/27` feature-mode rows completed.
- `0/27` teacher gates passed.
- `drift_plus_history` rolling windows: `0/45`.
- rolling-min partial MSE positive runs: `2/9`.
- rolling-min partial MSE average: `-0.006893`.
- test partial MSE positive runs: `5/9`.
- test partial MSE average: `0.000954`.
- best counterfactual modes for `drift_plus_history` rolling windows included
  `amplified_2x` (`15`), `conditional_random_matched_norm` (`10`), and
  `conditional_shuffled` (`9`).
- generator artifacts: `0`.

Interpretation:

V12e is stronger negative evidence than V12d. It shows that once residual
history obeys horizon-aware label release, `history_only` no longer produces a
large stable correction, and `drift_plus_history` does not recover a robust
incremental signal. The negative counterfactual margin means the real
`phi_v - phi_i` signal does not beat fixed-history conditional drift nulls.

The correct engineering decision is therefore to keep generator training
blocked. The next Stage III attempt must redesign the representation objective
so `phi_v` predicts delayed residual-release events or per-horizon residual
correction under a realistic label-availability schedule. Continuing to enlarge
the current generator would only train a parameter adapter on a teacher signal
that has not passed causal-history controls.

---

## 2026-06-01 Expert Update: V12d Context Teacher

V12d adds a direct control for a major confounder: residual autocorrelation.
The question is no longer only whether an aligned `phi_v - phi_i` drift signal
can predict forecast residuals, but whether it predicts anything that a causal
history of previous residuals does not already explain.

Implemented V12d code:

- `train/stage3_v12d_context_features.py`
- `models/stage3_residual_teacher_v12d.py`
- `train/run_stage3_v12d_context_teacher_probe.py`
- `train/run_stage3_v12d_context_teacher_matrix.py`
- `configs/stage3_v12d_context_teacher_probe.yaml`
- `tests/test_stage3_v12d_context_features.py`
- `tests/test_stage3_v12d_teacher_modes.py`

The V12d feature contract compares:

```text
drift_only          = V12c delta-vector signal
history_only        = lagged causal residual features
drift_plus_history  = concatenation of the two
```

The key scientific metric is the partial gain:

```text
partial_mse_gain_vs_history =
    MSE_gain(drift_plus_history) - MSE_gain(history_only)
```

V12d public-3 evidence:

- Manifest:
  `logs/s3v12dm/_manifests/v12dm_delta_20260601_203114_e5_ie5_b480_ew480`
- Detailed record:
  `docs/audit/STAGE3_V12D_CONTEXT_TEACHER_RECORD.md`
- `9/9` dataset-seed runs completed.
- `27/27` feature-mode rows completed.
- `0/27` teacher gates passed.
- `drift_plus_history` rolling windows: `6/45`.
- rolling-min partial MSE positive runs: `3/9`.
- rolling-min partial MSE average: `-0.043138`.
- test partial MSE positive runs: `4/9`.
- test partial MSE average: `-0.019044`.
- generator artifacts: `0`.

Interpretation:

```text
History-only residual features explain a large share of the apparent correction
signal. The current phi_v - phi_i drift representation does not robustly add
incremental predictive value beyond residual autocorrelation.
```

This is a stronger negative result than V12c. V12c showed that the drift signal
was weak; V12d shows that even when a correction signal exists, it is mostly
available from historical residuals. Stage III generator training remains
scientifically blocked.

---

## 2026-06-01 Expert Update: V12 Outcome-Aligned Transport Prototype

V12 was added after the V11 main matrix and V11 signal/horizon ablations both
failed to validate Stage III. The design target is narrower than a performance
claim:

```text
Fix the phi_v - phi_i coordinate problem first, then test whether the repaired
drift representation can support a nonzero train-only residual teacher.
```

Implemented V12 code is opt-in only:

- `models/stage3_alignment_v12.py` adds
  `TrainOnlyOutcomeAlignedTransportAlignerV12`.
- `models/stage3_residual_teacher_v12.py` adds
  `TrainOnlyOutcomeAlignedBlockResidualTeacherV12`, a V12 protocol wrapper
  around the strict V11 block/reliability teacher.
- `train/stage3_v12_artifacts.py` loads V12 aligner/teacher artifacts.
- `train/run_stage3_v12_sensitivity_audit.py` is an isolated V12 runner.
- `train/run_stage3_v12_teacher_audit_matrix.py` is a non-overwriting
  teacher-only matrix runner.
- `configs/stage3_v12_outcome_aligned_teacher_gate.yaml` selects
  `canonicalizer_method: outcome_aligned_transport_v12` and
  `residual_teacher.version: v12`.

The V12 aligner is materially different from V7/V10/V11:

1. It fits a train-only shared latent anchor from both `phi_i` and `phi_v`.
2. It projects `phi_v - phi_i` into that anchored tangent space before
   leakage removal.
3. It estimates invariant-predictable leakage as an orthogonal subspace and
   removes it with a zero-preserving projection.
4. It uses chronological residual-transfer calibration to select active
   dimensions.
5. It fails closed: if no dimension passes, transformed drift is zero and the
   downstream teacher must abstain.
6. It keeps `phi_i == phi_v -> transformed drift == 0` as a hard identity.

Verification so far:

```text
py_compile V12 modules and runners: passed
pytest tests/test_stage3_v12_alignment_protocol.py \
       tests/test_stage3_v12_teacher_protocol.py \
       tests/test_stage3_v12_artifacts.py -q: 6 passed
git diff --check: passed
```

Smoke evidence:

- `logs/s3v12_smoke/smoke_v12_panama_s42_20260601_03/panama/seed_42`
- `logs/s3v12_smoke_matrix2/_manifests/v12sm2_20260601_134230_e1_ie1_b8_ew8`

The smoke result is intentionally not a performance result. It used a tiny
budget and showed safety behavior: V12 alignment failed closed
(`active_rank=0`), the V12 teacher selected `alpha=0`, rolling validation was
`0/2`, and `teacher_only_abort.json` blocked generator training. No generator
artifact was produced.

Expert conclusion:

V12 is now a runnable alignment-repair prototype, but not yet evidence of
successful online adaptation. The next valid step is a pre-registered
teacher-only V12 matrix. A generator canary remains scientifically blocked
until a nonzero teacher first passes prospective rolling teacher-only gates.

Detailed record:

- `docs/audit/STAGE3_V12_OUTCOME_ALIGNED_TRANSPORT_RECORD.md`

## 2026-06-01 Expert Update: V12 Public-3 Matrix Negative Result

The V12 public-3 teacher-only matrix has now been completed:

- Manifest:
  `logs/s3v12m/_manifests/v12tm_20260601_134623_e5_ie5_b480_ew480/matrix_manifest.json`.
- Summary:
  `logs/s3v12m/_manifests/v12tm_20260601_134623_e5_ie5_b480_ew480/matrix_summary.csv`.
- Datasets: `panama`, `tetouan`, `spain_energy_weather`.
- Seeds: `42`, `43`, `44`.
- Epochs: `5`; validation windows: `5` per run.
- Teacher-only mode: enabled.

Verified result:

| Metric | Result |
| --- | ---: |
| Completed runs | `9/9` |
| Alignment gates | `9/9` |
| Teacher gates | `0/9` |
| Rolling validation windows | `0/45` |
| Nonzero teacher alpha | `0/9` |
| Teacher alpha source | all `fallback_block_gate` |
| Teacher-only aborts | `9/9` |
| Generator artifacts | `0` |

This result is scientifically useful but negative for the Stage III claim. It
separates two facts that the manuscript must not conflate:

1. **Representation progress**: V12 did not collapse in the larger run.
   Every run passed the alignment gate, with active rank `2` or `3`.
2. **Forecasting correction failure**: every run failed the teacher gate, every
   served alpha was zero, and no validation window passed.

The dominant failure pattern is not "no aligned signal exists"; the matrix
shows `alignment_oof_transfer_mse_gain` values ranging from small positive
signals to stronger values such as Panama seed42 `0.640201` and Tetouan seed43
`0.476276`. The dominant failure is that the downstream strict teacher does
not convert that alignment signal into a stable prospective residual
correction.

Mechanism diagnosis:

- V12 alignment improves the coordinate handling of `phi_v - phi_i`, but this
  is not a forecasting metric.
- V12 teacher is still essentially a V11 strict block/reliability teacher
  wrapped with V12 protocol metadata.
- The current teacher gate rejects all nonzero corrections by worst-block,
  MAE, counterfactual, and sign-agreement criteria.
- Therefore, training a generator after this result would train against an
  unvalidated teacher and would violate the evidence-first protocol.

Required V12b direction:

1. Keep the V12 shared-anchor aligner, but treat it as an upstream diagnostic,
   not as proof of adaptation.
2. Replace the wrapper-style V12 teacher with a genuinely outcome-aligned,
   out-of-fold teacher that is trained and selected on chronological residual
   transfer, not just final shrinkage of a V11-style solution.
3. Add per-horizon and per-block teacher diagnostics before any generator
   training.
4. Preserve zero-drift identity and fail-closed behavior.
5. Require public-3 teacher-only gate success before saving any generator
   artifact.

Current expert conclusion:

```text
V12 is a stronger representation-audit repair than V11, but it is not a
validated Stage III performance mechanism. The next implementation target is
V12b teacher redesign, not paper-claim upgrade and not generator training.
```

---

## 2026-06-01 Expert Update: V12b/V12c Teacher Redesign Result

V12b replaced the V12 wrapper teacher with a train-only chronological OOF
teacher and a shared signal contract. It fixed important protocol issues:

- train-only signal scale instead of per-batch abs-mean normalization;
- purged forward OOF folds;
- zero-drift identity;
- per-output alpha selection diagnostics;
- consistent teacher-only counterfactual path.

However, the V12b public-3 matrix remained negative:

- Manifest:
  `logs/s3v12bm/_manifests/v12btm_20260601_165710_e5_ie5_b480_ew480`.
- Completed runs: `9/9`.
- Teacher gates: `0/9`.
- Rolling windows: `0/45`.
- Main matrix alpha source: all `fallback_oof_gate`.
- Generator artifacts: `0`.

The V12b loose-alpha probe on Panama seed42 was especially diagnostic:

- Path:
  `logs/s3v12bprobe/probe_v12b_loose_panama_s42_20260601_01/panama/seed_42`.
- It forced nonzero exploratory alpha in `12/12` outputs.
- OOF MSE/MAE gains were tiny positive:
  `0.000924` / `0.000424`.
- Rolling validation still failed `0/5`.
- Worst sign agreement was only `0.126736`.
- `sign_flipped` was the best counterfactual in every validation window.

This ruled out the simple explanation that the strict gate alone caused the
failure. The residual mapping itself was not directionally stable.

V12c was then added as a diagnostic teacher, not a performance module. It
separates alpha selection into:

1. `ungated`: best OOF alpha, regardless of stability;
2. `relaxed`: enough chronological blocks pass;
3. `strict`: V12b-style worst-block safety conditions.

New opt-in files include:

- `models/stage3_residual_teacher_v12c.py`
- `train/stage3_v12c_signal_contract.py`
- `train/run_stage3_v12c_sensitivity_audit.py`
- `train/run_stage3_v12c_teacher_audit_matrix.py`
- `configs/stage3_v12c_multi_policy_teacher_probe.yaml`
- `configs/stage3_v12c_multi_policy_teacher_probe_nonnegative.yaml`
- `tests/test_stage3_v12c_teacher_policies.py`

Verification:

- V12c policy tests: `4 passed`.
- V12c + V12b targeted tests: `10 passed`.
- V12c modules/runners: `py_compile` passed.

V12c Panama seed42 8-combination probe:

- Root: `logs/s3v12c_probe8/_manifests`.
- Grid: `level_vector/delta_vector` x polarity `+/-1` x signed/nonnegative.
- All 8 manifests completed.
- All 8 teacher gates failed.
- `delta_vector` was the best signal family:
  worst rolling MSE gain `0.000909`, worst MAE gain `0.000707`,
  worst sign `0.466146`, test MSE/MAE gains `0.001881` / `0.000834`.

V12c best-delta public-3 matrix:

- Manifest:
  `logs/s3v12cm/_manifests/v12ctm_delta_nonneg_20260601_190259_e5_ie5_b480_ew480`.
- Completed runs: `9/9`.
- Teacher gates: `0/9`.
- Rolling windows: `0/45`.
- Generator artifacts: `0`.

The closest positive case was Panama seed43:

- worst MSE gain `0.005604` passes the `0.005` threshold;
- worst MAE gain `0.003026` passes the `0.002` threshold;
- worst counterfactual margin `0.002918` passes the `0.002` threshold;
- worst sign agreement `0.440451` fails the `0.55` threshold.

Expert conclusion:

```text
V12c improves the diagnostic resolution of Stage III: it can expose weak
drift-residual signals that V12b's all-zero strict gate hid. But V12c still
does not validate Stage III as a forecasting-improvement mechanism. The
current residual teacher lacks stable sign-consistent generalization across
datasets and seeds.
```

Therefore:

- Do not train a Stage III generator from V12b/V12c teacher evidence.
- Do not write that Stage III has learned concept drift.
- Do not claim significant online forecasting improvement from V12/V12b/V12c.
- The next model work should compare drift-only vs residual-history-only vs
  drift+history teachers under the same public-3 teacher-only gate.

Detailed record:

- `docs/audit/STAGE3_V12C_MULTI_POLICY_TEACHER_RECORD.md`

---

## 2026-06-01 Expert Update: V11 Block-Calibrated Teacher Matrix

V11 tested the next hypothesis after the V10 public-3 failure:

```text
Maybe V10 fails because aggregate shrinkage hides harmful chronological blocks;
a worst-block calibrated teacher should either find a stable nonzero correction
or safely abstain before generator training.
```

The V11 path was implemented as opt-in files only:

- `models/stage3_alignment_v11.py` adds
  `TrainSetCrossFittedResidualTransferPLSAlignerV11`.
- `models/stage3_residual_teacher_v11.py` adds
  `TrainOnlyRobustBlockResidualTeacherV11`.
- `train/run_stage3_v11_sensitivity_audit.py` uses explicit V9/V10/V11
  teacher dispatch and a hard `--teacher-only` guard.
- `train/run_stage3_v11_teacher_audit_matrix.py` runs public-3/seeds
  `42/43/44` without overwriting V10 artifacts.
- `configs/stage3_v11_block_calibrated_teacher_gate.yaml` keeps the strict
  teacher-only claim gate and uses `signal_mode: delta_vector`.

Verification passed:

- V11/V10/V9 targeted tests: `18 passed`.
- Project-local tests under `tests/`: `145 passed`.
- `git diff --check`: passed.
- Root-level `pytest -q` is not a valid project-local signal in the current
  workspace because it also collects unrelated external trees under
  `archive/external_sources/CLI-Anything/`, LibreOffice `archive/external_sources/core/`, and a permission-denied temporary
  directory. The collection failures are missing external packages such as
  `cli_anything.*`, `uno`, `officehelper`, and `uitest`, not V11 test failures.

The public-3 matrix did not validate V11:

- Manifest: `logs/s3v11m/_manifests/v11tm_20260601_100708_e5_ie5_b480_ew480`.
- Compact summary: `docs/audit/STAGE3_V11_TEACHER_MATRIX_SUMMARY.csv`.
- Teacher gates: `0/9`.
- Rolling validation windows: `0/45`.
- `teacher_alpha_source=fallback_block_gate` in all rows.
- `teacher_shrinkage_alpha=0.0` in all rows.
- No `generator*.pt` artifact exists under `ckpt/s3v11m`.

The important mechanism finding is not simply that V11 failed. It is that all
alignment gates passed while all teacher gates failed. That means the internal
alignment diagnostic is not equivalent to a usable residual correction. The
most likely current mismatch is:

1. The V11 aligner scores post-canonical residual-transfer delta.
2. The V11 teacher is fit on the actual generator input produced by
   `signal_mode: delta_vector`.
3. `delta_vector` applies temporal differencing and normalization before the
   teacher sees the signal.

Therefore high `alignment_teacher_residual_r2` can coexist with `alpha=0` in
the served teacher. The follow-up no-generator diagnostics compared
`delta_vector` against `level_vector` and also tested horizon-wise alpha.

Follow-up ablation results:

- `level_vector`: effective public-3 result remained `0/9` teacher gates,
  `0/45` rolling windows, and `0/9` nonzero alpha. This rules out temporal
  differencing in `delta_vector` as the sole cause.
- `block_horizon`: nonzero alpha appeared in `4/9` runs, but the variant still
  failed `0/9` teacher gates and `0/45` rolling windows. Mean worst MSE/MAE
  gains were `-0.000762`/`-0.000592`.

The ablation record is:

- `docs/audit/STAGE3_V11_SIGNAL_AND_HORIZON_ABLATION_RECORD.md`

Expert conclusion:

V11 is a useful fail-closed audit repair, not a performance repair. It should
not be described as a validated concept-drift learner. Generator training must
remain blocked until a nonzero teacher first passes the prospective rolling
teacher-only gate.

Detailed record:

- `docs/audit/STAGE3_V11_TEACHER_MATRIX_RECORD.md`

---

## 2026-05-27 Expert Update: V7 Residualized RRCCA

After V5/V6 failed the strict served-policy claim gate, a stronger V7 alignment
path was implemented and tested:

- `models/stage3_alignment_v7.py` adds
  `TrainSetChronologicalResidualizedReducedRankCCAAligner`.
- `train/run_stage3_v2_sensitivity_audit.py` supports it only when
  `canonicalizer_method: residualized_rrcca_v7`.
- `train/stage3_v2_artifacts.py` can reload the V7 canonicalizer by
  `class_name`.
- `configs/stage3_v7_residualized_rrcca_served_policy_gate.yaml` keeps the
  same strict served-policy claim gate as V5/V6.
- `tests/test_stage3_alignment_v7.py` verifies residualization, active-rank
  truncation, fail-closed behavior, and serialization.

V7 is methodologically stronger than V6 for the specific `phi_v - phi_i`
alignment concern because it does not force the entire `phi_v` distribution
into `phi_i` geometry. Instead, it:

1. computes `D_t = phi_v(t) - phi_i(t)`;
2. removes train-only invariant leakage by fitting `D_t ~ [phi_i(t), 1]`;
3. fits CCA between residual drift transitions and forecast-residual
   transitions on the inner chronological train-fit slice;
4. selects active dimensions only on an inner chronological train-calibration
   slice;
5. zeros inactive dimensions during validation/test.

However, the Panama seed42 `g5_b480` V7 canary still failed:

- V7 internal alignment gate passed with active rank `8`.
- residualizer R2 mean was `0.931512`, indicating that most raw `delta_phi_v`
  variance was explainable from `phi_i`.
- served-policy rolling validation was `0/5`.
- best candidate epoch `4` had worst-window MSE gain `0.000490`, MAE gain
  `0.000534`, and real-vs-counterfactual margin `0.000267`.
- no `generator_served_policy.pt` was saved.

Expert conclusion:

V7 strengthens the diagnosis but not the performance claim. It supports the
statement that the raw drift representation contains invariant leakage and that
alignment must be train-only and rank-gated. It does not support claiming that
Stage III learns useful concept-drift-conditioned parameter adaptation.

Detailed record:

- `docs/audit/STAGE3_V7_ALIGNMENT_RECORD.md`

---

## 2026-05-28 Expert Update: V8 HyperAdapter Negative Evidence

V8 tested a specific hypothesis left open by V7:

```text
Maybe the residualized RRCCA drift is useful, but the generator is too weak
because it only updates 2-D weights and cannot correct level-like output bias.
```

The implementation added an opt-in generator:

- `models/adaptation_v2.py`
  - `_HyperBiasParamAdapter`
  - `LayerAwareHyperAdapterGenerator`
- `configs/stage3_v8_hyperadapter_residualized_rrcca_served_policy_gate.yaml`
  - `stage3_v2.generator.type: layer_aware_hyperadapter_v8`
  - `target_names: [weight, bias]`
- `train/stage3_generator_training.py`
  - V8 checkpoint load support
  - `target_names` propagation
- `train/online_loop_stage3_v2.py`
  - serving-time `target_names` support

Verification:

```text
C:\Users\admin\anaconda3\python.exe -m pytest tests -q
127 passed in 5.59s

C:\Users\admin\anaconda3\python.exe -m py_compile models\adaptation_v2.py train\stage3_generator_training.py train\online_loop_stage3_v2.py train\run_stage3_v2_sensitivity_audit.py train\stage3_v2_artifacts.py
pass

git diff --check
pass
```

The same-budget Panama seed42 V8 canary was negative:

- logs: `logs/s3v8diag/v8ha_s42_g5b480_01/panama/seed_42`
- checkpoints: `ckpt/s3v8diag/panama/v8ha_s42_g5b480_01/seed_42`
- selected candidate: epoch `3`
- served-policy rolling validation: `0/5`
- worst-window real MSE gain: `-0.008390`
- worst-window real MAE gain: `-0.005176`
- worst real-vs-best-counterfactual margin: `-0.008390`
- final `generator_served_policy.pt`: not saved

This is worse than V7:

| Run | Best epoch | Rolling pass | Worst MSE gain | Worst MAE gain | Worst margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| V7 residualized RRCCA | 4 | `0/5` | `+0.000490` | `+0.000534` | `+0.000267` |
| V8 hyperadapter | 3 | `0/5` | `-0.008390` | `-0.005176` | `-0.008390` |

Expert conclusion:

V8 rejects the simple "more adapter capacity fixes Stage III" hypothesis. The
generator can now produce larger output shifts, but the learned update direction
is not reliable. The bottleneck is still the absence of a residual-specific
teacher/meta-objective that tells the generator what update would actually
correct the current forecasting residual.

Required next repair:

1. Build a train-only residual teacher update for `output_head` or a residual
   correction head.
2. Add generator losses that match teacher update direction and residual
   correction, not just blended MSE.
3. Improve `phi_v/phi_i` alignment through residual/update-supervised alignment
   or contrastive residual-transition calibration.
4. Keep V8 as negative diagnostic evidence, not manuscript performance evidence.

Detailed record:

- `docs/audit/STAGE3_V8_HYPERADAPTER_RECORD.md`

---

## 2026-05-28 Expert Update: V9 Residual-Teacher Negative Evidence

V9 tested the next hypothesis after V8:

```text
Maybe the generator fails because it has no residual-specific teacher telling it
which drift-conditioned correction direction would reduce the forecast error.
```

The implementation added an opt-in residual-teacher branch:

- `models/stage3_residual_teacher_v9.py`
  - `TrainOnlyResidualTeacherV9`
  - train-only `fit()` and `update()` checks
  - bias-free drift feature map so zero drift yields zero correction
- `models/stage3_alignment_v9.py`
  - `TrainSetTeacherResidualizedReducedRankCCAAligner`
  - teacher-residual-aware calibration on top of residualized RRCCA
  - low-rank invariant residualizer after the first smoke exposed rank collapse
- `models/adaptation_v9.py`
  - `ResidualBasisHyperGeneratorV9`
  - V8-style weight+bias adapters plus a residual correction head
- `configs/stage3_v9_residual_teacher_rrcca_served_policy_gate.yaml`
  - strict V7/V8 served-policy gate preserved
  - outputs isolated under `logs/s3v9*` and `ckpt/s3v9*`

Verification:

```text
C:\Users\admin\anaconda3\python.exe -m pytest tests\test_stage3_v9_teacher_protocol.py -q
7 passed

C:\Users\admin\anaconda3\python.exe -m pytest tests -q
134 passed
```

The first V9 smoke found an important failure mode: a high-dimensional residualizer
could overfit the small train slice and collapse drift to active rank `0`. This
was fixed by replacing it with a train-only low-rank PCA invariant residualizer.
Later smoke runs confirmed that zero drift stayed near identity and the teacher
path could reduce train residuals, but the claim gate still did not pass.

The same-budget Panama seed42 V9 canary was negative:

- logs: `logs/s3v9diag/v9rt_s42_g5b480_01/panama/seed_42`
- checkpoints: `ckpt/s3v9diag/panama/v9rt_s42_g5b480_01/seed_42`
- selected candidate: epoch `0`
- served-policy rolling validation: `0/5`
- worst-window real MSE gain: `-0.007082`
- worst-window real MAE gain: `-0.005109`
- worst real-vs-best-counterfactual margin: `-0.007082`
- max real reject rate: `0.0`
- final `generator_served_policy.pt`: not saved

The residual teacher was correctly train-only and reduced train residual MSE
from `0.1608959436` to `0.1450631768`, but its train R2 was only `0.092615`.
This is a weak signal: the current drift features explain only a small part of
the residual even on the training stream.

The window-level evidence is more damaging for the manuscript claim. For the
selected epoch `0`, all five validation windows had negative real-drift MSE and
MAE gains. In every window, zero drift was the best counterfactual. Therefore
V9 repaired several structural and protocol problems, but the real-drift
correction direction was harmful on the held-out rolling stream.

Comparison:

| Run | Best epoch | Rolling pass | Worst MSE gain | Worst MAE gain | Worst margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| V7 residualized RRCCA | 4 | `0/5` | `+0.000490` | `+0.000534` | `+0.000267` |
| V8 hyperadapter | 3 | `0/5` | `-0.008390` | `-0.005176` | `-0.008390` |
| V9 residual teacher | 0 | `0/5` | `-0.007082` | `-0.005109` | `-0.007082` |

Expert conclusion:

V9 is a meaningful audit/engineering repair, not a successful performance
repair. It confirms that Stage III needs train-only residual supervision and
zero-drift identity, but it also shows that those additions are insufficient.
The remaining bottleneck is a train-to-validation objective mismatch: generator
training loss decreases on train batches, while rolling validation gets worse.

Required next repair:

1. Run a teacher-only validation upper-bound before training another generator.
2. Export residual sign agreement between teacher correction and true validation
   residuals.
3. Replace train mini-batch blended-MSE optimization with a rolling-window or
   meta-objective that directly matches the served-policy validation criterion.
4. Keep V9 as negative diagnostic evidence and do not upgrade manuscript Stage
   III claims.

Detailed record:

- `docs/audit/STAGE3_V9_RESIDUAL_TEACHER_RECORD.md`

Follow-up teacher-only diagnostic:

- code: `train/stage3_v9_teacher_diagnostics.py`
- record: `docs/audit/STAGE3_V9_TEACHER_ONLY_DIAGNOSTIC_RECORD.md`
- same-budget logs: `logs/s3v9teacher_diag/v9teacher_s42_g5b480_01/panama/seed_42`

This diagnostic bypasses the generator and evaluates:

```text
y_teacher = y_base + teacher(z)
```

The result is negative:

- teacher-only rolling validation: `0/5`
- worst teacher-only real MSE gain: `-0.197784`
- worst teacher-only real MAE gain: `-0.118167`
- worst real-vs-best-counterfactual margin: `-0.197784`
- worst sign agreement: `0.515625`
- best counterfactual in every validation window: `zero`
- test MSE/MAE gain: `-0.097074` / `-0.052800`

This localizes the V9 failure upstream of generator capacity. The current
train-only residual teacher does not transfer to rolling validation, so training
a larger generator against this teacher direction is not scientifically
justified.

Additional code correction:

The V9 aligner's `teacher_residual_r2` field was updated to report an in-sample
ridge predictability R2 from residualized drift to residual, rather than a
misleading residual-centering quantity. This does not change serving behavior,
but makes alignment diagnostics less ambiguous.

---

## 2026-05-29 Expert Update: V10 Public-3 Teacher Matrix

V10 implemented the strongest Stage III representation repair currently present
in this repository:

1. a zero-preserving cross-fitted leakage-subspace projection for `phi_v - phi_i`;
2. train-only residual teacher fitting;
3. inner-train shrinkage calibration;
4. an explicit teacher gate that blocks generator training when the teacher does
   not transfer.

This directly addresses the user's concern that `phi_v - phi_i` was not
strictly aligned. V10 no longer subtracts an additive `f(phi_i)` term at serve
time. It estimates directions in drift space that are predictable from
invariant features and removes those directions with a linear projection:

```text
D_clean = D @ (I - B B^T)
```

This is important because it preserves the physical zero:

```text
D = 0  =>  D_clean = 0  =>  teacher correction = 0
```

However, the expanded matrix shows that better alignment is not sufficient.

Run scope:

- datasets: `panama`, `tetouan`, `spain_energy_weather`;
- seeds: `42`, `43`, `44`;
- budget: `5 epochs`, `5 invariant epochs`, `480` teacher train batches,
  `480` evaluation batches, `5` rolling validation windows;
- mode: `--teacher-only`, so no generator is trained even if a teacher passes.

Evidence:

- `docs/audit/STAGE3_V10_TEACHER_MATRIX_RECORD.md`
- `docs/audit/STAGE3_V10_TEACHER_MATRIX_SUMMARY.csv`
- `logs/s3v10teacher_matrix/_manifests/v10teacher_matrix_20260529_112849_e5_ie5_b480_ew480`
- `logs/s3v10m/_manifests/v10m_20260529_121906_e5_ie5_b480_ew480`

Matrix result:

| Dataset | Teacher Gate | Rolling Windows Passed | Mean Worst MSE Gain | Mean Worst MAE Gain |
| --- | ---: | ---: | ---: | ---: |
| Panama | `0/3` | `7/15` | `+0.007158` | `+0.000809` |
| Tetouan | `0/3` | `1/15` | `-0.144104` | `-0.096893` |
| Spain | `0/3` | `0/15` | `-0.025551` | `-0.018493` |
| All | `0/9` | `8/45` | `-0.054166` | `-0.038192` |

Expert conclusion:

V10 confirms that cross-fitted zero-preserving alignment is necessary but not
sufficient. It improves the Panama seed42 mechanism signal, but it does not
generalize across seeds or datasets. All nine effective runs fail the
pre-declared teacher gate, so V10 still cannot support a claim of validated
drift-conditioned online adaptation.

Required next repair:

1. Do not train the V10 generator as a manuscript candidate.
2. Build a V11 teacher before any generator work:
   - worst-block chronological calibration rather than aggregate shrinkage;
   - horizon-wise nonnegative MAE-aware shrinkage constrained by sign agreement;
   - selective correction gates based on train-only calibration reliability;
   - exact zero-drift identity at every layer.
3. Move representation learning from post-hoc cleanup toward residual-transfer
   supervision: the representation should be optimized or calibrated for
   out-of-fold residual correction, not only invariant leakage removal.
4. Only if a teacher-only run passes `5/5` rolling windows should generator
   canaries resume.

---

## 1. Executive Diagnosis

The current Stage III mechanism is not yet a defensible drift-conditioned adaptation module.

The core issue is not merely that the drift definition uses `phi_v(t) - phi_i(t)` instead of a temporal delta. The deeper issue is that the current implementation does not establish a reliable causal chain:

```text
meaningful drift representation
  -> trained generator reads drift
  -> zero drift gives zero/no adaptation
  -> drift direction changes generated update
  -> generated update improves prediction
```

The code currently implements a weaker chain:

```text
possibly unaligned embedding difference
  -> scalar compression in original Stage III
  -> random untrained generator
  -> nonzero update even for zero drift
  -> small transient perturbation
  -> blend with base prediction
```

This means Stage III may still produce small numerical gains, but current evidence does not justify claiming that the gains are caused by learned concept-drift conditioning.

---

## 2. Evidence From Code Re-Audit

### 2.1 `AdaptationGenerator` Is Instantiated Online

In `train/online_loop.py`, the generator is created directly inside `run_online_adaptation`:

```python
gen = AdaptationGenerator(
    drift_dim=online_cfg["stage2"]["lrp_bottleneck"],
    rank=online_cfg["stage2"]["lrp_bottleneck"],
    max_dim=512,
).to(device)
```

Location:

- `train/online_loop.py:97-102`

Repository search found generator creation/use here:

- `train/online_loop.py:98`
- `train/online_loop_stage3_delta.py:98`
- `train/online_loop_stage3_ablation.py:104`
- `train/online_loop_stage3_ablation.py:296`
- `train/analyze_tsne_components.py:190`
- `models/adaptation.py:76`

But search found no optimizer over `gen.parameters()` and no training loop that updates `w1`, `w2_alpha`, or `w2_beta`.

Relevant search result:

```text
rg "gen\\.parameters|generator_optimizer|optimizer.*Adaptation|stage3.*optimizer"
```

No training optimizer for `AdaptationGenerator` was found.

### Expert conclusion

The current generator is randomly initialized at online evaluation time and used directly. This is not a trained hypernetwork.

---

## 3. Is The Current `AdaptationGenerator` Architecture Good?

### 3.1 Current Architecture

Defined in `models/adaptation.py:76-94`:

```python
class AdaptationGenerator(nn.Module):
    def __init__(self, drift_dim: int, rank: int, max_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(drift_dim, rank)
        self.w2_alpha = nn.Linear(rank, max_dim)
        self.w2_beta = nn.Linear(rank, max_dim)

    def forward(self, drift: torch.Tensor, din: int, dout: int):
        h = torch.sigmoid(self.w1(drift))
        alpha_full = torch.sigmoid(self.w2_alpha(h)) + 1.0
        beta_full = torch.sigmoid(self.w2_beta(h)) + 1.0
        alpha = alpha_full[:din]
        beta = beta_full[:dout]
        return alpha, beta
```

Parameter deltas are built in `models/adaptation.py:97-116`:

```python
scale = alpha.unsqueeze(1) * beta.unsqueeze(0)
delta = (scale - 1.0) * p
```

### 3.2 What This Architecture Actually Does

For each 2D parameter matrix `p`, the generator creates row and column scaling vectors:

```text
alpha: one value per row
beta: one value per column
scale[row, col] = alpha[row] * beta[col]
delta = (scale - 1) * p
```

So this is not a general parameter update. It is a multiplicative low-rank-style modulation of the existing weight matrix.

### 3.3 Strengths

This design has some reasonable properties:

1. It is lightweight.
2. It avoids optimizing the full model online.
3. It can produce structured matrix-wise perturbations.
4. It can be fast if trained properly.
5. It resembles a hypernetwork / low-rank adaptation idea at a high level.

### 3.4 Critical Weaknesses

#### Weakness 1: No Identity Initialization

A good transient adaptation module should satisfy:

```text
if drift = 0, then delta ≈ 0
```

Current code does not satisfy this.

Because `nn.Linear` has bias:

```text
Linear(0) = bias
```

And because:

```python
alpha = sigmoid(...) + 1
beta = sigmoid(...) + 1
```

`alpha` and `beta` are usually between 1 and 2, so `scale - 1` is not zero.

#### Evidence

Diagnostic script result:

```text
zero delta_norm 161.546677 mean_abs_delta 0.99794 rel_diff_vs_zero 0.0
small_scalar delta_norm 161.514771 mean_abs_delta 0.997817 rel_diff_vs_zero 0.001894
large_scalar delta_norm 161.205505 mean_abs_delta 0.996501 rel_diff_vs_zero 0.022585
sign_flip_small delta_norm 161.579269 mean_abs_delta 0.998064 rel_diff_vs_zero 0.001898
random_vector delta_norm 161.568970 mean_abs_delta 0.997387 rel_diff_vs_zero 0.011249
```

Interpretation:

- Zero drift already generates a large delta.
- A small scalar drift changes the delta by only about `0.19%` relative to zero.
- A random vector drift changes the delta by only about `1.12%` relative to zero.

This is exactly why different drift definitions can produce almost identical results.

#### Weakness 2: Output Is Biased Toward Magnitude Increase

Since:

```text
alpha in (1, 2)
beta in (1, 2)
scale in (1, 4)
delta = (scale - 1) * p
```

The delta has the same sign as the original parameter `p`. For positive weights, it tends to increase them; for negative weights, it tends to make them more negative.

This is not a flexible learned update. It cannot naturally express:

- shrink this weight;
- reverse this feature direction;
- add a new residual direction independent of current weight sign;
- produce zero update at zero drift.

#### Weakness 3: Same Generator Serves All Layers

The same `gen` is used for all 2D parameters. It does not receive:

- layer name;
- layer type;
- whether the layer is expert, output head, VAE, or attention;
- input/output semantic role.

Two layers with similar shapes receive the same modulation pattern, except for their underlying weight values.

This is weak for a paper claim that Stage III performs meaningful drift-specific parameter adaptation.

#### Weakness 4: Layer Filtering Happens After Delta Construction

In Stage III:

```python
delta = build_param_delta(dict(model.named_parameters()), drift_vec, gen)
delta = _filter_named_tensors(delta, stage3_prefixes)
```

This constructs deltas for all 2D parameters first and filters later.

Risk:

- wasted compute;
- if a non-target 2D parameter dimension exceeds `max_dim=512`, `build_param_delta` can fail before filtering;
- the generator design is globally coupled to all model parameter shapes.

#### Weakness 5: Original Stage III Compresses Vector Drift To Scalar

Original Stage III:

```python
drift_scalar = d_v.mean()
drift_vec = (drift_scalar / (d_v.abs().mean() + 1e-6)).repeat(
    online_cfg["stage2"]["lrp_bottleneck"]
)
```

Location:

- `train/online_loop.py:200-203`

This discards directional information in the 64-dimensional invariant/variant embedding difference.

### Expert conclusion

The current architecture is acceptable as a rough prototype, but it is not strong enough for a top-journal claim about learned concept-drift-conditioned adaptation.

It needs at least:

1. a trained generator;
2. zero-drift identity behavior;
3. vector drift conditioning;
4. deterministic evaluation;
5. layer-aware or LoRA-style parameterization;
6. sensitivity diagnostics proving the drift input matters.

---

## 4. `phi_v - phi_i`: Why The Representation Space Is Not Yet Rigorous

### 4.1 Current Training Path

In `train/run_experiment.py`, two encoders are created:

```python
encoder_i = FlattenMLPEncoder(in_dim, repr_dim).to(device)
encoder_v = FlattenMLPEncoder(in_dim, repr_dim).to(device)
```

Location:

- `train/run_experiment.py:63-64`

`encoder_i` is trained inside MTIL:

```python
mtil = MTIL(encoder_i, reg).to(device)
```

Location:

- `train/run_experiment.py:68`

The invariant training objective is implemented in:

- `train/offline_invariant.py:12-56`
- `models/invariant_learning.py:183-212`

The variant encoder is trained separately:

```python
with torch.no_grad():
    _, y_inv = mtil(x)
    residual = y - y_inv
phi_v = encoder_v(x)
y_var = reg_v(phi_v)
loss = torch.mean((y_var - residual) ** 2)
```

Location:

- `train/run_experiment.py:93-114`

### 4.2 What Is Missing

The code does not enforce:

1. `phi_i` and `phi_v` live in the same coordinate system.
2. Dimension `k` of `phi_i` has the same semantic meaning as dimension `k` of `phi_v`.
3. `phi_v - phi_i` is a calibrated difference.
4. `phi_v` captures only variant factors.
5. `phi_i` captures only invariant factors.
6. `phi_v` is orthogonal or complementary to `phi_i`.
7. `phi_v - phi_i` correlates with actual distribution shift.

### 4.3 Why Direct Subtraction Is Risky

If two encoders are trained with different objectives, direct subtraction is like subtracting coordinates from two different maps.

Example:

```text
phi_i dimension 3 might encode "daily cycle"
phi_v dimension 3 might encode "residual temperature response"
```

Then:

```text
phi_v[3] - phi_i[3]
```

does not necessarily mean anything stable.

### 4.4 How To Fix Representation Alignment

There are several levels of repair.

#### Minimal repair: train-only standardization and projection

Use training data only to estimate:

```text
mean_i, std_i, mean_v, std_v
```

Then compute:

```text
phi_i_norm = (phi_i - mean_i) / std_i
phi_v_norm = (phi_v - mean_v) / std_v
drift = phi_v_norm - phi_i_norm
```

This improves scale comparability, but not semantic alignment.

#### Moderate repair: shared trunk with two heads

Instead of two independent MLPs:

```text
encoder_i = MLP_i(x)
encoder_v = MLP_v(x)
```

use:

```text
h = shared_encoder(x)
phi_i = invariant_head(h)
phi_v = variant_head(h)
```

This gives both branches a shared coordinate origin.

#### Strong repair: residual representation

Make variant representation explicitly residual:

```text
phi_i = invariant_head(h)
delta_phi_v = variant_residual_head(h)
phi_v = phi_i + delta_phi_v
drift = delta_phi_v
```

This makes the drift representation direct:

```text
drift = phi_v - phi_i = delta_phi_v
```

Now subtraction is meaningful by construction.

#### Add orthogonality and reconstruction constraints

Add losses:

```text
prediction = reg_i(phi_i) + reg_v(delta_phi_v)
L_pred = MSE(prediction, y)
L_orth = mean cosine similarity squared between phi_i and delta_phi_v
L_sparse = mean absolute value of delta_phi_v
L_smooth = temporal smoothness penalty on phi_i
L_variant_change = temporal sensitivity penalty/reward on delta_phi_v
```

Recommended objective:

```text
L = L_pred
  + lambda_inv * invariant_risk
  + lambda_orth * L_orth
  + lambda_sparse * L_sparse
  + lambda_smooth_i * smooth(phi_i)
  + lambda_temporal_v * controlled temporal variation(delta_phi_v)
```

### Expert conclusion

For a defensible paper mechanism, the best repair is not merely to change:

```text
phi_v - phi_i
```

to:

```text
[phi_v(t)-phi_i(t)] - [phi_v(t-1)-phi_i(t-1)]
```

The better repair is:

```text
build phi_v as phi_i + delta_phi_v, and use delta_phi_v or temporal difference of delta_phi_v
```

This gives `drift` a clearer mathematical meaning.

---

## 5. VAE Randomness During `eval()`: Correct Or Incorrect?

### 5.1 Current Code

In `models/factor_mechanism.py`:

```python
if self.use_vae:
    mu_z, log_var_z = self.vae_encoder(h)
    z = self._reparameterize(mu_z, log_var_z)
```

Location:

- `models/factor_mechanism.py:93-96`

Reparameterization:

```python
std = (0.5 * log_var).exp()
eps = torch.randn_like(std)
return mu + eps * std
```

Location:

- `models/factor_mechanism.py:133-137`

### 5.2 Is This Ever Correct?

Yes, stochastic VAE sampling is correct during training.

During training, the reparameterization trick is necessary because the VAE learns a distribution:

```text
q(z|x) = Normal(mu_z, sigma_z)
```

Sampling helps train the latent distribution.

### 5.3 Is It Correct For This Project's Evaluation?

For the current paper experiments: no, it is not appropriate unless explicitly treated as Monte Carlo evaluation.

Reason:

`model.eval()` in PyTorch disables dropout and batchnorm updates, but it does not automatically disable custom random sampling. Because `_reparameterize` always samples, evaluation remains stochastic.

Diagnostic result:

```text
eval_use_vae_true_max_y_diff 0.061709
eval_use_vae_true_max_z_diff 4.246341
eval_use_vae_false_max_y_diff 0.0
eval_use_vae_false_max_z_diff 0.0
```

This means the same model and same input can produce different predictions in eval mode.

### 5.4 Why It Matters For Stage III

Stage III compares:

```text
base prediction
candidate prediction under transient delta
```

But if both predictions contain random VAE noise, then:

```text
candidate - base
```

is partly:

```text
parameter update effect
```

and partly:

```text
random latent sampling effect
```

This contaminates:

- `stage3_pred_shift`;
- online MSE/MAE;
- ablation comparisons;
- claim that drift causes prediction changes.

### Correct options

#### Recommended option: deterministic eval

During training:

```python
z = mu + eps * std
```

During evaluation:

```python
z = mu
```

This is the standard practical choice for deterministic forecasting.

#### Alternative option: Monte Carlo evaluation

Run multiple samples:

```text
z_1, z_2, ..., z_K
```

Average predictions and report uncertainty.

But this requires:

- fixed `K`;
- repeated sampling;
- reported confidence intervals;
- much higher compute;
- explicit paper explanation.

### Expert conclusion

For this project, deterministic eval using `mu_z` is the right default. Stochastic eval should only be enabled for a separate uncertainty experiment.

---

## 6. Why The Four-Way Stage III Ablation Produced Identical Results

Expanded experiments showed that:

```text
level_scalar / delta_scalar / level_vector / delta_vector
```

produce essentially identical MSE/MAE.

Formal-config-aligned run:

```text
s3abl_pub3_s42_e5_inv1_v1 rows 12
panama max_mse_delta_vs_level_scalar 0.0 max_mae_delta_vs_level_scalar ~1e-6
spain_energy_weather max_mse_delta_vs_level_scalar 0.0 max_mae_delta_vs_level_scalar 0.0
tetouan max_mse_delta_vs_level_scalar 0.0 max_mae_delta_vs_level_scalar ~1e-6
```

Stronger invariant run:

```text
s3abl_pub3_s42_e5_v2 rows 12
all datasets max_mse_delta_vs_level_scalar 0.0
all datasets max_mae_delta_vs_level_scalar 0.0
```

Most likely explanation:

```text
drift signal differences are much smaller than the generator's drift-independent baseline update
```

In other words, the current generator acts mostly like:

```text
constant perturbation generator
```

not like:

```text
drift-conditioned adaptation generator
```

---

## 7. Recommended Repair Architecture

The repair should be staged. Do not jump directly to paper-scale experiments.

### Phase 0: Freeze claims

Do not claim:

```text
Stage III learns concept drift
```

until sensitivity tests prove it.

### Phase 1: Deterministic evaluation

Modify model forward behavior so that:

```text
training mode: sample z
eval mode: use mu_z
```

Add a config option:

```yaml
model:
  vae_eval_mode: mean
```

Possible values:

| Value | Meaning |
| --- | --- |
| `mean` | deterministic, use `mu_z` in eval |
| `sample` | stochastic, current behavior |
| `mc_mean` | average multiple latent samples |

Default should be `mean`.

### Phase 2: Generator identity behavior

Redesign generator so:

```text
drift = 0 -> delta = 0
```

Preferred parameterization:

```python
raw_alpha, raw_beta = generator(drift)
alpha = max_scale * tanh(raw_alpha)
beta = max_scale * tanh(raw_beta)
delta = scale * outer(alpha, beta)
```

or:

```python
delta = learned_lora_A(drift) @ learned_lora_B(drift)
```

Do not add `+1.0` in a way that makes zero drift nonzero.

### Phase 3: Vector drift

Stop using `d_v.mean()` as the main path.

Use one of:

```text
drift_vec = normalized(phi_v - phi_i)
```

or:

```text
drift_vec = temporal_delta(phi_v - phi_i)
```

But only after representation alignment improves.

### Phase 4: Representation alignment

Move from two independent encoders to:

```text
shared trunk + invariant head + variant residual head
```

Recommended:

```text
h = shared_encoder(x)
phi_i = invariant_head(h)
delta_phi_v = variant_head(h)
phi_v = phi_i + delta_phi_v
drift = delta_phi_v
```

Then temporal drift can be:

```text
drift_temporal = delta_phi_v(t) - delta_phi_v(t-1)
```

This matches your intuition and is mathematically cleaner.

### Phase 5: Train the generator

Train generator on training/validation stream episodes.

For each offline-trained model:

```text
freeze base model
freeze or partially freeze encoders
for each train stream batch:
  compute drift
  generator produces transient delta
  candidate prediction = model with transient delta
  loss = prediction loss + regularization
  update generator only
```

Suggested loss:

```text
L_gen = MSE(y_candidate, y)
      + lambda_delta * ||delta||_2^2
      + lambda_zero * ||G(0)||_2^2
      + lambda_smooth * ||delta_t - delta_{t-1}||_2^2
      + lambda_shift * max(0, candidate_shift - shift_budget)^2
```

### Phase 6: Sensitivity diagnostics

Run:

```text
real drift
zero drift
shuffled drift
sign-flipped drift
amplified drift
```

Expected healthy behavior:

```text
real drift better than zero/shuffled
sign-flipped changes prediction direction or hurts
amplified changes delta norm and candidate shift in controlled way
zero drift gives near-zero delta
```

### Phase 7: Paper-scale validation

Only after Phases 1-6 pass should paper tables be updated.

---

## 8. Decision Table

| Issue | Severity | Current status | Required fix before strong claim |
| --- | --- | --- | --- |
| Generator not trained | Critical | Confirmed | Add generator training phase |
| Zero drift nonzero update | Critical | Confirmed | Identity-preserving generator |
| Vector drift scalarized | High | Confirmed in original loop | Use vector/temporal vector drift |
| VAE eval stochastic | High | Confirmed | Deterministic eval or MC protocol |
| `phi_i/phi_v` unaligned | High | Confirmed by architecture | Shared trunk or residual variant representation |
| Lack of sensitivity audit | High | Confirmed | real/zero/shuffled/sign/amplified diagnostics |
| Stage III gains small and signal-insensitive | High | Confirmed by expanded runs | Re-run after repairs |

---

## 9. Recommended Claim After Repair

If repairs succeed, a defensible claim would be:

```text
We propose a drift-conditioned transient adaptation module in which a calibrated variant residual representation drives an identity-preserving low-rank generator. Sensitivity interventions show that the learned generator responds to the magnitude, direction, and temporal alignment of drift signals.
```

If repairs do not succeed, the safer claim is:

```text
We study transient online perturbation as a lightweight adaptation mechanism and find that the current implementation yields small online gains, but the drift-conditioning pathway requires further validation.
```

---

## 10. 2026-05-25 Protocol Re-Audit Addendum

二次核查后，需要把 Stage III 风险拆成两类：

1. mechanism risk：generator 和 drift representation 是否真的形成有效的 `drift -> delta -> better prediction` 链条。
2. protocol risk：实验数据流是否严格避免未来信息泄漏，validation/test 是否能支撑泛化结论。

当前 V2 已修复的机制和协议点包括：

- eval 阶段 VAE 默认使用 posterior mean；
- generator 有独立训练流程和 checkpoint；
- zero drift 在 V2 路径下保持 identity；
- drift 输入不再压缩成标量；
- raw split before windowing；
- online replay batch size = 1；
- validation gate 与 test report-only 分离；
- 新增 rolling validation gate；
- sensitivity row 区分 `streaming_causal` 与 `offline_diagnostic` intervention。

仍未解决的最高风险是：

```text
data/readers.py 在全量时间轴上执行 time_interpolate / ffill / bfill 后，V2 runner 才做 raw split。
```

因此目前不能声称完整 preprocessing pipeline 是 strictly causal 或 leakage-free。必须新增 split-safe imputation 后，才能把 validation/test 结果作为强证据。

当前论文措辞应继续降级：

```text
Stage III V2 is a repaired and audited drift-conditioned adaptation candidate, but current evidence does not yet validate reliable held-out drift-conditioned gains.
```

不能写：

```text
Stage III V2 effectively captures concept drift and improves online forecasting under strict causal evaluation.
```

---

## 11. 2026-05-25 V2.5 Evidence Update

V2.5 在 V2/V2.3 基础上继续修复了两个关键科研完整性问题：

1. **Generator checkpoint selection**：每个 generator epoch 保存候选 checkpoint，只在 rolling validation 上选择，不再默认使用最后一个 epoch。
2. **Train-only residual drift canonicalizer**：用训练集有序 stream 中的 `delta_phi_v[t] - delta_phi_v[t-1]` 与 forecast residual transition 拟合 canonical coordinates，validation/test 只加载冻结后的变换。

对应代码证据：

- `train/run_stage3_v2_sensitivity_audit.py`：保存 `generator_epoch_XXX.pt`，执行 rolling-validation checkpoint selection，并在 manifest 记录 selected epoch、candidate count、selection score 和 failure reasons。
- `models/aligned_representation_v2.py`：`TrainSetResidualDriftCanonicalizer` 用训练集 residual transition 拟合 canonical basis。
- `train/stage3_sensitivity_gate_v2.py`：`real_reject_rate` 已改为 shift/risk/missing 三类拒绝总和，避免大量 risk rejection 被漏算。
- `configs/stage3_v2_strong.yaml`：启用 `canonicalize_delta: true` 与 generator checkpoint selection。

较大预算 Panama 成对实验已经完成：

| seed | budget | test MSE gain | test MAE gain | rolling validation pass rate | test gate |
| ---: | --- | ---: | ---: | ---: | --- |
| 42 | b60 | -0.0938% | -0.3505% | 0.3333 | fail |
| 42 | b120 | +1.2656% | -0.4907% | 0.3333 | fail |
| 43 | b60 | -0.3250% | +0.1074% | 0.3333 | fail |
| 43 | b120 | +1.6361% | +1.1517% | 0.3333 | fail |

随后完成了更严格的 public-3 seed42 b240/full-eval 审计。该轮不限制 final evaluation sample count，只限制训练和 generator replay budget：

| dataset | budget | selected epoch | test MSE gain | test MAE gain | rolling pass rate | reject rate | test gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Panama | b240 | 0 | +0.0858% | +0.1118% | 0.0000 | 82.58% | fail |
| Tetouan | b240 | 5 | +0.0225% | +0.0180% | 0.0000 | 99.91% | fail |
| Spain Energy--Weather | b240 | 2 | +0.3882% | +0.3264% | 0.0000 | 32.46% | fail |
| Panama seed43 | b240 | 9 | +1.7843% | +0.9248% | 0.0000 | 58.04% | fail |
| Tetouan seed43 | b240 | 3 | -0.0181% | -0.0243% | 0.0000 | 99.04% | fail |
| Spain seed43 | b240 | 4 | -0.0075% | -0.0029% | 0.0000 | 99.87% | fail |
| Panama seed44 | b240 | 0 | +0.0005% | +0.0012% | 0.0000 | 99.98% | fail |
| Tetouan seed44 | b240 | 0 | +0.3384% | +0.2480% | 0.0000 | 83.10% | fail |
| Spain seed44 | b240 | 7 | +0.2603% | +0.1693% | 0.0000 | 91.46% | fail |

专家判断：

```text
V2.5 是有效的工程和审计修复，但不是有效的性能证明。
```

原因：

1. b120 只在 report-only test MSE 上产生局部改善；
2. rolling validation 在早期 Panama b60/b120 run 中都只有 1/3 通过，在 public-3 b240/full-eval 中全部为 0/5 通过；
3. seed42 b120 的 MAE 仍然退化；
4. seed43 b120 的 test gate 因 `reject_rate` 失败，说明部分收益来自过度拒绝；
5. 多个 run 中 counterfactual drift 接近或超过 real drift；
6. public-3 seed42 全部 test gate 失败，不能支撑跨数据集稳定性；
7. Panama seed43 虽然有明显 report-only gain，但仍因 counterfactual tie 和 reject rate 失败；
8. Tetouan seed43 复现了近乎完全拒绝和轻微退化，说明至少该数据集上机制不可用；
9. Spain seed43 没有复现 seed42 的正向信号，说明 Spain 的正结果具有 seed 依赖性；
10. Panama seed44 没有复现 seed43 的明显收益，且 best counterfactual 是 shuffled；
11. Tetouan seed44 虽然有 report-only gain，但仍因 reject rate 与 counterfactual tie 失败；
12. Spain seed44 虽然有 report-only gain 和方向信号，但 reject rate 达到 91.46%，且 amplified `2x/5x` 与 real 几乎并列；
13. 第一轮 public-3 seed42/43/44 b240/full-eval 共 9 个 run 全部 test gate 失败、rolling pass rate 全部为 0，平均 reject rate 约 82.94%。

因此论文中必须禁止以下表述：

```text
Stage III V2.5 significantly improves forecasting accuracy.
The canonical drift generator reliably captures concept drift.
The proposed method achieves SOTA performance under online drift.
```

可以保留的保守表述是：

```text
Stage III V2.5 introduces a split-aware, deterministic, counterfactual-audited protocol for testing drift-conditioned transient adaptation. In current claim-gated audits, it shows limited dataset-dependent report-only gains and some Spain direction sensitivity, but it does not pass rolling validation, counterfactual, gain-threshold, or rejection-rate criteria required for a strong performance claim.
```

---

## 12. 2026-05-26 V3 Repair Probe

V3 adds another repair layer on the opt-in V2/V2.5 path:

1. `TrainSetTemporalResidualCCAAligner` fits a train-only ridge-whitened CCA map between `delta_phi_v[t]-delta_phi_v[t-1]` and forecast-residual transitions.
2. Counterfactual interventions can now operate in generator-input space, so `amplified_2x/5x` is not cancelled by `normalize_vector=True`.
3. The runner now writes a resolved config snapshot and refuses existing output/checkpoint directories by default.
4. Alignment diagnostics are written for train/validation/test.

Verification:

```text
88 targeted Stage III V2/V3 tests passed.
```

Panama seed42 b120/full-eval probe:

| Metric | Value |
| --- | ---: |
| test MSE gain | +0.9338% |
| test MAE gain | +0.6660% |
| reject rate | 16.86% |
| best counterfactual | amplified_2x |
| rolling validation pass rate | 0/5 |
| test claim gate | fail |

Expert interpretation:

V3 improves the deployment profile compared with the V2.5 high-rejection runs. It also shows that real drift is better than zero, shuffled, sign-flipped, and random matched-norm on this single Panama probe.

However, `amplified_2x` still beats real drift. Because amplification is now applied in generator-input space, this is no longer a normalization artifact. It is direct evidence that the generator/risk-gate/blend stack is not yet calibrated to the natural magnitude of the real drift signal.

Therefore:

```text
V3 is a stronger repair and a promising probe, but it is still not a validated drift-conditioned performance mechanism.
```

Next technical repair:

1. Add per-step trace outputs for real vs amplified modes.
2. Diagnose whether amplified wins through larger delta norm, higher risk-gate acceptance, or blend/shift thresholds.
3. Add amplitude calibration or served-policy generator selection before expanding to multi-dataset experiments.

---

## 13. 2026-05-26 V4 Counterfactual Hard-Negative Full Run

V4 is a direct response to the V3 full-run diagnosis:

1. V3 amplified `5x` increased generator-input drift norm from `10.1432` to `50.7161`, but `delta_norm_pre_clamp` changed only from `1.7783` to `1.7787`.
2. V3 prediction shift also barely changed: real `0.203955`, amplified `5x` `0.204067`.
3. V3 risk score increased substantially under amplified drift: real `0.6850`, amplified `5x` `0.8821`.
4. Therefore V3 failed because the generator was not magnitude-calibrated and the gate treated amplified drift as safer, not because amplified produced a meaningfully different adaptation vector.

V4 changes:

1. Add amplified `2x/5x` drift as generator-training hard negatives.
2. Add `detach_negative_mse=false`, allowing the intervention-compare loss to backpropagate through negative counterfactual branches.
3. Add `would_apply_blended_mse/mae` to step traces so later common-mask diagnostics are possible.

Verification:

```text
97 tests passed.
py_compile passed for the modified Stage III files.
configs/stage3_v4_counterfactual_hard_negative.yaml parsed successfully.
```

Panama seed42 full run:

| Metric | V3 Full | V4 Full |
| --- | ---: | ---: |
| test real MSE gain | +4.1871% | +3.6507% |
| test real MAE gain | +3.1039% | +3.0443% |
| reject rate | 24.67% | 0.80% |
| best counterfactual | amplified_5x | amplified_2x |
| real-vs-best-CF margin | -0.000701 | +0.000138 |
| rolling pass rate | 0/5 | 0/5 |
| claim gate | fail | fail |

Expert judgment:

V4 is a real mechanism repair: it fixes the high-rejection behavior and reverses the amplified-vs-real direction on the test aggregate. But it is not enough for a manuscript claim, because the positive margin is far below the pre-declared `0.002` counterfactual margin and every rolling validation window still fails.

Safe interpretation:

```text
V4 provides partial evidence that counterfactual hard-negative training can improve Stage III deployment behavior, but it does not yet validate reliable concept-drift-conditioned adaptation.
```

Do not write:

```text
Stage III V4 proves the proposed drift generator learns concept drift.
Stage III V4 is ready to replace the original manuscript Stage III evidence.
```

---

## 14. 2026-05-27/28 V7 Residualized RRCCA And Served-policy Diagnostic Rerun

V7 addressed the representational-alignment issue directly:

1. Compute raw drift as `D = phi_v - phi_i`.
2. Fit a train-only ridge residualizer `D ~ phi_i` to remove invariant-explainable structure.
3. Build chronological inner fit/calibration splits.
4. Run reduced-rank CCA between residualized drift transitions and forecast-residual transitions.
5. Keep only dimensions that pass fit correlation, calibration correlation/R2, and bootstrap sign-stability filters.

Implementation entry points:

- `models/stage3_alignment_v7.py`
  - `TrainSetChronologicalResidualizedReducedRankCCAAligner`
- `train/run_stage3_v2_sensitivity_audit.py`
  - `canonicalizer_method: residualized_rrcca_v7`
  - `served_policy_candidate_window_breakdown.csv`
  - `served_policy_selected_candidate_window_breakdown.csv`
- `configs/stage3_v7_residualized_rrcca_served_policy_gate.yaml`

Test/verification:

```text
C:\Users\admin\anaconda3\python.exe -m pytest tests -q
121 passed in 3.99s

C:\Users\admin\anaconda3\python.exe -m py_compile models\stage3_alignment_v7.py train\run_stage3_v2_sensitivity_audit.py train\stage3_v2_artifacts.py
pass
```

Key alignment diagnostics from Panama seed42:

| Metric | Value |
| --- | ---: |
| active rank | 8 |
| inactive dim zeroed | 56 |
| residualizer R2 mean | 0.931512 |
| residualizer R2 max | 0.970671 |
| calibration top corr | 0.312118 |
| calibration transition R2 | 0.097418 |
| bootstrap sign stability min | 0.703125 |
| projection condition number | 1.242098 |

Expert interpretation of alignment:

```text
The user's concern was correct: raw phi_v - phi_i is heavily invariant-explainable.
V7 makes the drift representation more defensible as a diagnostic object.
```

However, the served-policy forecasting gate still failed.

Non-downgraded diagnostic rerun:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v2_sensitivity_audit.py --config configs\stage3_v7_residualized_rrcca_served_policy_gate.yaml --dataset panama --run-id v7ps42g5b480diag1 --seed 42 --device auto --modes real zero shuffled sign_flipped amplified_2x amplified_5x random_matched_norm --validation-windows 5 --drift-source delta_phi_v --drift-polarity 1.0 --signal-mode delta_vector --normalize-vector --epochs 5 --invariant-epochs 5 --generator-epochs 5 --max-generator-batches 480 --max-eval-batches 480 --out-root logs\s3v7pdiag --checkpoint-root ckpt\s3v7pdiag
```

Output:

- `logs/s3v7pdiag/v7ps42g5b480diag1/panama/seed_42`
- exit status: expected strict abstain
- final `generator_served_policy.pt`: not saved
- all-candidate breakdown rows: `125`

Selected served-policy candidate was epoch `4`:

| Window | Real MSE gain | Real MAE gain | Best counterfactual | Margin | Real reject rate | Gate |
| ---: | ---: | ---: | --- | ---: | ---: | --- |
| 0 | 0.000561 | 0.000534 | amplified_2x | 0.000556 | 0.033333 | fail |
| 1 | 0.002325 | 0.001505 | zero | 0.002325 | 0.031250 | fail |
| 2 | 0.001524 | 0.001651 | amplified_2x | 0.000900 | 0.037500 | fail |
| 3 | 0.000490 | 0.000634 | amplified_2x | 0.000267 | 0.037500 | fail |
| 4 | 0.001710 | 0.000719 | amplified_2x | 0.001684 | 0.045833 | fail |

Claim-gate thresholds were:

| Threshold | Value |
| --- | ---: |
| min real MSE gain | 0.005 |
| min real MAE gain | 0.002 |
| counterfactual margin | 0.002 |
| max real reject rate | 0.2 |
| rolling window pass rate | 1.0 |

Technical diagnosis:

1. V7 is no longer blocked by a completely insensitive drift pathway. Real drift has high use rate, and sign-flipped drift consistently hurts.
2. V7 is also no longer mainly blocked by excessive real-drift rejection; selected-candidate real reject rates are below `0.046`.
3. The remaining blocker is effect size and counterfactual separation. The best selected-window MSE gain is only `0.002325`, below the required `0.005`, and several windows have real-vs-amplified margins below `0.002`.
4. Therefore the current bottleneck is not merely implementation hygiene. It is the weak translation from cleaned representation drift into reliably useful parameter updates.

Scientific conclusion:

```text
V7 supports a representation-leakage diagnosis.
V7 does not support a manuscript claim that Stage III has learned reliable concept-drift-conditioned online adaptation.
```
