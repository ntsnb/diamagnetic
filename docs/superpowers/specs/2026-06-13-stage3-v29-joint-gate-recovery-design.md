# Stage III V29 Joint Gate Recovery Design

Date: 2026-06-13

## Purpose

This design defines the next predeclared V29 tuning matrix after the
history-gate recovery matrix exhausted without a teacher-gate pass. The goal is
to test whether frozen-topology V29 can jointly recover the three promotion
conditions that have failed together:

- positive partial-window count
- partial MSE/MAE floor safety versus `history_only`
- real-signal ordering versus counterfactual controls

This design does not change the V29 expert-local invariant-release architecture,
does not overwrite prior V29/V28 artifacts, and does not unlock generator
training unless the existing generator-unlock preflight later passes.

## Evidence Boundary

The immediate source evidence is the final HGR decomposition:

`artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_hgr04_gate_failure_decomposition.json`

Key facts from that artifact:

- `candidate_count=36`
- `gate_pass_count=0`
- `generator_unlocked_count=0`
- `matrix_decision=stop_local_scalar_matrix`
- `counterfactual_margin_negative=36/36`
- `min_partial_mse_below_floor=36/36`
- `min_partial_mae_below_floor=36/36`
- `positive_partial_windows_below_required=21/36`
- window `0` partial MSE floor failures are `36/36`
- window `2` has `0/36` partial-floor failures but `34/36` counterfactual-negative rows

The best selector-ranked HGR row is:

`v29hgr_conf015_03_cftie`

It improves the counterfactual margin to `-0.006009365114644338`, but still has
only `2/4` positive partial windows, negative partial floors, and a negative
counterfactual margin. Therefore the next matrix must not optimize only rank or
one metric. It must jointly test floor, positive-window, and counterfactual
constraints.

## Non-Goals

- Do not create V30 or modify `models/factor_mechanism_v29.py`.
- Do not change the native Stage III drift source away from
  `pooled_servable_change`.
- Do not train the generator from matrix generation alone.
- Do not rerun any completed HGR, repair, or teacher-gate scalar candidate.
- Do not claim accuracy improvement, online improvement, or paper-grade
  evidence from this matrix unless a future bounded run passes the existing
  selector, decomposition, and preflight gates.

## Recommended Approach

Create a new planner:

`train/plan_stage3_v29_joint_gate_recovery_matrix.py`

The planner reads the final HGR decomposition plus selector/window artifacts and
emits a short, path-safe, non-overwriting matrix under:

- `configs/stage3_v29_joint_gate_recovery_matrix/`
- `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_joint_gate_recovery_matrix.csv`
- `artifacts/stage3_v29_teacher_gate_tuning_matrix/v29_complete_conf015_joint_gate_recovery_matrix.md`

The matrix should contain four frozen-topology candidates. Each candidate must
modify multiple related invariant loss weights as a coherent mechanism test,
not as a blind scalar sweep.

Frozen topology means generated configs must not change the `model_v29` block,
the native `online.stage3.drift_source`, target modules, target names, or the
V29 runner entry point. The planner may change only `invariant` loss weights and
matrix metadata unless a later reviewed design explicitly expands the scope.

## Candidate Axes

### 1. `joint_floor_cf_balance`

Purpose: test whether partial floor and counterfactual ordering can be improved
together without increasing release amplitude enough to harm window `0`.

Expected overrides:

- small increase to counterfactual shuffle pressure and margin
- small increase to residual-release advantage
- small increase to support-conditioned advantage
- increased router/anchor stability to preserve window `0`

Promotion hypothesis:

This row is only useful if it improves counterfactual margin and min partial
floors together. A better rank with unchanged `2/4` positive windows is still a
negative result.

### 2. `window0_preserving_late_floor`

Purpose: retain the best HGR property, window `0` non-harm, while pushing
windows `1` and `4` away from history-dominant partial-floor failure.

Expected overrides:

- stronger late-window or delayed release pressure
- stronger support alignment
- conservative release-first increment
- anchor preservation kept at or above the HGR top row

Promotion hypothesis:

This row must not trade window `0` safety for aggregate partial gains. If
window `0` becomes harmful, the row should be rejected even if aggregate
validation improves.

### 3. `real_beats_shuffle_margin`

Purpose: attack the specific failure that real signal often loses to shuffled or
sign-flipped counterfactuals, especially in windows where partial floors are not
the dominant issue.

Expected overrides:

- increase counterfactual shuffle margin more than prior HGR rows
- increase local environment consistency to make real environment-conditioned
  signal less exchangeable with shuffled controls
- reduce release amplitude slightly to avoid counterfactual gains caused only by
  larger magnitude

Promotion hypothesis:

This row is useful only if real signal becomes best or near-best in more
windows. If counterfactual margin remains negative for the primary target, it
cannot unlock generator training.

### 4. `history_residual_rebalance`

Purpose: reduce the dominance of `history_only` while preserving residual
release enough to add incremental signal.

Expected overrides:

- lower history-after-release and history-budget pressure
- increase release-priority and support alignment
- increase suppression sparsity only mildly to avoid forced correction

Promotion hypothesis:

This row should increase positive partial-window count. If it only increases
aggregate raw forecast gain while min partial floors remain negative, it should
be rejected.

## Data Flow

1. Read the base V29 stability config:
   `configs/stage3_v29_expert_local_invariant_release_probe_tune_stability_release_deconstrained.yaml`.
2. Read final HGR decomposition:
   `v29_complete_conf015_hgr04_gate_failure_decomposition.json`.
3. Read final HGR selector and window artifacts:
   `v29_complete_conf015_hgr04_selection.csv` and
   `v29_complete_conf015_hgr04_selection_windows.csv`.
4. Refuse to generate rows unless the decomposition reports:
   `matrix_decision=stop_local_scalar_matrix`,
   `gate_pass_count=0`, and `generator_unlocked_count=0`.
5. Emit four candidate configs with:
   `joint_gate_recovery_matrix.protocol_version =
   stage3_v29_joint_gate_recovery_matrix_v1`.
6. Generate a non-overwriting execution plan with the existing execution-plan
   tool.
7. Run at most one bounded candidate at a time:
   `epochs=1`, `max_train_batches=60`, `max_generator_batches=60`,
   `max_eval_batches=60`, `validation_windows=5`.
8. After each run, refresh selector, generator-unlock preflight, and
   gate-failure decomposition before any next row.

## Error Handling And Safety

- Planner must refuse non-final, gate-positive, or generator-unlocked source
  decompositions.
- Planner must reject axes that duplicate old scalar, repair, or HGR axes.
- Generated run IDs must be short enough for Windows checkpoint path limits.
- Generated commands must omit `--overwrite`.
- Existing run or checkpoint directories must be treated as blockers by the
  execution-plan tool.
- Generator training remains blocked unless preflight reports
  `preflight_pass=true`.

## Testing Requirements

Add tests under:

`tests/test_stage3_v29_joint_gate_recovery_matrix.py`

Required assertions:

- non-final or gate-positive source decomposition returns no rows
- generated axes do not repeat old teacher-gate, repair, or HGR axes
- candidate rows preserve frozen V29 topology and only modify `invariant`
  weights plus matrix metadata
- generated configs retain `online.stage3.drift_source=pooled_servable_change`
- suggested commands are bounded and do not include `--overwrite`
- generated run IDs and predicted checkpoint paths remain path-safe when passed
  through the existing execution planner

## Review Criteria

The design is acceptable only if it keeps the V29 architecture frozen while
creating a stronger mechanism test than HGR. It should be rejected if it becomes
another one-axis scalar sweep or if it weakens the existing generator-unlock
preflight boundary.

## Expected Interpretation Of Outcomes

If all four candidates fail, the correct conclusion is not to keep tuning the
same axes. It would mean the current V29 teacher target and served correction
package still cannot pass the Stage III teacher gate under bounded weather
seed42 evidence. The next step would then be a target/mode-selection diagnostic
or a narrowly scoped serving-time abstention design, not a broad model rewrite.

If one candidate passes the primary teacher gate and generator preflight, it
becomes candidate evidence for a bounded generator run. It still does not become
paper-grade evidence until online validation, cross-dataset checks, and seed
checks exist.
