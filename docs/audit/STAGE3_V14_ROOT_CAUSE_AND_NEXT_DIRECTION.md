# Stage III V14 Root-Cause Diagnosis And Next Direction

## Scope

This document diagnoses why the V14 Panama b480 pilot failed and records the next direction before any V15 implementation or experiment is launched.

Primary evidence:

- `logs/s3v14stability_b480/v14stable_panama_s42_b480_01/panama/seed_42/teacher_only_abort.json`
- `logs/s3v14stability_b480/v14stable_panama_s42_b480_01/panama/seed_42/residual_teacher_v14_diagnostics.csv`
- `logs/s3v14stability_b480/v14stable_panama_s42_b480_01/panama/seed_42/v14_ridge_fh.csv`
- `logs/s3v14stability_b480/v14stable_panama_s42_b480_01/panama/seed_42/v14_alpha_blocks_fh.csv`
- `docs/audit/stage3_v14_pilot_failure_diagnostics/v14stable_panama_s42_b480_01/`

## What V14 Actually Tested

V14 did not test a new interpretation of the Stage III signal. It kept the corrected signal:

```text
fluctuation_delta(t) = delta_phi_v(t) - delta_phi_v(t-1)
```

where:

```text
delta_phi_v(t) = phi_v(t) - phi_i(t)
```

The main V14 changes were:

- replace V12b's default `[x, abs(x), x^2, norm(x)]` readout with train-only scaled linear context features;
- keep V12e per-horizon label availability for residual-history controls;
- set `selected_policy=strict`, so `predict_correction()` uses strict OOF-approved outputs by default;
- keep the full V13/V14 counterfactual gate.

The manifest confirms:

- `stage3_protocol_version = v14_stability_teacher_probe`
- `stage3_signal_mode = fluctuation_delta`
- `history_protocol = v12e_per_horizon_label_availability_history`

## Observed Failure

The run wrote:

`teacher_only_abort.json`

with stop reasons:

- `fluctuation_plus_history_teacher_gate_failed`
- `partial_mae_gain_vs_history_nonpositive`
- `partial_mse_gain_vs_history_nonpositive`

Rolling validation:

- `fluctuation_plus_history`: `0/5` windows passed.
- `fluctuation_only`: `0/5` windows passed.
- `history_only`: `0/5` windows passed, but its real MSE/MAE gains were positive before counterfactual/sign gates.

Partial gain versus `history_only`:

| Split | Partial MSE Gain | Partial MAE Gain |
| --- | ---: | ---: |
| Validation | `-0.014151` | `-0.016179` |
| Test | `-0.012672` | `-0.011822` |

This means the fluctuation signal did not add incremental value beyond causal residual-history controls.

## OOF Policy Diagnosis

V14 did not fail only because there was no numerical signal at all. The important distinction is between exploratory OOF gain and strict OOF stability.

For `fluctuation_plus_history`:

- `ungated` OOF MSE gain: `0.009183`
- `relaxed` OOF MSE gain: `0.005599`
- `strict` OOF MSE gain: `0.000000`
- `ungated` nonzero alpha fraction: `0.5`
- `relaxed` nonzero alpha fraction: `0.25`
- `strict` nonzero alpha fraction: `0.0`
- `strict` output pass count: `0/12`
- `relaxed` output pass count: `3/12`

For `history_only`:

- `ungated` OOF MSE gain: `0.045476`
- `relaxed` OOF MSE gain: `0.023319`
- `strict` OOF MSE gain: `0.016261`
- `ungated` nonzero alpha fraction: `1.0`
- `relaxed` nonzero alpha fraction: `0.333333`
- `strict` nonzero alpha fraction: `0.083333`
- `strict` output pass count: `1/12`
- `relaxed` output pass count: `4/12`

Interpretation:

`history_only` has a small but more stable residual-prediction signal than `fluctuation_plus_history`. Adding fluctuation features makes the teacher more exploratory but less stable under worst-block OOF criteria.

## Block-Level Diagnosis

The strongest `fluctuation_plus_history` candidate rows still fail strict safety because at least one chronological block becomes harmful.

Examples:

| Candidate | Fold Passes | Min MSE Gain | Min MAE Gain | Min Sign Agreement | Mean MSE Gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| output `1`, alpha `0.025` | `4/5` | `-0.074689` | `-0.032047` | `0.559524` | `0.005842` |
| output `0`, alpha `0.025` | `4/5` | `-0.179265` | `-0.051765` | `0.614458` | `-0.012527` |
| output `1`, alpha `0.050` | `4/5` | `-0.356210` | `-0.209011` | `0.559524` | `-0.031659` |

The failure is not a single all-or-nothing coding artifact. It is a stability problem: several outputs look helpful in some blocks but become harmful in the worst chronological block.

## Root-Cause Finding

The current best-supported root cause is:

> The existing `fluctuation_delta` representation is not stable enough as an incremental residual-correction input beyond causal residual history. V14's strict readout does what it was designed to do: it prevents unstable relaxed corrections from being served.

This is a direction-level finding, not a proof that all future Stage III designs are impossible.

What it does rule out:

- Simply relaxing the strict gate.
- Running public-3 on V14.
- Training a generator from V14.
- Claiming Stage III has learned fluctuation-driven adaptation from V14.

## Candidate Next Directions

### Option A: V15 Residual-Target Predictability Diagnostic

Build a non-overwriting V15 diagnostic that asks whether any train-only, chronologically calibrated readout can predict residual correction from:

- `history_only`
- `fluctuation_only`
- `fluctuation_plus_history`
- residual target variants, such as raw residual, residual sign, residual change, and horizon-specific residual blocks

This is not a generator experiment. It is a target/representation upper-bound diagnostic.

Pros:

- Directly tests whether the problem is the signal, the target, or the gate.
- Avoids another generator canary against an unstable teacher.
- Can be run first on Panama b480 and stopped if no target variant shows stable incremental value.

Cons:

- It may produce only negative evidence.
- It does not by itself create a deployable Stage III algorithm.

### Option B: V15 Representation-Training Redesign

Modify the aligned representation training objective so `delta_phi_v` is supervised by future residual-change or per-horizon residual-sign targets before any teacher/readout is fitted.

Pros:

- Addresses the upstream representation problem directly.
- Closer to a true algorithmic contribution if it works.

Cons:

- Higher risk and broader code impact.
- Easy to overfit if target definition is chosen after seeing validation/test behavior.
- Should not be launched before Option A clarifies which target is meaningful.

### Option C: Stop Stage III Algorithm Expansion

Freeze Stage III V2-V14 as audit evidence and rewrite the paper around safer contributions: shifted-stream protocol, frozen-vs-online improvement, backbone portability, deployment cost, and limitations of concept-drift adaptation.

Pros:

- Scientifically safest for manuscript integrity.
- Avoids compute spent on a repeatedly failed mechanism.

Cons:

- Does not satisfy the user's stated goal of continuing algorithm redesign.
- May reduce the apparent novelty of the paper unless reframed carefully.

## Recommendation

Do Option A first.

Do not start a V15 representation-training redesign until a target-predictability diagnostic shows a stable target that `fluctuation_delta` can help predict beyond `history_only`.

The next implementation should be non-overwriting and diagnostic-only:

- `train/run_stage3_v15_residual_target_predictability_probe.py`
- `configs/stage3_v15_residual_target_predictability_probe.yaml`
- `tests/test_stage3_v15_residual_target_predictability.py`
- `docs/audit/STAGE3_V15_RESIDUAL_TARGET_PREDICTABILITY_RECORD.md`

Stop rule:

If `fluctuation_plus_history` does not beat `history_only` for any pre-declared residual target on rolling validation, do not proceed to representation retraining or generator training.

