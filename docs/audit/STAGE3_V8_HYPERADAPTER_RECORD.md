# Stage III V8 HyperAdapter Experiment Record

Date: 2026-05-28

This record documents the opt-in V8 experiment that added bias adapters to the
Stage III generator. It is an audit record, not manuscript evidence for a
positive claim.

## 1. Why V8 Was Tried

V7 made the drift representation more defensible through residualized RRCCA, but
it still failed the served-policy validation gate. The main remaining diagnosis
was that the path

```text
cleaned drift -> generated parameter delta -> residual correction
```

was too weak. The V7 generator only updated 2-D weights in `output_head`. For
load forecasting, part of the online error can be level-like, so allowing a
small additive update to 1-D bias parameters was a reasonable hypothesis.

V8 therefore tested whether adding bias adapters improves the same Panama
seed42 same-budget canary without changing the legacy Stage III path.

## 2. Implementation Scope

V8 was implemented as an opt-in branch.

New or extended code paths:

- `models/adaptation_v2.py`
  - added `_HyperBiasParamAdapter`;
  - added `LayerAwareHyperAdapterGenerator`;
  - extended `build_stage3_v2_param_delta()` to support 1-D bias deltas only
    when this generator class is selected.
- `train/stage3_generator_training.py`
  - added `target_names` propagation into generator training;
  - added checkpoint loading support for `LayerAwareHyperAdapterGenerator`.
- `train/online_loop_stage3_v2.py`
  - reads `online.stage3.target_names` and applies bias deltas only when the
    config asks for them.
- `train/run_stage3_v2_sensitivity_audit.py`
  - reads `stage3_v2.generator.type`;
  - instantiates `LayerAwareHyperAdapterGenerator` only for
    `layer_aware_hyperadapter_v8`;
  - stores candidate checkpoints in a new V8 namespace.
- `configs/stage3_v8_hyperadapter_residualized_rrcca_served_policy_gate.yaml`
  - uses the V7 residualized RRCCA canonicalizer;
  - sets `target_names: [weight, bias]`;
  - keeps the same strict served-policy claim gate.

Legacy files were not replaced:

- `models/adaptation.py`
- `train/online_loop.py`
- original result tables and figures
- old logs and checkpoints

## 3. Verification Before Experiment Interpretation

The new V8 code was verified before reading the experiment result:

```text
C:\Users\admin\anaconda3\python.exe -m pytest tests -q
127 passed in 5.59s

C:\Users\admin\anaconda3\python.exe -m py_compile models\adaptation_v2.py train\stage3_generator_training.py train\online_loop_stage3_v2.py train\run_stage3_v2_sensitivity_audit.py train\stage3_v2_artifacts.py
pass

git diff --check
pass
```

The V8-specific tests include:

- zero drift produces zero deltas for weight and bias adapters;
- bias adapter receives gradients for nonzero drift;
- generator training can target bias adapters;
- V8 checkpoint round-trip preserves the bias adapter;
- artifact loading can reload a V8 generator;
- online replay can apply configured bias targets.

## 4. Smoke Run

Command:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v2_sensitivity_audit.py --config configs\stage3_v8_hyperadapter_residualized_rrcca_served_policy_gate.yaml --dataset panama --run-id v8ha_smoke_s42_g1b64_01 --seed 42 --device auto --modes real zero shuffled sign_flipped amplified_2x amplified_5x random_matched_norm --validation-windows 5 --drift-source delta_phi_v --drift-polarity 1.0 --signal-mode delta_vector --normalize-vector --epochs 1 --invariant-epochs 1 --generator-epochs 1 --max-generator-batches 64 --max-eval-batches 120 --out-root logs\s3v8smoke --checkpoint-root ckpt\s3v8smoke
```

Output:

- logs: `logs/s3v8smoke/v8ha_smoke_s42_g1b64_01/panama/seed_42`
- checkpoints: `ckpt/s3v8smoke/panama/v8ha_smoke_s42_g1b64_01/seed_42`
- exit status: expected strict abstain
- final served artifact: not saved
- checkpoint class: `LayerAwareHyperAdapterGenerator`
- target parameters:
  - `output_head.0.weight`
  - `output_head.0.bias`
  - `output_head.2.weight`
  - `output_head.2.bias`

Smoke selected only epoch `0`, which failed all validation windows:

| Metric | Value |
| --- | ---: |
| rolling pass | `0/5` |
| worst real MSE gain | `-0.002820` |
| worst real MAE gain | `-0.002019` |
| worst real-vs-best-CF margin | `-0.002820` |
| max real reject rate | `0.250000` |

This smoke run confirmed the V8 plumbing and artifact behavior, but it was
already a negative sign.

## 5. Same-Budget Canary

Command:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v2_sensitivity_audit.py --config configs\stage3_v8_hyperadapter_residualized_rrcca_served_policy_gate.yaml --dataset panama --run-id v8ha_s42_g5b480_01 --seed 42 --device auto --modes real zero shuffled sign_flipped amplified_2x amplified_5x random_matched_norm --validation-windows 5 --drift-source delta_phi_v --drift-polarity 1.0 --signal-mode delta_vector --normalize-vector --epochs 5 --invariant-epochs 5 --generator-epochs 5 --max-generator-batches 480 --max-eval-batches 480 --out-root logs\s3v8diag --checkpoint-root ckpt\s3v8diag
```

Output:

- logs: `logs/s3v8diag/v8ha_s42_g5b480_01/panama/seed_42`
- checkpoints: `ckpt/s3v8diag/panama/v8ha_s42_g5b480_01/seed_42`
- exit status: expected strict abstain
- final served artifact: not saved
- selected candidate by score: epoch `3`

Candidate selection:

| Epoch | Rolling pass | Worst MSE gain | Worst MAE gain | Worst margin | Max reject | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | `0/5` | `-0.022770` | `-0.011458` | `-0.024317` | `0.279167` | fail |
| 1 | `0/5` | `-0.013270` | `-0.008017` | `-0.013270` | `0.247917` | fail |
| 2 | `0/5` | `-0.012125` | `-0.009353` | `-0.012125` | `0.220833` | fail |
| 3 | `0/5` | `-0.008390` | `-0.005176` | `-0.008390` | `0.189583` | fail |
| 4 | `0/5` | `-0.014019` | `-0.009405` | `-0.014325` | `0.197917` | fail |

Selected epoch `3`, real drift windows:

| Window | Real MSE gain | Real MAE gain | Best CF | Best CF MSE gain | Margin | Real reject |
| ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 0 | `-0.008390` | `-0.005163` | `zero` | `0.000000` | `-0.008390` | `0.179167` |
| 1 | `-0.006742` | `-0.003497` | `zero` | `0.000000` | `-0.006742` | `0.164583` |
| 2 | `-0.005138` | `-0.005176` | `amplified_2x` | `0.000717` | `-0.005856` | `0.187500` |
| 3 | `-0.006986` | `-0.004135` | `amplified_2x` | `0.000791` | `-0.007777` | `0.189583` |
| 4 | `-0.004815` | `-0.002956` | `zero` | `0.000000` | `-0.004815` | `0.189583` |

Training metrics showed that V8 did increase update magnitude and prediction
shift:

| Epoch | Train base MSE | Train blended MSE | Delta pre-clamp | Delta post-clamp | Candidate shift |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `0.160896` | `0.161420` | `0.751472` | `0.491976` | `0.073096` |
| 1 | `0.160896` | `0.162144` | `1.341517` | `0.855565` | `0.152290` |
| 2 | `0.160896` | `0.161140` | `1.562330` | `0.774172` | `0.124570` |
| 3 | `0.160896` | `0.160383` | `1.789958` | `0.778211` | `0.124515` |
| 4 | `0.160896` | `0.159641` | `2.010090` | `0.842357` | `0.111995` |

So V8 did not fail because it was inert. It failed because the larger updates
were not reliably useful on rolling validation.

## 6. Comparison Against V7

The most relevant comparison is the same-budget V7 diagnostic run:

- V7 logs:
  `logs/s3v7pdiag/v7ps42g5b480diag1/panama/seed_42`
- V8 logs:
  `logs/s3v8diag/v8ha_s42_g5b480_01/panama/seed_42`

| Run | Best epoch | Rolling pass | Worst MSE gain | Worst MAE gain | Worst margin | Max reject |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V7 residualized RRCCA | 4 | `0/5` | `+0.000490` | `+0.000534` | `+0.000267` | `0.045833` |
| V8 hyperadapter | 3 | `0/5` | `-0.008390` | `-0.005176` | `-0.008390` | `0.189583` |

V8 is worse than V7 under the same validation gate. Therefore V8 should not be
expanded to public-3 or multi-seed evaluation as a performance candidate.

## 7. Diagnosis

V8 tested the hypothesis:

```text
V7 failed because the generator only updates weights; adding bias adapters will
let Stage III correct level-like residuals.
```

The evidence rejects that simple hypothesis. Bias capacity made the updates
stronger, but stronger updates were harmful. This points to a deeper issue:

```text
The generator is still not trained against a teacher update or residual-specific
adaptation target.
```

In other words, the model can now move the output more, but it still does not
know which direction of movement is a reliable correction for the current
drift.

## 8. Scientific Boundary

Supported:

- V8 is implemented as a tested opt-in branch.
- Bias adapters can be loaded, trained, checkpointed, and replayed.
- Increasing adapter capacity changes the Stage III behavior.
- The V8 branch is useful as a negative diagnostic.

Not supported:

- V8 improves forecasting.
- Bias adapters validate Stage III.
- Stage III has learned a reliable concept-drift-conditioned parameter update.
- The manuscript should upgrade Stage III claims from V8 evidence.

## 9. Next Repair Direction

The next version should not simply add more capacity or relax gates. The next
repair should introduce a residual-targeted meta-objective:

1. Construct a train-only residual teacher update for `output_head` or a
   residual correction head.
2. Train the generator to match the teacher update or the teacher residual
   correction, not only the final blended MSE.
3. Add explicit alignment loss between cleaned drift coordinates and residual
   transition/update directions.
4. Preserve zero-drift identity.
5. Keep served-policy abstention and counterfactual gates unchanged.

Until that is implemented and passes a validation-gated canary, Stage III must
remain an audited diagnostic branch rather than a manuscript performance claim.
