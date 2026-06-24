# Stage III V15 Residual-Target Predictability Record

## Scope

V15 is a non-overwriting Stage III diagnostic. It tests whether the corrected fluctuation signal has stable residual-target predictability beyond causal residual-history features.

V15 does not train a generator and does not update the manuscript with positive Stage III claims.

## Design And Plan

- Design spec: `docs/superpowers/specs/2026-06-03-stage3-v15-residual-target-predictability-design.md`
- Implementation plan: `docs/superpowers/plans/2026-06-03-stage3-v15-residual-target-predictability.md`

New V15 files:

- `configs/stage3_v15_residual_target_predictability_probe.yaml`
- `train/run_stage3_v15_residual_target_predictability_probe.py`
- `train/run_stage3_v15_residual_target_predictability_matrix.py`
- `tests/test_stage3_v15_residual_target_predictability.py`

No V12, V13, or V14 implementation file was modified.

## Protocol

Signal:

```text
delta_phi_v(t) = phi_v(t) - phi_i(t)
fluctuation_delta(t) = delta_phi_v(t) - delta_phi_v(t-1)
```

Primary targets:

- `raw_residual_vector`
- `residual_change_vector`

Secondary target:

- `residual_sign`

Feature groups:

- `history_only`
- `fluctuation_only`
- `fluctuation_plus_history`
- `level_offset_only`
- `level_offset_plus_history`

The primary gate asks whether `fluctuation_plus_history` beats `history_only` in rolling validation. A secondary target cannot unlock generator training.

## Verification

Commands run:

```powershell
pytest tests\test_stage3_v15_residual_target_predictability.py -q
```

Result:

```text
11 passed
```

Command:

```powershell
pytest tests\test_stage3_v15_residual_target_predictability.py tests\test_stage3_v14_teacher_protocol.py tests\test_stage3_v13_signal_contract.py tests\test_stage3_v12e_label_availability.py -q
```

Result:

```text
28 passed
```

Command:

```powershell
python -m py_compile train\run_stage3_v15_residual_target_predictability_probe.py train\run_stage3_v15_residual_target_predictability_matrix.py
```

Result: passed with zero exit code.

Command:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v15_residual_target_predictability_probe.py --help
C:\Users\admin\anaconda3\python.exe train\run_stage3_v15_residual_target_predictability_matrix.py --help
```

Result: both help commands completed successfully.

Note: the plain `python` interpreter does not have `torch` installed, so the probe help check must use `C:\Users\admin\anaconda3\python.exe`.

## Panama b480 Experiment

Command:

```powershell
C:\Users\admin\anaconda3\python.exe train\run_stage3_v15_residual_target_predictability_probe.py --config configs\stage3_v15_residual_target_predictability_probe.yaml --dataset panama --run-id v15target_panama_s42_b480_01 --seed 42 --epochs 5 --invariant-epochs 5 --device auto --max-train-batches 480 --max-generator-batches 480 --max-eval-batches 480 --validation-windows 5 --out-root logs\s3v15target_b480 --checkpoint-root ckpt\s3v15target_b480
```

Output directory:

`logs/s3v15target_b480/v15target_panama_s42_b480_01/panama/seed_42`

Checkpoint directory:

`ckpt/s3v15target_b480/panama/v15target_panama_s42_b480_01/seed_42`

Main result file:

`logs/s3v15target_b480/v15target_panama_s42_b480_01/panama/seed_42/teacher_only_abort.json`

Manifest confirms:

- `stage3_protocol_version = v15_residual_target_predictability_probe`
- `stage3_drift_source = delta_phi_v`
- `stage3_signal_mode = fluctuation_delta`
- `history_protocol = v12e_per_horizon_label_availability_history`
- `generator_training_allowed = false`
- `generator_artifact_count = 0`

## Target Gate Results

| Target | Role | Primary Gate | Positive Partial Windows | Min Partial MSE Gain | Min Partial MAE Gain | Counterfactual Margin | Counterfactual Pass |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `raw_residual_vector` | primary | failed | `0/5` | `-0.017731` | `-0.019379` | `0.000000` | false |
| `residual_change_vector` | primary | failed | `0/5` | `-0.007702` | `-0.005487` | `-0.003162` | false |
| `residual_sign` | secondary | failed | `5/5` | `0.000917` | `0.000472` | `-0.000611` | false |

Stop reasons:

- `do_not_train_generator`
- `no_primary_target_passed`

## Validation And Test Partial Gains

Validation partial gain of `fluctuation_plus_history` over `history_only`:

| Target | Partial MSE Gain | Partial MAE Gain | Partial Sign Agreement |
| --- | ---: | ---: | ---: |
| `raw_residual_vector` | `-0.014151` | `-0.016179` | `-0.057986` |
| `residual_change_vector` | `-0.007233` | `-0.004274` | `-0.006771` |
| `residual_sign` | `0.001417` | `0.000724` | `0.052431` |

Test partial gain of `fluctuation_plus_history` over `history_only`:

| Target | Partial MSE Gain | Partial MAE Gain | Partial Sign Agreement |
| --- | ---: | ---: | ---: |
| `raw_residual_vector` | `-0.012672` | `-0.011822` | `-0.054514` |
| `residual_change_vector` | `-0.009189` | `-0.005078` | `-0.014410` |
| `residual_sign` | `0.001566` | `0.000813` | `0.050000` |

The secondary `residual_sign` target showed small positive partial gains, but it failed the counterfactual margin and is not a primary target. It cannot unlock generator training.

## Interpretation

The primary V15 diagnostic is negative.

The current `fluctuation_delta` representation does not provide stable residual-target predictability beyond causal residual history for the two pre-declared primary targets on Panama seed42 b480. The result reinforces the V14 stop decision and blocks generator training.

This does not prove that all future Stage III designs are impossible. It does show that the current representation and readout path should not be promoted as learned fluctuation-driven adaptation.

## Allowed Next Step

The safest next step is not generator training. It is either:

- stop Stage III algorithm expansion and rewrite the manuscript with a conservative contribution framing; or
- design a new representation-training objective only after a fresh spec that directly targets residual predictability, with stricter anti-target-shopping rules.

Any future representation redesign must be a new non-overwriting version and must not cite V15 as positive performance evidence.
