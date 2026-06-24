# Stage III V5/V6 Served-policy Canary Record

Date: 2026-05-27

This record documents the first non-downgraded Panama seed42 canary after the
served-policy selection repair. These runs are audit evidence, not manuscript
performance evidence.

## Code/protocol changes before the canary

- Final served-policy generator selection now has real abstention semantics:
  if no candidate passes the rolling claim gate and `abstain_if_no_pass: true`,
  the runner writes selection CSVs but refuses to save a final served Stage III
  artifact.
- Served-policy checkpoint selection can now train and threshold-calibrate a
  risk gate per generator candidate. This prevents scoring one policy and then
  serving a different `generator + risk_gate + threshold` bundle.
- Selection and threshold calibration now exclude offline-only diagnostic modes
  by default. The selection modes used here were:
  `real;zero;sign_flipped;amplified_2x;amplified_5x`.
- Risk-gate training now names the previous-step negative as `previous_step`
  instead of overloading the sensitivity-audit `shuffled` mode.
- A V6 opt-in canonicalizer was added:
  `TrainSetWhitenedProcrustesTemporalResidualCCAAligner`, which aligns `phi_v`
  to `phi_i` by train-only whitening/CORAL plus orthogonal Procrustes before
  temporal residual CCA.
- Windows path length affected long run ids. The successful canaries therefore
  use short roots: `logs/s3v5g5`, `logs/s3v6g5`, `checkpoints/s3v5g5`, and
  `checkpoints/s3v6g5`.

## Verification

- `C:\Users\admin\anaconda3\python.exe -m pytest tests -q`
- Result: `115 passed`

## V5 g5_b480 canary

Run:

- Config: `configs/stage3_v5_served_policy_counterfactual_gate.yaml`
- Run id: `v5g5s42r1`
- Output: `logs/s3v5g5/v5g5s42r1/panama/seed_42`
- Checkpoints: `checkpoints/s3v5g5/panama/v5g5s42r1/seed_42`
- Budget: `epochs=5`, `invariant_epochs=5`, `generator_epochs=5`,
  `max_generator_batches=480`, `max_eval_batches=480`, `validation_windows=5`

Served-policy generator selection:

| Epoch | Pass | Window pass rate | Failure reasons | Worst MSE gain | Worst MAE gain | Worst real-vs-CF margin | Max reject |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |
| 0 | `False` | `0.0` | `window_pass_rate;real_mse_gain;real_mae_gain;counterfactual` | `-0.007112` | `-0.005336` | `-0.007112` | `0.172917` |
| 1 | `False` | `0.6` | `window_pass_rate;real_mse_gain;real_mae_gain;counterfactual` | `0.001599` | `-0.000439` | `0.000964` | `0.045833` |
| 2 | `False` | `0.2` | `window_pass_rate;real_mse_gain;counterfactual;real_mae_gain` | `-0.003118` | `-0.001445` | `-0.003185` | `0.066667` |
| 3 | `False` | `0.6` | `window_pass_rate;real_mse_gain;real_mae_gain` | `0.003363` | `0.001955` | `0.003363` | `0.058333` |
| 4 | `False` | `0.8` | `window_pass_rate;real_mse_gain` | `0.004620` | `0.004778` | `0.004575` | `0.095833` |

Interpretation:

- V5 g5_b480 substantially improves over V5.1 rolling behavior, from `2/5` to
  best candidate `4/5`.
- It still fails the pre-declared gate. The best candidate misses the worst
  window MSE-gain threshold (`0.004620 < 0.005`) and window pass rate (`4/5`
  rather than `5/5`).
- Because no candidate passed, the runner correctly abstained and did not save
  `generator_served_policy.pt`.
- This is a near-miss repair signal, not publishable Stage III validation.

## V5 g8_b960 canary

Run:

- Config: `configs/stage3_v5_served_policy_counterfactual_gate.yaml`
- Run id: `v5g8s42r1`
- Output: `logs/s3v5g8/v5g8s42r1/panama/seed_42`
- Checkpoints: `checkpoints/s3v5g8/panama/v5g8s42r1/seed_42`
- Budget: `epochs=5`, `invariant_epochs=5`, `generator_epochs=8`,
  `max_generator_batches=960`, `max_eval_batches=960`,
  `validation_windows=5`

Served-policy generator selection:

| Epoch | Pass | Window pass rate | Failure reasons | Worst MSE gain | Worst MAE gain | Worst real-vs-CF margin | Max reject |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |
| 0 | `False` | `0.2` | `window_pass_rate;real_mae_gain;real_mse_gain;counterfactual` | `-0.007176` | `-0.008458` | `-0.007444` | `0.049266` |
| 1 | `False` | `0.4` | `window_pass_rate;real_mse_gain;real_mae_gain;counterfactual` | `-0.001537` | `-0.003998` | `-0.001537` | `0.042977` |
| 2 | `False` | `0.6` | `window_pass_rate;real_mse_gain;real_mae_gain` | `0.003612` | `-0.002499` | `0.003612` | `0.028332` |
| 3 | `False` | `0.6` | `window_pass_rate;real_mse_gain;real_mae_gain;counterfactual` | `-0.006953` | `-0.008745` | `-0.007094` | `0.061845` |
| 4 | `False` | `0.6` | `window_pass_rate;real_mse_gain;real_mae_gain;counterfactual` | `-0.002373` | `-0.005942` | `-0.003847` | `0.064008` |
| 5 | `False` | `0.6` | `window_pass_rate;real_mse_gain;real_mae_gain` | `0.002033` | `-0.000243` | `0.002033` | `0.046122` |
| 6 | `False` | `0.6` | `window_pass_rate;real_mse_gain;real_mae_gain;counterfactual` | `-0.003231` | `-0.008009` | `-0.003258` | `0.027254` |
| 7 | `False` | `0.6` | `window_pass_rate;real_mse_gain;real_mae_gain;counterfactual` | `-0.003015` | `-0.006847` | `-0.003271` | `0.062959` |

Interpretation:

- Increasing the generator budget from `g5_b480` to `g8_b960` did not rescue
  Stage III under the same pre-declared served-policy gate.
- The best rolling pass rate dropped from `4/5` in `g5_b480` to `3/5` in
  `g8_b960`.
- The larger run repeatedly fails MAE and often fails MSE or the
  real-vs-counterfactual margin.
- The runner correctly abstained and did not save
  `generator_served_policy.pt`.
- This result makes the near-miss `g5_b480` less persuasive as a manuscript
  claim. It should be treated as a diagnostic clue, not evidence that more
  generator training is sufficient.

## V6 g5_b480 canary

Run:

- Config: `configs/stage3_v6_whitened_procrustes_served_policy_gate.yaml`
- Run id: `v6g5s42r1`
- Output: `logs/s3v6g5/v6g5s42r1/panama/seed_42`
- Checkpoints: `checkpoints/s3v6g5/panama/v6g5s42r1/seed_42`
- Same budget as V5 g5_b480.

Served-policy generator selection:

| Epoch | Pass | Window pass rate | Failure reasons | Worst MSE gain | Worst MAE gain | Worst real-vs-CF margin | Max reject |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |
| 0 | `False` | `0.0` | `window_pass_rate;real_mse_gain;real_mae_gain;counterfactual` | `-0.001276` | `-0.000819` | `-0.001321` | `0.081250` |
| 1 | `False` | `0.2` | `window_pass_rate;real_mse_gain;real_mae_gain;counterfactual` | `-0.000346` | `0.000299` | `-0.000346` | `0.104167` |
| 2 | `False` | `0.2` | `window_pass_rate;real_mse_gain;real_mae_gain;counterfactual` | `-0.003977` | `-0.002433` | `-0.003977` | `0.052083` |
| 3 | `False` | `0.2` | `window_pass_rate;real_mse_gain;real_mae_gain;counterfactual` | `-0.001718` | `-0.001431` | `-0.001718` | `0.041667` |
| 4 | `False` | `0.2` | `window_pass_rate;real_mse_gain;counterfactual;real_mae_gain` | `-0.000633` | `0.000473` | `-0.000775` | `0.027083` |

Alignment diagnostics:

- V6 was applied via `transform_phi_pair=True`.
- Train top transition-residual correlation decreased from `0.4381` to
  `0.3471`; validation top correlation increased from `0.2800` to `0.3232`;
  test top correlation decreased from `0.3448` to `0.3230`.
- V6 did not improve served-policy rolling evidence on Panama seed42.

Interpretation:

- The V6 alignment hypothesis is implemented and auditable, but this canary
  does not support using it as a manuscript improvement.
- V6 is worse than V5 on this evidence slice: best rolling pass rate is only
  `1/5`, and every candidate has negative worst-window MSE gain.

## Claim boundary after these canaries

- Supported: the repaired V5 served-policy protocol is much more scientifically
  honest; it can identify and block failed candidates, and V5 g5_b480 is close
  to the rolling gate on Panama seed42.
- Not supported: Stage III is a validated concept-drift learner; V6 alignment
  improves Stage III; Stage III should be promoted into the manuscript as a
  proven performance contribution.
- Next safe experiment, if continuing model-side work: do not spend public-3
  multi-seed compute yet. First inspect the failed V5 windows, especially the
  `g5_b480` epoch-4 near miss and the recurring `g8_b960` windows 3-4 failures,
  to determine whether the failure comes from drift representation, risk-gate
  over/under-filtering, or generator instability.
