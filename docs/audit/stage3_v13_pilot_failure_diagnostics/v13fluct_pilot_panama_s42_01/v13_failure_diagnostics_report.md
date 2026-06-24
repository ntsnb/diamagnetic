# Stage III V13 Failure Diagnostics

Input run directory: `logs\s3v13fluct_pilot\v13fluct_pilot_panama_s42_01\panama\seed_42`

## Window Gate Summary

- Diagnosed windows: 5.
- Failed windows: 2.
- Window 1: family `attribution`, reasons `counterfactual`, real MSE gain `0.522514`, real MAE gain `0.324470`, best counterfactual `zero`, margin `0.001857`.
- Window 3: family `real_behavior;incremental`, reasons `real_mae_gain;real_sign_agreement;partial_mse_gain_vs_history;partial_mae_gain_vs_history`, real MSE gain `0.011052`, real MAE gain `-0.008035`, best counterfactual `zero`, margin `0.039980`.

## Split Generalization

- Validation-to-test real MSE gain change: `-0.221147`.
- Validation-to-test real MAE gain change: `-0.106959`.
- Incremental value reversal versus history: `True`.

## Alpha Candidate Stability

- Candidate rows: 204.
- Rows with stability warnings: 179.

Interpretation: this is a derived diagnostic only. It does not validate V13 as a manuscript result.
