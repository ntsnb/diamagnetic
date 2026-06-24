# Stage III V13 Failure Diagnostics

Input run directory: `logs\s3v13fluct_b480\v13fluct_panama_s42_b480_01\panama\seed_42`

## Window Gate Summary

- Diagnosed windows: 5.
- Failed windows: 4.
- Window 1: family `attribution;real_behavior;incremental`, reasons `real_mse_gain;real_sign_agreement;counterfactual;partial_mse_gain_vs_history;partial_mae_gain_vs_history`, real MSE gain `0.001351`, real MAE gain `0.013692`, best counterfactual `zero`, margin `-0.023112`.
- Window 2: family `real_behavior;incremental`, reasons `real_mse_gain;real_mae_gain;real_sign_agreement;partial_mse_gain_vs_history;partial_mae_gain_vs_history`, real MSE gain `-0.001823`, real MAE gain `-0.000647`, best counterfactual `conditional_shuffled`, margin `0.021155`.
- Window 3: family `attribution;incremental`, reasons `counterfactual;partial_mse_gain_vs_history;partial_mae_gain_vs_history`, real MSE gain `0.019890`, real MAE gain `0.018770`, best counterfactual `zero`, margin `-0.051531`.
- Window 4: family `incremental`, reasons `partial_mae_gain_vs_history`, real MSE gain `0.041147`, real MAE gain `0.016023`, best counterfactual `shuffled`, margin `0.019605`.

## Split Generalization

- Validation-to-test real MSE gain change: `0.012205`.
- Validation-to-test real MAE gain change: `0.005657`.
- Incremental value reversal versus history: `False`.

## Alpha Candidate Stability

- Candidate rows: 204.
- Rows with stability warnings: 190.

Interpretation: this is a derived diagnostic only. It does not validate V13 as a manuscript result.
