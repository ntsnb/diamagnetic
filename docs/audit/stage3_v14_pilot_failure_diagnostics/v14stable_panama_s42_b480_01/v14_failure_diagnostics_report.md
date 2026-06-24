# Stage III V14 Failure Diagnostics

Input run directory: `logs\s3v14stability_b480\v14stable_panama_s42_b480_01\panama\seed_42`

## Window Gate Summary

- Diagnosed windows: 5.
- Failed windows: 5.
- Window 0: family `attribution;real_behavior;incremental`, reasons `real_mse_gain;real_mae_gain;real_sign_agreement;counterfactual;partial_mse_gain_vs_history;partial_mae_gain_vs_history`, real MSE gain `0.000000`, real MAE gain `0.000000`, best counterfactual `zero`, margin `0.000000`.
- Window 1: family `attribution;real_behavior;incremental`, reasons `real_mse_gain;real_mae_gain;real_sign_agreement;counterfactual;partial_mse_gain_vs_history;partial_mae_gain_vs_history`, real MSE gain `0.000000`, real MAE gain `0.000000`, best counterfactual `zero`, margin `0.000000`.
- Window 2: family `attribution;real_behavior;incremental`, reasons `real_mse_gain;real_mae_gain;real_sign_agreement;counterfactual;partial_mse_gain_vs_history;partial_mae_gain_vs_history`, real MSE gain `0.000000`, real MAE gain `0.000000`, best counterfactual `zero`, margin `0.000000`.
- Window 3: family `attribution;real_behavior;incremental`, reasons `real_mse_gain;real_mae_gain;real_sign_agreement;counterfactual;partial_mse_gain_vs_history;partial_mae_gain_vs_history`, real MSE gain `0.000000`, real MAE gain `0.000000`, best counterfactual `zero`, margin `0.000000`.
- Window 4: family `attribution;real_behavior;incremental`, reasons `real_mse_gain;real_mae_gain;real_sign_agreement;counterfactual;partial_mse_gain_vs_history;partial_mae_gain_vs_history`, real MSE gain `0.000000`, real MAE gain `0.000000`, best counterfactual `zero`, margin `0.000000`.

## Split Generalization

- Validation-to-test real MSE gain change: `0.000000`.
- Validation-to-test real MAE gain change: `0.000000`.
- Incremental value reversal versus history: `False`.

## Alpha Candidate Stability

- Candidate rows: 108.
- Rows with stability warnings: 108.

Interpretation: this is a derived diagnostic only. It does not validate V14 as a manuscript result.
