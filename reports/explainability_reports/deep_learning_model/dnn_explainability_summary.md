# DNN Explainability Summary

SHAP was not used because a stable, fast neural-network explainer is not guaranteed across optional TensorFlow/SHAP environments. Model-agnostic permutation importance is the controlled fallback.

| feature | permutation_pr_auc_drop |
| --- | --- |
| PAY_0 | 0.03897586669716013 |
| RecentPaymentDelay | 0.026749399645708882 |
| NumDelayedMonths | 0.020881936481903007 |
| PAY_6 | 0.016091732857956897 |
| BillToLimitRatio_2 | 0.014905631391798235 |
| MARRIAGE | 0.006003036132129269 |
| LIMIT_BAL | 0.004676304700612088 |
| PAY_AMT2 | 0.004419027684984145 |
| BILL_AMT6 | 0.00423229246191803 |
| PAY_AMT1 | 0.004153670231448747 |

Repayment-delay, utilization, billing, and payment variables in the leading features are economically plausible credit-risk signals; rankings remain associational.

DNN explanations are diagnostic and approximate. Model-agnostic explanations support trust but are not legal adverse-action reason codes.

The DNN is more operationally opaque than the tree benchmark, so additional complexity requires a material performance or fairness benefit.