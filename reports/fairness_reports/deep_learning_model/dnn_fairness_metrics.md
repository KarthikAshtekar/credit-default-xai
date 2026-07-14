# DNN Fairness Metrics

`SEX` is retained only for group fairness evaluation and is excluded from DNN training.

| model_name | threshold | protected_attribute | demographic_parity_difference | equalized_odds_difference | equal_opportunity_difference | disparate_impact_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| dnn_baseline | 0.5 | SEX | 0.03312040157040819 | 0.0472440035835342 | 0.013257088292721209 | 0.9629185568434344 |
| dnn_recall_optimized | 0.3 | SEX | 0.05373157176774379 | 0.05886278204169093 | 0.03128467553997616 | 0.9320962379632979 |

A DNN must be evaluated on fairness as well as performance. Improved recall that worsens fairness is an explicit tradeoff, not an automatic improvement.

These metrics are diagnostic and are not proof of legal compliance.