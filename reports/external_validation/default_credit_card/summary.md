# External Validation: UCI Default of Credit Card Clients (Taiwan credit-card default)

This benchmark trains separate models on the public UCI Default of Credit Card
Clients / Taiwan credit-card default dataset.
It does not apply the Dubai-trained model to the UCI schema and does not replace the
primary Dubai Arab Bank case-study results.
It is not evidence of production lending readiness or direct production generalization.

## Dataset

- Dataset: `UCI Default of Credit Card Clients / Taiwan credit-card default`
- Source: `uci`
- Rows: `30000`
- Target: `DEFAULT_PAYMENT_NEXT_MONTH`
- Features: `23`
- Protected attributes audited: `SEX`, `AGE_GROUP`

## Model Metrics

| model_name | accuracy | precision | recall | f1 | roc_auc |
| --- | --- | --- | --- | --- | --- |
| xgboost | 0.8177 | 0.6652 | 0.3534 | 0.4616 | 0.7791 |
| logistic_regression | 0.7735 | 0.4893 | 0.5531 | 0.5193 | 0.7588 |

## Fairness Metrics

| model_name | protected_attribute | demographic_parity_difference | equal_opportunity_difference | equalized_odds_difference | disparate_impact_ratio |
| --- | --- | --- | --- | --- | --- |
| logistic_regression | SEX | 0.0552 | 0.0477 | 0.0477 | 0.9285 |
| logistic_regression | AGE_GROUP | 0.1150 | 0.1570 | 0.1570 | 0.8537 |
| xgboost | SEX | 0.0054 | 0.0072 | 0.0286 | 0.9939 |
| xgboost | AGE_GROUP | 0.1137 | 0.0876 | 0.1531 | 0.8775 |

Fairness metrics are calculated on approval decisions derived from default
probabilities using threshold `0.5`.
They are saved separately from the Dubai fairness reports and should be read as
group-level diagnostics, not proof that the benchmark model is bias-free.
