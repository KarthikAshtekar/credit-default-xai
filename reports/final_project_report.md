# Final Project Report

## Current Primary Dataset

- Dataset: UCI Default of Credit Card Clients / Taiwan credit-card default
- UCI ID: `350`
- Loading method: `ucimlrepo`
- Rows: `30,000`
- Target: `Default_Flag`
- Target meaning: `1` means next-month default / bad outcome

The current project no longer uses a local private dataset as the final-result source.

## Feature Policy

Final active training feature set: `application_public`.

`SEX` is excluded from active final training features and retained for fairness auditing. `AGE`, `MARRIAGE`, and `EDUCATION` are included as profile variables and treated as audit-sensitive. Repayment-status fields `PAY_0` to `PAY_6` are historical predictors before the next-month target.

## Dataset Coverage

Saved coverage mapping:

- `reports/data_audit/five_block_dataset_mapping.md`
- `reports/data_audit/five_block_dataset_mapping.csv`

The dataset covers borrower profile, credit history, loan/exposure, financial health, and target blocks.

## Model Results

Held-out stratified split metrics:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost public | 0.8152 | 0.6584 | 0.3414 | 0.4496 | 0.7748 |
| Logistic public | 0.7408 | 0.4375 | 0.6036 | 0.5073 | 0.7527 |

Saved outputs:

- `reports/model_validation/public_credit_model_comparison.csv`
- `reports/model_validation/xgboost_public_model_metrics.json`
- `reports/model_validation/logistic_public_model_metrics.json`

No true temporal validation is reported because the UCI dataset has no application timestamp.

## Leakage Audit

Conclusion:

> No detected target leakage or train/test overlap in the public UCI pipeline based on implemented checks.

Implemented checks include target exclusion, ID exclusion, source-index overlap, duplicate selected-row overlap, target shuffle, mutual information, and UCI feature-timing review.

Target-shuffle ROC-AUC: `0.4922`.

Saved outputs:

- `reports/leakage_audit/leakage_audit_report.md`
- `reports/leakage_audit/leakage_audit_summary.json`

## Explainability

Saved artifacts:

- `reports/explainability_reports/application_model/xgboost_public_shap_summary.png`
- `reports/explainability_reports/application_model/xgboost_public_shap_local.png`
- `reports/explainability_reports/application_model/xgboost_public_lime_local.png`
- `reports/explainability_reports/application_model/xgboost_public_counterfactuals.json`

Main explanation themes are recent repayment delay, delayed-month count, credit limit, bill-to-limit utilization, and repayment amount patterns.

## Fairness And Mitigation

Primary protected attribute: `SEX`.

Favorable outcome: predicted non-default / low-risk approval decision.

| Metric | Value |
| --- | ---: |
| Demographic parity difference | 0.0220 |
| Equal opportunity difference | 0.0063 |
| Equalized odds difference | 0.0225 |
| Disparate impact ratio | 0.9754 |

Mitigation summary:

| Method | ROC-AUC | Demographic parity difference | Disparate impact ratio |
| --- | ---: | ---: | ---: |
| Baseline | 0.7748 | 0.0220 | 0.9754 |
| Reweighing | 0.7735 | 0.0226 | 0.9748 |
| Fairlearn post-processing | 0.6473 | 0.0127 | 0.9857 |

Saved outputs:

- `reports/fairness_reports/application_model/xgboost_public_fairness_metrics.json`
- `reports/fairness_reports/application_model/xgboost_public_fairness_metrics.csv`
- `reports/fairness_reports/application_model/xgboost_public_bias_mitigation_summary.json`
- `reports/fairness_reports/application_model/xgboost_public_fairness_accuracy_tradeoff.csv`

## Dashboard Status

The Streamlit dashboard has been migrated to UCI fields and artifacts.

Run:

```bash
streamlit run dashboard/app.py
```

The dashboard remains a demonstration tool, not a production decision system or regulatory scorecard.

## Future Scope

South German Credit, Bondora, and Home Credit are future scope only. They are not implemented as the current primary dataset.
