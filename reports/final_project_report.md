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

## Recall-Focused Threshold Tuning

Because missed defaults are costly in credit risk, the project includes a validation-only threshold and class-weight tuning workflow. The workflow splits the training fold into inner-train and validation data, selects thresholds on validation data only, and evaluates the chosen policy once on the untouched held-out test split.

| Policy | Threshold | Accuracy | Precision | Recall | F1 | F2 | PR-AUC | Approval-support rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline XGBoost | 0.50 | 0.8152 | 0.6584 | 0.3414 | 0.4496 | 0.3778 | 0.5415 | 0.8854 |
| Selected recall policy | 0.25 | 0.7669 | 0.4777 | 0.5810 | 0.5243 | 0.5569 | 0.5415 | 0.7311 |

Selected policy:

- Candidate: `xgboost_public_baseline_threshold_050`
- Rule: maximize recall subject to validation precision >= `0.50`
- Fallback used: `False`
- Test confusion counts: TP `771`, FP `843`, TN `3832`, FN `556`
- Best separated class-weight candidate: `scale_pos_weight=2.0`
- SMOTE: skipped because `imbalanced-learn` is not installed

Saved outputs:

- `reports/model_validation/threshold_tuning_report.csv`
- `reports/model_validation/threshold_selection_summary.csv`
- `reports/model_validation/selected_recall_policy.json`
- `reports/model_validation/class_weight_tuning_report.csv`
- `reports/model_validation/smote_experiment_report.csv`
- `reports/model_validation/recall_optimized_summary.md`
- `reports/model_validation/precision_recall_curve_baseline.png`
- `reports/model_validation/precision_recall_curve_recall_optimized.png`
- `reports/model_validation/precision_recall_curve_comparison.png`

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

## Deep Learning Benchmark

The controlled TensorFlow/Keras MLP benchmark uses the same application feature policy, excludes `SEX` from training while retaining it for fairness evaluation, and selects its operating threshold on validation data only.

| Experiment | Threshold | Precision | Recall | F2 | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DNN baseline | 0.50 | 0.6426 | 0.3482 | 0.3833 | 0.7657 | 0.5212 |
| DNN class-weighted | 0.50 | 0.4429 | 0.6021 | 0.5617 | 0.7664 | 0.5363 |
| DNN recall-optimized | 0.30 | 0.5149 | 0.5350 | 0.5309 | 0.7657 | 0.5212 |

XGBoost remains primary because the DNN did not materially improve ranking or PR-AUC and its selected policy captured fewer defaults than the XGBoost recall policy. DNN permutation importance is explicitly diagnostic and approximate; fairness metrics are diagnostic rather than proof of legal compliance.

## Future Scope

South German Credit, Bondora, and Home Credit are future scope only. They are not implemented as the current primary dataset.
