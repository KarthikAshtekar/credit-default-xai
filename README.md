# Explainable and Fair Credit-Card Default Risk Prediction

This repository implements a responsible AI workflow for credit-card default risk prediction using the public **UCI Default of Credit Card Clients / Taiwan credit-card default** dataset.

Current framing: **Explainable and fair credit-card default risk prediction using the public UCI Taiwan credit-card default dataset.**

The project is an academic and portfolio workflow, not production lending software, not a regulatory credit scorecard, and not a substitute for underwriting governance or legal review.

## Primary Dataset

- Dataset: UCI Default of Credit Card Clients / Taiwan credit-card default
- Source: public UCI dataset ID `350`
- Loader: `ucimlrepo`
- Rows: `30,000`
- Target: `Default_Flag`, where `1` means next-month default
- The project no longer requires a local private file for final results

Load the primary dataset:

```bash
python -m src.data_api_loader --source uci --dataset_name default_credit_card
```

The default project pipeline also loads this UCI dataset.

## Dataset Coverage

The UCI dataset covers the required project blocks:

| Block | Fields |
| --- | --- |
| Borrower profile | `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` |
| Credit history | `PAY_0`, `PAY_2`, `PAY_3`, `PAY_4`, `PAY_5`, `PAY_6` |
| Loan / exposure | `LIMIT_BAL` |
| Financial health | `BILL_AMT1`-`BILL_AMT6`, `PAY_AMT1`-`PAY_AMT6`, engineered ratios |
| Target | `Default_Flag` |

Saved mapping:

```text
reports/data_audit/five_block_dataset_mapping.md
reports/data_audit/five_block_dataset_mapping.csv
```

## Feature Policy

Final active training feature set: `application_public`.

- Includes public UCI fields available before the next-month default target
- Excludes the target
- Excludes direct protected attribute `SEX` from final active training features
- Retains `SEX` in the raw/audit dataframe for fairness analysis
- Includes `AGE`, `MARRIAGE`, and `EDUCATION` as profile variables, with the policy documented

`PAY_0` to `PAY_6` are historical repayment-status variables. They are valid predictors for the question "predict next-month default from prior repayment behavior" and are not treated as post-outcome leakage.

Engineered features include:

- `BillToLimitRatio_1` to `BillToLimitRatio_6`
- `AvgBillToLimitRatio`
- `AvgPaymentToBillRatio`
- `RecentPaymentDelay`
- `MaxPaymentDelay`
- `NumDelayedMonths`
- `AvgBillAmount`
- `AvgPaymentAmount`
- `PaymentToLimitRatio`

## Final Model Metrics

Final model: `models/xgboost_public.pkl`

Held-out stratified split metrics:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost public | 0.8152 | 0.6584 | 0.3414 | 0.4496 | 0.7748 |
| Logistic public | 0.7408 | 0.4375 | 0.6036 | 0.5073 | 0.7527 |

Saved metrics:

```text
reports/model_validation/public_credit_model_comparison.csv
reports/model_validation/xgboost_public_model_metrics.json
reports/model_validation/logistic_public_model_metrics.json
```

The UCI dataset does not include a true application timestamp. Temporal validation is therefore not invented; the report records that the final metrics use a stratified held-out split.

## Recall-Focused Threshold Tuning

The baseline XGBoost at the default `0.50` threshold has strong overall accuracy but low default-class recall:

| Policy | Threshold | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC | PR-AUC | Approval-support rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline XGBoost | 0.50 | 0.8152 | 0.6584 | 0.3414 | 0.4496 | 0.3778 | 0.7748 | 0.5415 | 0.8854 |
| Recall screening policy | 0.25 | 0.7669 | 0.4777 | 0.5810 | 0.5243 | 0.5569 | 0.7748 | 0.5415 | 0.7311 |

The selected recall policy keeps the baseline XGBoost model and lowers the screening threshold to `0.25`. Selection used validation data only, applying Rule A: maximize recall subject to validation precision >= `0.50`; fallback Rule B maximizes F2 if Rule A has no candidate.

Run the recall workflow:

```bash
python -m src.recall_optimization
```

Saved outputs:

```text
reports/model_validation/threshold_tuning_report.csv
reports/model_validation/threshold_selection_summary.csv
reports/model_validation/selected_recall_policy.json
reports/model_validation/class_weight_tuning_report.csv
reports/model_validation/smote_experiment_report.csv
reports/model_validation/recall_optimized_summary.md
reports/model_validation/precision_recall_curve_baseline.png
reports/model_validation/precision_recall_curve_recall_optimized.png
reports/model_validation/precision_recall_curve_comparison.png
reports/fairness_reports/application_model/threshold_fairness_comparison.csv
```

SMOTE is separated from the selected policy. If `imbalanced-learn` is not installed, the SMOTE experiment is skipped and documented instead of failing the workflow.

## Leakage Audit

Conclusion:

> No detected target leakage or train/test overlap in the public UCI pipeline based on implemented checks.

Implemented checks include:

- Target column not included in features
- ID columns not included in features
- Duplicate selected rows kept on the same side of the split
- Source-index train/test overlap check
- Target-shuffle sanity test
- Mutual-information review
- UCI feature-timing review

Saved outputs:

```text
reports/leakage_audit/leakage_audit_report.md
reports/leakage_audit/leakage_audit_summary.json
```

## Explainability

Explainability artifacts are regenerated for the UCI XGBoost model:

```text
reports/explainability_reports/application_model/xgboost_public_shap_summary.png
reports/explainability_reports/application_model/xgboost_public_shap_local.png
reports/explainability_reports/application_model/xgboost_public_lime_local.png
reports/explainability_reports/application_model/xgboost_public_counterfactuals.json
```

Expected strong drivers include recent repayment delay, number of delayed months, credit limit, bill-to-limit utilization, and repayment amount patterns.

## Fairness And Mitigation

Primary protected attribute: `SEX`.

Favorable outcome: predicted non-default / low-risk approval decision.

Saved XGBoost fairness metrics:

| Metric | Value |
| --- | ---: |
| Demographic parity difference | 0.0220 |
| Equal opportunity difference | 0.0063 |
| Equalized odds difference | 0.0225 |
| Disparate impact ratio | 0.9754 |

Mitigation experiments include reweighing and Fairlearn post-processing. Optional AIF360 TensorFlow/inFairness warnings are non-blocking.

Saved outputs:

```text
reports/fairness_reports/application_model/xgboost_public_fairness_metrics.json
reports/fairness_reports/application_model/xgboost_public_fairness_metrics.csv
reports/fairness_reports/application_model/xgboost_public_bias_mitigation_summary.json
reports/fairness_reports/application_model/xgboost_public_fairness_accuracy_tradeoff.csv
```

## Dashboard

Run:

```bash
streamlit run dashboard/app.py
```

Dashboard tabs:

- Project overview
- Applicant risk prediction
- Model performance
- Explainability
- Fairness analysis
- Counterfactual guidance
- Applicant risk scorecard report
- Leakage audit

The applicant prediction form uses UCI/Taiwan credit-card fields:

- `LIMIT_BAL`
- `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`
- `PAY_0`, `PAY_2`, `PAY_3`, `PAY_4`, `PAY_5`, `PAY_6`
- `BILL_AMT1` to `BILL_AMT6`
- `PAY_AMT1` to `PAY_AMT6`

Engineered features are computed internally.

## Reproduce

Install dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Run the full pipeline:

```bash
python -m src.run_pipeline
```

Run recall-focused threshold and class-weight validation:

```bash
python -m src.recall_optimization
```

Optional faster run:

```bash
python -m src.run_pipeline --skip-explainability --skip-counterfactuals --skip-mitigation
```

Development checks:

```bash
python -m compileall src dashboard
pytest
ruff check .
ruff format --check .
git diff --check
```

## Deep Learning Benchmark

A TensorFlow/Keras tabular MLP was added as a controlled model-family benchmark under the same public UCI dataset, application feature policy, stratified train/validation/test discipline, validation-only threshold selection, fairness audit, and explainability workflow. It tests whether additional complexity improves recall, PR-AUC, or ranking quality; it is not assumed to outperform tree boosting on structured credit data.

| DNN experiment | Threshold | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.50 | 0.8131 | 0.6426 | 0.3482 | 0.4516 | 0.3833 | 0.7657 | 0.5212 |
| Class-weighted | 0.50 | 0.7446 | 0.4429 | 0.6021 | 0.5104 | 0.5617 | 0.7664 | 0.5363 |
| Recall-optimized baseline | 0.30 | 0.7857 | 0.5149 | 0.5350 | 0.5248 | 0.5309 | 0.7657 | 0.5212 |

XGBoost remains the primary model: its ROC-AUC `0.7748` and PR-AUC `0.5415` exceed the DNN baseline, and its recall-optimized policy captures more defaults (`0.5810` recall). The DNN remains useful as evidence that final model selection is based on business-relevant metrics, fairness, explainability, and operational complexity rather than algorithm hype.

Run `python -m src.deep_learning_benchmark`; use `--quick`, `--epochs`, `--skip-class-weighted`, or `--skip-explainability` when needed. If TensorFlow is unavailable, the command exits successfully and records a clear skipped status.

## Future Scope

South German Credit, Bondora, and Home Credit are future-scope datasets only. They are not implemented as the current primary dataset.

Future work:

- Add additional public dataset adapters
- Add calibration and threshold governance
- Build a true scorecard track with WOE/IV/binning/PDO/base odds
- Add intersectional fairness where sample sizes support it
- Add monitoring, drift checks, and deployment controls
