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

## Future Scope

South German Credit, Bondora, and Home Credit are future-scope datasets only. They are not implemented as the current primary dataset.

Future work:

- Add additional public dataset adapters
- Add calibration and threshold governance
- Build a true scorecard track with WOE/IV/binning/PDO/base odds
- Add intersectional fairness where sample sizes support it
- Add monitoring, drift checks, and deployment controls
