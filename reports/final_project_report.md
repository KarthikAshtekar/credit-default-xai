# Final Project Report

## 1. Executive Summary

This project builds a responsible AI workflow for credit-card default risk prediction using the public UCI Taiwan credit-card default dataset. The final primary model is XGBoost, supported by recall-focused threshold tuning, SHAP/LIME explanations, fairness diagnostics, leakage checks, a DNN benchmark, and a Streamlit decision-support dashboard.

Final conclusion: XGBoost remains the primary model. The DNN is retained as a benchmark showing that additional complexity did not materially improve the business objective.

## 2. Business Problem

Credit-risk teams need to identify cardholders with elevated default risk while keeping the workflow explainable enough for manual review. Accuracy alone is not sufficient because the default class is the business-critical class. The project therefore emphasizes recall, PR-AUC, threshold tradeoffs, explainability, and diagnostic fairness metrics.

## 3. Dataset

- Dataset: UCI Default of Credit Card Clients / Taiwan credit-card default
- UCI ID: `350`
- Loader: `ucimlrepo`
- Rows: `30,000`
- Target: `Default_Flag`
- Target meaning: `1` means next-month default / bad outcome

The dataset has no true application timestamp, so final metrics use a stratified held-out split. Temporal validation is not fabricated.

## 4. Preprocessing and Feature Engineering

The final active feature set is `application_public`. It includes UCI fields available before the next-month default target and engineered ratios based on historical bill, payment, limit, and repayment-status behavior.

Feature policy:

- `SEX` is excluded from active training features and retained for fairness auditing.
- `AGE`, `MARRIAGE`, and `EDUCATION` are included as profile variables with explicit audit caveats.
- `PAY_0` to `PAY_6` are treated as historical repayment-status variables, not post-outcome leakage.
- Engineered features include utilization ratios, repayment ratios, recent delay, maximum delay, delayed-month count, average bill amount, and average payment amount.

## 5. Leakage Controls

The leakage audit reports no detected target leakage or train/test overlap based on implemented checks.

Implemented checks:

- Target and ID exclusion from active features
- Duplicate selected-row split handling
- Source-index train/test overlap check
- Target-shuffle sanity test
- Mutual-information review
- UCI feature-timing review

Target-shuffle ROC-AUC: `0.4922`.

## 6. Models Compared

### Logistic Regression

Logistic regression is retained as a transparent linear benchmark.

### XGBoost

XGBoost is the primary model family because it performs strongly on structured tabular credit data and remains compatible with SHAP/LIME explanations.

### Recall-Optimized XGBoost

The selected screening policy keeps the XGBoost model and lowers the operating threshold from `0.50` to `0.25`. The threshold was selected using validation data only.

### Deep Neural Network Benchmark

The DNN benchmark uses a TensorFlow/Keras MLP architecture `64 -> 32 -> 16 -> 1`. It was added as a controlled benchmark, not a forced replacement.

## 7. Model Evaluation Metrics

| Model / policy | Threshold | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic public | 0.50 | 0.7408 | 0.4375 | 0.6036 | 0.5073 |  | 0.7527 |  |
| XGBoost baseline | 0.50 | 0.8152 | 0.6584 | 0.3414 | 0.4496 | 0.3778 | 0.7748 | 0.5415 |
| XGBoost recall policy | 0.25 | 0.7669 | 0.4777 | 0.5810 | 0.5243 | 0.5569 | 0.7748 | 0.5415 |
| DNN baseline | 0.50 | 0.8131 | 0.6426 | 0.3482 | 0.4516 | 0.3833 | 0.7657 | 0.5212 |
| DNN class-weighted | 0.50 | 0.7446 | 0.4429 | 0.6021 | 0.5104 | 0.5617 | 0.7664 | 0.5363 |
| DNN recall policy | 0.30 | 0.7857 | 0.5149 | 0.5350 | 0.5248 | 0.5309 | 0.7657 | 0.5212 |

## 8. Final Model Selection

XGBoost remains the final model because it has better ROC-AUC and PR-AUC than the DNN baseline and its recall-optimized policy captures more defaults than the DNN recall policy.

Final model statement:

> The project uses XGBoost as the primary model and uses the recall-optimized threshold as a manual-review screening policy. The DNN is retained as a benchmark showing that additional complexity did not materially improve ranking quality or recall-policy performance.

## 9. Explainability

### SHAP

SHAP is used for global and local explanations of the XGBoost model.

### LIME

LIME provides an additional local explanation view for applicant-level predictions.

### Counterfactual Guidance

Counterfactual guidance focuses on repayment timeliness, utilization, and repayment relative to bill amounts. It is decision-support guidance, not a guarantee of a lending outcome.

### DNN Permutation Importance

DNN explainability uses model-agnostic permutation importance as a fallback. This is diagnostic and approximate, and does not replace the clearer tree-model explanation workflow.

## 10. Fairness Analysis

Primary protected attribute: `SEX`.

Favorable outcome: predicted non-default / lower-risk manual-review support.

| Model / policy | Threshold | DP difference | Equal opportunity difference | Equalized odds difference | Disparate impact ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost baseline | 0.50 | 0.0220 | 0.0063 | 0.0225 | 0.9754 |
| XGBoost recall policy | 0.25 | 0.0691 |  |  | 0.9089 |
| DNN baseline | 0.50 | 0.0331 | 0.0133 | 0.0472 | 0.9629 |
| DNN recall policy | 0.30 | 0.0537 | 0.0313 | 0.0589 | 0.9321 |

These are diagnostic fairness metrics. They do not prove legal compliance or a bias-free model.

## 11. Dashboard User Journey

The Streamlit dashboard has three user-facing tabs:

- Applicant Report: enter applicant details, get default risk, risk band, manual-review flag, maximum advisable credit exposure, drivers, and downloadable report.
- Improvement Guidance: review shortcomings and simulate potential improvements.
- Model Governance: inspect model metrics, threshold tradeoffs, leakage audit, fairness diagnostics, explainability artifacts, and ML-vs-DL comparisons.

The default applicant prediction uses XGBoost. DNN appears only in governance/advanced comparison.

## 12. Business Relevance for Banks/NBFCs

The workflow demonstrates how a bank or NBFC could structure a model-supported credit-risk review process:

- Use the model score for manual-review support.
- Use threshold tuning to improve default capture.
- Use explanations to understand key risk drivers.
- Use fairness diagnostics to inspect group-level tradeoffs.
- Keep human oversight for final lending decisions.

## 13. Limitations

- Public academic benchmark dataset, not production bank data
- No true timestamp for temporal validation
- No reject inference
- No calibrated regulatory scorecard
- No production monitoring or drift controls
- Fairness metrics are diagnostic and observational
- The dashboard is not a production lending system

## 14. Future Scope

- Add probability calibration and threshold governance.
- Add a scorecard track with WOE/IV/binning/PDO/base odds.
- Add additional public datasets only as separate validation sources.
- Add intersectional fairness diagnostics where sample sizes support them.
- Add drift monitoring and model monitoring before production use.

## 15. Final Conclusion

XGBoost remains the primary model. The recall-optimized threshold is the recommended screening policy for manual-review support. The DNN benchmark is useful because it verifies that a more complex neural network does not materially outperform the simpler and more explainable XGBoost workflow on this structured credit dataset.
