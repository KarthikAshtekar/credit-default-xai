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

Primary protected attribute: `SEX`/gender.

Verified UCI coding:

| SEX code | Group |
| ---: | --- |
| 1 | Male |
| 2 | Female |

Favorable outcome: predicted non-default / lower-risk manual-review support.

| Model / policy | Threshold | DP difference | Equal opportunity difference | Equalized odds difference | Disparate impact ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost baseline | 0.50 | 0.0220 | 0.0063 | 0.0225 | 0.9754 |
| XGBoost recall policy | 0.25 | 0.0691 | 0.0468 | 0.0723 | 0.9089 |
| DNN baseline | 0.50 | 0.0331 | 0.0133 | 0.0472 | 0.9629 |
| DNN recall policy | 0.30 | 0.0537 | 0.0313 | 0.0589 | 0.9321 |

Did we find discrimination? No legal discrimination or causal bias finding is made here. The metrics are diagnostic, not proof of legal compliance or complete fairness.

What we did find is a fairness tradeoff. The baseline XGBoost threshold showed small group-level differences on `SEX`. The recall-optimized threshold improved default capture, but it widened demographic parity, equal opportunity, equalized odds, and disparate-impact gaps. That makes the recall policy a governance-sensitive choice: it may be useful for manual-review screening, but it should not be treated as fairness-neutral.

### Protected-Attribute Fairness Deep Dive: SEX/Gender

The deeper protected-attribute audit adds group outcome analysis, group error analysis, calibration analysis, proxy-risk analysis, feature association checks, SHAP driver comparison by group, a threshold fairness frontier, individual SEX sensitivity, and a nearest-neighbour individual fairness diagnostic.

Main findings:

- Group outcome analysis: at threshold `0.50`, high-risk flag rates were `0.1280` for Male applicants (SEX=1) and `0.1060` for Female applicants (SEX=2); at threshold `0.25`, they were `0.3109` and `0.2419`. Male applicants therefore had higher high-risk flag rates at both thresholds. These are governance diagnostics and do not prove legal discrimination.
- Group error analysis: at threshold `0.50`, false-positive rates were `0.0542` for Male applicants and `0.0479` for Female applicants; false-negative rates were `0.6459` and `0.6684`. At threshold `0.25`, false-positive rates rose to `0.2094` and `0.1626`, while false-negative rates fell to `0.3782` and `0.4505`. Male applicants had higher FPR, while Female applicants had higher FNR.
- Error-harm interpretation: a higher FPR means more actual non-defaulters are flagged high-risk, which may unnecessarily push reliable customers into manual review or lower credit support. A higher FNR means more actual defaulters are missed, which may increase lender default-risk exposure. These are different error harms and are treated as a fairness-governance signal, not proof of legal discrimination.
- Threshold tradeoff: lowering XGBoost from threshold `0.50` to `0.25` improved recall from `0.3414` to `0.5810`, but widened demographic parity difference from `0.0220` to `0.0691` and equalized odds difference from `0.0225` to `0.0723`.
- Calibration check: the largest absolute bin-level calibration gap was `0.0863`, so score calibration by group should be monitored before any real operational use.
- Proxy-risk analysis: `SEX`/gender was moderately predictable from non-sensitive features. Random forest proxy ROC-AUC was `0.6476`; logistic proxy ROC-AUC was `0.5924`. This means removing `SEX` prevents direct use, but it is not a complete fairness strategy.
- Feature association: the top proxy-associated features by standardized mean difference include `BillToLimitRatio_1`, `BillToLimitRatio_2`, `PAY_2`, `AvgBillToLimitRatio`, and `BillToLimitRatio_3`.
- SHAP driver comparison completed. The strongest common drivers remained repayment and utilization variables such as `MaxPaymentDelay`, `PAY_0`, `BILL_AMT1`, `NumDelayedMonths`, and payment-amount features.
- Threshold fairness frontier: threshold choice is a governance lever. Lowering the threshold improves default capture but can widen group-level fairness diagnostics and manual-review burden.
- Individual SEX sensitivity: flipping `SEX` only caused maximum probability change `0.00000000` and zero baseline/recall-policy decision changes. This verifies no direct use of `SEX` in the active XGBoost prediction path, but it does not eliminate proxy concerns.
- Nearest-neighbour diagnostic: median cross-group nearest-neighbour probability difference was `0.0278`, 90th percentile was `0.1156`, and `764` matched pairs exceeded `0.10` absolute probability difference. This is diagnostic and depends on the chosen distance metric.

Final fairness interpretation: the protected-attribute deep dive does not establish legal discrimination or causal bias. It shows a diagnostic fairness-governance signal. Male applicants (SEX=1) had higher high-risk flag rates than Female applicants (SEX=2), and this gap widened under the recall-focused threshold. The recall-focused policy improved default capture but also widened demographic parity and equalized-odds differences. Individual sensitivity testing showed that flipping SEX alone did not change XGBoost predictions, confirming no direct use of SEX in the active prediction path. However, proxy analysis showed that SEX/gender is moderately predictable from non-sensitive credit variables, so removing SEX alone is not a complete fairness strategy. Therefore, the model should remain a decision-support tool requiring threshold governance, human oversight, and ongoing fairness monitoring.

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
