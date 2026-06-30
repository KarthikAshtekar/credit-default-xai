# Interview Defense

## 30-Second Pitch

This project builds a responsible AI workflow for credit default prediction. The main result is not the highest possible model score; it is the disciplined decision to reject near-perfect hindsight-driven results and present an honest application-time XGBoost underwriting model. The workflow combines leakage auditing, explainability, fairness diagnostics, mitigation experiments, a Streamlit dashboard, and a separate UCI external-validation benchmark.

## 2-Minute Technical Walkthrough

I started with the Dubai Arab Bank case-study dataset and framed the business question as application-time default prediction. The dataset included borrower, loan, macroeconomic, and post-loan behavioral fields, so I separated features into application-time, behavioral-monitoring, and full-diagnostic sets.

The full-feature XGBoost model produced near-perfect results, but I treated that as suspicious. I audited target leakage, ID leakage, train/test overlap, feature-target correlations, mutual information, single-feature AUC, feature importance, SHAP signals, and target-shuffle performance. The audit showed that the main issue was feature timing: repayment and behavioral monitoring variables gave the model information that would not exist at origination.

The final model is therefore `xgboost_application.pkl`, trained only on application-time features. It achieved `0.7825` ROC-AUC on the random split and `0.7841` ROC-AUC on temporal validation. I then added SHAP, LIME, counterfactual explanations, fairness metrics, mitigation experiments, and a Streamlit dashboard to communicate the model and its caveats.

## Core Technical Contribution

The core contribution is a complete responsible AI credit risk workflow that combines predictive modeling with feature-timing discipline, leakage audit evidence, explainability, fairness diagnostics, mitigation tradeoff reporting, and dashboard communication.

## Why The Full-Feature XGBoost Model Was Rejected

The full-feature XGBoost model reached near-perfect performance because it used behavioral and monitoring variables that reflected post-loan outcomes or stress signals. Those features can be useful for portfolio monitoring, but using them for loan origination would create hindsight leakage. Rejecting that model is a strength because it shows the project prioritizes valid decision timing over inflated performance.

## Why The Application-Time XGBoost Model Is Final

The application-time XGBoost model uses only information expected to be available when a lending decision is made. Its ROC-AUC is lower than the behavioral model, but it is the honest underwriting result. Logistic regression remains a benchmark, while behavioral and full-diagnostic models are retained only as diagnostic evidence.

## How Leakage Audit Was Performed

The leakage audit checked that `Default_Flag` was not used as a feature, `CustomerID` and `LoanID` were dropped, and train/test rows and IDs did not overlap. It also reviewed suspicious features through correlations, mutual information, single-feature AUC, XGBoost importance, SHAP signals, and target-shuffle testing. Target shuffle dropped to near-random performance, while post-loan behavioral variables explained the unrealistic full-feature score.

## How SHAP, LIME, And Counterfactuals Are Used

SHAP is used for global feature importance and local applicant-level drivers. LIME provides a second local explanation view for an individual prediction. Counterfactuals provide decision-support guidance by showing how selected input changes could move a high-risk case toward a lower-risk classification. These explanations support interpretation; they are not a substitute for legally governed adverse-action reason codes.

## How Fairness Was Evaluated

Fairness was evaluated on approval decisions derived from predicted default probabilities. The saved application-model report uses `Gender` as the protected attribute and includes demographic parity difference, equal opportunity difference, equalized odds difference, and disparate impact ratio. Mitigation experiments include reweighing and Fairlearn post-processing. These are group-level diagnostics and tradeoff evidence, not proof that the model is bias-free.

## What External Validation Does And Does Not Prove

External validation benchmarks the workflow on the public UCI Default of Credit Card Clients / Taiwan credit-card default dataset. It trains fresh logistic regression and XGBoost models on the UCI schema and saves separate metrics under `reports/external_validation/default_credit_card/`.

It does not force the Dubai-trained model onto UCI data, and it does not prove production generalization of the Dubai model. The schemas, target framing, feature availability, and population differ, so the external results are useful as a public benchmark only.

## Current Limitations

- The primary dataset is a case-study dataset, not production banking data.
- The final model is not a calibrated regulatory credit scorecard.
- External validation is a separate benchmark, not direct validation of the Dubai-trained model.
- Fairness metrics are observational and group-level.
- Reject inference, causal fairness, threshold governance, monitoring, legal review, and adverse-action compliance are not implemented.
- The dashboard is a local demo app, not a production service.

## Future Work

- Extend external validation to additional public credit datasets.
- Add probability calibration, threshold design, and business-cost analysis.
- Build a true scorecard track with binning, WOE/IV, PDO, base odds, and reason codes.
- Add intersectional fairness analysis where sample sizes are adequate.
- Add model monitoring, drift checks, governance documentation, and deployment controls.
