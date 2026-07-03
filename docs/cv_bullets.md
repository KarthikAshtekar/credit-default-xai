# CV Bullets

## Resume Bullets

- Built an explainable and fair credit-card default risk workflow on the public UCI Taiwan dataset, training logistic regression and XGBoost models with leakage-audited held-out validation.
- Refactored the project from a private local dataset dependency to a reproducible `ucimlrepo` UCI pipeline with target normalization, feature engineering, fairness audit retention, and dashboard migration.
- Engineered utilization and repayment-history features including bill-to-limit ratios, repayment-to-bill ratios, recent delay, maximum delay, delayed-month count, average bill amount, and payment-to-limit ratio.
- Generated SHAP, LIME, counterfactual, fairness, mitigation, leakage audit, and Streamlit dashboard artifacts for the public UCI XGBoost model.
- Documented protected-attribute policy by excluding `SEX` from active final training features while retaining it for fairness analysis.

## 3-Pointer CV Version

- Built a responsible AI credit-card default workflow using the public UCI Taiwan dataset and reproducible `ucimlrepo` loading.
- Trained logistic and XGBoost models, with final XGBoost held-out ROC-AUC of `0.7748` and a leakage audit showing no detected target leakage or train/test overlap.
- Delivered SHAP, LIME, counterfactual guidance, fairness metrics on `SEX`, mitigation tradeoff reports, and a UCI-compatible Streamlit dashboard.

## 4-Pointer CV Version

- Migrated the credit-default project to the public UCI Taiwan credit-card default dataset as the primary dataset.
- Implemented UCI schema normalization, `Default_Flag` target standardization, utilization/payment feature engineering, and protected-attribute audit handling.
- Trained logistic regression and XGBoost models with held-out metrics, leakage audit, fairness diagnostics, and mitigation experiments.
- Built a Streamlit dashboard for UCI-style applicant risk prediction, local SHAP explanations, counterfactual guidance, scorecard-style reporting, and audit review.

## LinkedIn / GitHub Description

Explainable and fair credit-card default risk prediction using the public UCI Taiwan dataset. The project includes UCI data loading, feature engineering, logistic and XGBoost modeling, leakage audit, SHAP/LIME/counterfactual explanations, fairness metrics, mitigation tradeoffs, and a Streamlit dashboard. It is framed as an academic responsible AI workflow, not production lending software.
