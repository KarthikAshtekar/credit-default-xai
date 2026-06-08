# CV Bullets

## Resume Bullets
- Built a leakage-audited credit default risk model using XGBoost, achieving `0.7825` ROC-AUC on application-time features while rejecting hindsight-driven near-perfect results.
- Developed an end-to-end responsible AI workflow for credit risk covering preprocessing, leakage diagnosis, explainability, fairness analysis, mitigation experiments, and dashboard delivery.
- Separated underwriting features from post-loan behavioral monitoring signals to produce an honest application-time model suitable for portfolio presentation.
- Integrated SHAP, LIME, and DiCE counterfactual explanations to translate model predictions into global, local, and actionable decision support views.
- Designed fairness reporting for approval decisions using demographic parity, equal opportunity, equalized odds, and disparate impact metrics.

## Technical Resume Bullets
- Implemented feature-set controls in Python to distinguish application-time, behavioral-monitoring, and full diagnostic model variants within a shared credit risk pipeline.
- Added a leakage audit script that checks target leakage, duplicate train/test overlap, feature-target correlation, mutual information, single-feature AUC, XGBoost importance, SHAP signals, and target-shuffle sanity performance.
- Validated the final XGBoost application model with both random and temporal splits, confirming stable ROC-AUC (`0.7825` random, `0.7841` temporal).
- Built a Streamlit dashboard that loads saved model-validation, explainability, fairness, and counterfactual artifacts with graceful fallbacks for missing files.
- Produced notebook-driven and markdown-based reporting artifacts tied directly to saved JSON, CSV, and PNG outputs rather than duplicating model-training code.

## Project Title Variants
- Explainable and Fair Credit Default Risk Prediction
- Leakage-Audited Credit Risk Modeling with Explainability and Fairness
- Responsible AI for Credit Default Prediction

## LinkedIn / GitHub Description
Built a leakage-audited credit default risk project using Python, XGBoost, SHAP, LIME, counterfactual explanations, and fairness metrics. The work explicitly rejected near-perfect diagnostic results caused by post-loan behavioral features and reframed the final model around application-time underwriting variables, delivering a more honest `0.7825` ROC-AUC XGBoost model supported by temporal validation, bias analysis, mitigation experiments, notebooks, and a Streamlit dashboard.
