# Explainable and Fair Credit Default Risk Prediction

Built a leakage-audited credit default risk model using XGBoost, SHAP, LIME, counterfactual explanations, and fairness metrics.

## Business Problem
Credit decisions affect access to financial opportunity. A useful credit risk model must be accurate enough to support underwriting, transparent enough to explain decisions, and fair enough to surface group-level risk disparities instead of hiding them.

This project asks:

> Can we build a credit default risk model that is accurate, explainable, and fair at loan application time?

## Dataset
- Case-study Dubai Arab Bank dataset
- 10,000 rows
- 28 original columns
- Binary target: `Default_Flag`

The repository now uses three feature framings:
- `application`: variables available at loan start only
- `behavioral`: application variables plus post-loan repayment or monitoring signals
- `full_diagnostic`: the broader mixed feature set kept only for leakage diagnosis

## Methodology
Pipeline:

`EDA -> preprocessing -> feature engineering -> leakage audit -> modeling -> explainability -> fairness -> mitigation -> dashboard`

Key modeling guardrails:
- `CustomerID` and `LoanID` are dropped
- `Default_Flag` is not used as an input feature
- Train/test overlap checks are part of the leakage audit
- The hindsight-style `LoanAgeDays` feature was removed
- Post-loan behavioral features are excluded from the final application model

## Final Model Result
Final recommended model: `models/xgboost_application.pkl`

Application-time metrics:
- Accuracy: `0.7105`
- Precision: `0.6579`
- Recall: `0.7503`
- F1: `0.7011`
- ROC-AUC: `0.7825`

Temporal validation for the same model remained stable:
- Application XGBoost temporal ROC-AUC: `0.7841`

## Leakage Audit Result
The original full-feature XGBoost model achieved near-perfect performance. That result was rejected as the final project outcome because it was suspiciously strong for an application-time underwriting use case.

Audit findings:
- No direct target-column leakage was found
- No train/test row overlap or duplicate IDs were found
- Target shuffle dropped to ROC-AUC `0.5262`, which supports honest validation
- The real issue was feature timing: post-loan behavioral features created hindsight leakage for application-time prediction

Behavioral or monitoring-only features excluded from the final application model:
- `OnTimePayments_Last12M`
- `MissedPayments_Last12M`
- `MissedEMIs_Last6M`
- `AvgMonthlyDebit_AED`
- `StdMonthlyDebit_AED`
- `SalaryDropFlag`
- `SpendingSpikeFlag`
- `StressSignalCount`
- `HistoricalRiskScore`
- `MissedPaymentRate`

Interpretation:
- `xgboost_behavioral` and `xgboost_full_diagnostic` are diagnostic or monitoring models only
- `xgboost_application` is the honest final underwriting model

## Explainability
The final application-time model is paired with:
- SHAP global summary
- SHAP local explanation
- LIME local explanation
- DiCE counterfactual guidance

Saved artifacts live under:
- `reports/explainability_reports/application_model/`

These artifacts support both portfolio presentation and stakeholder-facing interpretation.

## Fairness
Fairness analysis is centered on the final application-time model.

Protected attribute coverage:
- `Gender` is the default protected attribute used in the saved fairness report
- `Age` / age grouping and `Nationality` are discussed where usable

Metrics used:
- Demographic parity difference
- Equal opportunity difference
- Equalized odds difference
- Disparate impact ratio

Saved fairness outputs live under:
- `reports/fairness_reports/application_model/`

Bias mitigation experiments include:
- Reweighing
- Fairlearn post-processing

The project presents mitigation honestly as a tradeoff exercise, not a guaranteed free improvement.

## Dashboard
Run the presentation dashboard with:

```bash
streamlit run dashboard/app.py
```

The dashboard includes:
- Project overview
- Applicant risk prediction
- Clean model performance
- SHAP explainability
- Fairness analysis
- Counterfactual guidance
- Leakage audit summary

## How To Run
Install dependencies:

```bash
pip install -r requirements.txt
```

Generate the validated artifacts:

```bash
python -m src.train_logistic
python -m src.train_xgboost
python -m src.evaluate_models
python -m src.leakage_audit
python -m src.shap_explainer
python -m src.lime_explainer
python -m src.fairness_metrics
python -m src.bias_mitigation
python -m src.counterfactuals
```

Launch the dashboard:

```bash
streamlit run dashboard/app.py
```

## Project Structure
```text
credit-default-xai/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── ...
│   └── 13_fairness_vs_accuracy_tradeoff.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── leakage_audit.py
│   ├── train_logistic.py
│   ├── train_xgboost.py
│   ├── evaluate_models.py
│   ├── shap_explainer.py
│   ├── lime_explainer.py
│   ├── counterfactuals.py
│   ├── fairness_metrics.py
│   └── bias_mitigation.py
├── dashboard/
│   ├── app.py
│   └── common.py
├── reports/
│   ├── explainability_reports/
│   ├── fairness_reports/
│   ├── leakage_audit/
│   ├── model_validation/
│   └── final_project_report.md
├── docs/
│   └── cv_bullets.md
└── models/
    ├── logistic_application.pkl
    ├── logistic_behavioral.pkl
    ├── xgboost_application.pkl
    ├── xgboost_behavioral.pkl
    └── xgboost_full_diagnostic.pkl
```

## Limitations
- Academic demonstration rather than production credit scoring
- Case-study or simulated dataset
- No causal fairness analysis
- No external validation yet on real public datasets
- No reject inference or calibrated scorecard development

## Future Work
- UCI Taiwan credit card default dataset
- HMDA fair-lending style evaluation
- External validation on public real-world credit data
- Causal fairness methods
- Reject inference
- Scorecard calibration and threshold design

## Final Framing
This project should be presented as a responsible AI credit risk workflow, not as a production banking deployment. The core contribution is the combination of leakage audit, clean application-time model selection, explainability, fairness analysis, mitigation, and dashboard communication in one coherent portfolio project.
