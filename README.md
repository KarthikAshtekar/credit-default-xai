# Explainable and Fair Credit Default Risk Prediction

## Project Motivation
Financial institutions increasingly rely on machine learning to predict credit default risk, but high accuracy alone is not enough. In regulated settings, models must also be interpretable and fair across demographic groups.

This project builds an end-to-end ML system that jointly optimizes:
- Predictive performance
- Explainability (SHAP, LIME, counterfactuals)
- Fairness (pre- and post-mitigation)
- Usability via a Streamlit dashboard

## Business Problem
Banks need early and reliable default-risk prediction to:
- Reduce non-performing loans
- Improve underwriting decisions
- Provide responsible credit access

The challenge is balancing model performance against ethical and regulatory concerns.

## AI Ethics Motivation
Credit scoring can amplify historical bias if sensitive proxies are learned unchecked. This repository measures and mitigates group disparities, then quantifies fairness-performance tradeoffs for transparent governance.

## Research Question
Can a credit default prediction model remain accurate while also being fair and explainable across different demographic groups?

## Repository Structure
```text
credit-default-xai/
├── README.md
├── requirements.txt
├── .gitignore
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
│   ├── train_logistic.py
│   ├── train_xgboost.py
│   ├── evaluate_models.py
│   ├── shap_explainer.py
│   ├── lime_explainer.py
│   ├── counterfactuals.py
│   ├── fairness_metrics.py
│   ├── bias_mitigation.py
│   └── utils.py
├── dashboard/
│   ├── app.py
│   └── pages/
│       ├── prediction.py
│       ├── shap_dashboard.py
│       ├── fairness_dashboard.py
│       └── counterfactual_dashboard.py
├── reports/
│   ├── figures/
│   ├── fairness_reports/
│   └── explainability_reports/
└── models/
    ├── logistic_model.pkl
    └── xgboost_model.pkl
```

## Dataset
Primary dataset is auto-detected from `data/raw/`.

Supported raw formats:
- CSV (`.csv`)
- Excel (`.xlsx`, `.xls`)

Expected target:
- `Default_Flag`

Columns dropped by default:
- `CustomerID`
- `LoanID`

## Modeling
Implemented models:
1. Logistic Regression
2. XGBoost Classifier

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- Confusion Matrix

## Explainability
Implemented:
1. SHAP global explanations
2. SHAP local explanations
3. LIME local explanations
4. DiCE counterfactual explanations

Artifacts are saved into `reports/explainability_reports/` and `reports/figures/`.

## Fairness Analysis
Computed metrics:
- Demographic Parity Difference
- Equal Opportunity Difference
- Equalized Odds Difference
- Disparate Impact Ratio

Pre- and post-mitigation comparisons are generated.

## Bias Mitigation
Methods included:
1. Reweighing (AIF360-compatible + fallback implementation)
2. Fairlearn post-processing (`ThresholdOptimizer`)

## Dashboard
Run Streamlit app:
```bash
streamlit run dashboard/app.py
```

Pages:
1. Credit Risk Prediction
2. SHAP Explanations
3. Fairness Dashboard
4. Counterfactual Dashboard

## Quickstart
1. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```
2. Train baseline models:
```bash
python -m src.train_logistic
python -m src.train_xgboost
```
3. Evaluate and generate explainability artifacts:
```bash
python -m src.evaluate_models
python -m src.shap_explainer
python -m src.lime_explainer
python -m src.fairness_metrics
python -m src.bias_mitigation
```

## Team Guardrails
This repository includes collaboration guardrails to support team forking and scaling:
- `CONTRIBUTING.md` for branch/PR workflow
- `CODE_OF_CONDUCT.md` for collaboration standards
- `SECURITY.md` for responsible vulnerability reporting
- `.github/workflows/ci.yml` for lint + smoke checks
- `.github/pull_request_template.md` and `.github/ISSUE_TEMPLATE/*` for consistent contribution quality
- `.pre-commit-config.yaml` + `pyproject.toml` for local quality automation
- `.github/CODEOWNERS` for review routing (update placeholders with your real team handles)

## Typical Results (What To Report)
- XGBoost usually improves ROC-AUC over Logistic Regression.
- Fairness mitigation often reduces group disparity while slightly impacting accuracy.
- SHAP and LIME highlight key drivers (income, repayment behavior, obligations).

## Fairness Findings (Template)
Document the change in:
- Demographic parity difference
- Equal opportunity difference
- Equalized odds difference
- Disparate impact ratio
before and after mitigation, then discuss business/regulatory implications.

## Explainability Findings (Template)
Summarize:
- Global top features via SHAP
- Local explanations for high-risk customers
- Counterfactual changes needed to reduce risk

## Future Improvements
- Add hyperparameter optimization and calibration
- Add additional datasets (Home Credit, German Credit, Lending Club)
- Add model monitoring and drift checks
- Add human-in-the-loop review workflow

## License
This repository is intended for educational and portfolio use. Add a project license if publishing publicly.
