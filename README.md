# Explainable and Fair Credit Default Risk Prediction

This repository implements an end-to-end responsible AI workflow for credit default risk prediction. The final model is intentionally framed as an **application-time underwriting model**, supported by leakage auditing, model validation, explainability, fairness analysis, mitigation experiments, notebooks, tests, and a Streamlit dashboard.

The key project decision is to reject near-perfect diagnostic model results that depended on post-loan behavioral signals. The final recommended model uses only information that should be available at loan application time.

## Current Project Status

Status: **portfolio-ready academic case study, not production-ready lending software**.

Completed:
- Data understanding, cleaning, feature engineering, and narrative notebooks from `01` through `13`
- Application-time, behavioral, and full-diagnostic feature framings
- Logistic regression and XGBoost model training
- Random split and temporal split validation
- Leakage audit covering target leakage, ID leakage, train/test overlap, target shuffle testing, suspicious feature ranking, and feature-timing review
- Final application-time XGBoost model selection
- SHAP global and local explanation artifacts
- LIME local explanation artifact
- DiCE counterfactual artifact
- Fairness metrics for the final application model
- Bias mitigation experiments using reweighing and Fairlearn post-processing
- Streamlit dashboard for prediction, model performance, explainability, fairness, counterfactual guidance, scorecard-style applicant reporting, and leakage audit review
- Dataset loader support for local files, direct CSV/Excel URLs, and selected UCI datasets
- Separate external validation workflow for the UCI Default of Credit Card Clients dataset
- Test suite for repository structure, pipeline orchestration, data loading, dashboard prediction helpers, and report generation
- Governance and collaboration files including contribution, security, maintainer, CI, issue template, pull request template, and code-owner guardrails

Not completed or intentionally out of scope:
- Production lending deployment
- Regulatory credit scorecard calibration
- Causal fairness analysis
- Reject inference
- Legal or adverse-action compliance review
- Direct comparison between Dubai case-study metrics and external public-dataset metrics
- Hosted dashboard deployment
- Dashboard screenshot image files under `assets/dashboard/`

## Business Problem

Credit decisions affect access to financial opportunity. A useful credit risk model must be accurate enough to support underwriting, transparent enough to explain decisions, and disciplined enough to avoid hidden leakage or unfair group-level outcomes.

This project asks:

> Can we build a credit default risk model that is accurate, explainable, and fair using only loan application-time information?

## Dataset

Default project dataset:
- Dubai Arab Bank case-study dataset
- 10,000 rows
- 28 original columns
- Binary target: `Default_Flag`

Place the local case-study file at:

```text
data/raw/Afors Consulting_Dubai Arab Bank Dataset_MDI.xlsx
```

Raw data files and trained model pickle files are intentionally ignored by Git. A fresh clone can regenerate model artifacts after the dataset is placed in `data/raw/`.

The project uses three feature framings:
- `application`: borrower profile, affordability, bureau, loan, date-part, and macroeconomic variables available at loan start
- `behavioral`: application variables plus repayment and account-behavior monitoring signals
- `full_diagnostic`: the broad mixed feature set retained for leakage diagnosis only

## Dataset Loading

The default project workflow uses the local case-study file. The repository also includes a lightweight loader for future validation and comparison work.

Local case-study dataset:

```bash
python -m src.data_api_loader --source local
```

UCI dataset loading:

```bash
python -m src.data_api_loader --source uci --dataset_name default_credit_card
python -m src.data_api_loader --source uci --dataset_name south_german_credit
```

Direct URL loading:

```bash
python -m src.data_api_loader --source url --url "<direct_csv_or_excel_url>"
```

Important limitation: UCI and URL loading are available as dataset access utilities, but the saved final model metrics in this repository are based on the local Dubai Arab Bank case-study dataset.

## External Validation

Primary reported project results still come from the Dubai Arab Bank case-study dataset. External validation is implemented as a separate benchmark workflow under `src.external_validation`; it does not overwrite, replace, or mix into the main Dubai final model table.

External datasets have different schemas, so the Dubai-trained model is not forced onto UCI data. Instead, the workflow trains fresh benchmark models on the public dataset while applying the same responsible-AI ideas: clean preprocessing, logistic regression and XGBoost comparison, default-risk metrics, and approval-decision fairness metrics.

Currently supported dataset:
- UCI Default of Credit Card Clients / Taiwan credit-card default dataset

Run:

```bash
python -m src.external_validation --dataset default_credit_card
```

Outputs are saved under:

```text
reports/external_validation/default_credit_card/
```

Saved external-validation outputs:
- `metrics.json`
- `model_comparison.csv`
- `fairness_metrics.json`
- `fairness_metrics.csv`
- `summary.md`

External validation reports are useful as public benchmark evidence, but they are not directly comparable to the Dubai Arab Bank case-study results because the schemas, population, target framing, and feature availability differ.

## Methodology

Pipeline:

```text
EDA -> preprocessing -> feature engineering -> leakage audit -> modeling -> validation -> explainability -> fairness -> mitigation -> dashboard
```

Core guardrails:
- `CustomerID` and `LoanID` are dropped before modeling
- `Default_Flag` is never used as an input feature
- The hindsight-style `LoanAgeDays` feature was removed
- Train/test overlap checks are part of the leakage audit
- Post-loan behavioral variables are excluded from the final application model
- Temporal validation is reported alongside random split validation
- Fairness mitigation is treated as a tradeoff exercise, not as proof that the model is bias-free

## Final Model

Final recommended model: `models/xgboost_application.pkl`

Application-time validation metrics:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.7105 |
| Precision | 0.6579 |
| Recall | 0.7503 |
| F1 | 0.7011 |
| ROC-AUC | 0.7825 |

Temporal validation for the same application-time XGBoost framing remained stable:

| Model | Split | ROC-AUC |
| --- | --- | ---: |
| XGBoost application | random | 0.7825 |
| XGBoost application | temporal | 0.7841 |
| Logistic application | random | 0.7639 |
| Logistic application | temporal | 0.7623 |

Interpretation:
- `xgboost_application` is the honest final underwriting model
- `logistic_application` is retained as a transparent benchmark
- `xgboost_behavioral` and `xgboost_full_diagnostic` are retained only as monitoring or leakage-diagnostic evidence

## Leakage Audit Result

The original full-feature XGBoost model achieved near-perfect performance. That result was rejected as the final project outcome because it depended on information that would not be available at loan origination.

Audit findings:
- No direct target-column leakage was found
- No duplicate train/test rows, duplicate customer IDs, or duplicate loan IDs were found
- Target shuffle dropped to ROC-AUC `0.5262`, supporting that the validation split itself was not the main issue
- The main issue was feature timing: post-loan behavioral variables created hindsight leakage for an application-time underwriting use case

Behavioral or monitoring-only variables excluded from the final application model:
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

`PastDefaults` is also excluded from the application-time model because the dataset does not explicitly state whether it refers only to defaults before the current loan.

## Model Comparison

Saved model comparison:

| Model | Feature Set | Accuracy | F1 | ROC-AUC | Project Use |
| --- | --- | ---: | ---: | ---: | --- |
| Logistic regression | application | 0.6925 | 0.6751 | 0.7639 | benchmark |
| XGBoost | application | 0.7105 | 0.7011 | 0.7825 | final model |
| Logistic regression | behavioral | 0.8635 | 0.8525 | 0.9476 | diagnostic only |
| XGBoost | behavioral | 1.0000 | 1.0000 | 1.0000 | rejected for origination |
| XGBoost | full diagnostic | 0.9995 | 0.9994 | 1.0000 | leakage diagnostic only |

The gap between the application-time and behavioral/full-diagnostic results is the central evidence behind the feature-timing decision.

## Explainability

Completed explainability artifacts for the final application model:
- SHAP global summary plot
- SHAP local waterfall plot
- LIME local explanation plot
- DiCE counterfactual output

Saved artifacts:

```text
reports/explainability_reports/application_model/
```

Dashboard prediction also computes current applicant-level SHAP drivers after a user submits the applicant form, using the final application-time model.

## Fairness And Mitigation

The saved fairness report uses `Gender` as the protected attribute for the final application model. Approval decisions are derived from predicted default probabilities at a `0.50` threshold.

Saved fairness metrics:

| Metric | Value |
| --- | ---: |
| Demographic parity difference | 0.0151 |
| Equalized odds difference | 0.0138 |
| Equal opportunity difference | 0.0008 |
| Disparate impact ratio | 0.9693 |

Bias mitigation experiments:

| Method | ROC-AUC | Demographic Parity Difference | Disparate Impact Ratio |
| --- | ---: | ---: | ---: |
| Baseline | 0.7825 | 0.0151 | 0.9693 |
| Reweighing | 0.7801 | 0.0361 | 0.9279 |
| Fairlearn post-processing | 0.6964 | 0.0074 | 0.9871 |

Interpretation:
- Reweighing kept predictive performance close to baseline but did not improve the saved fairness metrics
- Fairlearn post-processing reduced demographic parity difference but materially reduced ROC-AUC
- These results should be presented as fairness-performance tradeoffs, not as evidence that the model is fully fair

Saved fairness outputs:

```text
reports/fairness_reports/application_model/
```

## Dashboard

Run the Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

Dashboard tabs and features:
- Project overview
- Applicant risk prediction
- Validated model performance
- SHAP-based explainability
- Fairness analysis
- Counterfactual guidance
- Applicant risk scorecard report
- Leakage audit summary

The applicant form:
- Asks only for application-time applicant and loan details
- Computes EMI and affordability ratios internally
- Excludes post-loan repayment and monitoring fields
- Displays applicant-level SHAP drivers after prediction
- Generates a scorecard-style applicant report for presentation purposes

Important caveat: the scorecard-style report is not a calibrated regulatory credit scorecard.

## Dashboard Preview

Dashboard screenshot support is prepared through the `assets/dashboard/` folder. Screenshot image files are not currently committed.

Dashboard screenshots should be saved under `assets/dashboard/` using these filenames:

| View | Expected file |
| --- | --- |
| Project Overview | `assets/dashboard/overview.png` |
| Applicant Risk Prediction | `assets/dashboard/prediction.png` |
| Applicant-Level SHAP Explanation | `assets/dashboard/applicant_shap.png` |
| Validated Model Performance | `assets/dashboard/model_performance.png` |
| Fairness Analysis | `assets/dashboard/fairness_analysis.png` |
| Applicant Risk Scorecard Report | `assets/dashboard/scorecard_report.png` |
| Leakage Audit | `assets/dashboard/leakage_audit.png` |

## Notebooks

The notebook sequence documents the project workflow:

| Notebook | Focus |
| --- | --- |
| `01_data_understanding.ipynb` | dataset overview and business framing |
| `02_data_cleaning.ipynb` | cleaning choices and reproducibility notes |
| `03_feature_engineering.ipynb` | engineered ratios and final feature framing |
| `04_logistic_regression.ipynb` | benchmark model |
| `05_xgboost_model.ipynb` | final application-time XGBoost model |
| `06_model_comparison.ipynb` | model comparison across feature framings |
| `07_shap_analysis.ipynb` | global and local SHAP explanations |
| `08_lime_analysis.ipynb` | local LIME explanation |
| `09_counterfactuals.ipynb` | counterfactual guidance |
| `10_fairness_analysis.ipynb` | fairness metrics |
| `11_bias_mitigation.ipynb` | mitigation experiments |
| `12_final_results.ipynb` | final project summary |
| `13_fairness_vs_accuracy_tradeoff.ipynb` | fairness-performance tradeoff |

## How To Reproduce

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

Install development dependencies when running tests and lint checks:

```bash
pip install -r requirements-dev.txt
```

Place the local case-study dataset in `data/raw/`, then regenerate artifacts:

```bash
python -m src.run_pipeline
```

Optional pipeline shortcuts:

```bash
python -m src.run_pipeline --skip-explainability
python -m src.run_pipeline --skip-counterfactuals
python -m src.run_pipeline --skip-mitigation
```

Individual stages:

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

Run the separate public-dataset validation benchmark:

```bash
python -m src.external_validation --dataset default_credit_card
```

Development checks:

```bash
python -m compileall src dashboard
pytest
ruff check .
ruff format --check .
```

Core Ruff checks intentionally exclude notebooks. Treat notebooks as narrative companions and use notebook-specific tooling separately if notebook linting is required.

## Generated Artifacts

Tracked report artifacts:
- `reports/model_validation/clean_feature_model_comparison.*`
- `reports/model_validation/temporal_split_comparison.csv`
- `reports/model_validation/*_model_metrics.json`
- `reports/leakage_audit/*`
- `reports/explainability_reports/application_model/*`
- `reports/fairness_reports/application_model/*`
- `reports/external_validation/default_credit_card/*`
- `reports/final_project_report.md`

Ignored local artifacts:
- Raw datasets under `data/raw/`
- Trained model pickle files under `models/`
- Additional generated figures under ignored report paths

## Project Structure

```text
credit-default-xai/
|-- .github/
|   |-- workflows/
|   |-- ISSUE_TEMPLATE/
|   |-- CODEOWNERS
|   `-- pull_request_template.md
|-- assets/
|   `-- dashboard/
|-- data/
|   |-- raw/
|   `-- processed/
|-- dashboard/
|   |-- app.py
|   |-- common.py
|   |-- prediction_helpers.py
|   `-- report_utils.py
|-- docs/
|   |-- cv_bullets.md
|   `-- model_card.md
|-- models/
|-- notebooks/
|   |-- 01_data_understanding.ipynb
|   `-- 13_fairness_vs_accuracy_tradeoff.ipynb
|-- reports/
|   |-- explainability_reports/
|   |-- external_validation/
|   |-- fairness_reports/
|   |-- leakage_audit/
|   |-- model_validation/
|   `-- final_project_report.md
|-- src/
|   |-- data_api_loader.py
|   |-- data_preprocessing.py
|   |-- feature_engineering.py
|   |-- run_pipeline.py
|   |-- train_logistic.py
|   |-- train_xgboost.py
|   |-- evaluate_models.py
|   |-- external_validation.py
|   |-- leakage_audit.py
|   |-- shap_explainer.py
|   |-- lime_explainer.py
|   |-- counterfactuals.py
|   |-- fairness_metrics.py
|   `-- bias_mitigation.py
|-- tests/
|-- requirements.txt
|-- requirements-dev.txt
|-- pyproject.toml
`-- README.md
```

## Current Limitations

- The dataset is a case-study dataset and appears suitable for academic demonstration, not direct production validation.
- Raw data is not committed, so a fresh clone requires the local dataset file before the full pipeline can be rerun.
- External validation is a separate public-dataset benchmark and is not direct validation of the Dubai-trained model.
- UCI Taiwan external-validation results are not directly comparable to Dubai case-study metrics because the schemas and populations differ.
- The final model is not calibrated as a lending scorecard and does not include WOE/IV binning, PDO, base odds, score scaling, or regulatory scorecard documentation.
- Threshold selection is fixed for the saved fairness report and has not been optimized against business loss, approval-rate, or policy constraints.
- Fairness analysis is observational and group-level; it does not establish causal fairness or legal compliance.
- The saved fairness report focuses on `Gender`; deeper intersectional analysis across age, nationality, city, and employment status remains future work.
- Reject inference is not implemented, so the project does not correct for selection bias from previously approved or rejected applicants.
- Explainability artifacts are useful for model interpretation but are not a substitute for compliant adverse-action reason code generation.
- The Streamlit dashboard is a local demo app, not a hardened production service.
- Model monitoring, drift detection, model registry, access control, audit logging, and deployment automation are not implemented.
- Dashboard screenshot assets referenced by earlier README versions are not currently present in the repository.

## Future Scope

High-priority next steps:
- Extend external validation beyond UCI Taiwan to datasets such as South German Credit.
- Add dataset-adapter logic for more public datasets while keeping external metrics separate from Dubai headline results.
- Calibrate predicted probabilities and evaluate Brier score, calibration curves, expected calibration error, and threshold stability.
- Build a true scorecard track with binning, WOE/IV, PDO, base odds, points allocation, reason codes, and policy cutoffs.
- Design threshold strategies using business costs, approval-rate constraints, and fairness guardrails.

Fairness and governance extensions:
- Add intersectional fairness analysis across `Gender`, age bands, nationality, city, and employment status where sample sizes are adequate.
- Explore causal fairness methods and document assumptions clearly.
- Add HMDA-style fair-lending evaluation where relevant data is available.
- Develop adverse-action reason code logic aligned with explainability outputs and domain governance.
- Add a model monitoring plan covering drift, stability, fairness, and calibration over time.

Engineering extensions:
- Add a hosted dashboard deployment option.
- Add dashboard screenshot generation and commit stable images under `assets/dashboard/`.
- Add model versioning and artifact metadata.
- Add CI jobs that run lightweight pipeline smoke tests with synthetic fixtures.
- Package the pipeline as a reusable command-line workflow with clearer dataset configuration.

## Interview Defense

- The near-perfect full-feature XGBoost model was rejected because its performance depended on post-loan behavioral information that would not exist at loan application time.
- The lower application-time ROC-AUC is the honest underwriting result because it uses only fields available when a loan decision would be made.
- Behavioral features can be useful for portfolio monitoring, but using them for origination creates hindsight leakage.
- Fairness mitigation can reduce model performance because it changes training weights or decision rules.
- Production readiness would require real-world validation, calibration, threshold governance, adverse-action review, security review, model monitoring, legal review, reject inference strategy, and deployment controls.

## Final Framing

This project should be presented as a responsible AI credit risk workflow, not as a production banking deployment. The core contribution is the disciplined combination of leakage audit, application-time model selection, explainability, fairness analysis, mitigation, and dashboard communication in one coherent portfolio project.
