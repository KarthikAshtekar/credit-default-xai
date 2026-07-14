# Explainable and Fair Credit Default Risk Prediction

This project builds an explainable credit-card default risk workflow on the public UCI Taiwan credit-card default dataset. It uses XGBoost as the primary model, validation-only recall threshold tuning for manual-review screening, SHAP/LIME explanations, fairness diagnostics on `SEX`, a TensorFlow/Keras DNN benchmark, and a Streamlit dashboard for applicant-level decision support.

The project is for educational and portfolio use. It is not a production credit approval engine, not a regulatory scorecard, and not a substitute for underwriting governance or legal review.

## Why This Matters

Banks and NBFCs need to reduce default losses while avoiding unnecessary rejection of reliable customers. A useful credit-risk workflow must therefore look beyond accuracy: it should detect more actual defaults, explain the drivers behind model scores, audit fairness tradeoffs, and keep human review in the loop.

## Dataset

- Dataset: UCI Default of Credit Card Clients / Taiwan credit-card default
- Source: public UCI dataset ID `350`
- Loader: `ucimlrepo`
- Rows: `30,000`
- Target: `Default_Flag`, where `1` means next-month default
- Final active feature set: `application_public`
- Protected attribute policy: `SEX` is excluded from active model training and retained for fairness analysis

The UCI dataset covers borrower profile, repayment history, credit exposure, bill/payment behavior, and the default target. Engineered features include bill-to-limit ratios, payment-to-bill ratios, recent repayment delay, maximum delay, delayed-month count, average bill amount, average payment amount, and payment-to-limit ratio.

## Final Model Snapshot

Final primary model: `models/xgboost_public.pkl`

| Policy | Threshold | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoost baseline | 0.50 | 0.8152 | 0.6584 | 0.3414 | 0.4496 | 0.3778 | 0.7748 | 0.5415 |
| XGBoost recall policy | 0.25 | 0.7669 | 0.4777 | 0.5810 | 0.5243 | 0.5569 | 0.7748 | 0.5415 |

The recall policy keeps the same XGBoost model and lowers the screening threshold to `0.25`. The threshold was selected on validation data only by maximizing recall subject to validation precision >= `0.50`, then evaluated once on the held-out test split.

## ML vs DL Conclusion

A TensorFlow/Keras MLP benchmark was added as a controlled comparison, not as a forced replacement.

| Model / policy | Threshold | Accuracy | Precision | Recall | F2 | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DNN baseline | 0.50 | 0.8131 | 0.6426 | 0.3482 | 0.3833 | 0.7657 | 0.5212 |
| DNN class-weighted | 0.50 | 0.7446 | 0.4429 | 0.6021 | 0.5617 | 0.7664 | 0.5363 |
| DNN recall policy | 0.30 | 0.7857 | 0.5149 | 0.5350 | 0.5309 | 0.7657 | 0.5212 |

XGBoost remains primary because it has stronger ROC-AUC, PR-AUC, and recall-policy performance. The DNN is retained as evidence that additional model complexity did not materially improve the business objective on structured tabular credit data.

## Dashboard

Run:

```bash
streamlit run dashboard/app.py
```

The dashboard is decision-support only. It does not guarantee loan approval, rejection, or regulatory compliance.

Main user journey:

- Enter UCI-style applicant details.
- Generate a model-based default probability using XGBoost.
- See the risk band and manual-review flag.
- Estimate a model-supported maximum advisable credit exposure through scenario simulation.
- Review shortcomings through local SHAP drivers and plain-English explanations.
- Simulate improvement scenarios such as timelier repayment, lower utilization, and stronger repayment amounts.
- Download the applicant report when a prediction has been generated.
- Review governance artifacts such as validation metrics, leakage checks, fairness diagnostics, and the DNN benchmark.

## Responsible AI

- Leakage audit: no detected leakage or train/test overlap based on implemented checks; target-shuffle ROC-AUC is `0.4922`.
- Explainability: SHAP, LIME, counterfactual guidance, and DNN permutation importance.
- Fairness: diagnostic metrics on `SEX`, including demographic parity difference, equal opportunity difference, equalized odds difference, and disparate impact ratio.
- Threshold governance: recall improvements are reported with precision, approval-support, and fairness tradeoffs.
- Human oversight: outputs are framed as manual-review support, not automated lending decisions.

## Key Artifacts

- `reports/final_project_report.md`
- `docs/model_card.md`
- `docs/interview_defense.md`
- `docs/cv_bullets.md`
- `reports/artifact_index.md`
- `reports/model_validation/`
- `reports/leakage_audit/`
- `reports/fairness_reports/`
- `reports/explainability_reports/`
- `dashboard/app.py`

## Reproduce

Create or activate the local environment:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Run the full pipeline:

```bash
python -m src.run_pipeline
```

Run recall optimization:

```bash
python -m src.recall_optimization
```

Run the DNN benchmark:

```bash
python -m src.deep_learning_benchmark
```

Run the dashboard:

```bash
streamlit run dashboard/app.py
```

Run validation checks:

```bash
python -m compileall src dashboard
pytest
ruff check .
ruff format --check .
git diff --check
```

## Screenshots

Dashboard screenshots are intentionally not linked unless image files are present in the repository. To capture fresh screenshots, run the Streamlit app and save the Applicant Report, Improvement Guidance, and Model Governance views under a tracked screenshots folder.

## Future Scope

- Add probability calibration and threshold monitoring.
- Add true time-based validation if a dataset with application timestamps is introduced.
- Build a scorecard track with WOE/IV/binning/PDO/base odds.
- Add intersectional fairness diagnostics where sample sizes support them.
- Add drift monitoring and production governance controls before any real deployment.
