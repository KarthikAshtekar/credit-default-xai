# Final Artifact Index

| Artifact | Purpose | Contents | Audience |
| --- | --- | --- | --- |
| `README.md` | Fast project overview | Summary, final metrics, dashboard journey, commands, responsible AI framing | Recruiter-facing |
| `reports/final_project_report.md` | Full project narrative | Dataset, preprocessing, leakage controls, models, evaluation, explainability, fairness, dashboard, limitations | Technical-facing |
| `docs/model_card.md` | Governance summary | Intended use, non-intended use, metrics, fairness, explainability, leakage controls, limitations, oversight | Governance-facing |
| `docs/interview_defense.md` | Interview preparation | Crisp answers for dataset, model choice, recall tuning, DNN, leakage, explainability, fairness, dashboard, production readiness | Recruiter-facing |
| `docs/cv_bullets.md` | Resume/portfolio wording | Two-, three-, and four-bullet versions plus short project description | Recruiter-facing |
| `reports/model_validation/` | Model validation evidence | Model metrics, threshold tuning, recall policy, prediction exports, ML-vs-DL comparison, PR curves | Technical-facing |
| `reports/leakage_audit/` | Leakage-control evidence | Leakage audit report and summary JSON, including target-shuffle sanity result | Governance-facing |
| `reports/fairness_reports/` | Fairness diagnostics | XGBoost fairness metrics, threshold fairness comparison, mitigation tradeoffs, DNN fairness diagnostics | Governance-facing |
| `reports/explainability_reports/` | Explanation artifacts | SHAP, LIME, counterfactuals, and DNN permutation importance | Technical-facing |
| `dashboard/app.py` | Local dashboard | Applicant report, improvement guidance, model governance views, report download | Recruiter-facing |

The final project framing is public UCI Taiwan credit-card default data, XGBoost as the primary model, recall-optimized thresholding for manual-review screening, DNN as a benchmark, and dashboard outputs as decision support only.
