# Final Artifact Index

| Artifact | Purpose | Contents | Audience |
| --- | --- | --- | --- |
| `README.md` | Fast project overview | Summary, final metrics, dashboard journey, commands, responsible AI framing | Recruiter-facing |
| `project_end_to_end.html` | Standalone project explainer | Dataset, features, preprocessing, leakage, explainability, business logic, models, metrics, tuning, results, dashboard roadmap, future scope | Outsider-facing |
| `reports/final_project_report.md` | Full project narrative | Dataset, preprocessing, leakage controls, models, evaluation, explainability, fairness, dashboard, limitations | Technical-facing |
| `docs/model_card.md` | Governance summary | Intended use, non-intended use, metrics, fairness, explainability, leakage controls, limitations, oversight | Governance-facing |
| `docs/interview_defense.md` | Interview preparation | Crisp answers for dataset, model choice, recall tuning, DNN, leakage, explainability, fairness, dashboard, production readiness | Recruiter-facing |
| `docs/cv_bullets.md` | Resume/portfolio wording | Two-, three-, and four-bullet versions plus short project description | Recruiter-facing |
| `reports/model_validation/` | Model validation evidence | Model metrics, threshold tuning, recall policy, prediction exports, ML-vs-DL comparison, PR curves | Technical-facing |
| `reports/leakage_audit/` | Leakage-control evidence | Leakage audit report and summary JSON, including target-shuffle sanity result | Governance-facing |
| `reports/fairness_reports/` | Fairness diagnostics | XGBoost fairness metrics, threshold fairness comparison, mitigation tradeoffs, DNN fairness diagnostics, protected-attribute fairness deep dive on `SEX`/gender | Governance-facing |
| `reports/explainability_reports/` | Explanation artifacts | SHAP, LIME, counterfactuals, and DNN permutation importance | Technical-facing |
| `dashboard/app.py` | Local dashboard | Applicant report, improvement guidance, model governance views, report download | Recruiter-facing |

The final project framing is public UCI Taiwan credit-card default data, XGBoost as the primary model, recall-optimized thresholding for manual-review screening, DNN as a benchmark, and dashboard outputs as decision support only.

## Fairness Deep-Dive Artifacts

These governance-facing artifacts use the verified UCI mapping `SEX=1` = Male and `SEX=2` = Female. They are diagnostic fairness-governance evidence, not legal conclusions, and they preserve numeric `sex_code` fields for reproducibility.

| Artifact | Purpose | Contents | Audience |
| --- | --- | --- | --- |
| `src/protected_attributes.py` | Centralized protected-attribute mapping | Defines `SEX_GROUP_LABELS = {1: "Male", 2: "Female"}` and readable display labels | Technical-facing |
| `src/fairness_deep_dive.py` | Reproducible fairness-governance CLI | Generates protected-attribute reports for `SEX`/gender without changing the final model or applicant workflow | Technical-facing |
| `reports/fairness_reports/application_model/fairness_deep_dive_summary.md` | Executive fairness summary | Scope, protected attribute rationale, core findings, governance recommendations, limitations | Governance-facing |
| `reports/fairness_reports/application_model/group_outcome_analysis_by_sex.csv` | Group outcome audit | Actual default rate, predicted risk, high-risk flag rate, approval-support rate by Male (SEX=1) and Female (SEX=2) and policy | Governance-facing |
| `reports/fairness_reports/application_model/group_error_analysis_by_sex.csv` | Error-rate audit | TP, FP, TN, FN, recall, FPR, FNR, PPV, NPV by Male (SEX=1) and Female (SEX=2) and policy | Governance-facing |
| `reports/fairness_reports/application_model/group_calibration_by_sex.csv` | Calibration check | Probability-bin calibration gaps by Male (SEX=1) and Female (SEX=2) | Governance-facing |
| `reports/fairness_reports/application_model/proxy_sex_predictability.csv` | Proxy-risk analysis | Predictability of `SEX`/gender from non-sensitive variables | Governance-facing |
| `reports/fairness_reports/application_model/feature_association_with_sex.csv` | Feature association diagnostics | Group means, medians, standardized mean differences, correlations, mutual information by verified group labels | Technical-facing |
| `reports/fairness_reports/application_model/shap_driver_comparison_by_sex.csv` | SHAP driver comparison | Mean absolute SHAP by Male (SEX=1) and Female (SEX=2) group for XGBoost | Technical-facing |
| `reports/fairness_reports/application_model/threshold_fairness_frontier.csv` | Threshold governance | Performance and fairness metrics across XGBoost thresholds `0.10` to `0.70` | Governance-facing |
| `reports/fairness_reports/application_model/individual_sex_sensitivity.csv` | Direct-use sensitivity | Original vs SEX-flipped XGBoost probabilities and decision changes, with readable Male/Female labels | Governance-facing |
| `reports/fairness_reports/application_model/nearest_neighbour_individual_fairness.csv` | Individual fairness diagnostic | Cross-group nearest-neighbour score differences | Technical-facing |
