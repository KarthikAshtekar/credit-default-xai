# Final Project Report

## 1. Abstract
This project develops a responsible-AI credit default risk workflow centered on **application-time prediction** rather than hindsight-driven monitoring. After a leakage audit revealed that near-perfect XGBoost results were driven by post-loan behavioral features, the final model was locked to `xgboost_application.pkl`, which achieved **0.7105 accuracy** and **0.7825 ROC-AUC** on application-time features, with **0.7841 ROC-AUC** under temporal validation. The project combines clean validation, SHAP, LIME, counterfactual explanations, fairness analysis, bias mitigation, and a Streamlit dashboard into a single portfolio-ready case study.

## 2. Business Context
Credit risk models influence lending decisions, financial inclusion, and portfolio losses. For a model to be useful in underwriting, it must do more than rank applicants well. It must also support explanation and governance. This project therefore treats predictive performance, interpretability, and fairness as linked requirements instead of isolated add-ons.

## 3. Dataset
- Source: Dubai Arab Bank case-study dataset
- Rows: 10,000
- Original columns: 28
- Target: `Default_Flag`

The dataset mixes borrower profile, financial profile, loan terms, macroeconomic variables, and post-loan behavioral fields. That made it useful for both modeling and leakage diagnosis, but it also required strict feature-timing discipline.

Three feature framings were used:
- `application`: borrower profile, financial profile, bureau score, loan terms, and macro variables available at loan start
- `behavioral`: application variables plus repayment and account-behavior signals
- `full_diagnostic`: the broad mixed feature set kept only for diagnostic comparison

## 4. Leakage Audit
The original full-feature XGBoost result was nearly perfect and was treated as suspicious.

Audit findings:
- `Default_Flag` was not included as a feature
- `CustomerID` and `LoanID` were removed
- No train/test overlap or duplicate IDs were found
- The hindsight-style `LoanAgeDays` feature was removed
- Target shuffle dropped to **0.5262 ROC-AUC**, arguing against direct split leakage
- The primary issue was feature timing: post-loan behavioral signals created hindsight leakage for an application-time underwriting use case

Examples of behavioral or post-loan monitoring features excluded from the final application model:
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

## 5. Modeling
### Application-Time Models
- `logistic_application`: accuracy `0.6925`, ROC-AUC `0.7639`
- `xgboost_application`: accuracy `0.7105`, ROC-AUC `0.7825`

### Diagnostic-Only Models
- `logistic_behavioral`: accuracy `0.8635`, ROC-AUC `0.9476`
- `xgboost_behavioral`: accuracy `1.0000`, ROC-AUC `1.0000`
- `xgboost_full_diagnostic`: accuracy `0.9995`, ROC-AUC `1.0000`

### Temporal Validation
- `xgboost_application`: ROC-AUC `0.7841`
- `logistic_application`: ROC-AUC `0.7623`

Interpretation:
- `xgboost_application` is the final selected model because it is the best-performing **clean application-time** model
- Behavioral and full-diagnostic XGBoost results are retained as evidence from the leakage audit, not as the final headline result

## 6. Explainability
The final model is supported by:
- SHAP global summary
- SHAP local explanation
- LIME local explanation
- DiCE counterfactual explanation

Key explainability takeaway:
- The application-time model is primarily driven by bureau score and loan-burden related variables rather than post-loan signals

Counterfactual output shows how a high-risk case could move toward approval by improving variables such as:
- `BureauScore`
- burden-related features such as `LoanToAnnualIncome` or related affordability signals

## 7. Fairness Analysis
Fairness analysis was run on the final application model using approval decisions derived from predicted default probability.

Saved fairness metrics for the application model:
- Demographic parity difference: `0.0151`
- Equalized odds difference: `0.0138`
- Equal opportunity difference: `0.0008`
- Disparate impact ratio: `0.9693`

Interpretation:
- Group-level disparities are present but modest in the saved report
- The fairness results are suitable for transparent discussion, not for claiming the model is bias-free

## 8. Bias Mitigation
Two mitigation strategies were evaluated:
- Reweighing
- Fairlearn post-processing

Observed tradeoff:
- Reweighing kept performance close to baseline but did not improve the saved fairness metrics
- Fairlearn post-processing reduced demographic parity difference more strongly but reduced ROC-AUC to `0.6954`

This is the correct project-level conclusion: fairness mitigation is a decision tradeoff, not a guaranteed performance-neutral improvement.

## 9. Dashboard
The Streamlit dashboard presents the final project in a demo-friendly format:
- project overview
- applicant risk prediction
- model performance comparison
- explainability section
- fairness analysis
- counterfactual guidance
- leakage audit summary

Run with:

```bash
streamlit run dashboard/app.py
```

## 10. Limitations
- The dataset appears to be case-study or simulated rather than production banking data
- The project is an academic demonstration, not a deployment-ready scorecard
- Causal fairness methods are not implemented
- External validation on public real-world credit datasets remains future work
- Reject inference, calibration, and policy threshold design are out of scope

## 11. Conclusion
This project demonstrates a responsible AI workflow for credit risk:

`data validation -> leakage audit -> clean modeling -> explainability -> fairness analysis -> mitigation -> dashboard communication`

The most important result is not the diagnostic near-perfect XGBoost score. It is the disciplined decision to reject that result, isolate the leakage framing issue, and present `xgboost_application.pkl` as the final honest underwriting model.
