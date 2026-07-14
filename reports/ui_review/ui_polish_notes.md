# Streamlit Dashboard UI Polish Notes

Scope: UI/UX-only polish for the Streamlit dashboard. Model training, model-selection logic, and final model choice were not changed.

## Iteration 1 - pre-polish baseline

Screenshots:

- `01_applicant_report_initial.png`
- `02_improvement_guidance_initial.png`
- `03_model_governance_initial.png`
- `04_applicant_report_filled_initial.png`

Findings:

- Dashboard used the default light visual treatment and felt less polished than the project report quality.
- Applicant-facing labels exposed raw UCI feature names such as monthly payment and bill columns.
- Guidance and governance sections were too sparse or too technical at first glance.
- Static governance images existed, but the first screen did not provide concise reviewer-oriented metrics.

## Iteration 2 - dark-mode structure and simplified journey

Screenshots:

- `01_applicant_report_iteration1.png`
- `02_improvement_guidance_iteration1.png`
- `03_model_governance_iteration1.png`
- `04_applicant_report_filled_iteration1.png`

Changes:

- Added dark theme configuration and shared Streamlit CSS helpers.
- Simplified the dashboard to three tabs: Applicant Report, Improvement Guidance, and Model Governance.
- Kept XGBoost as the applicant-facing model and moved DNN language to governance-only.
- Added compact cards for default risk, risk band, manual review signal, and maximum advisable credit exposure.
- Added Plotly chart builders for threshold tradeoffs, model comparison, fairness diagnostics, PR curves, and scenario curves.

Follow-up issues:

- Governance card contrast needed tightening.
- The generated applicant screenshot was not landing on the report output area.
- The screenshot script was selecting the sidebar combobox instead of the applicant profile selector.

## Iteration 3 - output-state capture and contrast pass

Screenshots:

- `01_applicant_report_iteration2.png`
- `02_improvement_guidance_iteration2.png`
- `03_model_governance_iteration2.png`
- `04_applicant_report_filled_iteration2.png`
- `01_applicant_report_iteration3.png`
- `02_improvement_guidance_iteration3.png`
- `03_model_governance_iteration3.png`
- `04_applicant_report_filled_iteration3.png`

Changes:

- Strengthened dark-mode contrast for headings, metric cards, expander labels, and muted text.
- Updated screenshot capture to scroll to generated output sections.
- Updated applicant-profile selection in Playwright to target the Demo applicant profile control.
- Verified the high-delay applicant path shows High Risk, Manual review recommended, and scenario-based maximum advisable credit exposure.

Follow-up issue:

- The guidance tab expanded too much content before the scenario section, making the what-if controls less discoverable.

## Iteration 4 - final review pass

Screenshots:

- `01_applicant_report_final.png`
- `02_improvement_guidance_final.png`
- `03_model_governance_final.png`
- `04_applicant_report_filled_final.png`
- `05_improvement_scenario_final.png`
- `01_applicant_report_final_1440.png`
- `02_improvement_guidance_final_1440.png`
- `03_model_governance_final_1440.png`
- `04_applicant_report_filled_final_1440.png`
- `05_improvement_scenario_final_1440.png`
- `01_applicant_report_final_1920.png`
- `02_improvement_guidance_final_1920.png`
- `03_model_governance_final_1920.png`
- `04_applicant_report_filled_final_1920.png`
- `05_improvement_scenario_final_1920.png`

Final state:

- The first screen is dark, compact, and applicant-facing.
- Advanced UCI fields remain available for audit demos but are hidden by default.
- Improvement Guidance shows shortcomings first and includes a separate scenario screenshot for risk and advisable-exposure changes.
- Governance starts with concise reviewer metrics and keeps detailed artifacts in expanders.
- Terminology uses decision-support language rather than approval or rejection language.
