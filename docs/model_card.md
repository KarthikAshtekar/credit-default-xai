# Model Card: Application-Time Credit Default XGBoost

## Intended Use
This model supports a portfolio/interview demonstration of responsible AI credit risk modeling. It estimates default risk from application-time borrower, affordability, bureau, loan, and macroeconomic fields.

## Excluded Use
This is not a production lending decision engine, not a regulatory credit scorecard, and not a substitute for underwriting governance, legal review, or adverse-action processes.

## Data And Target
- Dataset: Dubai Arab Bank case-study dataset
- Rows: 10,000
- Target: `Default_Flag`
- Final feature policy: use application-time variables only

## Model Family And Final Choice
The final recommended model is `xgboost_application.pkl`. Logistic regression is retained as a benchmark. Behavioral and full-diagnostic models are retained only as monitoring or leakage-diagnostic artifacts.

## Leakage Audit Decision
The full-feature XGBoost model produced near-perfect metrics, but this result was rejected. The audit found no direct target-column leakage or train/test overlap; the issue was feature timing. Post-loan repayment and monitoring fields made the model too informed for an application-time underwriting use case.

## Application-Time Feature Policy
The final applicant UI excludes post-loan fields such as missed payments, salary-drop flags, spending-spike flags, debit behavior, stress signal count, and historical risk score. EMI and burden ratios are computed internally from application inputs.

## Fairness And Mitigation
Saved application-model fairness analysis uses `Gender` as the default protected attribute. Metrics include demographic parity difference, equal opportunity difference, equalized odds difference, and disparate impact ratio. Reweighing and Fairlearn post-processing are presented as fairness-performance tradeoff experiments, not guaranteed improvements.

## Limitations
- Academic/portfolio case study
- Case-study or simulated data
- No causal fairness claims
- No reject inference
- No public external validation yet
- No calibrated scorecard with WOE/IV/binning/PDO/base odds

## Ethical Caveats
Credit models can affect access to financial opportunity. This project is framed as a transparency and governance workflow, not as proof that the model is fair, lawful, or production-ready.

## Reproducibility
Place the case-study dataset in `data/raw/`, install dependencies, and run:

```bash
python -m src.run_pipeline
```

Then launch the dashboard:

```bash
streamlit run dashboard/app.py
```

## Interview Defense Notes
- Rejecting the near-perfect model is a strength because it shows feature-timing discipline.
- Honest application-time ROC-AUC is better than inflated hindsight performance.
- Behavioral features belong in monitoring or collections contexts, not origination unless they are truly available at application time.
- Fairness mitigation may reduce ROC-AUC because it changes optimization or thresholding behavior.
- Production readiness would require governance, calibration, model monitoring, adverse-action compliance, security review, and real-world validation.
