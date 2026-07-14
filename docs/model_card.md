# Model Card: UCI Public Credit-Card Default XGBoost

## Intended Use

This model supports an academic/portfolio demonstration of responsible AI credit risk modeling. It estimates next-month credit-card default risk using the public UCI Taiwan credit-card default dataset.

## Excluded Use

This is not a production lending decision engine, not a regulatory credit scorecard, and not a substitute for underwriting governance, legal review, calibration, monitoring, or adverse-action processes.

## Data And Target

- Dataset: UCI Default of Credit Card Clients / Taiwan credit-card default
- UCI ID: `350`
- Rows: `30,000`
- Target: `Default_Flag`
- Target meaning: `1` means next-month default / bad outcome
- Loading method: `ucimlrepo`

## Model Family And Final Choice

The final comparison also includes a TensorFlow/Keras MLP benchmark (`64-32-16-1` dense architecture with dropout and batch normalization). The baseline DNN achieved ROC-AUC `0.7657` and PR-AUC `0.5212`; its validation-selected `0.30` threshold achieved test recall `0.5350` at precision `0.5149`. Because these results do not materially improve on XGBoost, the DNN remains an experimental benchmark and is not used for applicant predictions.

The final recommended model is `xgboost_public.pkl`. Logistic regression is retained as a transparent benchmark.

Final held-out metrics for `xgboost_public`:

- Accuracy: `0.8152`
- Precision: `0.6584`
- Recall: `0.3414`
- F1: `0.4496`
- ROC-AUC: `0.7748`

## Recall-Focused Operating Policy

The saved model remains `xgboost_public.pkl`. For manual-review screening, a separate validation-based threshold policy is available:

- Baseline threshold `0.50`: recall `0.3414`, precision `0.6584`, F2 `0.3778`, PR-AUC `0.5415`
- Selected recall threshold `0.25`: recall `0.5810`, precision `0.4777`, F2 `0.5569`, PR-AUC `0.5415`
- Selection rule: maximize validation recall subject to validation precision >= `0.50`
- Fallback rule: maximize F2 if no threshold satisfies the precision floor

This policy improves default capture for screening while lowering accuracy, precision, and approval-support rate. It is not a lending decision rule by itself.

## Feature Policy

The final active feature set is `application_public`.

- `SEX` is excluded from active model training and retained for fairness auditing.
- `AGE`, `MARRIAGE`, and `EDUCATION` are included as profile variables and treated as audit-sensitive.
- `PAY_0` to `PAY_6` are historical repayment-status variables before the next-month target.
- Engineered features summarize utilization, repayment ratios, delayed-month count, and average bill/payment behavior.

## Leakage Audit Decision

No detected target leakage or train/test overlap in the public UCI pipeline based on implemented checks.

The audit does not claim leakage is impossible. It checks target exclusion, ID exclusion, duplicate selected-row overlap, source-index overlap, target shuffle, mutual information, and UCI feature timing.

## Fairness And Mitigation

Primary protected attribute: `SEX`.

Favorable outcome: predicted non-default / low-risk approval decision.

Saved fairness metrics include demographic parity difference, equal opportunity difference, equalized odds difference, and disparate impact ratio. Reweighing and Fairlearn post-processing are reported as fairness-performance tradeoff experiments, not guaranteed improvements.

## Explainability

SHAP, LIME, and counterfactual artifacts are generated for the public UCI XGBoost model. Explanation text focuses on repayment delay, delayed-month count, credit limit, utilization, and repayment relative to bill amount.

## Limitations

- Public academic benchmark dataset, not production bank data
- No true chronological application timestamp for temporal validation
- No reject inference
- No causal fairness claims
- No calibrated scorecard with WOE/IV/binning/PDO/base odds
- Dashboard is a local demonstration app

## Reproducibility

```bash
python -m src.data_api_loader --source uci --dataset_name default_credit_card
python -m src.recall_optimization
python -m src.run_pipeline
streamlit run dashboard/app.py
```
