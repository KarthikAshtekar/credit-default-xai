# Interview Defense

## 30-Second Pitch

This project builds an explainable and fair credit-card default risk workflow using the public UCI Taiwan credit-card default dataset. It trains a logistic benchmark and an XGBoost model, audits leakage, generates SHAP/LIME/counterfactual artifacts, evaluates group fairness on `SEX`, and presents the results in a Streamlit dashboard.

## Technical Walkthrough

The primary dataset is UCI Default of Credit Card Clients, fetched through `ucimlrepo` with dataset ID `350`. The adapter normalizes the target to `Default_Flag`, where `1` means next-month default, and creates utilization and repayment features such as `AvgBillToLimitRatio`, `AvgPaymentToBillRatio`, `RecentPaymentDelay`, `MaxPaymentDelay`, and `NumDelayedMonths`.

The final feature set is `application_public`. It excludes `SEX` from active training while retaining it for fairness auditing. `AGE`, `MARRIAGE`, and `EDUCATION` are included as profile variables with the policy documented.

The final XGBoost model achieved `0.7748` ROC-AUC on a held-out stratified split. Logistic regression remains the benchmark at `0.7527` ROC-AUC.

## Recall Optimization Defense

The baseline XGBoost achieved ROC-AUC 0.7748 and accuracy 0.8152, but default-class recall at the 0.50 threshold was 0.3414. Since missed defaults are costly in credit risk, I added validation-based threshold tuning, F2-score, class-weight tuning, PR-AUC, and SMOTE experiments. The selected recall policy improves default capture while explicitly reporting the tradeoff in precision, approval-support rate, and fairness metrics.

For the selected policy, the model remains XGBoost but the screening threshold moves from `0.50` to `0.25`. Held-out recall increases from `0.3414` to `0.5810`, while precision moves from `0.6584` to `0.4777` and approval-support rate moves from `0.8854` to `0.7311`. The threshold was selected on validation data only and then evaluated once on the untouched test split.

## Leakage Defense

The audit checks target exclusion, ID exclusion, duplicate selected-row overlap, source-index overlap, target shuffle, mutual information, and feature timing. The result is:

> No detected target leakage or train/test overlap in the public UCI pipeline based on implemented checks.

`PAY_0` to `PAY_6` are historical repayment-status variables before the next-month default target. They are strong predictors, but they are not treated as post-outcome leakage for this prediction question.

## Explainability

SHAP provides global and local model drivers. LIME provides an additional local explanation. Counterfactual guidance is framed around reducing repayment delays, lowering bill-to-limit utilization, increasing repayment relative to bill amount, and maintaining timely recent repayment.

## Fairness

Fairness is evaluated on favorable outcomes, defined as predicted non-default / low-risk approval decisions. The primary protected attribute is `SEX`. Metrics include demographic parity difference, equal opportunity difference, equalized odds difference, and disparate impact ratio.

The threshold comparison is also audited. At the baseline `0.50` threshold, demographic parity difference is `0.0220` and disparate impact ratio is `0.9754`. At the recall policy threshold `0.25`, demographic parity difference is `0.0691` and disparate impact ratio is `0.9089`, so the recall improvement is reported with a fairness tradeoff instead of claimed as a free improvement.

## Limitations

- The dataset has no true application timestamp, so temporal validation is not fabricated.
- The model is not a calibrated regulatory scorecard.
- Fairness metrics are observational and group-level.
- No causal fairness, reject inference, production monitoring, adverse-action compliance, or deployment controls are implemented.

## Future Work

- Add South German Credit, Bondora, and Home Credit as future-scope datasets.
- Add probability calibration and threshold governance.
- Build a scorecard track with WOE/IV/binning/PDO/base odds.
- Add intersectional fairness where sample sizes are adequate.
