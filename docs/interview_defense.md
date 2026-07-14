# Interview Defense

## 1. Why this dataset?

I used the public UCI Taiwan credit-card default dataset because it is reproducible, widely known, and contains borrower profile, repayment history, bill amounts, payment amounts, credit exposure, and a clear next-month default target. It avoids dependence on a private local file and makes the project easier for reviewers to rerun.

## 2. Why XGBoost?

XGBoost is strong on structured tabular data, handles nonlinear interactions well, and works cleanly with SHAP explanations. For this dataset, it outperformed the DNN benchmark on ROC-AUC, PR-AUC, and recall-policy performance, so it remains the final model.

## 3. Why not accuracy only?

Accuracy hides minority-class performance. In credit default prediction, missing actual defaulters is costly, so I evaluated recall, precision, F1, F2, ROC-AUC, PR-AUC, confusion counts, and approval-support tradeoffs.

## 4. Why recall optimization?

The baseline XGBoost had good accuracy but low default-class recall at the default `0.50` threshold. Recall optimization makes the model more useful for screening by capturing more actual defaults, while explicitly reporting the cost in precision and manual-review volume.

## 5. Why threshold 0.25?

The `0.25` threshold was selected on validation data, not test data. The rule was to maximize validation recall subject to validation precision >= `0.50`. After selection, the policy was evaluated once on the held-out test set.

## 6. Why add Deep Learning?

The DNN was added as a controlled benchmark, not a forced replacement. Since credit data is structured tabular data, I expected XGBoost to be strong. The DNN tested whether additional model complexity improves recall, PR-AUC, or ranking quality enough to justify added opacity.

## 7. Why did the DNN not become final?

The DNN did not materially outperform XGBoost. XGBoost had better ROC-AUC (`0.7748` vs `0.7657`), better PR-AUC (`0.5415` vs `0.5212`), and better recall-policy performance (`0.5810` recall vs `0.5350`). Because the DNN added complexity without better business performance, XGBoost remains final.

## 8. How did you avoid leakage?

I excluded the target and ID-style fields from features, kept `SEX` out of active model training, used train/test split controls, checked source-index overlap, checked duplicate selected rows, reviewed mutual information, and ran a target-shuffle sanity test. The result was no detected leakage based on implemented checks.

## 9. How did you use SHAP and LIME?

SHAP is used for global and local XGBoost explanations, showing which features move risk up or down. LIME provides a second local explanation view. In the dashboard, local SHAP drivers are converted into plain-English risk reasons and improvement guidance.

## 10. What fairness metrics did you use?

I used demographic parity difference, equal opportunity difference, equalized odds difference, and disparate impact ratio on `SEX`. The favorable outcome is predicted non-default / lower-risk manual-review support. These are diagnostic metrics, not proof of legal compliance.

## 11. Why does threshold affect fairness?

Changing the threshold changes who receives a favorable predicted outcome. A lower threshold catches more likely defaults, but it can also shift false positives differently across groups. That is why the project reports threshold-performance and threshold-fairness tradeoffs together.

## 12. What does the dashboard do?

The dashboard lets a user enter applicant details, generate an XGBoost-based default probability, view a risk band, see whether manual review is recommended, estimate model-supported advisable credit exposure, inspect risk drivers, simulate improvement scenarios, download an applicant report, and review governance artifacts.

## 13. Is this production-ready?

No. It is an educational and portfolio workflow. It lacks production monitoring, calibration governance, adverse-action compliance, reject inference, live data validation, access controls, and formal model risk management.

## 14. How is the report generated: model-based or rule-based?

The default probability is model-based. Risk bands and manual-review flags are threshold-based. Maximum advisable credit exposure is estimated through scenario simulation. Shortcomings and recommendations are generated using SHAP/rule-based mappings, making the report reproducible rather than black-box text generation.

## 15. What would you improve next?

I would add probability calibration, threshold governance, a scorecard track with WOE/IV/binning/PDO/base odds, intersectional fairness diagnostics where sample sizes support them, drift monitoring, and validation on additional public credit datasets.
