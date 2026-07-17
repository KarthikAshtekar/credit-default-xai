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

## 12. Did your model discriminate against male or female applicants?

No. I did not conclude legal discrimination or causal bias. What I found was a diagnostic fairness-governance signal. Male applicants had higher high-risk flag rates, especially after lowering the threshold for recall. Female applicants had higher false negative rates. These are group-level error and outcome differences, not proof of legal discrimination.

## 13. Which group was more affected by false positives?

Male applicants had higher false positive rates. Under the recall-focused threshold, male FPR was `0.2094` versus female FPR `0.1626`. In credit-risk terms, this means more actual non-defaulters in the male group were flagged high-risk.

## 14. Which group was more affected by false negatives?

Female applicants had higher false negative rates. Under the recall-focused threshold, female FNR was `0.4505` versus male FNR `0.3782`. In credit-risk terms, this means more actual defaulters in the female group were missed.

## 15. Why is removing SEX not enough?

Because non-sensitive variables can still encode protected-group information. In the proxy audit, `SEX`/gender was moderately predictable from non-sensitive credit variables, with ROC-AUC `0.6476`. So excluding `SEX` prevents direct use, but it does not eliminate proxy risk.

## 16. Did flipping SEX change predictions?

No. The individual sensitivity test showed zero probability change and zero decision changes after flipping `SEX` alone. This verifies that `SEX` is not directly used in the active XGBoost prediction path. However, it does not rule out indirect proxy effects.

## 17. What is the most important fairness lesson?

Fairness is not only a model-training issue. It is also affected by the operating threshold and business policy. Lowering the threshold improved recall but widened some group-level fairness metrics, so threshold selection needs governance review.

## 18. What does the dashboard do?

The dashboard lets a user enter applicant details, generate an XGBoost-based default probability, view a risk band, see whether manual review is recommended, estimate model-supported advisable credit exposure, inspect risk drivers, simulate improvement scenarios, download an applicant report, and review governance artifacts.

## 19. Is this production-ready?

No. It is an educational and portfolio workflow. It lacks production monitoring, calibration governance, adverse-action compliance, reject inference, live data validation, access controls, and formal model risk management.

## 20. How is the report generated: model-based or rule-based?

The default probability is model-based. Risk bands and manual-review flags are threshold-based. Maximum advisable credit exposure is estimated through scenario simulation. Shortcomings and recommendations are generated using SHAP/rule-based mappings, making the report reproducible rather than black-box text generation.

## 21. What would you improve next?

I would add probability calibration, threshold governance, a scorecard track with WOE/IV/binning/PDO/base odds, intersectional fairness diagnostics where sample sizes support them, drift monitoring, and validation on additional public credit datasets.
