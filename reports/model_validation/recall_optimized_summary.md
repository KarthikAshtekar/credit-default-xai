# Recall-Optimized Summary

ROC-AUC measures ranking over both classes. PR-AUC is especially useful when default is the minority class. Recall-focused screening should inspect the precision-recall tradeoff, not only ROC-AUC.

| Candidate | Threshold | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC | PR-AUC | Approval-support rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current_baseline_threshold_050 | 0.50 | 0.8152 | 0.6584 | 0.3414 | 0.4496 | 0.3778 | 0.7748 | 0.5415 | 0.8854 |
| xgboost_public_baseline_threshold_050 | 0.25 | 0.7669 | 0.4777 | 0.5810 | 0.5243 | 0.5569 | 0.7748 | 0.5415 | 0.7311 |
| xgboost_public_weighted_spw_2_0 | 0.40 | 0.7699 | 0.4830 | 0.5780 | 0.5262 | 0.5561 | 0.7732 | 0.5426 | 0.7354 |

Recommended wording: For risk-screening use, the recall-optimized threshold/model captures more actual defaulters at the cost of more false positives. The original 0.50 threshold remains a conservative baseline, while the recall-optimized policy is more appropriate for manual-review triage.

SMOTE note: imbalanced-learn is not installed; SMOTE experiment skipped.
