# Group Error Analysis by SEX

Verified UCI protected-attribute mapping: `SEX=1` is Male and `SEX=2` is Female.

Positive class = default / high-risk flag.

- False positive: an actual non-defaulter is flagged high risk. This can unnecessarily route reliable customers into manual review or lower credit support.
- False negative: an actual defaulter is not flagged high risk. This can miss actual default risk and increase bank/NBFC loss.

Different error patterns across groups are a fairness-governance concern. They are not proof of legal discrimination.

Baseline threshold 0.50: Male applicants had slightly higher false-positive rate (`0.0542` vs `0.0479`), while Female applicants had slightly higher false-negative rate (`0.6684` vs `0.6459`).

Recall threshold 0.25: Male applicants had higher false-positive rate (`0.2094` vs `0.1626`), while Female applicants had higher false-negative rate (`0.4505` vs `0.3782`).

These are different error harms: higher FPR affects actual non-defaulters through more high-risk flags, while higher FNR affects lender risk exposure by missing more actual defaulters.

## Primary XGBoost policies

| policy | sex_group | sex_code | group | n | true_positives | false_positives | true_negatives | false_negatives | precision | recall | false_positive_rate | false_negative_rate | negative_predictive_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost_baseline_threshold_050 | Male | 1 | Male (SEX=1) | 2351 | 205 | 96 | 1676 | 374 | 0.6810631229235881 | 0.3540587219343696 | 0.05417607223476298 | 0.6459412780656304 | 0.817560975609756 |
| xgboost_baseline_threshold_050 | Female | 2 | Female (SEX=2) | 3651 | 248 | 139 | 2764 | 500 | 0.6408268733850129 | 0.3315508021390374 | 0.0478815018945918 | 0.6684491978609626 | 0.8468137254901961 |
| xgboost_recall_threshold_025 | Male | 1 | Male (SEX=1) | 2351 | 360 | 371 | 1401 | 219 | 0.49247606019151846 | 0.6217616580310881 | 0.20936794582392776 | 0.37823834196891193 | 0.8648148148148148 |
| xgboost_recall_threshold_025 | Female | 2 | Female (SEX=2) | 3651 | 411 | 472 | 2431 | 337 | 0.4654586636466591 | 0.5494652406417112 | 0.1625904236996211 | 0.4505347593582888 | 0.8782514450867052 |
