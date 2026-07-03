# Leakage Audit Report

## Conclusion

No detected target leakage or train/test overlap in the public UCI pipeline based on implemented checks.

This does not mean leakage is impossible; it means the implemented checks did not find target leakage, ID leakage, duplicate train/test rows, or train/test overlap.

## UCI Feature-Timing Review

`PAY_0` to `PAY_6` are historical repayment-status variables used to predict the next-month default target. They are treated as valid historical predictors for this modeling question, not post-outcome leakage.

## Implemented Checks

- Target excluded from model features: `True`
- ID columns excluded from model features: `True`
- Source-index overlap: `0`
- Duplicate selected rows across train/test: `0`
- Repeated feature signatures excluding target across train/test: `10`
- Target-shuffle ROC-AUC: `0.4922`

## Top Mutual Information Features

| Feature | Mutual information |
| --- | ---: |
| PAY_0 | 0.075529 |
| RecentPaymentDelay | 0.075491 |
| NumDelayedMonths | 0.071421 |
| MaxPaymentDelay | 0.070565 |
| PAY_2 | 0.049999 |
| PAY_3 | 0.036233 |
| PAY_4 | 0.029670 |
| PAY_5 | 0.028025 |
| PAY_6 | 0.025874 |
| AvgPaymentAmount | 0.022745 |

High mutual information is reviewed as a suspiciousness signal, not automatic proof of leakage. In this dataset, strong repayment-history signals are expected because they summarize recent historical payment behavior before the next-month target.
