# Threshold Selection Summary

Preferred rule: `maximize_recall_precision_050`.
Fallback rule: `maximize_f2` if no threshold satisfies precision >= 0.50.

Selected policy: `xgboost_public_baseline_threshold_050` at threshold `0.25`.
Selection rule used: `maximize_recall_precision_050`.
Fallback used: `False`.

All selection decisions are based on validation data only. Final test metrics are evaluated after selection on the untouched held-out test split.
