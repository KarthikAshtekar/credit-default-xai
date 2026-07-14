# Threshold Tuning Report

Thresholds were evaluated on validation data only, from `0.10` to `0.70` in `0.05` steps.

`0.50` is a default classification threshold, not a business law. Lowering the threshold usually increases default-class recall, but lowers precision and approval-support rate because more borrowers are flagged as high risk.

Final threshold choice should use business loss, fairness guardrails, and operational review capacity.

F1 balances precision and recall equally. F2 gives higher weight to recall and is useful when missing actual defaulters is more costly than flagging some good borrowers for review.
