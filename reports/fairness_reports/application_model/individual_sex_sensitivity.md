# Individual SEX Sensitivity Diagnostic

Verified UCI protected-attribute mapping: `SEX=1` is Male and `SEX=2` is Female.

This diagnostic flips `SEX` only, keeps all other features fixed, and compares XGBoost probabilities.

This is not a causal counterfactual fairness test. It is a sensitivity diagnostic. If `SEX` is excluded and predictions do not change when `SEX` is flipped, that only rules out direct use in the prediction path. It does not rule out proxy effects through other variables.

Maximum absolute probability change: `0.00000000`.
Baseline-threshold decision changes: `0`.
Recall-threshold decision changes: `0`.

Interpretation: SEX/gender is not directly used in the active XGBoost prediction path, but proxy effects through non-sensitive variables remain possible.
