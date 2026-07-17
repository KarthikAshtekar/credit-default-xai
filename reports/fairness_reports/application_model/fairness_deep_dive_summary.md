# Fairness Deep Dive Summary: Protected Attribute SEX

## 1. What was analyzed

This deep dive analyzed group outcomes, group errors, calibration, proxy predictability, feature associations, SHAP drivers, threshold fairness frontier, individual sensitivity, and nearest-neighbour individual fairness for `SEX`.

## 2. Why SEX was used as the protected attribute

`SEX` is available in the public UCI dataset and is excluded from the active XGBoost training feature set while retained for fairness diagnostics. Verified mapping: `SEX=1` is Male and `SEX=2` is Female.

## 3. Baseline XGBoost fairness results

At threshold 0.50, demographic parity difference was `0.0220`, equalized odds difference was `0.0225`, and disparate impact ratio was `0.9754`.

## 4. Recall-threshold fairness tradeoff

At threshold 0.25, demographic parity difference was `0.0691`, equalized odds difference was `0.0723`, and disparate impact ratio was `0.9089`. The recall policy improves default capture but widens several fairness diagnostics.

## 5. Group-wise outcome findings

Group-wise outcome analysis shows differences in actual default rates, mean predicted default probabilities, high-risk flag rates, and low-risk support rates. These are governance diagnostics, not proof of legal discrimination or causal bias.

Male applicants (SEX=1) had higher high-risk flag rates than Female applicants (SEX=2): `0.1280` vs `0.1060` at threshold 0.50, and `0.3109` vs `0.2419` at threshold 0.25.

## 6. Group-wise error findings

Error analysis compares false positives and false negatives by group. False positives may unnecessarily push reliable customers into manual review or lower credit support; false negatives may miss actual default risk.

At threshold 0.50, Male false-positive rate was `0.0542` vs Female `0.0479`; Female false-negative rate was `0.6684` vs Male `0.6459`.

At threshold 0.25, Male false-positive rate was `0.2094` vs Female `0.1626`; Female false-negative rate was `0.4505` vs Male `0.3782`.

## 7. Calibration findings

Largest absolute bin-level calibration gap: `0.0863`. Calibration gaps should be monitored because the same score may correspond to different observed default rates across groups.

## 8. Proxy predictability findings

Best proxy model: `proxy_random_forest` with ROC-AUC `0.6476`. Predictability of `SEX`/gender from non-sensitive features indicates proxy risk, not proof of legal discrimination.

## 9. Feature association findings

The feature association artifact ranks variables by standardized mean difference and mutual information with `SEX`, helping explain why group-level outcome differences may occur.

## 10. SHAP driver comparison

Status: `completed`. SHAP driver comparison completed.

## 11. Threshold fairness frontier findings

Lowering the threshold from 0.50 to 0.25 improves recall but widens several group-level fairness diagnostics.

## 12. Individual sensitivity findings

Maximum probability change when flipping `SEX` only: `0.00000000`. Baseline decision changes: `0`. Recall-policy decision changes: `0`.

## 13. Nearest-neighbour findings

Status: `completed`. Median nearest-neighbour probability difference: `0.0278`.

## 14. Final interpretation

The protected-attribute deep dive does not establish legal discrimination or causal bias. It shows a diagnostic fairness-governance signal. Male applicants (SEX=1) had higher high-risk flag rates than Female applicants (SEX=2), and this gap widened under the recall-focused threshold. The recall-focused policy improved default capture but also widened demographic parity and equalized-odds differences. Individual sensitivity testing showed that flipping SEX alone did not change XGBoost predictions, confirming no direct use of SEX in the active prediction path. However, proxy analysis showed that SEX/gender is moderately predictable from non-sensitive credit variables, so removing SEX alone is not a complete fairness strategy. Therefore, the model should remain a decision-support tool requiring threshold governance, human oversight, and ongoing fairness monitoring.

## 15. Governance recommendations

- Keep XGBoost as decision-support, not automated approval.
- Review threshold changes with fairness guardrails and manual-review capacity.
- Monitor fairness metrics over time and by relevant intersections where sample sizes support them.
- Treat proxy-risk analysis as evidence that removing `SEX` alone is not a complete fairness strategy.
- Add calibration and threshold governance before any real operational use.

## 16. Limitations

- Public academic dataset, not production bank data.
- Diagnostic fairness metrics are observational and not causal.
- No legal compliance conclusion is made.
- Nearest-neighbour individual fairness depends on the chosen distance metric.
- SHAP comparisons are approximate and environment-sensitive.
