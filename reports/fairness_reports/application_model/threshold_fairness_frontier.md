# Threshold Fairness Frontier

Verified UCI protected-attribute mapping: `SEX=1` is Male and `SEX=2` is Female.

Threshold is a governance lever. Lower thresholds can improve default capture while changing group-level disparity and manual-review volume.

Moving the XGBoost threshold from 0.50 to 0.25 improved recall from `0.3414` to `0.5810` but widened demographic parity difference from `0.0220` to `0.0691` and equalized odds difference from `0.0225` to `0.0723`.

The recall-optimized threshold should therefore be reviewed with fairness guardrails before operational use.

## Baseline vs recall threshold

| threshold | accuracy | precision | recall | f2 | approval_support_rate | demographic_parity_difference | disparate_impact_ratio | equal_opportunity_difference | equalized_odds_difference | false_positive_rate_difference | false_negative_rate_difference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 0.8152282572475842 | 0.6584302325581395 | 0.3413715146948003 | 0.3777518345563709 | 0.8853715428190603 | 0.022032268651218212 | 0.9753554494958341 | 0.006294570340171113 | 0.022507919795332176 | 0.006294570340171175 | 0.022507919795332176 |
| 0.25 | 0.766911029656781 | 0.47769516728624534 | 0.5810097965335342 | 0.5569199653279399 | 0.7310896367877374 | 0.06907997098153773 | 0.9088833186222565 | 0.04677752212430675 | 0.07229641738937687 | 0.046777522124306664 | 0.07229641738937687 |
