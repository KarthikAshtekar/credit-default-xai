# Group Outcome Analysis by SEX

Verified UCI protected-attribute mapping: `SEX=1` is Male and `SEX=2` is Female.

This report compares observed default rates, predicted default probabilities, high-risk flag rates, and low-risk / approval-support rates by `SEX`/gender.

Outcome differences show whether groups receive different model-driven high-risk flag rates. They are governance diagnostics and do not prove legal discrimination or causal bias.

Male applicants (SEX=1) had higher high-risk flag rates than Female applicants (SEX=2): `0.1280` vs `0.1060` at threshold 0.50, and `0.3109` vs `0.2419` at the recall threshold 0.25.

## Primary XGBoost policies

| policy | sex_group | sex_code | group | n | actual_default_rate | mean_predicted_default_probability | predicted_high_risk_rate | approval_support_rate | demographic_parity_difference | disparate_impact_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost_baseline_threshold_050 | Male | 1 | Male (SEX=1) | 2351 | 0.24627817949808592 | 0.2373718076758826 | 0.12803062526584433 | 0.8719693747341557 | 0.022032268651218212 | 0.9753554494958341 |
| xgboost_baseline_threshold_050 | Female | 2 | Female (SEX=2) | 3651 | 0.20487537660914817 | 0.2100556876140783 | 0.10599835661462613 | 0.8940016433853739 | 0.022032268651218212 | 0.9753554494958341 |
| xgboost_recall_threshold_025 | Male | 1 | Male (SEX=1) | 2351 | 0.24627817949808592 | 0.2373718076758826 | 0.3109315185027648 | 0.6890684814972352 | 0.06907997098153773 | 0.9088833186222565 |
| xgboost_recall_threshold_025 | Female | 2 | Female (SEX=2) | 3651 | 0.20487537660914817 | 0.2100556876140783 | 0.24185154752122706 | 0.7581484524787729 | 0.06907997098153773 | 0.9088833186222565 |

The fairness differences use the favorable outcome, defined as predicted non-default / lower-risk support, to stay consistent with the project model card.
