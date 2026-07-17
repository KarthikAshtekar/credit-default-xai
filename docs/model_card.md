# Model Card: Public UCI Credit Default Risk Model

## Model Purpose

Estimate next-month credit-card default risk for an educational responsible AI workflow using the public UCI Taiwan credit-card default dataset.

## Intended Use

- Portfolio and academic demonstration
- Model-supported manual-review triage
- Explainability, fairness, leakage, and threshold-governance analysis
- Applicant-level decision-support reporting in a local Streamlit dashboard

## Non-Intended Use

This model is not a production credit approval engine, not a regulatory scorecard, not an adverse-action reason generator, and not a substitute for underwriting governance, legal review, monitoring, or human oversight.

## Dataset

- Dataset: UCI Default of Credit Card Clients / Taiwan credit-card default
- UCI ID: `350`
- Rows: `30,000`
- Loader: `ucimlrepo`
- Target: `Default_Flag`
- Target meaning: `1` means next-month default / bad outcome

## Features

Final active feature set: `application_public`.

- Includes credit limit, historical repayment status, bill amounts, payment amounts, and engineered utilization/payment features.
- Excludes `SEX` from active training features.
- Retains `SEX` in the raw/audit data for fairness diagnostics.
- Includes `AGE`, `MARRIAGE`, and `EDUCATION` with audit caveats.

## Final Model

Primary model: XGBoost (`models/xgboost_public.pkl`).

Logistic regression is retained as a benchmark. A TensorFlow/Keras DNN is retained as a model-family benchmark only.

## Baseline Metrics

XGBoost at threshold `0.50` on the held-out test split:

- Accuracy: `0.8152`
- Precision: `0.6584`
- Recall: `0.3414`
- F1: `0.4496`
- F2: `0.3778`
- ROC-AUC: `0.7748`
- PR-AUC: `0.5415`

## Recall-Optimized Policy Metrics

XGBoost recall policy at threshold `0.25`:

- Accuracy: `0.7669`
- Precision: `0.4777`
- Recall: `0.5810`
- F1: `0.5243`
- F2: `0.5569`
- ROC-AUC: `0.7748`
- PR-AUC: `0.5415`

Selection rule: maximize validation recall subject to validation precision >= `0.50`.

## DNN Benchmark Results

MLP architecture: `64 -> 32 -> 16 -> 1`.

| DNN policy | Threshold | Precision | Recall | F2 | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.50 | 0.6426 | 0.3482 | 0.3833 | 0.7657 | 0.5212 |
| Class-weighted | 0.50 | 0.4429 | 0.6021 | 0.5617 | 0.7664 | 0.5363 |
| Recall-optimized | 0.30 | 0.5149 | 0.5350 | 0.5309 | 0.7657 | 0.5212 |

The DNN did not materially improve ranking quality or recall-policy performance. XGBoost remains the primary model.

## Fairness Metrics

Primary protected attribute: `SEX`/gender.

Verified UCI coding:

| SEX code | Group |
| ---: | --- |
| 1 | Male |
| 2 | Female |

Favorable outcome: predicted non-default / lower-risk manual-review support.

Baseline XGBoost at threshold `0.50`:

- Demographic parity difference: `0.0220`
- Equal opportunity difference: `0.0063`
- Equalized odds difference: `0.0225`
- Disparate impact ratio: `0.9754`

Recall XGBoost at threshold `0.25`:

- Demographic parity difference: `0.0691`
- Equal opportunity difference: `0.0468`
- Equalized odds difference: `0.0723`
- Disparate impact ratio: `0.9089`

These fairness metrics are diagnostic. They do not prove legal compliance or absence of bias.

Fairness conclusion: the audit did not prove discrimination. The baseline XGBoost policy showed small group-level differences on `SEX`, while the recall-optimized threshold widened those differences. The noteworthy result is therefore a fairness-governance tradeoff, not a confirmed discrimination finding.

## Protected Attribute: SEX/Gender

- Source coding: `1 = Male`, `2 = Female`.
- Intended fairness use: audit only. `SEX` is retained for diagnostic fairness reporting and is not an active final XGBoost prediction feature.
- Direct-use check: individual sensitivity testing showed maximum probability change `0.00000000` and zero decision changes when `SEX` alone was flipped.
- Proxy-risk check: `SEX`/gender was moderately predictable from non-sensitive variables; the best proxy model ROC-AUC was `0.6476`.
- Threshold risk: lowering the XGBoost threshold from `0.50` to `0.25` improved recall but widened demographic parity and equalized-odds differences.
- Monitoring requirement: group-level outcome monitoring, group-level error monitoring, calibration monitoring, proxy-risk monitoring, and threshold governance.

## Fairness Deep Dive on Protected Attribute SEX

Additional diagnostic artifacts under `reports/fairness_reports/application_model/` evaluate:

- Group outcomes by `SEX`
- Group error rates by `SEX`
- Calibration by `SEX`
- Proxy predictability of `SEX` from non-sensitive variables
- Feature association with `SEX`
- SHAP driver comparison by `SEX`
- Threshold fairness frontier
- Individual SEX sensitivity
- Nearest-neighbour individual fairness

Deep-dive findings:

- Baseline threshold `0.50`: DP difference `0.0220`, equalized odds difference `0.0225`, disparate impact ratio `0.9754`.
- Recall threshold `0.25`: DP difference `0.0691`, equalized odds difference `0.0723`, disparate impact ratio `0.9089`.
- Outcome audit shows Male applicants (SEX=1) had higher high-risk flag rates than Female applicants (SEX=2): `0.1280` vs `0.1060` at threshold `0.50`, and `0.3109` vs `0.2419` at threshold `0.25`.
- Error-rate audit shows Male applicants had higher FPR and Female applicants had higher FNR. Under the recall-focused threshold, Male FPR was `0.2094` vs Female FPR `0.1626`, while Female FNR was `0.4505` vs Male FNR `0.3782`.
- Calibration check found largest bin-level calibration gap `0.0863`.
- Proxy-risk analysis found the best proxy model could predict `SEX`/gender with ROC-AUC `0.6476`; this is not proof of legal discrimination, but it shows that direct removal of `SEX` is not a complete fairness strategy.
- Individual SEX sensitivity found maximum probability change `0.00000000` and zero decision changes when flipping `SEX` only, confirming no direct use of `SEX` in the active XGBoost prediction path.

Governance conclusion: the project found a diagnostic fairness-governance signal and threshold-governance tradeoff. It found no proof of legal discrimination and makes no causal bias claim. The model requires threshold governance, proxy-risk monitoring, calibration monitoring, and human oversight before any real operational use.

## Explainability Tools

- SHAP: global and local XGBoost explanations
- LIME: local XGBoost explanation
- Counterfactual guidance: repayment, utilization, and repayment-amount improvement suggestions
- DNN fallback explainability: model-agnostic permutation importance

## Leakage Controls

The audit reports no detected target leakage or train/test overlap based on implemented checks.

Checks include target exclusion, ID exclusion, duplicate selected-row handling, source-index overlap checks, target shuffle, mutual-information review, and UCI feature-timing review.

## Limitations

- Public academic dataset, not live production bank data
- No true chronological application timestamp
- No reject inference
- No causal fairness analysis
- No calibrated regulatory scorecard
- No production monitoring or drift detection
- Dashboard is a local decision-support demo

## Human Oversight Requirement

Human review is required for any real credit decision. The model output should be interpreted as a screening signal with explanations and caveats, not as an automated lending decision.

## Ethical Considerations

Threshold changes can improve default capture while changing group-level fairness metrics and manual-review volume. Any real use would require policy review, model risk governance, monitoring, calibration, and legally appropriate explanation processes.

## Future Monitoring Needs

- Data drift and feature distribution monitoring
- Calibration monitoring
- Threshold-performance monitoring
- Fairness monitoring across groups and intersections
- Explanation stability review
- Periodic retraining and challenger-model review
