# ML vs DL Comparison

| model_name | threshold | accuracy | precision | recall | f1 | f2 | roc_auc | pr_auc | approval_support_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost_full_public_diagnostic | 0.5 | 0.8193935354881706 | 0.6753246753246753 | 0.3526752072343632 | 0.4633663366336634 |  | 0.7805110639172432 |  |  |
| xgboost_public | 0.5 | 0.8152282572475842 | 0.6584302325581395 | 0.3413715146948003 | 0.4496277915632754 |  | 0.7747718507831988 |  |  |
| logistic_public | 0.5 | 0.740753082305898 | 0.4374658656471873 | 0.6036171816126601 | 0.5072830905636478 |  | 0.7526831057147118 |  |  |
| xgboost_public_recall_optimized | 0.25 | 0.766911029656781 | 0.47769516728624534 | 0.5810097965335342 | 0.5243114586875213 | 0.5569199653279399 | 0.7747718507831988 | 0.5415000164539201 | 0.7310896367877374 |
| dnn_baseline | 0.5 | 0.8130623125624792 | 0.6425591098748261 | 0.34815373021853807 | 0.45161290322580644 | 0.3832752613240418 | 0.7657170490310258 | 0.5212098829700869 | 0.8802065978007331 |
| dnn_class_weighted | 0.5 | 0.7445851382872376 | 0.4429046563192905 | 0.6021100226073851 | 0.510380070265091 | 0.561726659167604 | 0.7663732676738573 | 0.5362943957494026 | 0.6994335221592802 |
| dnn_recall_optimized | 0.3 | 0.7857380873042319 | 0.5148658448150834 | 0.5350414468726451 | 0.524759793052476 | 0.5308808135187678 | 0.7657170490310258 | 0.5212098829700869 | 0.7702432522492503 |

## Findings

- ROC-AUC: DNN `0.7657` vs XGBoost `0.7748`; DNN does not improve ranking.
- PR-AUC: DNN `0.5212` vs XGBoost `0.5415`; DNN does not improve minority-class ranking.
- Recall policy: DNN recall `0.5350` at precision `0.5149` vs XGBoost recall `0.5810` at precision `0.4777`. The DNN meets the validation precision rule but captures fewer test defaults.

## Decision

XGBoost remains the primary model.

XGBoost remains the primary model unless the DNN demonstrates materially better recall/PR-AUC under comparable fairness and explainability constraints.

The comparison considers ROC-AUC, PR-AUC, recall at acceptable precision, fairness, explainability, and operational complexity. Tree boosting often remains strong on structured tabular credit data even when a DNN is useful as a learning benchmark.

XGBoost saved probabilities were unavailable; the ML-vs-DL curve contains DNN only.