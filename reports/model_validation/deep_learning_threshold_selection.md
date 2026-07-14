# Deep Learning Threshold Selection

- Source experiment: `dnn_baseline`
- Selected threshold: `0.30`
- Rule: `maximize_recall_precision_050`
- Fallback used: `False`
- Threshold selection used validation data only; the test split was evaluated once.

## Untouched Test Metrics

| threshold | accuracy | precision | recall | specificity | f1 | f2 | false_positive_rate | false_negative_rate | true_positives | false_positives | true_negatives | false_negatives | predicted_default_rate | predicted_non_default_rate | approval_support_rate | default_capture_rate | expected_cost | roc_auc | pr_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3 | 0.7857380873042319 | 0.5148658448150834 | 0.5350414468726451 | 0.8568983957219252 | 0.524759793052476 | 0.5308808135187678 | 0.14310160427807486 | 0.46495855312735496 | 710 | 669 | 4006 | 617 | 0.22975674775074975 | 0.7702432522492503 | 0.7702432522492503 | 0.5350414468726451 | 3754 | 0.7657170490310258 | 0.5212098829700869 |