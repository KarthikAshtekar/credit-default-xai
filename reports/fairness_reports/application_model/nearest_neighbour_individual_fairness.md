# Nearest-Neighbour Individual Fairness Diagnostic

Verified UCI protected-attribute mapping: `SEX=1` is Male and `SEX=2` is Female.

This approximates individual fairness: similar individuals should receive similar scores. It depends heavily on the chosen standardized Euclidean distance metric and is diagnostic, not conclusive.

| average_probability_difference | median_probability_difference | p90_probability_difference | large_difference_count_gt_0_10 |
| --- | --- | --- | --- |
| 0.049561672261912694 | 0.027756245 | 0.11557635000000013 | 764 |
