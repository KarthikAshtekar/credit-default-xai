# Class-Weight Tuning Report

`scale_pos_weight` candidates were trained on inner-train data for validation-based selection. After threshold selection, each setting was refit on the full training split and evaluated once on the untouched test split.

The computed imbalance ratio candidate is `n_negative / n_positive` from the inner-train split.
