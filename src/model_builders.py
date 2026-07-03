"""Reusable estimator builders for validation experiments."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def build_logistic_estimator() -> LogisticRegression:
    return LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )


def build_xgboost_estimator() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
