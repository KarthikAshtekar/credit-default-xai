"""Fairness metric computation for binary classification."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    true_positive_rate,
)

from .data_preprocessing import TARGET_COL, prepare_modeling_table
from .utils import (
    MODELS_DIR,
    REPORTS_DIR,
    infer_protected_attribute,
    load_dataset_auto,
    load_model,
    save_json,
)


def _disparate_impact_ratio(y_pred: np.ndarray, sensitive: pd.Series) -> float:
    rates = (
        pd.DataFrame({"y_pred": y_pred, "sensitive": sensitive})
        .groupby("sensitive")["y_pred"]
        .mean()
    )
    if len(rates) < 2:
        return 1.0
    min_rate = rates.min()
    max_rate = rates.max()
    return float(min_rate / max_rate) if max_rate > 0 else 1.0


def _equal_opportunity_difference(y_true, y_pred, sensitive):
    groups = pd.Series(sensitive).dropna().unique()
    tprs = []
    for g in groups:
        mask = sensitive == g
        if mask.sum() == 0:
            continue
        tprs.append(true_positive_rate(y_true[mask], y_pred[mask]))
    if not tprs:
        return 0.0
    return float(max(tprs) - min(tprs))


def compute_fairness_metrics(y_true, y_pred, sensitive) -> Dict[str, float]:
    metrics = {
        "demographic_parity_difference": float(
            demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive)
        ),
        "equalized_odds_difference": float(
            equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive)
        ),
        "equal_opportunity_difference": float(
            _equal_opportunity_difference(y_true, y_pred, sensitive)
        ),
        "disparate_impact_ratio": float(_disparate_impact_ratio(y_pred, sensitive)),
    }
    return metrics


def run(model_path: Path | None = None) -> Dict:
    model_path = model_path or (
        MODELS_DIR / "xgboost_model.pkl"
        if (MODELS_DIR / "xgboost_model.pkl").exists()
        else MODELS_DIR / "logistic_model.pkl"
    )

    df_raw, _ = load_dataset_auto()
    protected_col = infer_protected_attribute(df_raw)
    prepared = prepare_modeling_table(df_raw, target_col=TARGET_COL)

    y = prepared[TARGET_COL]
    X = prepared.drop(columns=[TARGET_COL])

    if protected_col not in X.columns:
        # fallback if transformed/dropped due preprocessing decisions
        sensitive = df_raw[protected_col].astype(str)
    else:
        sensitive = X[protected_col].astype(str)

    model = load_model(model_path)
    y_pred = model.predict(X)

    fairness = compute_fairness_metrics(y.values, y_pred, sensitive.values)

    payload = {
        "model": str(model_path),
        "protected_attribute": protected_col,
        "fairness_metrics": fairness,
    }

    out_json = REPORTS_DIR / "fairness_reports" / f"{model_path.stem}_fairness_metrics.json"
    out_csv = REPORTS_DIR / "fairness_reports" / f"{model_path.stem}_fairness_metrics.csv"

    save_json(payload, out_json)
    pd.DataFrame([fairness]).to_csv(out_csv, index=False)
    return payload


if __name__ == "__main__":
    result = run()
    print("Fairness metrics computed.")
    print(result)
