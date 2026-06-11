"""Fairness metric computation for approval decisions on held-out data."""

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

from .data_preprocessing import (
    FEATURE_SET_APPLICATION,
    FEATURE_SET_BEHAVIORAL,
    FEATURE_SET_FULL_DIAGNOSTIC,
    TARGET_COL,
    get_dataset_split,
)
from .utils import (
    MODELS_DIR,
    REPORTS_DIR,
    infer_protected_attribute,
    load_dataset_auto,
    load_model,
    project_relative_path,
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


def _model_context(model_path: Path) -> tuple[str, Path]:
    stem = model_path.stem
    if "behavioral" in stem:
        return FEATURE_SET_BEHAVIORAL, REPORTS_DIR / "fairness_reports" / "behavioral_model"
    if "full_diagnostic" in stem:
        return FEATURE_SET_FULL_DIAGNOSTIC, REPORTS_DIR / "fairness_reports" / "full_diagnostic"
    return FEATURE_SET_APPLICATION, REPORTS_DIR / "fairness_reports" / "application_model"


def run(model_path: Path | None = None, threshold: float = 0.50) -> Dict:
    model_path = model_path or (
        MODELS_DIR / "xgboost_application.pkl"
        if (MODELS_DIR / "xgboost_application.pkl").exists()
        else MODELS_DIR / "logistic_application.pkl"
    )

    feature_set, report_dir = _model_context(model_path)
    report_dir.mkdir(parents=True, exist_ok=True)

    df_raw, _ = load_dataset_auto()
    protected_col = infer_protected_attribute(df_raw)
    split = get_dataset_split(df_raw, target_col=TARGET_COL, feature_set=feature_set)
    sensitive = df_raw.loc[split.test_indices, protected_col].astype(str)

    model = load_model(model_path)
    default_proba = model.predict_proba(split.X_test)[:, 1]
    approval_pred = (default_proba < threshold).astype(int)
    approval_true = (1 - split.y_test).astype(int)

    fairness = compute_fairness_metrics(approval_true.values, approval_pred, sensitive.values)

    payload = {
        "model": project_relative_path(model_path),
        "feature_set": feature_set,
        "protected_attribute": protected_col,
        "approval_threshold": threshold,
        "fairness_metrics": fairness,
        "test_rows": int(len(split.X_test)),
    }

    out_json = report_dir / f"{model_path.stem}_fairness_metrics.json"
    out_csv = report_dir / f"{model_path.stem}_fairness_metrics.csv"

    save_json(payload, out_json)
    pd.DataFrame([fairness]).to_csv(out_csv, index=False)
    return payload


if __name__ == "__main__":
    result = run()
    print("Fairness metrics computed.")
    print(result)
