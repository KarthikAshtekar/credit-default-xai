"""Model evaluation utilities and comparison helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .utils import REPORTS_DIR, save_json


def evaluate_classification(y_true, y_pred, y_proba) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def plot_confusion_matrix(y_true, y_pred, model_name: str) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    out_path = REPORTS_DIR / "figures" / f"{model_name.lower()}_confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return cm


def compare_model_metrics(metrics_map: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(metrics_map).T.sort_values(by="roc_auc", ascending=False)
    out_csv = REPORTS_DIR / "fairness_reports" / "model_performance_summary.csv"
    df.to_csv(out_csv, index=True)

    save_json(metrics_map, REPORTS_DIR / "fairness_reports" / "model_performance_summary.json")
    return df


def run() -> pd.DataFrame:
    from .data_preprocessing import TARGET_COL, train_test_data
    from .utils import MODELS_DIR, load_dataset_auto, load_model

    df_raw, _ = load_dataset_auto()
    _, X_test, _, y_test = train_test_data(df_raw, target_col=TARGET_COL)

    metrics_map = {}
    for name, model_file in [("logistic", "logistic_model.pkl"), ("xgboost", "xgboost_model.pkl")]:
        model_path = MODELS_DIR / model_file
        if not model_path.exists():
            continue

        model = load_model(model_path)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics_map[name] = evaluate_classification(y_test, y_pred, y_proba)
        plot_confusion_matrix(y_test, y_pred, name)

    if not metrics_map:
        raise FileNotFoundError("No trained model found in models/. Train models first.")

    return compare_model_metrics(metrics_map)


if __name__ == "__main__":
    comparison_df = run()
    print(comparison_df)
