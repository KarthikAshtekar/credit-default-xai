"""SHAP explainability utilities for global and local explanations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .data_preprocessing import TARGET_COL, train_test_data
from .utils import MODELS_DIR, REPORTS_DIR, ensure_directories, load_dataset_auto, load_model


def _transformed_feature_names(pipeline, X: pd.DataFrame):
    preprocessor = pipeline.named_steps["preprocessor"]
    return preprocessor.get_feature_names_out()


def _transform_X(pipeline, X: pd.DataFrame):
    preprocessor = pipeline.named_steps["preprocessor"]
    return preprocessor.transform(X)


def generate_shap_artifacts(model_path: Path, sample_size: int = 500) -> dict:
    ensure_directories()
    df, _ = load_dataset_auto()
    _, X_test, _, _ = train_test_data(df, target_col=TARGET_COL)

    model_pipeline = load_model(model_path)

    X_sample = X_test.sample(min(sample_size, len(X_test)), random_state=42)
    Xt = _transform_X(model_pipeline, X_sample)

    feature_names = _transformed_feature_names(model_pipeline, X_sample)
    feature_frame = pd.DataFrame(
        Xt.toarray() if hasattr(Xt, "toarray") else Xt, columns=feature_names
    )

    estimator = model_pipeline.named_steps["classifier"]

    if estimator.__class__.__name__.lower().startswith("xgb"):
        explainer = shap.TreeExplainer(estimator)
        raw_shap = explainer.shap_values(feature_frame)
        if isinstance(raw_shap, list):
            shap_values = raw_shap[-1]
            base_value = (
                explainer.expected_value[-1]
                if isinstance(explainer.expected_value, list)
                else explainer.expected_value
            )
        else:
            shap_values = raw_shap
            base_value = explainer.expected_value
    else:
        explainer = shap.Explainer(estimator, feature_frame)
        explanation = explainer(feature_frame)
        shap_values = explanation.values
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, -1]
        base_value = (
            explanation.base_values[0]
            if hasattr(explanation, "base_values")
            else float(np.mean(shap_values))
        )

    # Global summary
    summary_path = REPORTS_DIR / "figures" / f"{model_path.stem}_shap_summary.png"
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, feature_frame, show=False)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=150)
    plt.close()

    # Local explanation for first sampled record
    local_path = REPORTS_DIR / "figures" / f"{model_path.stem}_shap_local.png"
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[0],
            base_values=float(base_value),
            data=feature_frame.iloc[0].values,
            feature_names=feature_frame.columns.tolist(),
        ),
        show=False,
    )
    plt.tight_layout()
    plt.savefig(local_path, dpi=150)
    plt.close()

    return {
        "model": str(model_path),
        "summary_plot": str(summary_path),
        "local_plot": str(local_path),
        "rows_explained": int(len(feature_frame)),
    }


def run() -> dict:
    xgb_path = MODELS_DIR / "xgboost_model.pkl"
    log_path = MODELS_DIR / "logistic_model.pkl"

    if xgb_path.exists():
        return generate_shap_artifacts(xgb_path)
    if log_path.exists():
        return generate_shap_artifacts(log_path)

    raise FileNotFoundError("No trained model found in models/. Train a model first.")


if __name__ == "__main__":
    result = run()
    print("SHAP artifacts generated.")
    print(result)
