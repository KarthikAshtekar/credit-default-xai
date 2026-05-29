"""LIME local explanation utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from .data_preprocessing import TARGET_COL, train_test_data
from .utils import MODELS_DIR, REPORTS_DIR, ensure_directories, load_dataset_auto, load_model


def generate_lime_explanation(model_path: Path, instance_index: int = 0) -> dict:
    ensure_directories()

    df, _ = load_dataset_auto()
    X_train, X_test, y_train, _ = train_test_data(df, target_col=TARGET_COL)

    model_pipeline = load_model(model_path)
    preprocessor = model_pipeline.named_steps["preprocessor"]
    estimator = model_pipeline.named_steps["classifier"]

    Xt_train = preprocessor.fit_transform(X_train)
    Xt_test = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()
    Xt_train_dense = Xt_train.toarray() if hasattr(Xt_train, "toarray") else Xt_train
    Xt_test_dense = Xt_test.toarray() if hasattr(Xt_test, "toarray") else Xt_test

    explainer = LimeTabularExplainer(
        training_data=Xt_train_dense,
        feature_names=feature_names.tolist(),
        class_names=["No Default", "Default"],
        mode="classification",
    )

    idx = min(instance_index, len(Xt_test_dense) - 1)
    exp = explainer.explain_instance(
        Xt_test_dense[idx],
        estimator.predict_proba,
        num_features=min(12, len(feature_names)),
    )

    fig = exp.as_pyplot_figure()
    out_path = REPORTS_DIR / "figures" / f"{model_path.stem}_lime_local.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return {
        "model": str(model_path),
        "lime_plot": str(out_path),
        "instance_index": idx,
    }


def run() -> dict:
    xgb_path = MODELS_DIR / "xgboost_model.pkl"
    log_path = MODELS_DIR / "logistic_model.pkl"

    if xgb_path.exists():
        return generate_lime_explanation(xgb_path)
    if log_path.exists():
        return generate_lime_explanation(log_path)

    raise FileNotFoundError("No trained model found in models/. Train a model first.")


if __name__ == "__main__":
    result = run()
    print("LIME explanation generated.")
    print(result)
