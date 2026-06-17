"""LIME local explanation utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

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
    ensure_directories,
    load_dataset_auto,
    load_model,
    project_relative_path,
)


def _model_context(model_path: Path) -> tuple[str, Path]:
    stem = model_path.stem
    if "behavioral" in stem:
        return FEATURE_SET_BEHAVIORAL, REPORTS_DIR / "explainability_reports" / "behavioral_model"
    if "full_diagnostic" in stem:
        return (
            FEATURE_SET_FULL_DIAGNOSTIC,
            REPORTS_DIR / "explainability_reports" / "full_diagnostic",
        )
    return FEATURE_SET_APPLICATION, REPORTS_DIR / "explainability_reports" / "application_model"


def generate_lime_explanation(model_path: Path, instance_index: int = 0) -> dict:
    ensure_directories()

    df, _ = load_dataset_auto()
    feature_set, report_dir = _model_context(model_path)
    report_dir.mkdir(parents=True, exist_ok=True)
    split = get_dataset_split(df, target_col=TARGET_COL, feature_set=feature_set)
    X_train, X_test = split.X_train, split.X_test

    model_pipeline = load_model(model_path)
    preprocessor = model_pipeline.named_steps["preprocessor"]
    estimator = model_pipeline.named_steps["classifier"]

    Xt_train = preprocessor.transform(X_train)
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
    out_path = report_dir / f"{model_path.stem}_lime_local.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return {
        "model": project_relative_path(model_path),
        "lime_plot": project_relative_path(out_path),
        "instance_index": idx,
    }


def run() -> dict:
    xgb_path = MODELS_DIR / "xgboost_application.pkl"
    log_path = MODELS_DIR / "logistic_application.pkl"
    behavioral_path = MODELS_DIR / "xgboost_behavioral.pkl"

    if xgb_path.exists():
        return generate_lime_explanation(xgb_path)
    if behavioral_path.exists():
        return generate_lime_explanation(behavioral_path)
    if log_path.exists():
        return generate_lime_explanation(log_path)

    raise FileNotFoundError("No trained model found in models/. Train a model first.")


if __name__ == "__main__":
    result = run()
    print("LIME explanation generated.")
    print(result)
