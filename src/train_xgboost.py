"""Train and persist XGBoost application, behavioral, and diagnostic models."""

from __future__ import annotations

from pathlib import Path

from .data_preprocessing import (
    FEATURE_SET_APPLICATION,
    FEATURE_SET_BEHAVIORAL,
    FEATURE_SET_FULL_DIAGNOSTIC,
)
from .evaluate_models import run_model_experiment
from .model_builders import build_xgboost_estimator
from .utils import MODELS_DIR, REPORTS_DIR, ensure_directories, load_dataset_auto, save_json


def run(output_path: Path | None = None) -> dict:
    ensure_directories()
    df, data_path = load_dataset_auto()

    application_path = output_path or (MODELS_DIR / "xgboost_application.pkl")
    application = run_model_experiment(
        df,
        build_xgboost_estimator(),
        "xgboost_application",
        FEATURE_SET_APPLICATION,
        model_output_path=application_path,
    )
    behavioral = run_model_experiment(
        df,
        build_xgboost_estimator(),
        "xgboost_behavioral",
        FEATURE_SET_BEHAVIORAL,
        model_output_path=MODELS_DIR / "xgboost_behavioral.pkl",
    )
    full_diagnostic = run_model_experiment(
        df,
        build_xgboost_estimator(),
        "xgboost_full_diagnostic",
        FEATURE_SET_FULL_DIAGNOSTIC,
        model_output_path=MODELS_DIR / "xgboost_full_diagnostic.pkl",
    )

    payload = {
        "dataset": str(data_path),
        "models": {
            "xgboost_application": {
                "model_path": str(application_path),
                "metrics": application["metrics"],
            },
            "xgboost_behavioral": {
                "model_path": str(MODELS_DIR / "xgboost_behavioral.pkl"),
                "metrics": behavioral["metrics"],
            },
            "xgboost_full_diagnostic": {
                "model_path": str(MODELS_DIR / "xgboost_full_diagnostic.pkl"),
                "metrics": full_diagnostic["metrics"],
            },
        },
    }
    save_json(payload, REPORTS_DIR / "model_validation" / "xgboost_training_summary.json")
    return payload


if __name__ == "__main__":
    result = run()
    print("XGBoost models trained.")
    print(result)
