"""Train and persist the public UCI XGBoost final model."""

from __future__ import annotations

from pathlib import Path

from .data_preprocessing import (
    FEATURE_SET_APPLICATION,
)
from .evaluate_models import run_model_experiment
from .model_builders import build_xgboost_estimator
from .utils import (
    MODELS_DIR,
    REPORTS_DIR,
    ensure_directories,
    load_dataset_auto,
    project_relative_path,
    save_json,
)


def run(output_path: Path | None = None) -> dict:
    ensure_directories()
    df, data_path = load_dataset_auto()

    model_path = output_path or (MODELS_DIR / "xgboost_public.pkl")
    result = run_model_experiment(
        df,
        build_xgboost_estimator(),
        "xgboost_public",
        FEATURE_SET_APPLICATION,
        model_output_path=model_path,
    )

    payload = {
        "dataset": project_relative_path(data_path),
        "feature_set": FEATURE_SET_APPLICATION,
        "model_path": project_relative_path(model_path),
        "metrics": result["metrics"],
        "feature_columns": result["feature_columns"],
    }
    save_json(payload, REPORTS_DIR / "model_validation" / "xgboost_training_summary.json")
    save_json(payload, REPORTS_DIR / "model_validation" / "xgboost_public_model_metrics.json")
    return payload


if __name__ == "__main__":
    result = run()
    print("XGBoost public final model trained.")
    print(result)
