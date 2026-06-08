"""Train and persist logistic regression application and behavioral models."""

from __future__ import annotations

from pathlib import Path

from .data_preprocessing import FEATURE_SET_APPLICATION, FEATURE_SET_BEHAVIORAL
from .evaluate_models import run_model_experiment
from .model_builders import build_logistic_estimator
from .utils import MODELS_DIR, REPORTS_DIR, ensure_directories, load_dataset_auto, save_json


def run(output_path: Path | None = None) -> dict:
    ensure_directories()
    df, data_path = load_dataset_auto()

    application_path = output_path or (MODELS_DIR / "logistic_application.pkl")
    application = run_model_experiment(
        df,
        build_logistic_estimator(),
        "logistic_application",
        FEATURE_SET_APPLICATION,
        model_output_path=application_path,
    )
    behavioral = run_model_experiment(
        df,
        build_logistic_estimator(),
        "logistic_behavioral",
        FEATURE_SET_BEHAVIORAL,
        model_output_path=MODELS_DIR / "logistic_behavioral.pkl",
    )

    payload = {
        "dataset": str(data_path),
        "models": {
            "logistic_application": {
                "model_path": str(application_path),
                "metrics": application["metrics"],
            },
            "logistic_behavioral": {
                "model_path": str(MODELS_DIR / "logistic_behavioral.pkl"),
                "metrics": behavioral["metrics"],
            },
        },
    }
    save_json(payload, REPORTS_DIR / "model_validation" / "logistic_training_summary.json")
    return payload


if __name__ == "__main__":
    result = run()
    print("Logistic regression models trained.")
    print(result)
