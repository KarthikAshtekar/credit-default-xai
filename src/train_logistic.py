"""Train and persist Logistic Regression baseline."""

from __future__ import annotations

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .data_preprocessing import TARGET_COL, build_preprocessor, train_test_data
from .evaluate_models import evaluate_classification, plot_confusion_matrix
from .utils import MODELS_DIR, REPORTS_DIR, ensure_directories, load_dataset_auto, save_json, save_model


def train_logistic_pipeline(X_train, y_train) -> Pipeline:
    preprocessor = build_preprocessor(X_train)
    model = LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def run(output_path: Path | None = None) -> dict:
    ensure_directories()
    df, data_path = load_dataset_auto()
    X_train, X_test, y_train, y_test = train_test_data(df, target_col=TARGET_COL)

    pipeline = train_logistic_pipeline(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = evaluate_classification(y_test, y_pred, y_proba)
    plot_confusion_matrix(y_test, y_pred, "logistic")

    model_path = output_path or (MODELS_DIR / "logistic_model.pkl")
    save_model(pipeline, model_path)

    payload = {
        "dataset": str(data_path),
        "model_path": str(model_path),
        "metrics": metrics,
    }
    save_json(payload, REPORTS_DIR / "fairness_reports" / "logistic_metrics.json")
    return payload


if __name__ == "__main__":
    result = run()
    print("Logistic regression trained.")
    print(result)
