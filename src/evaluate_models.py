"""Model evaluation utilities and validation experiment runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from .data_preprocessing import (
    BUREAU_FINANCIAL_FEATURES,
    DEMOGRAPHIC_FEATURES,
    FEATURE_SET_APPLICATION,
    FEATURE_SET_BEHAVIORAL,
    FEATURE_SET_FULL_DIAGNOSTIC,
    FINANCIAL_BURDEN_FEATURES,
    PAST_DEFAULTS_ASSUMPTION,
    TARGET_COL,
    build_preprocessor,
    get_dataset_split,
    get_feature_columns,
    prepare_modeling_table,
)
from .model_builders import build_logistic_estimator, build_xgboost_estimator
from .utils import (
    REPORTS_DIR,
    ensure_directories,
    load_dataset_auto,
    project_relative_path,
    save_json,
    save_model,
)


def evaluate_classification(y_true, y_pred, y_proba) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def plot_confusion_matrix(
    y_true, y_pred, model_name: str, out_dir: Path | None = None
) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    target_dir = out_dir or (REPORTS_DIR / "figures")
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"{model_name.lower()}_confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return cm


def fit_pipeline(
    estimator, X_train: pd.DataFrame, y_train: pd.Series, sample_weight=None
) -> Pipeline:
    preprocessor = build_preprocessor(X_train)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clone(estimator)),
        ]
    )

    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["classifier__sample_weight"] = sample_weight

    pipeline.fit(X_train, y_train, **fit_kwargs)
    return pipeline


def run_model_experiment(
    df: pd.DataFrame,
    estimator,
    model_name: str,
    feature_set: str,
    split_strategy: str = "random",
    feature_columns: list[str] | None = None,
    model_output_path: Path | None = None,
) -> Dict[str, Any]:
    split = get_dataset_split(
        df,
        target_col=TARGET_COL,
        feature_set=feature_set,
        split_strategy=split_strategy,
        feature_columns=feature_columns,
    )
    pipeline = fit_pipeline(estimator, split.X_train, split.y_train)

    y_pred = pipeline.predict(split.X_test)
    y_proba = pipeline.predict_proba(split.X_test)[:, 1]
    metrics = evaluate_classification(split.y_test, y_pred, y_proba)
    plot_confusion_matrix(split.y_test, y_pred, model_name)

    if model_output_path is not None:
        save_model(pipeline, model_output_path)

    return {
        "model_name": model_name,
        "feature_set": feature_set,
        "split_strategy": split_strategy,
        "metrics": metrics,
        "train_rows": int(len(split.X_train)),
        "test_rows": int(len(split.X_test)),
        "feature_count": int(len(split.feature_columns)),
        "feature_columns": split.feature_columns,
        "train_indices": split.train_indices.tolist(),
        "test_indices": split.test_indices.tolist(),
        "model_path": project_relative_path(model_output_path),
        "pipeline": pipeline,
    }


def _serializable_result(result: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(result)
    payload.pop("pipeline", None)
    payload.pop("train_indices", None)
    payload.pop("test_indices", None)
    return payload


def _flatten_results(results: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in results:
        row = {
            "model_name": result["model_name"],
            "feature_set": result["feature_set"],
            "split_strategy": result["split_strategy"],
            "train_rows": result["train_rows"],
            "test_rows": result["test_rows"],
            "feature_count": result["feature_count"],
        }
        row.update(result["metrics"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        by=["split_strategy", "roc_auc", "accuracy"],
        ascending=[True, False, False],
    )


def compare_model_metrics(metrics_map: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(metrics_map).T.sort_values(by="roc_auc", ascending=False)
    out_csv = REPORTS_DIR / "model_validation" / "clean_feature_model_comparison.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=True)
    save_json(
        metrics_map,
        REPORTS_DIR / "model_validation" / "clean_feature_model_comparison.json",
    )
    return df


def _standard_experiments(df: pd.DataFrame) -> list[Dict[str, Any]]:
    return [
        run_model_experiment(
            df,
            build_logistic_estimator(),
            "logistic_application",
            FEATURE_SET_APPLICATION,
        ),
        run_model_experiment(
            df,
            build_xgboost_estimator(),
            "xgboost_application",
            FEATURE_SET_APPLICATION,
        ),
        run_model_experiment(
            df,
            build_logistic_estimator(),
            "logistic_behavioral",
            FEATURE_SET_BEHAVIORAL,
        ),
        run_model_experiment(
            df,
            build_xgboost_estimator(),
            "xgboost_behavioral",
            FEATURE_SET_BEHAVIORAL,
        ),
        run_model_experiment(
            df,
            build_xgboost_estimator(),
            "xgboost_full_diagnostic",
            FEATURE_SET_FULL_DIAGNOSTIC,
        ),
    ]


def _write_feature_set_json(
    path: Path,
    feature_set: str,
    results: list[Dict[str, Any]],
    notes: list[str] | None = None,
) -> None:
    payload = {
        "feature_set": feature_set,
        "split_strategy": "random",
        "models": {result["model_name"]: _serializable_result(result) for result in results},
    }
    if notes:
        payload["notes"] = notes
    save_json(payload, path)


def run_temporal_split_comparison(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    experiments = [
        ("logistic_application", FEATURE_SET_APPLICATION, build_logistic_estimator()),
        ("xgboost_application", FEATURE_SET_APPLICATION, build_xgboost_estimator()),
        ("logistic_behavioral", FEATURE_SET_BEHAVIORAL, build_logistic_estimator()),
        ("xgboost_behavioral", FEATURE_SET_BEHAVIORAL, build_xgboost_estimator()),
        ("xgboost_full_diagnostic", FEATURE_SET_FULL_DIAGNOSTIC, build_xgboost_estimator()),
    ]
    for name, feature_set, estimator in experiments:
        results.append(
            run_model_experiment(
                df,
                estimator,
                f"{name}_random",
                feature_set,
                split_strategy="random",
            )
        )
        results.append(
            run_model_experiment(
                df,
                estimator,
                f"{name}_temporal",
                feature_set,
                split_strategy="temporal",
            )
        )

    output = _flatten_results(results)
    out_path = REPORTS_DIR / "model_validation" / "temporal_split_comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)
    return output


def run_ablation_study(df: pd.DataFrame) -> pd.DataFrame:
    prepared = prepare_modeling_table(df, target_col=TARGET_COL)
    baseline_features = get_feature_columns(prepared, FEATURE_SET_FULL_DIAGNOSTIC)

    experiments = [
        ("baseline_full_diagnostic", baseline_features),
        (
            "without_payment_behavior",
            [
                col
                for col in baseline_features
                if col
                not in {
                    "OnTimePayments_Last12M",
                    "MissedPayments_Last12M",
                    "MissedEMIs_Last6M",
                    "MissedPaymentRate",
                    "HistoricalRiskScore",
                }
            ],
        ),
        (
            "without_demographics",
            [col for col in baseline_features if col not in set(DEMOGRAPHIC_FEATURES)],
        ),
        (
            "without_burden_features",
            [col for col in baseline_features if col not in set(FINANCIAL_BURDEN_FEATURES)],
        ),
        (
            "bureau_financial_only",
            [col for col in baseline_features if col in set(BUREAU_FINANCIAL_FEATURES)],
        ),
    ]

    results = []
    for name, feature_columns in experiments:
        result = run_model_experiment(
            df,
            build_xgboost_estimator(),
            f"xgboost_{name}",
            FEATURE_SET_FULL_DIAGNOSTIC,
            feature_columns=feature_columns,
        )
        rows = {
            "ablation": name,
            "feature_count": len(feature_columns),
        }
        rows.update(result["metrics"])
        results.append(rows)

    output = pd.DataFrame(results).sort_values(by="roc_auc", ascending=False)
    out_path = REPORTS_DIR / "model_validation" / "ablation_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)
    return output


def run() -> pd.DataFrame:
    ensure_directories()
    df_raw, _ = load_dataset_auto()
    standard_results = _standard_experiments(df_raw)
    standard_df = _flatten_results(standard_results)

    clean_feature_metrics = {result["model_name"]: result["metrics"] for result in standard_results}
    compare_model_metrics(clean_feature_metrics)

    application_results = [
        result for result in standard_results if result["feature_set"] == FEATURE_SET_APPLICATION
    ]
    behavioral_results = [
        result for result in standard_results if result["feature_set"] == FEATURE_SET_BEHAVIORAL
    ]
    full_results = [
        result
        for result in standard_results
        if result["feature_set"] == FEATURE_SET_FULL_DIAGNOSTIC
    ]

    model_validation_dir = REPORTS_DIR / "model_validation"
    model_validation_dir.mkdir(parents=True, exist_ok=True)
    _write_feature_set_json(
        model_validation_dir / "application_model_metrics.json",
        FEATURE_SET_APPLICATION,
        application_results,
        notes=[PAST_DEFAULTS_ASSUMPTION],
    )
    _write_feature_set_json(
        model_validation_dir / "behavioral_model_metrics.json",
        FEATURE_SET_BEHAVIORAL,
        behavioral_results,
        notes=[
            "Behavioral monitoring features include repayment history and account-behavior signals.",
            PAST_DEFAULTS_ASSUMPTION,
        ],
    )
    _write_feature_set_json(
        model_validation_dir / "full_diagnostic_model_metrics.json",
        FEATURE_SET_FULL_DIAGNOSTIC,
        full_results,
        notes=[
            "Full diagnostic results are not the final credit-origination headline model.",
            "This feature set includes post-outcome behavioral features and is diagnostic only.",
        ],
    )

    standard_df.to_csv(model_validation_dir / "clean_feature_model_comparison.csv", index=False)
    run_temporal_split_comparison(df_raw)
    run_ablation_study(df_raw)
    return standard_df


if __name__ == "__main__":
    comparison_df = run()
    print(comparison_df.to_string(index=False))
