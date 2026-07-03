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
    FEATURE_POLICY_NOTE,
    FEATURE_SET_APPLICATION,
    FEATURE_SET_FULL_DIAGNOSTIC,
    FINANCIAL_BURDEN_FEATURES,
    TARGET_COL,
    UCI_TIMING_NOTE,
    build_preprocessor,
    get_dataset_split,
    get_feature_columns,
    prepare_modeling_table,
)
from .dataset_adapters import (
    PAY_STATUS_COLUMNS,
    UCI_DEFAULT_CREDIT_CARD_DISPLAY_NAME,
)
from .model_builders import build_logistic_estimator, build_xgboost_estimator
from .utils import (
    DATA_PROCESSED_DIR,
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
        by=["roc_auc", "accuracy"],
        ascending=[False, False],
    )


def compare_model_metrics(metrics_map: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(metrics_map).T.sort_values(by="roc_auc", ascending=False)
    out_csv = REPORTS_DIR / "model_validation" / "public_credit_model_comparison.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=True)
    save_json(
        metrics_map,
        REPORTS_DIR / "model_validation" / "public_credit_model_comparison.json",
    )
    return df


def _standard_experiments(df: pd.DataFrame) -> list[Dict[str, Any]]:
    return [
        run_model_experiment(
            df,
            build_logistic_estimator(),
            "logistic_public",
            FEATURE_SET_APPLICATION,
        ),
        run_model_experiment(
            df,
            build_xgboost_estimator(),
            "xgboost_public",
            FEATURE_SET_APPLICATION,
        ),
        run_model_experiment(
            df,
            build_xgboost_estimator(),
            "xgboost_full_public_diagnostic",
            FEATURE_SET_FULL_DIAGNOSTIC,
        ),
    ]


def _write_model_json(path: Path, result: Dict[str, Any], notes: list[str]) -> None:
    save_json(
        {
            "dataset": UCI_DEFAULT_CREDIT_CARD_DISPLAY_NAME,
            "target": TARGET_COL,
            "notes": notes,
            **_serializable_result(result),
        },
        path,
    )


def write_temporal_validation_note() -> pd.DataFrame:
    output = pd.DataFrame(
        [
            {
                "validation": "temporal_split",
                "status": "not_run",
                "reason": (
                    "The UCI Taiwan credit-card default dataset has no true application "
                    "timestamp, so the final reported metrics use a stratified random "
                    "held-out test split. Dates were not invented."
                ),
            }
        ]
    )
    out_path = REPORTS_DIR / "model_validation" / "temporal_split_comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)
    return output


def run_ablation_study(df: pd.DataFrame) -> pd.DataFrame:
    prepared = prepare_modeling_table(df, target_col=TARGET_COL)
    baseline_features = get_feature_columns(prepared, FEATURE_SET_APPLICATION)

    experiments = [
        ("baseline_application_public", baseline_features),
        (
            "without_repayment_status",
            [col for col in baseline_features if col not in set(PAY_STATUS_COLUMNS)],
        ),
        (
            "without_utilization_ratios",
            [col for col in baseline_features if col not in set(FINANCIAL_BURDEN_FEATURES)],
        ),
        (
            "without_profile_fields",
            [col for col in baseline_features if col not in {"AGE", "MARRIAGE", "EDUCATION"}],
        ),
    ]

    rows = []
    for name, feature_columns in experiments:
        result = run_model_experiment(
            df,
            build_xgboost_estimator(),
            f"xgboost_{name}",
            FEATURE_SET_APPLICATION,
            feature_columns=feature_columns,
        )
        row = {
            "ablation": name,
            "feature_count": len(feature_columns),
        }
        row.update(result["metrics"])
        rows.append(row)

    output = pd.DataFrame(rows).sort_values(by="roc_auc", ascending=False)
    out_path = REPORTS_DIR / "model_validation" / "ablation_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)
    return output


def run() -> pd.DataFrame:
    ensure_directories()
    df_raw, _ = load_dataset_auto()
    processed = prepare_modeling_table(df_raw, target_col=TARGET_COL)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed.to_csv(DATA_PROCESSED_DIR / "uci_taiwan_credit_default_processed.csv", index=False)

    standard_results = _standard_experiments(df_raw)
    standard_df = _flatten_results(standard_results)

    metrics_map = {result["model_name"]: result["metrics"] for result in standard_results}
    compare_model_metrics(metrics_map)

    model_validation_dir = REPORTS_DIR / "model_validation"
    model_validation_dir.mkdir(parents=True, exist_ok=True)
    standard_df.to_csv(model_validation_dir / "public_credit_model_comparison.csv", index=False)
    standard_df.to_csv(model_validation_dir / "clean_feature_model_comparison.csv", index=False)
    save_json(metrics_map, model_validation_dir / "clean_feature_model_comparison.json")

    result_by_name = {result["model_name"]: result for result in standard_results}
    _write_model_json(
        model_validation_dir / "logistic_public_model_metrics.json",
        result_by_name["logistic_public"],
        [FEATURE_POLICY_NOTE, UCI_TIMING_NOTE],
    )
    _write_model_json(
        model_validation_dir / "xgboost_public_model_metrics.json",
        result_by_name["xgboost_public"],
        [FEATURE_POLICY_NOTE, UCI_TIMING_NOTE],
    )
    _write_model_json(
        model_validation_dir / "full_public_diagnostic_model_metrics.json",
        result_by_name["xgboost_full_public_diagnostic"],
        [
            "Diagnostic-only feature set includes protected attribute SEX.",
            UCI_TIMING_NOTE,
        ],
    )

    # Compatibility filenames are overwritten so old local-case metrics are not left as headlines.
    _write_model_json(
        model_validation_dir / "application_model_metrics.json",
        result_by_name["xgboost_public"],
        [FEATURE_POLICY_NOTE, UCI_TIMING_NOTE],
    )
    write_temporal_validation_note()
    run_ablation_study(df_raw)
    return standard_df


if __name__ == "__main__":
    comparison_df = run()
    print(comparison_df.to_string(index=False))
