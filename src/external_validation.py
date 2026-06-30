"""External public-dataset validation workflows.

These workflows train separate benchmark models on public datasets. They do not apply the
Dubai-trained model to external schemas and do not alter the primary project results.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_api_loader import load_dataset
from .fairness_metrics import compute_fairness_metrics
from .utils import REPORTS_DIR, project_relative_path, save_json

DATASET_DEFAULT_CREDIT_CARD = "default_credit_card"
SUPPORTED_DATASETS = [DATASET_DEFAULT_CREDIT_CARD]
TARGET_COLUMN = "DEFAULT_PAYMENT_NEXT_MONTH"
APPROVAL_THRESHOLD = 0.50

DEFAULT_CREDIT_CARD_COLUMN_MAP = {
    "X1": "LIMIT_BAL",
    "X2": "SEX",
    "X3": "EDUCATION",
    "X4": "MARRIAGE",
    "X5": "AGE",
    "X6": "PAY_0",
    "X7": "PAY_2",
    "X8": "PAY_3",
    "X9": "PAY_4",
    "X10": "PAY_5",
    "X11": "PAY_6",
    "X12": "BILL_AMT1",
    "X13": "BILL_AMT2",
    "X14": "BILL_AMT3",
    "X15": "BILL_AMT4",
    "X16": "BILL_AMT5",
    "X17": "BILL_AMT6",
    "X18": "PAY_AMT1",
    "X19": "PAY_AMT2",
    "X20": "PAY_AMT3",
    "X21": "PAY_AMT4",
    "X22": "PAY_AMT5",
    "X23": "PAY_AMT6",
    "Y": TARGET_COLUMN,
    "DEFAULT_PAYMENT_NEXT_MONTH": TARGET_COLUMN,
}

DEFAULT_CREDIT_CARD_FEATURES = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]

DEFAULT_CREDIT_CARD_CATEGORICAL_FEATURES = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
]


@dataclass(frozen=True)
class ExternalDataset:
    name: str
    source: str
    target_column: str
    X: pd.DataFrame
    y: pd.Series
    sensitive_features: dict[str, pd.Series]


def _clean_column_key(column: object) -> str:
    key = str(column).strip().replace(".", "_").replace("-", "_").replace(" ", "_")
    while "__" in key:
        key = key.replace("__", "_")
    return key.upper()


def normalize_default_credit_card_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize UCI Taiwan credit-card columns to business-readable names."""

    rename_map = {}
    for column in df.columns:
        key = _clean_column_key(column)
        rename_map[column] = DEFAULT_CREDIT_CARD_COLUMN_MAP.get(key, key)
    return df.rename(columns=rename_map)


def _build_age_group(age: pd.Series) -> pd.Series:
    age_numeric = pd.to_numeric(age, errors="coerce")
    bins = [0, 26, 36, 46, 56, 66, np.inf]
    labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]
    grouped = pd.cut(age_numeric, bins=bins, labels=labels, right=False)
    return grouped.astype("object").where(grouped.notna(), "unknown").astype(str)


def _build_sex_group(sex: pd.Series) -> pd.Series:
    sex_numeric = pd.to_numeric(sex, errors="coerce")
    labels = sex_numeric.map({1: "male", 2: "female"})
    return labels.where(labels.notna(), "unknown").astype(str)


def prepare_default_credit_card_dataset(
    df: pd.DataFrame,
    source: str = "uci",
) -> ExternalDataset:
    """Prepare the UCI Default of Credit Card Clients dataset for benchmarking."""

    out = normalize_default_credit_card_columns(df)
    required_columns = DEFAULT_CREDIT_CARD_FEATURES + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in out.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Default credit-card dataset is missing required columns: {missing}")

    out = out[required_columns].copy()
    for column in required_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=[TARGET_COLUMN])
    out[TARGET_COLUMN] = out[TARGET_COLUMN].astype(int)

    X = out[DEFAULT_CREDIT_CARD_FEATURES].copy()
    for column in DEFAULT_CREDIT_CARD_CATEGORICAL_FEATURES:
        X[column] = X[column].astype("Int64").astype(str)

    y = out[TARGET_COLUMN].astype(int)
    sensitive_features = {
        "SEX": _build_sex_group(out["SEX"]),
        "AGE_GROUP": _build_age_group(out["AGE"]),
    }

    return ExternalDataset(
        name=DATASET_DEFAULT_CREDIT_CARD,
        source=source,
        target_column=TARGET_COLUMN,
        X=X,
        y=y,
        sensitive_features=sensitive_features,
    )


def build_external_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = [
        column for column in DEFAULT_CREDIT_CARD_CATEGORICAL_FEATURES if column in X.columns
    ]
    numeric_cols = [column for column in X.columns if column not in categorical_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )


def build_logistic_pipeline(X: pd.DataFrame) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_external_preprocessor(X)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def build_xgboost_pipeline(X: pd.DataFrame) -> Pipeline | None:
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None

    return Pipeline(
        steps=[
            ("preprocessor", build_external_preprocessor(X)),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=150,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_binary_classifier(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float | None]:
    try:
        roc_auc = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        roc_auc = None

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
    }


def compute_external_fairness_metrics(
    y_default_true: pd.Series,
    default_proba: np.ndarray,
    sensitive: pd.Series,
    threshold: float = APPROVAL_THRESHOLD,
) -> dict[str, float]:
    """Compute approval-decision fairness metrics from default probabilities."""

    approval_pred = (default_proba < threshold).astype(int)
    approval_true = (1 - y_default_true).astype(int)
    return compute_fairness_metrics(
        approval_true.values, approval_pred, sensitive.astype(str).values
    )


def _fit_external_models(
    dataset: ExternalDataset,
    include_xgboost: bool = True,
    test_size: float = 0.20,
    random_state: int = 42,
) -> dict[str, Any]:
    stratify = dataset.y if dataset.y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X,
        dataset.y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    model_pipelines: dict[str, Pipeline] = {
        "logistic_regression": build_logistic_pipeline(dataset.X)
    }
    if include_xgboost:
        xgboost_pipeline = build_xgboost_pipeline(dataset.X)
        if xgboost_pipeline is not None:
            model_pipelines["xgboost"] = xgboost_pipeline

    model_metrics = {}
    fairness_metrics = {}
    for model_name, pipeline in model_pipelines.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        model_metrics[model_name] = evaluate_binary_classifier(y_test, y_pred, y_proba)

        model_fairness = {}
        for attribute, sensitive in dataset.sensitive_features.items():
            sensitive_test = sensitive.loc[X_test.index]
            model_fairness[attribute] = compute_external_fairness_metrics(
                y_default_true=y_test,
                default_proba=y_proba,
                sensitive=sensitive_test,
            )
        fairness_metrics[model_name] = model_fairness

    return {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_columns": dataset.X.columns.tolist(),
        "model_metrics": model_metrics,
        "fairness_metrics": fairness_metrics,
    }


def _model_comparison_frame(model_metrics: dict[str, dict[str, float | None]]) -> pd.DataFrame:
    rows = []
    for model_name, metrics in model_metrics.items():
        row = {"model_name": model_name}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(by=["roc_auc", "accuracy"], ascending=False)


def _fairness_frame(fairness_metrics: dict[str, dict[str, dict[str, float]]]) -> pd.DataFrame:
    rows = []
    for model_name, attribute_metrics in fairness_metrics.items():
        for protected_attribute, metrics in attribute_metrics.items():
            row = {
                "model_name": model_name,
                "protected_attribute": protected_attribute,
            }
            row.update(metrics)
            rows.append(row)
    return pd.DataFrame(rows)


def _format_metric(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.4f}"


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(_format_metric(value) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_external_validation_outputs(
    result: dict[str, Any],
    report_dir: Path,
) -> dict[str, str]:
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "dataset": result["dataset"],
        "source": result["source"],
        "target_column": result["target_column"],
        "row_count": result["row_count"],
        "train_rows": result["train_rows"],
        "test_rows": result["test_rows"],
        "feature_columns": result["feature_columns"],
        "model_metrics": result["model_metrics"],
        "notes": [
            "External validation trains separate models on the public dataset.",
            "These metrics do not replace the Dubai Arab Bank case-study headline results.",
        ],
    }
    fairness_payload = {
        "dataset": result["dataset"],
        "decision_target": "approval",
        "approval_threshold": APPROVAL_THRESHOLD,
        "protected_attributes": list(result["sensitive_attributes"]),
        "fairness_metrics": result["fairness_metrics"],
    }

    metrics_json = report_dir / "metrics.json"
    fairness_json = report_dir / "fairness_metrics.json"
    fairness_csv = report_dir / "fairness_metrics.csv"
    comparison_csv = report_dir / "model_comparison.csv"
    summary_md = report_dir / "summary.md"

    save_json(metrics_payload, metrics_json)
    save_json(fairness_payload, fairness_json)

    comparison_df = _model_comparison_frame(result["model_metrics"])
    fairness_df = _fairness_frame(result["fairness_metrics"])
    comparison_df.to_csv(comparison_csv, index=False)
    fairness_df.to_csv(fairness_csv, index=False)

    summary_lines = [
        "# External Validation: UCI Default of Credit Card Clients (Taiwan credit-card default)",
        "",
        "This benchmark trains separate models on the public UCI Default of Credit Card",
        "Clients / Taiwan credit-card default dataset.",
        "It does not apply the Dubai-trained model to the UCI schema and does not replace the",
        "primary Dubai Arab Bank case-study results.",
        "It is not evidence of production lending readiness or direct production generalization.",
        "",
        "## Dataset",
        "",
        "- Dataset: `UCI Default of Credit Card Clients / Taiwan credit-card default`",
        f"- Source: `{result['source']}`",
        f"- Rows: `{result['row_count']}`",
        f"- Target: `{result['target_column']}`",
        f"- Features: `{len(result['feature_columns'])}`",
        "- Protected attributes audited: `SEX`, `AGE_GROUP`",
        "",
        "## Model Metrics",
        "",
        *_markdown_table(
            comparison_df,
            ["model_name", "accuracy", "precision", "recall", "f1", "roc_auc"],
        ),
        "",
        "## Fairness Metrics",
        "",
        *_markdown_table(
            fairness_df,
            [
                "model_name",
                "protected_attribute",
                "demographic_parity_difference",
                "equal_opportunity_difference",
                "equalized_odds_difference",
                "disparate_impact_ratio",
            ],
        ),
        "",
        "Fairness metrics are calculated on approval decisions derived from default",
        f"probabilities using threshold `{APPROVAL_THRESHOLD}`.",
        "They are saved separately from the Dubai fairness reports and should be read as",
        "group-level diagnostics, not proof that the benchmark model is bias-free.",
    ]
    summary_md.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return {
        "metrics_json": project_relative_path(metrics_json) or str(metrics_json),
        "fairness_json": project_relative_path(fairness_json) or str(fairness_json),
        "fairness_csv": project_relative_path(fairness_csv) or str(fairness_csv),
        "model_comparison_csv": project_relative_path(comparison_csv) or str(comparison_csv),
        "summary_md": project_relative_path(summary_md) or str(summary_md),
    }


def run_default_credit_card_validation(
    df: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
    output_dir: Path | None = None,
    include_xgboost: bool = True,
) -> dict[str, Any]:
    metadata = metadata or {}
    dataset = prepare_default_credit_card_dataset(
        df,
        source=str(metadata.get("source", "provided")),
    )
    training_result = _fit_external_models(dataset, include_xgboost=include_xgboost)
    result = {
        "dataset": DATASET_DEFAULT_CREDIT_CARD,
        "source": dataset.source,
        "target_column": dataset.target_column,
        "row_count": int(len(dataset.X)),
        "sensitive_attributes": dataset.sensitive_features.keys(),
        **training_result,
    }

    report_dir = output_dir or (REPORTS_DIR / "external_validation" / DATASET_DEFAULT_CREDIT_CARD)
    result["output_files"] = write_external_validation_outputs(result, report_dir)
    return result


def run_external_validation(
    dataset: str = DATASET_DEFAULT_CREDIT_CARD,
    output_dir: Path | None = None,
    include_xgboost: bool = True,
) -> dict[str, Any]:
    if dataset != DATASET_DEFAULT_CREDIT_CARD:
        raise ValueError(f"Unsupported external validation dataset: {dataset}")

    df, metadata = load_dataset(source="uci", dataset_name=DATASET_DEFAULT_CREDIT_CARD)
    return run_default_credit_card_validation(
        df,
        metadata=metadata,
        output_dir=output_dir,
        include_xgboost=include_xgboost,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run separate external validation on public credit datasets."
    )
    parser.add_argument(
        "--dataset", default=DATASET_DEFAULT_CREDIT_CARD, choices=SUPPORTED_DATASETS
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional report output directory. Defaults to reports/external_validation/<dataset>/.",
    )
    parser.add_argument(
        "--no-xgboost",
        action="store_true",
        help="Skip the optional XGBoost benchmark.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_external_validation(
            dataset=args.dataset,
            output_dir=args.output_dir,
            include_xgboost=not args.no_xgboost,
        )
    except Exception as exc:
        print(f"External validation failed: {exc}")
        return 1

    print("External validation completed.")
    for label, path in result["output_files"].items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
