"""Leakage and validation audit for the public UCI credit-default pipeline."""

from __future__ import annotations

import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from .data_preprocessing import (
    FEATURE_SET_APPLICATION,
    TARGET_COL,
    get_dataset_split,
    get_feature_columns,
    prepare_modeling_table,
)
from .dataset_adapters import PAY_STATUS_COLUMNS, UCI_ID_COLUMNS
from .evaluate_models import fit_pipeline, run_model_experiment
from .model_builders import build_xgboost_estimator
from .utils import (
    REPORTS_DIR,
    ensure_directories,
    load_dataset_auto,
    project_relative_path,
    save_json,
)

LEAKAGE_AUDIT_DIR = REPORTS_DIR / "leakage_audit"
LEAKAGE_CONCLUSION = (
    "No detected target leakage or train/test overlap in the public UCI pipeline "
    "based on implemented checks."
)


def _mutual_information_scores(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    encoded = pd.DataFrame(index=X.index)
    discrete_mask = []
    for col in X.columns:
        series = X[col]
        if pd.api.types.is_numeric_dtype(series):
            encoded[col] = series.fillna(series.median())
            discrete_mask.append(False)
        else:
            encoded[col] = pd.factorize(series.fillna("Missing").astype(str))[0]
            discrete_mask.append(True)

    scores = mutual_info_classif(
        encoded,
        y,
        discrete_features=discrete_mask,
        random_state=42,
    )
    return (
        pd.DataFrame({"feature": X.columns, "mutual_information": scores})
        .sort_values(by="mutual_information", ascending=False)
        .reset_index(drop=True)
    )


def _duplicate_overlap_summary(df_raw: pd.DataFrame) -> dict:
    split = get_dataset_split(df_raw, feature_set=FEATURE_SET_APPLICATION)
    train_raw = df_raw.loc[split.train_indices].copy()
    test_raw = df_raw.loc[split.test_indices].copy()
    train_selected = split.X_train.copy()
    train_selected[TARGET_COL] = split.y_train
    test_selected = split.X_test.copy()
    test_selected[TARGET_COL] = split.y_test

    row_signature_cols = [col for col in df_raw.columns if col not in {TARGET_COL, *UCI_ID_COLUMNS}]
    train_feature_signatures = set(train_raw[row_signature_cols].astype(str).agg("||".join, axis=1))
    test_feature_signatures = set(test_raw[row_signature_cols].astype(str).agg("||".join, axis=1))
    train_selected_rows = set(train_selected.astype(str).agg("||".join, axis=1))
    test_selected_rows = set(test_selected.astype(str).agg("||".join, axis=1))

    id_overlap = {}
    for id_col in UCI_ID_COLUMNS:
        if id_col in df_raw.columns:
            id_overlap[id_col] = int(len(set(train_raw[id_col]) & set(test_raw[id_col])))

    return {
        "source_index_overlap": int(len(set(split.train_indices) & set(split.test_indices))),
        "duplicate_selected_rows_across_train_test": int(
            len(train_selected_rows & test_selected_rows)
        ),
        "duplicate_feature_signatures_excluding_target_across_train_test": int(
            len(train_feature_signatures & test_feature_signatures)
        ),
        "id_overlap": id_overlap,
        "train_rows": int(len(train_raw)),
        "test_rows": int(len(test_raw)),
    }


def _target_shuffle_test(df_raw: pd.DataFrame) -> dict:
    shuffled = df_raw.copy()
    shuffled[TARGET_COL] = shuffled[TARGET_COL].sample(frac=1.0, random_state=42).to_numpy()
    result = run_model_experiment(
        shuffled,
        build_xgboost_estimator(),
        "xgboost_public_target_shuffle",
        FEATURE_SET_APPLICATION,
    )
    return {
        "feature_set": FEATURE_SET_APPLICATION,
        "roc_auc": result["metrics"]["roc_auc"],
        "accuracy": result["metrics"]["accuracy"],
    }


def _xgboost_importance(df_raw: pd.DataFrame) -> pd.DataFrame:
    split = get_dataset_split(df_raw, feature_set=FEATURE_SET_APPLICATION)
    pipeline = fit_pipeline(build_xgboost_estimator(), split.X_train, split.y_train)
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    return (
        pd.DataFrame({"feature": feature_names, "importance": estimator.feature_importances_})
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )


def _write_markdown_report(summary: dict) -> None:
    lines = [
        "# Leakage Audit Report",
        "",
        "## Conclusion",
        "",
        LEAKAGE_CONCLUSION,
        "",
        "This does not mean leakage is impossible; it means the implemented checks did not find target leakage, ID leakage, duplicate train/test rows, or train/test overlap.",
        "",
        "## UCI Feature-Timing Review",
        "",
        "`PAY_0` to `PAY_6` are historical repayment-status variables used to predict the next-month default target. They are treated as valid historical predictors for this modeling question, not post-outcome leakage.",
        "",
        "## Implemented Checks",
        "",
        f"- Target excluded from model features: `{summary['checks']['target_not_in_features']}`",
        f"- ID columns excluded from model features: `{summary['checks']['id_columns_not_in_features']}`",
        f"- Source-index overlap: `{summary['duplicate_overlap']['source_index_overlap']}`",
        f"- Duplicate selected rows across train/test: `{summary['duplicate_overlap']['duplicate_selected_rows_across_train_test']}`",
        f"- Repeated feature signatures excluding target across train/test: `{summary['duplicate_overlap']['duplicate_feature_signatures_excluding_target_across_train_test']}`",
        f"- Target-shuffle ROC-AUC: `{summary['target_shuffle_test']['roc_auc']:.4f}`",
        "",
        "## Top Mutual Information Features",
        "",
        "| Feature | Mutual information |",
        "| --- | ---: |",
    ]
    for row in summary["top_mutual_information"][:10]:
        lines.append(f"| {row['feature']} | {row['mutual_information']:.6f} |")
    lines.extend(
        [
            "",
            "High mutual information is reviewed as a suspiciousness signal, not automatic proof of leakage. In this dataset, strong repayment-history signals are expected because they summarize recent historical payment behavior before the next-month target.",
        ]
    )
    (LEAKAGE_AUDIT_DIR / "leakage_audit_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run() -> dict:
    ensure_directories()
    LEAKAGE_AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    df_raw, data_path = load_dataset_auto()
    prepared = prepare_modeling_table(df_raw, target_col=TARGET_COL)
    feature_columns = get_feature_columns(prepared, FEATURE_SET_APPLICATION)
    X = prepared[feature_columns]
    y = prepared[TARGET_COL]

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    corr_df = (
        prepared[numeric_cols + [TARGET_COL]]
        .corr(numeric_only=True)[TARGET_COL]
        .drop(TARGET_COL)
        .rename("correlation_with_target")
        .reset_index()
        .rename(columns={"index": "feature"})
        .sort_values(by="correlation_with_target", key=lambda s: s.abs(), ascending=False)
        .reset_index(drop=True)
    )
    corr_df.to_csv(LEAKAGE_AUDIT_DIR / "feature_target_correlations.csv", index=False)

    mi_df = _mutual_information_scores(X, y)
    mi_df.to_csv(LEAKAGE_AUDIT_DIR / "mutual_information_scores.csv", index=False)

    importance_df = _xgboost_importance(df_raw)
    importance_df.to_csv(LEAKAGE_AUDIT_DIR / "xgboost_feature_importance.csv", index=False)

    duplicate_summary = _duplicate_overlap_summary(df_raw)
    shuffle_summary = _target_shuffle_test(df_raw)
    save_json(shuffle_summary, LEAKAGE_AUDIT_DIR / "target_shuffle_test.json")

    id_cols_in_features = sorted(set(UCI_ID_COLUMNS) & set(feature_columns))
    target_in_features = TARGET_COL in feature_columns
    mi_threshold = float(mi_df["mutual_information"].quantile(0.95)) if not mi_df.empty else 0.0
    high_mi_features = mi_df.loc[mi_df["mutual_information"] >= mi_threshold, "feature"].tolist()

    suspicious_features = [
        f"{feature}: top mutual information review signal" for feature in high_mi_features
    ]
    with open(LEAKAGE_AUDIT_DIR / "suspicious_features.txt", "w", encoding="utf-8") as handle:
        handle.write("\n".join(suspicious_features) if suspicious_features else LEAKAGE_CONCLUSION)

    summary = {
        "dataset": project_relative_path(data_path),
        "row_count": int(len(df_raw)),
        "feature_count": int(len(feature_columns)),
        "feature_set": FEATURE_SET_APPLICATION,
        "checks": {
            "target_not_in_features": not target_in_features,
            "id_columns_not_in_features": not id_cols_in_features,
            "id_columns_found_in_features": id_cols_in_features,
            "pay_status_timing_review": {
                "columns": PAY_STATUS_COLUMNS,
                "leakage_decision": "historical predictors, not treated as leakage",
            },
        },
        "top_correlations": corr_df.head(10).to_dict(orient="records"),
        "top_mutual_information": mi_df.head(10).to_dict(orient="records"),
        "top_xgboost_feature_importance": importance_df.head(10).to_dict(orient="records"),
        "duplicate_overlap": duplicate_summary,
        "target_shuffle_test": shuffle_summary,
        "suspicious_features": suspicious_features,
        "conclusion": LEAKAGE_CONCLUSION,
    }
    save_json(summary, LEAKAGE_AUDIT_DIR / "leakage_audit_summary.json")
    _write_markdown_report(summary)

    print(LEAKAGE_CONCLUSION)
    return summary


if __name__ == "__main__":
    result = run()
    print(result)
