"""Leakage and validation audit for the credit default pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.feature_selection import mutual_info_classif

from .data_preprocessing import (
    FEATURE_SET_FULL_DIAGNOSTIC,
    POST_OUTCOME_BEHAVIORAL_FEATURES,
    TARGET_COL,
    get_dataset_split,
    get_feature_columns,
    prepare_modeling_table,
)
from .evaluate_models import fit_pipeline, run_model_experiment
from .model_builders import build_xgboost_estimator
from .utils import REPORTS_DIR, ensure_directories, load_dataset_auto, save_json

LEAKAGE_AUDIT_DIR = REPORTS_DIR / "leakage_audit"


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


def _single_feature_auc(df_raw: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    rows = []
    for feature in feature_columns:
        result = run_model_experiment(
            df_raw,
            build_xgboost_estimator(),
            f"single_feature_{feature}",
            FEATURE_SET_FULL_DIAGNOSTIC,
            feature_columns=[feature],
        )
        rows.append(
            {
                "feature": feature,
                "roc_auc": result["metrics"]["roc_auc"],
                "accuracy": result["metrics"]["accuracy"],
            }
        )

    return pd.DataFrame(rows).sort_values(by="roc_auc", ascending=False).reset_index(drop=True)


def _xgboost_importance_and_shap(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = get_dataset_split(df_raw, feature_set=FEATURE_SET_FULL_DIAGNOSTIC)
    pipeline = fit_pipeline(build_xgboost_estimator(), split.X_train, split.y_train)

    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["classifier"]
    Xt_test = preprocessor.transform(split.X_test)
    feature_names = preprocessor.get_feature_names_out()
    feature_frame = pd.DataFrame(
        Xt_test.toarray() if hasattr(Xt_test, "toarray") else Xt_test,
        columns=feature_names,
        index=split.X_test.index,
    )

    importance_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": estimator.feature_importances_,
            }
        )
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )

    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(feature_frame)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    shap_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            }
        )
        .sort_values(by="mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return importance_df, shap_df


def _duplicate_overlap_summary(df_raw: pd.DataFrame) -> dict:
    split = get_dataset_split(df_raw, feature_set=FEATURE_SET_FULL_DIAGNOSTIC)
    train_raw = df_raw.loc[split.train_indices].copy()
    test_raw = df_raw.loc[split.test_indices].copy()

    row_signature_cols = [col for col in df_raw.columns if col != TARGET_COL]
    train_rows = set(train_raw.astype(str).agg("||".join, axis=1))
    test_rows = set(test_raw.astype(str).agg("||".join, axis=1))
    train_rows_no_target = set(train_raw[row_signature_cols].astype(str).agg("||".join, axis=1))
    test_rows_no_target = set(test_raw[row_signature_cols].astype(str).agg("||".join, axis=1))

    summary = {
        "duplicate_rows_across_train_test": int(len(train_rows & test_rows)),
        "duplicate_rows_excluding_target_across_train_test": int(
            len(train_rows_no_target & test_rows_no_target)
        ),
        "duplicate_customer_ids_across_train_test": 0,
        "duplicate_loan_ids_across_train_test": 0,
    }
    if "CustomerID" in df_raw.columns:
        summary["duplicate_customer_ids_across_train_test"] = int(
            len(set(train_raw["CustomerID"]) & set(test_raw["CustomerID"]))
        )
    if "LoanID" in df_raw.columns:
        summary["duplicate_loan_ids_across_train_test"] = int(
            len(set(train_raw["LoanID"]) & set(test_raw["LoanID"]))
        )
    return summary


def _target_shuffle_test(df_raw: pd.DataFrame) -> dict:
    shuffled = df_raw.copy()
    shuffled[TARGET_COL] = shuffled[TARGET_COL].sample(frac=1.0, random_state=42).to_numpy()
    result = run_model_experiment(
        shuffled,
        build_xgboost_estimator(),
        "xgboost_full_diagnostic_target_shuffle",
        FEATURE_SET_FULL_DIAGNOSTIC,
    )
    return {
        "feature_set": FEATURE_SET_FULL_DIAGNOSTIC,
        "roc_auc": result["metrics"]["roc_auc"],
        "accuracy": result["metrics"]["accuracy"],
    }


def run() -> dict:
    ensure_directories()
    LEAKAGE_AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    df_raw, data_path = load_dataset_auto()
    prepared = prepare_modeling_table(df_raw, target_col=TARGET_COL)
    feature_columns = get_feature_columns(prepared, FEATURE_SET_FULL_DIAGNOSTIC)
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

    single_feature_auc_df = _single_feature_auc(df_raw, feature_columns)
    single_feature_auc_df.to_csv(LEAKAGE_AUDIT_DIR / "single_feature_auc.csv", index=False)

    importance_df, shap_df = _xgboost_importance_and_shap(df_raw)
    importance_df.to_csv(LEAKAGE_AUDIT_DIR / "xgboost_feature_importance.csv", index=False)

    duplicate_summary = _duplicate_overlap_summary(df_raw)
    shuffle_summary = _target_shuffle_test(df_raw)
    save_json(shuffle_summary, LEAKAGE_AUDIT_DIR / "target_shuffle_test.json")

    suspicious_features: list[str] = []
    corr_hits = corr_df.loc[corr_df["correlation_with_target"].abs() > 0.80, "feature"].tolist()
    auc_hits = single_feature_auc_df.loc[single_feature_auc_df["roc_auc"] > 0.90, "feature"].tolist()
    mi_threshold = float(mi_df["mutual_information"].quantile(0.95)) if not mi_df.empty else 0.0
    mi_hits = mi_df.loc[mi_df["mutual_information"] >= mi_threshold, "feature"].tolist()
    behavior_hits = [feature for feature in POST_OUTCOME_BEHAVIORAL_FEATURES if feature in feature_columns]

    suspicious_features.extend([f"{feature}: abs(correlation) > 0.80" for feature in corr_hits])
    suspicious_features.extend([f"{feature}: single-feature AUC > 0.90" for feature in auc_hits])
    suspicious_features.extend([f"{feature}: top mutual information" for feature in mi_hits])
    suspicious_features.extend(
        [f"{feature}: post-loan behavioral feature" for feature in behavior_hits]
    )
    suspicious_features = sorted(set(suspicious_features))

    with open(LEAKAGE_AUDIT_DIR / "suspicious_features.txt", "w", encoding="utf-8") as handle:
        handle.write("\n".join(suspicious_features) if suspicious_features else "No suspicious features found.")

    summary = {
        "dataset": str(data_path),
        "row_count": int(len(df_raw)),
        "feature_count": int(len(feature_columns)),
        "top_correlations": corr_df.head(10).to_dict(orient="records"),
        "top_mutual_information": mi_df.head(10).to_dict(orient="records"),
        "top_single_feature_auc": single_feature_auc_df.head(10).to_dict(orient="records"),
        "top_xgboost_feature_importance": importance_df.head(10).to_dict(orient="records"),
        "top_shap_features": shap_df.head(10).to_dict(orient="records"),
        "duplicate_overlap": duplicate_summary,
        "target_shuffle_test": shuffle_summary,
        "suspicious_features": suspicious_features,
    }
    save_json(summary, LEAKAGE_AUDIT_DIR / "leakage_audit_summary.json")

    warnings = []
    if corr_hits:
        warnings.append(f"High correlation features: {', '.join(corr_hits)}")
    if auc_hits:
        warnings.append(f"High single-feature AUC features: {', '.join(auc_hits)}")
    if mi_hits:
        warnings.append(f"Top mutual information features: {', '.join(mi_hits[:5])}")
    if behavior_hits:
        warnings.append(f"Post-loan behavioral features present: {', '.join(behavior_hits[:5])}")

    if warnings:
        print("WARNING: potential leakage signals detected.")
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("No major leakage warnings triggered by the configured thresholds.")

    return summary


if __name__ == "__main__":
    result = run()
    print(result)
