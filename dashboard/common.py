"""Shared dashboard utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / "reports"

from src.data_preprocessing import (  # noqa: E402
    FEATURE_SET_APPLICATION,
    TARGET_COL,
    get_feature_columns,
    prepare_modeling_table,
)
from src.train_logistic import run as train_logistic_run  # noqa: E402
from src.train_xgboost import run as train_xgboost_run  # noqa: E402
from src.utils import MODELS_DIR, load_dataset_auto, load_model  # noqa: E402


def load_data_for_dashboard() -> pd.DataFrame:
    df, _ = load_dataset_auto()
    return df


def ensure_model(model_choice: str):
    if model_choice == "Logistic Regression":
        path = MODELS_DIR / "logistic_public.pkl"
        if not path.exists():
            train_logistic_run(path)
    else:
        path = MODELS_DIR / "xgboost_public.pkl"
        if not path.exists():
            train_xgboost_run(path)
    return load_model(path), path


def get_feature_table() -> pd.DataFrame:
    raw = load_data_for_dashboard()
    prepared = prepare_modeling_table(raw, target_col=TARGET_COL)
    feature_columns = get_feature_columns(prepared, feature_set=FEATURE_SET_APPLICATION)
    return prepared[feature_columns].copy()


def get_application_artifact_paths() -> dict[str, Path]:
    return {
        "performance": REPORTS_DIR / "model_validation" / "public_credit_model_comparison.csv",
        "temporal": REPORTS_DIR / "model_validation" / "temporal_split_comparison.csv",
        "fairness_csv": REPORTS_DIR
        / "fairness_reports"
        / "application_model"
        / "xgboost_public_fairness_metrics.csv",
        "fairness_json": REPORTS_DIR
        / "fairness_reports"
        / "application_model"
        / "xgboost_public_fairness_metrics.json",
        "mitigation": REPORTS_DIR
        / "fairness_reports"
        / "application_model"
        / "xgboost_public_fairness_accuracy_tradeoff.csv",
        "recall_summary": REPORTS_DIR / "model_validation" / "recall_optimized_summary.json",
        "selected_recall_policy": REPORTS_DIR / "model_validation" / "selected_recall_policy.json",
        "threshold_tuning": REPORTS_DIR / "model_validation" / "threshold_tuning_report.csv",
        "threshold_selection": REPORTS_DIR / "model_validation" / "threshold_selection_summary.csv",
        "precision_recall_curve": REPORTS_DIR
        / "model_validation"
        / "precision_recall_curve_comparison.png",
        "threshold_fairness_comparison": REPORTS_DIR
        / "fairness_reports"
        / "application_model"
        / "threshold_fairness_comparison.csv",
        "fairness_deep_dive_summary": REPORTS_DIR
        / "fairness_reports"
        / "application_model"
        / "fairness_deep_dive_summary.json",
        "fairness_group_outcome": REPORTS_DIR
        / "fairness_reports"
        / "application_model"
        / "group_outcome_analysis_by_sex.csv",
        "fairness_group_error": REPORTS_DIR
        / "fairness_reports"
        / "application_model"
        / "group_error_analysis_by_sex.csv",
        "fairness_group_calibration": REPORTS_DIR
        / "fairness_reports"
        / "application_model"
        / "group_calibration_by_sex.csv",
        "fairness_proxy_predictability": REPORTS_DIR
        / "fairness_reports"
        / "application_model"
        / "proxy_sex_predictability.csv",
        "fairness_feature_association": REPORTS_DIR
        / "fairness_reports"
        / "application_model"
        / "feature_association_with_sex.csv",
        "fairness_threshold_frontier": REPORTS_DIR
        / "fairness_reports"
        / "application_model"
        / "threshold_fairness_frontier.csv",
        "fairness_individual_sensitivity": REPORTS_DIR
        / "fairness_reports"
        / "application_model"
        / "individual_sex_sensitivity.csv",
        "fairness_nearest_neighbour": REPORTS_DIR
        / "fairness_reports"
        / "application_model"
        / "nearest_neighbour_individual_fairness.csv",
        "leakage": REPORTS_DIR / "leakage_audit" / "leakage_audit_summary.json",
        "leakage_report": REPORTS_DIR / "leakage_audit" / "leakage_audit_report.md",
        "counterfactual": REPORTS_DIR
        / "explainability_reports"
        / "application_model"
        / "xgboost_public_counterfactuals.json",
        "shap_summary": REPORTS_DIR
        / "explainability_reports"
        / "application_model"
        / "xgboost_public_shap_summary.png",
        "shap_local": REPORTS_DIR
        / "explainability_reports"
        / "application_model"
        / "xgboost_public_shap_local.png",
        "lime_local": REPORTS_DIR
        / "explainability_reports"
        / "application_model"
        / "xgboost_public_lime_local.png",
        "deep_learning_metrics": REPORTS_DIR / "model_validation" / "deep_learning_metrics.json",
        "deep_learning_comparison": REPORTS_DIR
        / "model_validation"
        / "deep_learning_comparison.csv",
        "ml_vs_dl_comparison": REPORTS_DIR / "model_validation" / "ml_vs_dl_comparison.csv",
        "ml_vs_dl_precision_recall_curve": REPORTS_DIR
        / "model_validation"
        / "ml_vs_dl_precision_recall_curve.png",
        "logistic_test_predictions": REPORTS_DIR
        / "model_validation"
        / "logistic_test_predictions.csv",
        "xgboost_test_predictions": REPORTS_DIR
        / "model_validation"
        / "xgboost_test_predictions.csv",
        "dnn_test_predictions": REPORTS_DIR / "model_validation" / "dnn_test_predictions.csv",
        "deep_learning_policy": REPORTS_DIR
        / "model_validation"
        / "deep_learning_selected_policy.json",
        "deep_learning_fairness": REPORTS_DIR
        / "fairness_reports"
        / "deep_learning_model"
        / "dnn_fairness_metrics.csv",
    }
