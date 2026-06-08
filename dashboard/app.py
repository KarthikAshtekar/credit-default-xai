"""Presentation-ready Streamlit dashboard for the validated application model."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .common import ensure_model, get_feature_table
except ImportError:
    from common import ensure_model, get_feature_table


REPORTS_DIR = ROOT / "reports"


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def show_warning_if_missing(label: str, path: Path) -> None:
    st.warning(f"{label} is missing: `{path}`")


def risk_band(probability: float) -> str:
    if probability < 0.30:
        return "Low Risk"
    if probability <= 0.60:
        return "Medium Risk"
    return "High Risk"


def format_counterfactual_changes(counterfactual_payload: dict[str, Any]) -> pd.DataFrame:
    feature_names = counterfactual_payload["feature_names"]
    original_values = counterfactual_payload["test_data"][0][0][: len(feature_names)]
    counterfactual_values = counterfactual_payload["cfs_list"][0][0][: len(feature_names)]
    original = pd.Series(original_values, index=feature_names, name="original")
    first_cf = pd.Series(
        counterfactual_values,
        index=feature_names,
        name="counterfactual",
    )
    changes = pd.DataFrame({"original": original, "counterfactual": first_cf})
    return changes[changes["original"].astype(str) != changes["counterfactual"].astype(str)]


st.set_page_config(
    page_title="Explainable and Fair Credit Default Risk Prediction",
    page_icon="📊",
    layout="wide",
)

st.title("Explainable and Fair Credit Default Risk Prediction")
st.caption(
    "Leakage-audited application-time credit default risk modeling with XGBoost, SHAP, "
    "LIME, counterfactual explanations, and fairness analysis."
)

performance_path = REPORTS_DIR / "model_validation" / "clean_feature_model_comparison.csv"
temporal_path = REPORTS_DIR / "model_validation" / "temporal_split_comparison.csv"
fairness_csv_path = (
    REPORTS_DIR / "fairness_reports" / "application_model" / "xgboost_application_fairness_metrics.csv"
)
fairness_json_path = (
    REPORTS_DIR / "fairness_reports" / "application_model" / "xgboost_application_fairness_metrics.json"
)
mitigation_path = (
    REPORTS_DIR
    / "fairness_reports"
    / "application_model"
    / "xgboost_application_fairness_accuracy_tradeoff.csv"
)
leakage_path = REPORTS_DIR / "leakage_audit" / "leakage_audit_summary.json"
counterfactual_path = (
    REPORTS_DIR
    / "explainability_reports"
    / "application_model"
    / "xgboost_application_counterfactuals.json"
)
shap_summary_path = (
    REPORTS_DIR
    / "explainability_reports"
    / "application_model"
    / "xgboost_application_shap_summary.png"
)
shap_local_path = (
    REPORTS_DIR
    / "explainability_reports"
    / "application_model"
    / "xgboost_application_shap_local.png"
)

tab_overview, tab_prediction, tab_performance, tab_explainability, tab_fairness, tab_counterfactual, tab_leakage = (
    st.tabs(
        [
            "Project Overview",
            "Applicant Risk Prediction",
            "Model Performance",
            "Explainability",
            "Fairness Analysis",
            "Counterfactual Guidance",
            "Leakage Audit",
        ]
    )
)

with tab_overview:
    st.subheader("Business Problem")
    st.write(
        "Credit default models influence access to financial opportunity. This project focuses on "
        "loan application-time prediction, where the model must be accurate enough to support "
        "underwriting while also remaining transparent and fair."
    )
    st.subheader("Responsible AI Framing")
    st.write(
        "The project explicitly rejects the tempting near-perfect model that uses post-loan "
        "behavioral signals. The final model is `xgboost_application.pkl`, built only from "
        "variables available at loan start."
    )
    st.info(
        "Final application model metrics: accuracy 0.7105, precision 0.6579, recall 0.7503, "
        "F1 0.7011, ROC-AUC 0.7825."
    )

with tab_prediction:
    st.subheader("Applicant Risk Prediction")
    model, model_path = ensure_model("XGBoost")
    feature_table = get_feature_table()
    st.write("Input simplified applicant details to score application-time default risk.")
    numeric_cols = feature_table.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = feature_table.select_dtypes(exclude=["number"]).columns.tolist()

    user_input: dict[str, Any] = {}
    left, right = st.columns(2)
    for idx, col in enumerate(numeric_cols):
        default_val = float(feature_table[col].median())
        target_col = left if idx % 2 == 0 else right
        user_input[col] = target_col.number_input(col, value=default_val)
    for idx, col in enumerate(categorical_cols):
        options = sorted(feature_table[col].dropna().astype(str).unique().tolist()) or ["Unknown"]
        target_col = left if idx % 2 == 0 else right
        user_input[col] = target_col.selectbox(col, options=options)

    input_df = pd.DataFrame([user_input])
    if st.button("Predict Application Risk"):
        probability = float(model.predict_proba(input_df)[:, 1][0])
        band = risk_band(probability)
        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("Default Probability", f"{probability:.2%}")
        metric_b.metric("Risk Category", band)
        metric_c.metric("Model", model_path.name)
        st.caption(
            "Low Risk: p < 0.30 | Medium Risk: 0.30 <= p <= 0.60 | High Risk: p > 0.60"
        )

with tab_performance:
    st.subheader("Validated Model Performance")
    performance_df = load_csv(performance_path)
    if performance_df is None:
        show_warning_if_missing("Model comparison file", performance_path)
    else:
        st.dataframe(performance_df, width="stretch")
        roc_plot = performance_df[["model_name", "roc_auc"]].set_index("model_name")
        st.bar_chart(roc_plot)
        st.write(
            "Use the application-time rows as the main presentation result. Behavioral and full "
            "diagnostic variants remain in the table to document the leakage audit outcome."
        )

    temporal_df = load_csv(temporal_path)
    if temporal_df is None:
        show_warning_if_missing("Temporal comparison file", temporal_path)
    else:
        st.subheader("Temporal Validation")
        st.dataframe(
            temporal_df[temporal_df["model_name"].str.contains("application")],
            width="stretch",
        )

with tab_explainability:
    st.subheader("SHAP-Based Explainability")
    if shap_summary_path.exists():
        st.image(str(shap_summary_path), caption="Global SHAP summary for the final application model")
    else:
        show_warning_if_missing("SHAP summary image", shap_summary_path)

    if shap_local_path.exists():
        st.image(str(shap_local_path), caption="Local SHAP explanation for one applicant")
    else:
        show_warning_if_missing("Local SHAP image", shap_local_path)

    st.write(
        "SHAP shows how each application-time feature pushes a prediction up or down relative to "
        "the model's baseline risk. In this project, bureau score and loan burden are the main "
        "drivers of higher or lower predicted default probability."
    )

with tab_fairness:
    st.subheader("Fairness Analysis")
    fairness_df = load_csv(fairness_csv_path)
    fairness_json = load_json(fairness_json_path)
    mitigation_df = load_csv(mitigation_path)

    if fairness_df is None or fairness_json is None:
        show_warning_if_missing("Fairness report", fairness_json_path)
    else:
        st.write(f"Protected attribute used in the saved report: `{fairness_json['protected_attribute']}`")
        st.dataframe(fairness_df, width="stretch")
        fairness_plot = fairness_df.T.reset_index()
        fairness_plot.columns = ["metric", "value"]
        st.bar_chart(fairness_plot.set_index("metric"))
        st.write(
            "Demographic parity difference closer to 0 and disparate impact closer to 1 indicate "
            "smaller group-level approval disparities."
        )

    if mitigation_df is None:
        show_warning_if_missing("Fairness mitigation tradeoff file", mitigation_path)
    else:
        st.subheader("Fairness vs Performance Tradeoff")
        st.dataframe(mitigation_df, width="stretch")
        tradeoff_chart = mitigation_df.set_index("method")[
            ["perf_roc_auc", "fair_demographic_parity_difference"]
        ]
        st.line_chart(tradeoff_chart)

with tab_counterfactual:
    st.subheader("Counterfactual Guidance")
    counterfactual_payload = load_json(counterfactual_path)
    if counterfactual_payload is None:
        show_warning_if_missing("Counterfactual file", counterfactual_path)
    else:
        changes = format_counterfactual_changes(counterfactual_payload)
        st.dataframe(changes.astype(str), width="stretch")
        st.write(
            "Counterfactual explanations turn a rejection into guidance. In the saved example, "
            "higher bureau score and lower loan burden are the most direct levers for reducing risk."
        )

with tab_leakage:
    st.subheader("Leakage Audit")
    leakage_summary = load_json(leakage_path)
    if leakage_summary is None:
        show_warning_if_missing("Leakage audit summary", leakage_path)
    else:
        st.write(
            "The original full-feature XGBoost model looked almost perfect. That was not accepted at "
            "face value. The audit found no target-column leakage or train/test overlap, but it did "
            "find that post-loan behavioral features created hindsight leakage for the application-time use case."
        )
        st.write("Target shuffle ROC-AUC:", round(leakage_summary["target_shuffle_test"]["roc_auc"], 4))
        st.write("Key suspicious features:")
        for item in leakage_summary["suspicious_features"][:10]:
            st.write(f"- {item}")
        st.success("Final honest model: `xgboost_application.pkl`")
