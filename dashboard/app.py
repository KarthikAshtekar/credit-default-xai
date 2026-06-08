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
    from .prediction_helpers import (
        EXCLUDED_USER_INPUT_FIELDS,
        build_applicant_model_row,
        build_defaults,
        build_local_shap_figure,
        compute_local_shap_analysis,
        decision_support_recommendation,
        generate_counterfactual_guidance,
        generate_plain_english_explanation,
        risk_band,
    )
except ImportError:
    from common import ensure_model, get_feature_table
    from prediction_helpers import (
        EXCLUDED_USER_INPUT_FIELDS,
        build_applicant_model_row,
        build_defaults,
        build_local_shap_figure,
        compute_local_shap_analysis,
        decision_support_recommendation,
        generate_counterfactual_guidance,
        generate_plain_english_explanation,
        risk_band,
    )


REPORTS_DIR = ROOT / "reports"
DATASET_SOURCE_OPTIONS = [
    "Local Case Dataset",
    "UCI Default Credit Card",
    "UCI South German Credit",
    "Direct URL",
]


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

selected_dataset_source = st.sidebar.selectbox("Dataset Source", DATASET_SOURCE_OPTIONS)
if selected_dataset_source != "Local Case Dataset":
    st.sidebar.info(
        "Available for external validation / future extension. The main dashboard currently "
        "uses the local case-study schema and the saved local application model."
    )
    if selected_dataset_source == "Direct URL":
        st.sidebar.text_input("Direct dataset URL", value="", disabled=True)

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
    defaults = build_defaults(feature_table)
    st.write("Input applicant details to score default risk using only application-time features.")
    st.caption(
        "Post-loan behavioral fields such as missed payments, salary-drop flags, and spending "
        "spike signals are intentionally excluded from this form."
    )
    applicant_col, loan_col = st.columns(2)

    with applicant_col:
        st.markdown("**Applicant Details**")
        age = st.number_input("Age", min_value=18, value=int(defaults["Age"]), step=1)
        gender = st.selectbox(
            "Gender",
            options=sorted(feature_table["Gender"].dropna().astype(str).unique().tolist()),
        )
        nationality = st.selectbox(
            "Nationality",
            options=sorted(feature_table["Nationality"].dropna().astype(str).unique().tolist()),
        )
        city = st.selectbox(
            "City",
            options=sorted(feature_table["City"].dropna().astype(str).unique().tolist()),
        )
        employment_status = st.selectbox(
            "EmploymentStatus",
            options=sorted(feature_table["EmploymentStatus"].dropna().astype(str).unique().tolist()),
        )
        annual_income = st.number_input(
            "AnnualIncome_AED",
            min_value=0.0,
            value=float(defaults["AnnualIncome_AED"]),
        )
        other_obligations = st.number_input(
            "OtherObligations_AED",
            min_value=0.0,
            value=float(defaults["OtherObligations_AED"]),
        )
        bureau_score = st.number_input(
            "BureauScore",
            min_value=0.0,
            value=float(defaults["BureauScore"]),
        )
        unemployment_pct = st.number_input(
            "Unemployment_pct",
            min_value=0.0,
            value=float(defaults["Unemployment_pct"]),
        )
        inflation_pct = st.number_input(
            "Inflation_pct",
            min_value=0.0,
            value=float(defaults["Inflation_pct"]),
        )

    with loan_col:
        st.markdown("**Loan Details**")
        loan_type = st.selectbox(
            "LoanType",
            options=sorted(feature_table["LoanType"].dropna().astype(str).unique().tolist()),
        )
        loan_amount = st.number_input(
            "LoanAmount_AED",
            min_value=0.0,
            value=float(defaults["LoanAmount_AED"]),
        )
        loan_tenure = st.number_input(
            "LoanTenureMonths",
            min_value=1,
            value=int(defaults["LoanTenureMonths"]),
            step=1,
        )
        interest_rate = st.number_input(
            "InterestRate_pct",
            min_value=0.0,
            value=float(defaults["InterestRate_pct"]),
        )
        st.info(
            "Derived fields such as EMI and loan burden ratios are computed internally from these inputs."
        )

    applicant_inputs = {
        "Age": age,
        "Gender": gender,
        "Nationality": nationality,
        "City": city,
        "EmploymentStatus": employment_status,
        "AnnualIncome_AED": annual_income,
        "OtherObligations_AED": other_obligations,
        "BureauScore": bureau_score,
        "LoanType": loan_type,
        "LoanAmount_AED": loan_amount,
        "LoanTenureMonths": loan_tenure,
        "InterestRate_pct": interest_rate,
        "Unemployment_pct": unemployment_pct,
        "Inflation_pct": inflation_pct,
    }
    applicant_df, computed_ratios = build_applicant_model_row(applicant_inputs, feature_table)

    st.markdown("**Computed Financial Ratios**")
    ratios_col1, ratios_col2, ratios_col3, ratios_col4, ratios_col5 = st.columns(5)
    ratios_col1.metric("EMI_AED", f"{computed_ratios['EMI_AED']:.2f}")
    ratios_col2.metric("LoanToAnnualIncome", f"{computed_ratios['LoanToAnnualIncome']:.3f}")
    ratios_col3.metric("DebtToIncomeRatio", f"{computed_ratios['DebtToIncomeRatio']:.3f}")
    ratios_col4.metric("EMIToIncomeRatio", f"{computed_ratios['EMIToIncomeRatio']:.3f}")
    ratios_col5.metric("LoanBurdenRatio", f"{computed_ratios['LoanBurdenRatio']:.3f}")

    if st.button("Predict Application Risk"):
        probability = float(model.predict_proba(applicant_df)[:, 1][0])
        band = risk_band(probability)
        recommendation = decision_support_recommendation(probability)
        shap_result: dict[str, Any] | None = None
        shap_warning: str | None = None

        try:
            shap_result = compute_local_shap_analysis(model, applicant_df, feature_table)
        except Exception as exc:
            shap_warning = f"Local SHAP explanation could not be generated for this applicant: {exc}"

        positive_drivers = shap_result["positive_drivers"] if shap_result else []
        negative_drivers = shap_result["negative_drivers"] if shap_result else []
        plot_df = shap_result["plot_df"] if shap_result else pd.DataFrame()
        explanation = generate_plain_english_explanation(
            probability,
            positive_drivers,
            negative_drivers,
        )
        guidance = generate_counterfactual_guidance(applicant_df, positive_drivers)

        st.session_state["current_prediction_result"] = {
            "probability": probability,
            "risk_band": band,
            "recommendation": recommendation,
            "applicant_df": applicant_df,
            "computed_ratios": computed_ratios,
            "positive_drivers": positive_drivers,
            "negative_drivers": negative_drivers,
            "plot_df": plot_df,
            "explanation": explanation,
            "guidance": guidance,
            "shap_warning": shap_warning,
        }

    prediction_result = st.session_state.get("current_prediction_result")
    if prediction_result:
        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("Default Probability", f"{prediction_result['probability']:.2%}")
        metric_b.metric("Risk Category", prediction_result["risk_band"])
        metric_c.metric("Decision Support Recommendation", prediction_result["recommendation"])
        st.caption(
            "Decision support only. This dashboard does not make final credit approval or rejection decisions."
        )
        st.write(prediction_result["explanation"])

        if prediction_result["shap_warning"]:
            st.warning(prediction_result["shap_warning"])

        st.markdown("**Current Application Risk Drivers**")
        drivers_left, drivers_right = st.columns(2)
        with drivers_left:
            st.write("Top factors increasing default risk")
            if prediction_result["positive_drivers"]:
                for idx, driver in enumerate(prediction_result["positive_drivers"][:3], start=1):
                    st.write(
                        f"{idx}. {driver['display_name']} ({driver['shap_value']:+.4f})"
                    )
                    st.caption(driver["interpretation"])
            else:
                st.warning("No positive local SHAP drivers were recovered for this applicant.")
        with drivers_right:
            st.write("Top factors reducing default risk")
            if prediction_result["negative_drivers"]:
                for idx, driver in enumerate(prediction_result["negative_drivers"][:3], start=1):
                    st.write(
                        f"{idx}. {driver['display_name']} ({driver['shap_value']:+.4f})"
                    )
                    st.caption(driver["interpretation"])
            else:
                st.warning("No negative local SHAP drivers were recovered for this applicant.")

        figure = build_local_shap_figure(prediction_result["plot_df"])
        if figure is not None:
            st.plotly_chart(figure, width="stretch")
        else:
            st.warning("Local SHAP plot is unavailable for this applicant.")

        st.markdown("**Counterfactual Guidance**")
        for item in prediction_result["guidance"]:
            st.write(f"- {item}")

    st.caption(
        f"Excluded from user input: {', '.join(EXCLUDED_USER_INPUT_FIELDS[:5])}, plus other post-loan fields."
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
    prediction_result = st.session_state.get("current_prediction_result")
    if prediction_result:
        st.write("Current applicant guidance based on live local SHAP drivers:")
        for item in prediction_result["guidance"]:
            st.write(f"- {item}")
        st.caption("These are decision-support suggestions only and do not promise approval.")
    else:
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
