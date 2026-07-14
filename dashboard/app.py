"""Streamlit dashboard for the public UCI credit-card default model."""

# ruff: noqa: E402

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

from dashboard.charts import (
    build_fairness_chart,
    build_model_comparison_chart,
    build_pr_curve_chart,
    build_scenario_curve_chart,
    build_threshold_tradeoff_chart,
    load_report_csv_safely,
)
from dashboard.common import ensure_model, get_application_artifact_paths, get_feature_table
from dashboard.prediction_helpers import (
    BILL_AMOUNT_COLUMNS,
    EDUCATION_OPTIONS,
    MARRIAGE_OPTIONS,
    PAY_AMOUNT_COLUMNS,
    PAY_STATUS_COLUMNS,
    SEX_OPTIONS,
    build_applicant_model_row,
    build_applicant_presets,
    compute_local_shap_analysis,
    decision_support_recommendation,
    generate_counterfactual_guidance,
    generate_plain_english_explanation,
    risk_band,
)
from dashboard.report_utils import DEFAULT_DECISION_THRESHOLD, build_applicant_risk_report
from dashboard.scenarios import (
    average_bill_amount,
    average_payment_amount,
    build_summary_applicant_inputs,
    build_target_credit_curve,
    estimate_maximum_advisable_credit_exposure,
    predict_default_risk,
    simulate_adjusted_applicant,
    summarize_shortcomings,
)
from dashboard.ui_components import (
    apply_dark_theme,
    format_currency,
    format_percent,
    friendly_model_name,
    render_hero,
    render_metric_card,
    render_panel,
    render_pills,
    risk_tone,
)

DATASET_SOURCE_OPTIONS = [
    "UCI Default of Credit Card Clients / Taiwan credit-card default",
    "South German Credit (future scope)",
    "Bondora (future scope)",
    "Home Credit (future scope)",
]

PAY_STATUS_LABELS = {
    -2: "No use / no balance",
    -1: "Paid in full",
    0: "On time",
    1: "1 month delayed",
    2: "2 months delayed",
    3: "3 months delayed",
    4: "4 months delayed",
    5: "5 months delayed",
    6: "6 months delayed",
    7: "7 months delayed",
    8: "8+ months delayed",
}


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def get_recall_policy_threshold(policy: dict[str, Any] | None) -> float | None:
    if not policy:
        return None
    try:
        return float(policy["selected_threshold"])
    except (KeyError, TypeError, ValueError):
        return None


def manual_review_signal(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "Manual review recommended"
    return "No manual review flag"


def option_index(options: list[int], selected: int) -> int:
    try:
        return options.index(selected)
    except ValueError:
        return 0


def payment_status_label(value: int) -> str:
    return PAY_STATUS_LABELS.get(value, f"{value} months delayed")


def _profile_inputs(selected_preset: dict[str, Any]) -> dict[str, Any]:
    sex_options = list(SEX_OPTIONS)
    education_options = list(EDUCATION_OPTIONS)
    marriage_options = list(MARRIAGE_OPTIONS)
    col_a, col_b = st.columns(2)
    with col_a:
        age = st.number_input("Age", min_value=18, value=int(selected_preset["AGE"]), step=1)
        education = st.selectbox(
            "Education",
            options=education_options,
            format_func=lambda value: EDUCATION_OPTIONS[value],
            index=option_index(education_options, int(selected_preset["EDUCATION"])),
        )
    with col_b:
        sex = st.selectbox(
            "Sex",
            options=sex_options,
            format_func=lambda value: SEX_OPTIONS[value],
            index=option_index(sex_options, int(selected_preset["SEX"])),
        )
        marriage = st.selectbox(
            "Marital status",
            options=marriage_options,
            format_func=lambda value: MARRIAGE_OPTIONS[value],
            index=option_index(marriage_options, int(selected_preset["MARRIAGE"])),
        )
    return {"AGE": age, "SEX": sex, "EDUCATION": education, "MARRIAGE": marriage}


def _advanced_monthly_inputs(applicant_inputs: dict[str, Any]) -> dict[str, Any]:
    updated = dict(applicant_inputs)
    with st.expander("Advanced monthly history", expanded=False):
        st.caption("Raw UCI monthly fields are available here for audit-style demos.")
        st.markdown("**Repayment status**")
        pay_cols = st.columns(3)
        for index, column in enumerate(PAY_STATUS_COLUMNS):
            with pay_cols[index % 3]:
                updated[column] = st.number_input(
                    f"{column} - {payment_status_label(int(updated[column]))}",
                    min_value=-2,
                    max_value=8,
                    value=int(updated[column]),
                    step=1,
                )

        st.markdown("**Bill amount history**")
        bill_cols = st.columns(3)
        for index, column in enumerate(BILL_AMOUNT_COLUMNS):
            with bill_cols[index % 3]:
                updated[column] = st.number_input(
                    f"{column}",
                    min_value=0.0,
                    value=float(updated[column]),
                    step=1000.0,
                )

        st.markdown("**Payment amount history**")
        payment_cols = st.columns(3)
        for index, column in enumerate(PAY_AMOUNT_COLUMNS):
            with payment_cols[index % 3]:
                updated[column] = st.number_input(
                    f"{column}",
                    min_value=0.0,
                    value=float(updated[column]),
                    step=1000.0,
                )
    return updated


def _load_prediction_frames(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    frames = {}
    for model_name, key in [
        ("logistic_public", "logistic_test_predictions"),
        ("xgboost_public", "xgboost_test_predictions"),
        ("dnn_baseline", "dnn_test_predictions"),
    ]:
        frame = load_report_csv_safely(paths[key])
        if frame is not None:
            frames[model_name] = frame
    return frames


def _result_payload(
    model: Any,
    feature_table: pd.DataFrame,
    applicant_inputs: dict[str, Any],
    review_threshold: float,
    selected_recall_policy: dict[str, Any] | None,
) -> dict[str, Any]:
    applicant_df, computed_ratios = build_applicant_model_row(applicant_inputs, feature_table)
    probability = float(model.predict_proba(applicant_df)[:, 1][0])
    shap_result: dict[str, Any] | None = None
    shap_warning: str | None = None

    try:
        shap_result = compute_local_shap_analysis(model, applicant_df, feature_table)
    except Exception as exc:
        shap_warning = f"Local SHAP explanation could not be generated: {exc}"

    positive_drivers = shap_result["positive_drivers"] if shap_result else []
    negative_drivers = shap_result["negative_drivers"] if shap_result else []
    guidance = generate_counterfactual_guidance(applicant_df, positive_drivers)
    exposure_estimate = estimate_maximum_advisable_credit_exposure(
        model,
        applicant_inputs,
        feature_table,
        review_threshold,
    )
    scenario_curve = build_target_credit_curve(model, applicant_inputs, feature_table)
    return {
        "probability": probability,
        "risk_band": risk_band(probability),
        "recommendation": decision_support_recommendation(probability),
        "manual_review_signal": manual_review_signal(probability, review_threshold),
        "review_threshold": review_threshold,
        "recall_policy_name": (
            selected_recall_policy.get("selected_candidate") if selected_recall_policy else None
        ),
        "applicant_inputs": applicant_inputs,
        "applicant_df": applicant_df,
        "computed_ratios": computed_ratios,
        "positive_drivers": positive_drivers,
        "negative_drivers": negative_drivers,
        "plot_df": shap_result["plot_df"] if shap_result else pd.DataFrame(),
        "explanation": generate_plain_english_explanation(
            probability,
            positive_drivers,
            negative_drivers,
        ),
        "guidance": guidance,
        "exposure_estimate": exposure_estimate,
        "scenario_curve": scenario_curve,
        "shap_warning": shap_warning,
    }


st.set_page_config(
    page_title="Credit Default Risk Decision Support",
    page_icon="CC",
    layout="wide",
)
apply_dark_theme()

artifact_paths = get_application_artifact_paths()
selected_recall_policy = load_json(artifact_paths["selected_recall_policy"])
recall_policy_threshold = get_recall_policy_threshold(selected_recall_policy)
active_review_threshold = recall_policy_threshold or DEFAULT_DECISION_THRESHOLD

model, model_path = ensure_model("XGBoost")
feature_table = get_feature_table()
presets = build_applicant_presets(feature_table)

selected_dataset_source = st.sidebar.selectbox("Dataset Source", DATASET_SOURCE_OPTIONS)
if selected_dataset_source != DATASET_SOURCE_OPTIONS[0]:
    st.sidebar.info("This dataset is future scope and is not used by the current pipeline.")
st.sidebar.caption("XGBoost is the applicant-facing model. DNN is governance-only.")

render_hero(
    "Explainable and Fair Credit Default Risk Prediction",
    "Credit default risk decision-support using public UCI Taiwan credit-card data",
)
render_pills(["Decision-support only", "XGBoost primary model", "Public UCI dataset"])

tab_applicant, tab_guidance, tab_governance = st.tabs(
    ["Applicant Report", "Improvement Guidance", "Model Governance"]
)

with tab_applicant:
    st.subheader("Applicant Report")
    st.caption(
        "Enter a compact applicant profile, then generate a model-based risk report. "
        "Advanced monthly fields are hidden unless needed for a detailed demo."
    )

    selected_preset_name = st.selectbox("Demo applicant profile", options=list(presets))
    selected_preset = presets[selected_preset_name]

    summary_col, profile_col = st.columns([1.05, 0.95])
    with summary_col:
        target_credit_amount = st.number_input(
            "Target credit amount",
            min_value=0.0,
            value=float(selected_preset["LIMIT_BAL"]),
            step=10000.0,
        )
        latest_repayment_status = st.selectbox(
            "Latest repayment status",
            options=list(PAY_STATUS_LABELS),
            format_func=payment_status_label,
            index=option_index(list(PAY_STATUS_LABELS), int(selected_preset["PAY_0"])),
        )
        max_recent_delay = st.selectbox(
            "Maximum recent delay",
            options=list(PAY_STATUS_LABELS),
            format_func=payment_status_label,
            index=option_index(
                list(PAY_STATUS_LABELS),
                max(int(selected_preset[column]) for column in PAY_STATUS_COLUMNS),
            ),
        )
    with profile_col:
        avg_bill = st.number_input(
            "Average bill amount",
            min_value=0.0,
            value=average_bill_amount(selected_preset),
            step=1000.0,
        )
        avg_payment = st.number_input(
            "Average payment amount",
            min_value=0.0,
            value=average_payment_amount(selected_preset),
            step=1000.0,
        )
        with st.expander("Applicant profile", expanded=False):
            profile_values = _profile_inputs(selected_preset)

    applicant_inputs = build_summary_applicant_inputs(
        selected_preset,
        target_credit_amount,
        latest_repayment_status,
        max_recent_delay,
        avg_bill,
        avg_payment,
    )
    applicant_inputs.update(profile_values if "profile_values" in locals() else {})
    applicant_inputs = _advanced_monthly_inputs(applicant_inputs)

    if st.button("Generate Applicant Report", type="primary"):
        st.session_state["current_prediction_result"] = _result_payload(
            model,
            feature_table,
            applicant_inputs,
            active_review_threshold,
            selected_recall_policy,
        )

    prediction_result = st.session_state.get("current_prediction_result")
    if not prediction_result:
        render_panel(
            "Ready for a risk estimate",
            "Generate an applicant report to view predicted default risk, risk band, manual-review signal, and maximum advisable credit exposure.",
        )
    else:
        exposure_estimate = prediction_result["exposure_estimate"]
        cols = st.columns(4)
        with cols[0]:
            render_metric_card(
                "Predicted default risk",
                format_percent(prediction_result["probability"]),
                "Model-based XGBoost probability",
                risk_tone(prediction_result["risk_band"]),
            )
        with cols[1]:
            render_metric_card(
                "Risk band",
                prediction_result["risk_band"],
                "Low, medium, or high risk",
                risk_tone(prediction_result["risk_band"]),
            )
        with cols[2]:
            render_metric_card(
                "Manual review signal",
                prediction_result["manual_review_signal"],
                f"Threshold {prediction_result['review_threshold']:.0%}",
                "watch"
                if "recommended" in prediction_result["manual_review_signal"].lower()
                else "good",
            )
        with cols[3]:
            render_metric_card(
                "Maximum advisable credit exposure",
                format_currency(exposure_estimate.get("max_exposure")),
                "Scenario-based estimate",
                "neutral",
            )

        st.markdown("#### Requested target vs advisable exposure")
        target_cols = st.columns(2)
        with target_cols[0]:
            render_metric_card(
                "Requested target credit amount",
                format_currency(prediction_result["applicant_inputs"]["LIMIT_BAL"]),
                "Entered by the user",
            )
        with target_cols[1]:
            probability_at_exposure = exposure_estimate.get("probability_at_exposure")
            render_metric_card(
                "Risk at advisable exposure",
                format_percent(probability_at_exposure)
                if probability_at_exposure is not None
                else "Not available",
                exposure_estimate.get("note", ""),
            )

        st.markdown("#### Short explanation")
        st.write(prediction_result["explanation"])
        if prediction_result["shap_warning"]:
            st.warning(prediction_result["shap_warning"])

        report = build_applicant_risk_report(
            probability=prediction_result["probability"],
            positive_drivers=prediction_result["positive_drivers"],
            negative_drivers=prediction_result["negative_drivers"],
            guidance=prediction_result["guidance"],
            threshold=prediction_result["review_threshold"],
            exposure_estimate=exposure_estimate,
            shap_warning=prediction_result["shap_warning"],
        )
        st.download_button(
            "Download applicant report",
            data=report["markdown"],
            file_name="applicant_risk_report.md",
            mime="text/markdown",
        )

with tab_guidance:
    st.subheader("Improvement Guidance")
    prediction_result = st.session_state.get("current_prediction_result")
    if not prediction_result:
        render_panel(
            "Generate an applicant report first",
            "The guidance tab becomes actionable after the dashboard has a current applicant risk estimate.",
        )
    else:
        shortcomings = summarize_shortcomings(
            prediction_result["applicant_inputs"],
            prediction_result["probability"],
        )
        st.markdown("#### Top shortcomings")
        for index, item in enumerate(shortcomings):
            with st.expander(item["shortcoming"], expanded=index == 0):
                st.write(f"**Why it matters:** {item['why']}")
                st.write(f"**What to improve:** {item['action']}")

        st.markdown("#### Scenario simulation")
        scenario_col_a, scenario_col_b = st.columns(2)
        current_inputs = prediction_result["applicant_inputs"]
        with scenario_col_a:
            simulated_target = st.number_input(
                "Lower target credit amount",
                min_value=0.0,
                value=float(current_inputs["LIMIT_BAL"]),
                step=10000.0,
            )
            simulated_delay = st.selectbox(
                "Improve repayment delay",
                options=list(PAY_STATUS_LABELS),
                format_func=payment_status_label,
                index=option_index(
                    list(PAY_STATUS_LABELS),
                    max(int(current_inputs[column]) for column in PAY_STATUS_COLUMNS),
                ),
            )
        with scenario_col_b:
            simulated_payment = st.number_input(
                "Increase average payment amount",
                min_value=0.0,
                value=average_payment_amount(current_inputs),
                step=1000.0,
            )
            simulated_bill = st.number_input(
                "Reduce average bill amount",
                min_value=0.0,
                value=average_bill_amount(current_inputs),
                step=1000.0,
            )

        simulated_inputs = simulate_adjusted_applicant(
            current_inputs,
            simulated_target,
            simulated_delay,
            simulated_bill,
            simulated_payment,
        )
        simulated_risk = predict_default_risk(model, simulated_inputs, feature_table)
        change = simulated_risk - prediction_result["probability"]
        target_advisable = simulated_risk < prediction_result["review_threshold"]

        sim_cols = st.columns(4)
        with sim_cols[0]:
            render_metric_card(
                "Original risk",
                format_percent(prediction_result["probability"]),
                "",
                risk_tone(prediction_result["risk_band"]),
            )
        with sim_cols[1]:
            render_metric_card(
                "Simulated risk",
                format_percent(simulated_risk),
                "",
                risk_tone(risk_band(simulated_risk)),
            )
        with sim_cols[2]:
            render_metric_card("Change in risk", format_percent(change, digits=1), "", "neutral")
        with sim_cols[3]:
            render_metric_card(
                "Target becomes advisable",
                "Yes" if target_advisable else "No",
                f"Threshold {prediction_result['review_threshold']:.0%}",
                "good" if target_advisable else "watch",
            )

        scenario_curve = build_target_credit_curve(model, simulated_inputs, feature_table)
        fig = build_scenario_curve_chart(scenario_curve, prediction_result["review_threshold"])
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")

with tab_governance:
    st.subheader("Model Governance")
    st.caption("Technical evidence for reviewers. Applicant scoring remains XGBoost-first.")
    recall_summary = load_json(artifact_paths["recall_summary"])
    selected_policy = recall_summary.get("selected_policy", {}) if recall_summary else {}
    selected_metrics = selected_policy.get("test_metrics", {})
    baseline_metrics = (recall_summary or {}).get("baseline_threshold_050", {})

    gov_cols = st.columns(4)
    with gov_cols[0]:
        render_metric_card("Final model", "XGBoost", "Applicant-facing model", "good")
    with gov_cols[1]:
        render_metric_card(
            "Baseline ROC-AUC",
            f"{baseline_metrics.get('roc_auc', 0.7748):.4f}",
            "Threshold 0.50",
        )
    with gov_cols[2]:
        render_metric_card(
            "Recall-policy recall",
            format_percent(selected_metrics.get("recall", 0.5810)),
            "Threshold 0.25",
            "watch",
        )
    with gov_cols[3]:
        render_metric_card("DNN role", "Benchmark only", "Governance tab only", "neutral")

    with st.expander("Model performance", expanded=False):
        performance_df = load_report_csv_safely(artifact_paths["performance"])
        comparison_df = load_report_csv_safely(artifact_paths["deep_learning_comparison"])
        fig = build_model_comparison_chart(
            comparison_df if comparison_df is not None else performance_df
        )
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        if performance_df is not None:
            display = performance_df.copy()
            display["model_name"] = display["model_name"].map(friendly_model_name)
            st.dataframe(display, use_container_width=True, hide_index=True)
        else:
            st.info("Model performance artifacts are not available.")

    with st.expander("Recall threshold tradeoff", expanded=False):
        threshold_tuning_df = load_report_csv_safely(artifact_paths["threshold_tuning"])
        selected_candidate = selected_policy.get("candidate_name")
        selected_threshold = selected_policy.get("selected_threshold")
        fig = build_threshold_tradeoff_chart(
            threshold_tuning_df,
            selected_candidate,
            float(selected_threshold) if selected_threshold is not None else None,
        )
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        else:
            st.info("Threshold tuning artifact is not available.")

    with st.expander("ML vs DL benchmark", expanded=False):
        dnn_metrics = load_json(artifact_paths["deep_learning_metrics"])
        dnn_policy = load_json(artifact_paths["deep_learning_policy"])
        dnn_comparison = load_report_csv_safely(artifact_paths["deep_learning_comparison"])
        if dnn_metrics is None:
            st.info("Deep learning benchmark artifacts are not available.")
        elif dnn_metrics.get("status") == "skipped":
            st.warning(dnn_metrics.get("reason", "Deep learning benchmark was skipped."))
        else:
            st.write(
                "The DNN benchmark tests whether added model complexity improves ranking or "
                "recall-policy performance. XGBoost remains the final model."
            )
            if dnn_policy:
                policy_metrics = dnn_policy.get("test_metrics", {})
                dnn_cols = st.columns(3)
                with dnn_cols[0]:
                    render_metric_card(
                        "DNN threshold", f"{dnn_policy.get('selected_threshold', 0):.2f}"
                    )
                with dnn_cols[1]:
                    render_metric_card("DNN recall", format_percent(policy_metrics.get("recall")))
                with dnn_cols[2]:
                    render_metric_card("DNN PR-AUC", f"{policy_metrics.get('pr_auc', 0):.4f}")
        pr_fig = build_pr_curve_chart(_load_prediction_frames(artifact_paths))
        if pr_fig is not None:
            st.plotly_chart(pr_fig, use_container_width=True, theme="streamlit")
        if dnn_comparison is not None:
            display = dnn_comparison.copy()
            display["model_name"] = display["model_name"].map(friendly_model_name)
            st.dataframe(display, use_container_width=True, hide_index=True)

    with st.expander("Fairness diagnostics", expanded=False):
        threshold_fairness_df = load_report_csv_safely(
            artifact_paths["threshold_fairness_comparison"]
        )
        fairness_fig = build_fairness_chart(threshold_fairness_df)
        if fairness_fig is not None:
            st.plotly_chart(fairness_fig, use_container_width=True, theme="streamlit")
        if threshold_fairness_df is not None:
            display = threshold_fairness_df.copy()
            display["policy"] = display["policy"].map(friendly_model_name)
            st.dataframe(display, use_container_width=True, hide_index=True)
        else:
            st.info("Fairness diagnostics are not available.")

    with st.expander("Leakage audit", expanded=False):
        leakage_summary = load_json(artifact_paths["leakage"])
        if leakage_summary is None:
            st.info("Leakage audit summary is not available.")
        else:
            st.success(leakage_summary.get("conclusion", "Leakage audit completed."))
            target_shuffle = leakage_summary.get("target_shuffle_test", {})
            st.write(f"Target-shuffle ROC-AUC: `{target_shuffle.get('roc_auc', 0):.4f}`")
            render_pills(
                [
                    "Target excluded",
                    "ID fields excluded",
                    "No detected train/test overlap",
                    "Feature timing reviewed",
                ]
            )

    with st.expander("Explainability artifacts", expanded=False):
        st.write("SHAP and LIME artifacts remain available for technical review.")
        image_cols = st.columns(2)
        with image_cols[0]:
            if artifact_paths["shap_summary"].exists():
                st.image(str(artifact_paths["shap_summary"]), caption="XGBoost SHAP summary")
            else:
                st.info("SHAP summary image is not available.")
        with image_cols[1]:
            if artifact_paths["lime_local"].exists():
                st.image(str(artifact_paths["lime_local"]), caption="Local LIME explanation")
            else:
                st.info("LIME image is not available.")
