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

from dashboard.common import ensure_model, get_application_artifact_paths, get_feature_table
from dashboard.prediction_helpers import (
    BILL_AMOUNT_COLUMNS,
    EDUCATION_OPTIONS,
    EXCLUDED_USER_INPUT_FIELDS,
    MARRIAGE_OPTIONS,
    PAY_AMOUNT_COLUMNS,
    PAY_STATUS_COLUMNS,
    SEX_OPTIONS,
    USER_INPUT_FIELDS,
    build_applicant_model_row,
    build_applicant_presets,
    build_local_shap_figure,
    compute_local_shap_analysis,
    decision_support_recommendation,
    generate_counterfactual_guidance,
    generate_plain_english_explanation,
    risk_band,
)
from dashboard.report_utils import DEFAULT_DECISION_THRESHOLD, build_applicant_risk_report

DATASET_SOURCE_OPTIONS = [
    "UCI Default of Credit Card Clients / Taiwan credit-card default",
    "South German Credit (future scope)",
    "Bondora (future scope)",
    "Home Credit (future scope)",
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


def get_recall_policy_threshold(policy: dict[str, Any] | None) -> float | None:
    if not policy:
        return None
    try:
        return float(policy["selected_threshold"])
    except (KeyError, TypeError, ValueError):
        return None


def screening_flag_text(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "Manual review recommended"
    return "No manual review flag"


def option_index(options: list[int], selected: int) -> int:
    try:
        return options.index(selected)
    except ValueError:
        return 0


def format_counterfactual_changes(counterfactual_payload: dict[str, Any]) -> pd.DataFrame:
    try:
        feature_names = counterfactual_payload["feature_names"]
        original_values = counterfactual_payload["test_data"][0][0][: len(feature_names)]
        counterfactual_values = counterfactual_payload["cfs_list"][0][0][: len(feature_names)]
    except (KeyError, IndexError, TypeError):
        return pd.DataFrame()
    original = pd.Series(original_values, index=feature_names, name="original")
    first_cf = pd.Series(counterfactual_values, index=feature_names, name="counterfactual")
    changes = pd.DataFrame({"original": original, "counterfactual": first_cf})
    return changes[changes["original"].astype(str) != changes["counterfactual"].astype(str)]


def _number_input_grid(
    columns: list[str],
    selected_preset: dict[str, Any],
    min_value: float | None = None,
) -> dict[str, float]:
    values = {}
    grid_cols = st.columns(3)
    for index, column in enumerate(columns):
        with grid_cols[index % 3]:
            values[column] = st.number_input(
                column,
                min_value=min_value,
                value=float(selected_preset[column]),
                step=1000.0 if "AMT" in column else 1.0,
            )
    return values


def estimate_maximum_advisable_credit_exposure(
    model: Any,
    applicant_inputs: dict[str, Any],
    feature_table: pd.DataFrame,
    threshold: float,
) -> dict[str, Any]:
    """Estimate a model-supported exposure cap by varying LIMIT_BAL only."""

    current_limit = max(float(applicant_inputs["LIMIT_BAL"]), 0.0)
    if current_limit <= 0:
        return {
            "max_exposure": 0.0,
            "threshold": threshold,
            "probability_at_exposure": None,
            "note": "No positive credit exposure was entered for simulation.",
        }

    start = min(10000.0, current_limit)
    if current_limit == start:
        limits = [current_limit]
    else:
        points = min(60, max(2, int(current_limit // 10000) + 1))
        limits = [start + (current_limit - start) * i / (points - 1) for i in range(points)]

    rows = []
    for candidate_limit in limits:
        scenario = dict(applicant_inputs)
        scenario["LIMIT_BAL"] = float(candidate_limit)
        scenario_df, _ = build_applicant_model_row(scenario, feature_table)
        probability = float(model.predict_proba(scenario_df)[:, 1][0])
        rows.append({"limit": float(candidate_limit), "probability": probability})

    feasible = [row for row in rows if row["probability"] < threshold]
    if feasible:
        selected = max(feasible, key=lambda row: row["limit"])
        return {
            "max_exposure": selected["limit"],
            "threshold": threshold,
            "probability_at_exposure": selected["probability"],
            "note": (
                "Largest simulated LIMIT_BAL up to the entered exposure that stayed below "
                "the active manual-review threshold."
            ),
        }

    lowest_risk = min(rows, key=lambda row: row["probability"])
    return {
        "max_exposure": 0.0,
        "threshold": threshold,
        "probability_at_exposure": lowest_risk["probability"],
        "note": (
            "No simulated LIMIT_BAL up to the entered exposure stayed below the active "
            "manual-review threshold."
        ),
    }


def simulate_improvement_scenarios(
    model: Any,
    applicant_inputs: dict[str, Any],
    feature_table: pd.DataFrame,
    current_probability: float,
) -> pd.DataFrame:
    scenarios: list[tuple[str, dict[str, Any]]] = []

    timely_repayment = dict(applicant_inputs)
    for column in PAY_STATUS_COLUMNS:
        timely_repayment[column] = min(int(timely_repayment[column]), 0)
    scenarios.append(("Timelier recent repayment", timely_repayment))

    lower_utilization = dict(applicant_inputs)
    for column in BILL_AMOUNT_COLUMNS:
        lower_utilization[column] = max(float(lower_utilization[column]) * 0.80, 0.0)
    scenarios.append(("20% lower bill balances", lower_utilization))

    stronger_payments = dict(applicant_inputs)
    for column in PAY_AMOUNT_COLUMNS:
        stronger_payments[column] = max(float(stronger_payments[column]) * 1.20, 0.0)
    scenarios.append(("20% higher repayment amounts", stronger_payments))

    rows = []
    for scenario_name, scenario_inputs in scenarios:
        scenario_df, _ = build_applicant_model_row(scenario_inputs, feature_table)
        probability = float(model.predict_proba(scenario_df)[:, 1][0])
        rows.append(
            {
                "scenario": scenario_name,
                "default_probability": probability,
                "change_vs_current": probability - current_probability,
            }
        )
    return pd.DataFrame(rows)


st.set_page_config(
    page_title="Public Credit-Card Default Risk Model",
    page_icon="CC",
    layout="wide",
)

st.title("Explainable and Fair Credit-Card Default Risk Prediction")
st.caption(
    "Primary dataset: public UCI Taiwan credit-card default data. Target: next-month default."
)
st.caption(
    "This tool provides model-based decision support for educational/portfolio purposes. "
    "It does not guarantee loan approval, rejection, or regulatory compliance."
)

selected_dataset_source = st.sidebar.selectbox("Dataset Source", DATASET_SOURCE_OPTIONS)
if selected_dataset_source != DATASET_SOURCE_OPTIONS[0]:
    st.sidebar.info(
        "This dataset is documented as future scope and is not used by the current pipeline."
    )

artifact_paths = get_application_artifact_paths()
performance_path = artifact_paths["performance"]
temporal_path = artifact_paths["temporal"]
fairness_csv_path = artifact_paths["fairness_csv"]
fairness_json_path = artifact_paths["fairness_json"]
mitigation_path = artifact_paths["mitigation"]
recall_summary_path = artifact_paths["recall_summary"]
selected_recall_policy_path = artifact_paths["selected_recall_policy"]
threshold_tuning_path = artifact_paths["threshold_tuning"]
threshold_selection_path = artifact_paths["threshold_selection"]
precision_recall_curve_path = artifact_paths["precision_recall_curve"]
threshold_fairness_comparison_path = artifact_paths["threshold_fairness_comparison"]
leakage_path = artifact_paths["leakage"]
counterfactual_path = artifact_paths["counterfactual"]
shap_summary_path = artifact_paths["shap_summary"]
shap_local_path = artifact_paths["shap_local"]
lime_local_path = artifact_paths["lime_local"]
deep_learning_metrics_path = artifact_paths["deep_learning_metrics"]
deep_learning_comparison_path = artifact_paths["deep_learning_comparison"]
ml_vs_dl_curve_path = artifact_paths["ml_vs_dl_precision_recall_curve"]
deep_learning_policy_path = artifact_paths["deep_learning_policy"]
deep_learning_fairness_path = artifact_paths["deep_learning_fairness"]

tab_applicant, tab_guidance, tab_governance = st.tabs(
    ["Applicant Report", "Improvement Guidance", "Model Governance"]
)

with tab_governance:
    st.subheader("Project Framing")
    st.write(
        "This project models next-month credit-card default using the public UCI Default of "
        "Credit Card Clients / Taiwan dataset. The pipeline uses a reproducible public dataset "
        "loaded through `ucimlrepo`, not an earlier local private file."
    )
    st.subheader("Feature Policy")
    st.write(
        "`SEX` is retained for fairness auditing and excluded from the final active training "
        "features. `AGE`, `MARRIAGE`, and `EDUCATION` are included as profile variables and "
        "called out as audit-sensitive fields."
    )
    st.info(
        "`PAY_0` to `PAY_6` are historical repayment-status variables used to predict "
        "next-month default; they are not treated as leakage for this modeling question."
    )

with tab_applicant:
    st.subheader("Applicant Risk Prediction")
    model, model_path = ensure_model("XGBoost")
    feature_table = get_feature_table()
    selected_recall_policy = load_json(selected_recall_policy_path)
    recall_policy_threshold = get_recall_policy_threshold(selected_recall_policy)
    presets = build_applicant_presets(feature_table)
    selected_preset_name = st.selectbox("Demo Cardholder Preset", options=list(presets))
    selected_preset = presets[selected_preset_name]

    profile_col, exposure_col = st.columns(2)
    with profile_col:
        st.markdown("**Profile**")
        limit_bal = st.number_input(
            "LIMIT_BAL",
            min_value=0.0,
            value=float(selected_preset["LIMIT_BAL"]),
            step=10000.0,
        )
        age = st.number_input("AGE", min_value=18, value=int(selected_preset["AGE"]), step=1)
        sex_options = list(SEX_OPTIONS)
        sex = st.selectbox(
            "SEX",
            options=sex_options,
            format_func=lambda value: f"{value} - {SEX_OPTIONS[value]}",
            index=option_index(sex_options, int(selected_preset["SEX"])),
        )
        education_options = list(EDUCATION_OPTIONS)
        education = st.selectbox(
            "EDUCATION",
            options=education_options,
            format_func=lambda value: f"{value} - {EDUCATION_OPTIONS[value]}",
            index=option_index(education_options, int(selected_preset["EDUCATION"])),
        )
        marriage_options = list(MARRIAGE_OPTIONS)
        marriage = st.selectbox(
            "MARRIAGE",
            options=marriage_options,
            format_func=lambda value: f"{value} - {MARRIAGE_OPTIONS[value]}",
            index=option_index(marriage_options, int(selected_preset["MARRIAGE"])),
        )

    with exposure_col:
        st.markdown("**Recent Repayment Status**")
        pay_status_inputs = {}
        pay_cols = st.columns(3)
        for index, column in enumerate(PAY_STATUS_COLUMNS):
            with pay_cols[index % 3]:
                pay_status_inputs[column] = st.number_input(
                    column,
                    min_value=-2,
                    max_value=8,
                    value=int(selected_preset[column]),
                    step=1,
                )

    st.markdown("**Bill Amount History**")
    bill_inputs = _number_input_grid(BILL_AMOUNT_COLUMNS, selected_preset)
    st.markdown("**Payment Amount History**")
    payment_inputs = _number_input_grid(PAY_AMOUNT_COLUMNS, selected_preset, min_value=0.0)

    applicant_inputs = {
        "LIMIT_BAL": limit_bal,
        "SEX": sex,
        "EDUCATION": education,
        "MARRIAGE": marriage,
        "AGE": age,
        **pay_status_inputs,
        **bill_inputs,
        **payment_inputs,
    }
    applicant_df, computed_ratios = build_applicant_model_row(applicant_inputs, feature_table)

    st.markdown("**Computed UCI Features**")
    ratio_cols = st.columns(4)
    ratio_cols[0].metric("AvgBillToLimitRatio", f"{computed_ratios['AvgBillToLimitRatio']:.3f}")
    ratio_cols[1].metric("AvgPaymentToBillRatio", f"{computed_ratios['AvgPaymentToBillRatio']:.3f}")
    ratio_cols[2].metric("MaxPaymentDelay", f"{computed_ratios['MaxPaymentDelay']:.0f}")
    ratio_cols[3].metric("NumDelayedMonths", f"{computed_ratios['NumDelayedMonths']:.0f}")

    if st.button("Predict Default Risk"):
        probability = float(model.predict_proba(applicant_df)[:, 1][0])
        band = risk_band(probability)
        recommendation = decision_support_recommendation(probability)
        shap_result: dict[str, Any] | None = None
        shap_warning: str | None = None

        try:
            shap_result = compute_local_shap_analysis(model, applicant_df, feature_table)
        except Exception as exc:
            shap_warning = f"Local SHAP explanation could not be generated: {exc}"

        positive_drivers = shap_result["positive_drivers"] if shap_result else []
        negative_drivers = shap_result["negative_drivers"] if shap_result else []
        plot_df = shap_result["plot_df"] if shap_result else pd.DataFrame()
        explanation = generate_plain_english_explanation(
            probability,
            positive_drivers,
            negative_drivers,
        )
        guidance = generate_counterfactual_guidance(applicant_df, positive_drivers)
        baseline_screening_flag = probability >= DEFAULT_DECISION_THRESHOLD
        recall_screening_flag = (
            probability >= recall_policy_threshold if recall_policy_threshold is not None else None
        )
        active_review_threshold = recall_policy_threshold or DEFAULT_DECISION_THRESHOLD
        exposure_estimate = estimate_maximum_advisable_credit_exposure(
            model,
            applicant_inputs,
            feature_table,
            active_review_threshold,
        )
        improvement_scenarios = simulate_improvement_scenarios(
            model,
            applicant_inputs,
            feature_table,
            probability,
        )

        st.session_state["current_prediction_result"] = {
            "probability": probability,
            "threshold": DEFAULT_DECISION_THRESHOLD,
            "baseline_screening_flag": baseline_screening_flag,
            "recall_policy_threshold": recall_policy_threshold,
            "recall_screening_flag": recall_screening_flag,
            "recall_policy_name": (
                selected_recall_policy.get("selected_candidate") if selected_recall_policy else None
            ),
            "recall_policy_rule": (
                selected_recall_policy.get("selection_rule") if selected_recall_policy else None
            ),
            "risk_band": band,
            "recommendation": recommendation,
            "applicant_df": applicant_df,
            "computed_ratios": computed_ratios,
            "positive_drivers": positive_drivers,
            "negative_drivers": negative_drivers,
            "plot_df": plot_df,
            "explanation": explanation,
            "guidance": guidance,
            "exposure_estimate": exposure_estimate,
            "improvement_scenarios": improvement_scenarios,
            "applicant_inputs": applicant_inputs,
            "shap_warning": shap_warning,
            "model_path": str(model_path),
        }

    prediction_result = st.session_state.get("current_prediction_result")
    if prediction_result:
        metric_a, metric_b = st.columns(2)
        metric_a.metric("Default Probability", f"{prediction_result['probability']:.2%}")
        metric_b.metric("Risk Category", prediction_result["risk_band"])
        baseline_col, recall_col = st.columns(2)
        baseline_col.metric(
            "Baseline 0.50 Screening",
            screening_flag_text(
                prediction_result["probability"],
                DEFAULT_DECISION_THRESHOLD,
            ),
        )
        recall_threshold = prediction_result.get("recall_policy_threshold")
        if recall_threshold is None:
            recall_col.info(
                "Run `python -m src.recall_optimization` to add recall-optimized screening."
            )
        else:
            recall_col.metric(
                f"Recall Policy {recall_threshold:.2f}",
                screening_flag_text(prediction_result["probability"], recall_threshold),
            )
            st.caption(
                f"Recall policy: `{prediction_result.get('recall_policy_name')}` selected by "
                f"`{prediction_result.get('recall_policy_rule')}`."
            )
        st.warning(f"Decision-support recommendation: **{prediction_result['recommendation']}**")
        st.caption("Decision support only. This is not a regulatory credit scorecard.")
        st.write(prediction_result["explanation"])

        exposure_estimate = prediction_result.get("exposure_estimate", {})
        exposure_cols = st.columns(2)
        exposure_cols[0].metric(
            "Maximum Advisable Credit Exposure",
            f"{float(exposure_estimate.get('max_exposure', 0.0)):,.0f}",
        )
        probability_at_exposure = exposure_estimate.get("probability_at_exposure")
        exposure_cols[1].metric(
            "Risk At Simulated Exposure",
            "Not available"
            if probability_at_exposure is None
            else f"{float(probability_at_exposure):.2%}",
        )
        st.caption(exposure_estimate.get("note", "Exposure simulation unavailable."))

        if prediction_result["shap_warning"]:
            st.warning(prediction_result["shap_warning"])

        st.markdown("**Current Risk Drivers**")
        drivers_left, drivers_right = st.columns(2)
        with drivers_left:
            st.write("Top factors increasing default risk")
            for idx, driver in enumerate(prediction_result["positive_drivers"][:3], start=1):
                st.write(f"{idx}. {driver['display_name']} ({driver['shap_value']:+.4f})")
                st.caption(driver["interpretation"])
        with drivers_right:
            st.write("Top factors reducing default risk")
            for idx, driver in enumerate(prediction_result["negative_drivers"][:3], start=1):
                st.write(f"{idx}. {driver['display_name']} ({driver['shap_value']:+.4f})")
                st.caption(driver["interpretation"])

        figure = build_local_shap_figure(prediction_result["plot_df"])
        if figure is not None:
            st.plotly_chart(figure, width="stretch")

        st.markdown("**Counterfactual Guidance**")
        for item in prediction_result["guidance"]:
            st.write(f"- {item}")

    st.caption(
        f"Input fields: {len(USER_INPUT_FIELDS)} UCI fields. Computed internally: "
        f"{', '.join(EXCLUDED_USER_INPUT_FIELDS[:5])}."
    )

with tab_governance:
    st.subheader("Held-Out Model Performance")
    performance_df = load_csv(performance_path)
    if performance_df is None:
        show_warning_if_missing("Model comparison file", performance_path)
    else:
        st.dataframe(performance_df, width="stretch")
        if {"model_name", "roc_auc"}.issubset(performance_df.columns):
            roc_plot = performance_df[["model_name", "roc_auc"]].set_index("model_name")
            st.bar_chart(roc_plot)

    temporal_df = load_csv(temporal_path)
    if temporal_df is None:
        show_warning_if_missing("Temporal comparison note", temporal_path)
    else:
        st.subheader("Temporal Validation")
        st.dataframe(temporal_df, width="stretch")

    st.subheader("Threshold and Recall Tradeoff")
    recall_summary = load_json(recall_summary_path)
    threshold_tuning_df = load_csv(threshold_tuning_path)
    threshold_selection_df = load_csv(threshold_selection_path)
    selected_policy = recall_summary.get("selected_policy", {}) if recall_summary else {}

    if recall_summary is None:
        show_warning_if_missing("Recall optimization summary", recall_summary_path)
        st.info("Run `python -m src.recall_optimization` to generate threshold tradeoff reports.")
    else:
        selected_metrics = selected_policy.get("test_metrics", {})
        metric_cols = st.columns(4)
        metric_cols[0].metric(
            "Selected Threshold",
            f"{selected_policy.get('selected_threshold', 0):.2f}",
        )
        metric_cols[1].metric("Test Recall", f"{selected_metrics.get('recall', 0):.2%}")
        metric_cols[2].metric("Test Precision", f"{selected_metrics.get('precision', 0):.2%}")
        metric_cols[3].metric("Test F2", f"{selected_metrics.get('f2', 0):.4f}")

        comparison_df = pd.DataFrame(recall_summary.get("comparisons", []))
        if not comparison_df.empty:
            display_columns = [
                "candidate_name",
                "selected_threshold",
                "test_accuracy",
                "test_precision",
                "test_recall",
                "test_f1",
                "test_f2",
                "test_pr_auc",
                "test_approval_support_rate",
                "notes",
            ]
            st.dataframe(
                comparison_df[[col for col in display_columns if col in comparison_df.columns]],
                width="stretch",
            )

    if threshold_tuning_df is not None and not threshold_tuning_df.empty:
        selected_candidate = selected_policy.get("candidate_name")
        candidate_options = threshold_tuning_df["candidate_name"].dropna().unique().tolist()
        default_index = (
            candidate_options.index(selected_candidate)
            if selected_candidate in candidate_options
            else 0
        )
        tradeoff_candidate = st.selectbox(
            "Threshold Tradeoff Candidate",
            options=candidate_options,
            index=default_index,
        )
        tradeoff_df = threshold_tuning_df[
            threshold_tuning_df["candidate_name"] == tradeoff_candidate
        ].sort_values("threshold")
        chart_columns = [
            "precision",
            "recall",
            "f2",
            "approval_support_rate",
        ]
        st.line_chart(tradeoff_df.set_index("threshold")[chart_columns])
    else:
        show_warning_if_missing("Threshold tuning report", threshold_tuning_path)

    if threshold_selection_df is not None:
        with st.expander("View threshold-selection rules"):
            st.dataframe(threshold_selection_df, width="stretch")

    if precision_recall_curve_path.exists():
        st.image(str(precision_recall_curve_path), caption="Held-out precision-recall comparison")

with tab_governance:
    st.subheader("SHAP And LIME Explainability")
    if shap_summary_path.exists():
        st.image(str(shap_summary_path), caption="Global SHAP summary for the UCI XGBoost model")
    else:
        show_warning_if_missing("SHAP summary image", shap_summary_path)

    image_cols = st.columns(2)
    with image_cols[0]:
        if shap_local_path.exists():
            st.image(str(shap_local_path), caption="Local SHAP explanation")
        else:
            show_warning_if_missing("Local SHAP image", shap_local_path)
    with image_cols[1]:
        if lime_local_path.exists():
            st.image(str(lime_local_path), caption="Local LIME explanation")
        else:
            show_warning_if_missing("Local LIME image", lime_local_path)

    st.write(
        "Expected strong drivers include recent repayment delay, number of delayed months, "
        "credit limit, bill-to-limit utilization, and repayment amount patterns."
    )

with tab_governance:
    st.subheader("Fairness Analysis")
    fairness_df = load_csv(fairness_csv_path)
    fairness_json = load_json(fairness_json_path)
    mitigation_df = load_csv(mitigation_path)
    threshold_fairness_df = load_csv(threshold_fairness_comparison_path)

    if fairness_json is None:
        show_warning_if_missing("Fairness report", fairness_json_path)
    else:
        st.write(
            f"Protected attribute used in the saved report: `{fairness_json['protected_attribute']}`"
        )
        st.caption(fairness_json.get("favorable_outcome", "Favorable outcome: non-default."))
        fairness_metrics = fairness_json.get("fairness_metrics", {})
        required_metrics = {
            "demographic_parity_difference",
            "equalized_odds_difference",
            "equal_opportunity_difference",
            "disparate_impact_ratio",
        }
        if required_metrics.issubset(fairness_metrics):
            difference_col1, difference_col2 = st.columns(2)
            difference_col1.metric(
                "Demographic Parity Difference",
                f"{fairness_metrics['demographic_parity_difference']:.4f}",
            )
            difference_col2.metric(
                "Equalized Odds Difference",
                f"{fairness_metrics['equalized_odds_difference']:.4f}",
            )
            opportunity_col, ratio_col = st.columns(2)
            opportunity_col.metric(
                "Equal Opportunity Difference",
                f"{fairness_metrics['equal_opportunity_difference']:.4f}",
            )
            ratio_col.metric(
                "Disparate Impact Ratio",
                f"{fairness_metrics['disparate_impact_ratio']:.4f}",
            )

    if fairness_df is None:
        show_warning_if_missing("Fairness metrics CSV", fairness_csv_path)
    else:
        with st.expander("View saved fairness metrics"):
            st.dataframe(fairness_df, width="stretch")

    if mitigation_df is None:
        show_warning_if_missing("Fairness mitigation tradeoff file", mitigation_path)
    else:
        st.subheader("Fairness vs Performance Tradeoff")
        st.dataframe(mitigation_df, width="stretch")

    if threshold_fairness_df is None:
        show_warning_if_missing(
            "Threshold fairness comparison file",
            threshold_fairness_comparison_path,
        )
    else:
        st.subheader("Baseline vs Recall-Optimized Threshold Fairness")
        st.dataframe(threshold_fairness_df, width="stretch")

with tab_guidance:
    st.subheader("Counterfactual Guidance")
    prediction_result = st.session_state.get("current_prediction_result")
    if prediction_result:
        st.write("Current applicant guidance based on live local SHAP drivers:")
        for item in prediction_result["guidance"]:
            st.write(f"- {item}")
        improvement_scenarios = prediction_result.get("improvement_scenarios")
        if isinstance(improvement_scenarios, pd.DataFrame) and not improvement_scenarios.empty:
            st.markdown("**Improvement scenario simulation**")
            scenario_display = improvement_scenarios.copy()
            scenario_display["default_probability"] = scenario_display["default_probability"].map(
                lambda value: f"{value:.2%}"
            )
            scenario_display["change_vs_current"] = scenario_display["change_vs_current"].map(
                lambda value: f"{value:+.2%}"
            )
            st.dataframe(scenario_display, width="stretch")
        st.caption("These are decision-support suggestions only and do not promise approval.")
    else:
        counterfactual_payload = load_json(counterfactual_path)
        if counterfactual_payload is None:
            show_warning_if_missing("Counterfactual file", counterfactual_path)
        else:
            changes = format_counterfactual_changes(counterfactual_payload)
            if changes.empty:
                st.json(counterfactual_payload)
            else:
                st.dataframe(changes.astype(str), width="stretch")

with tab_applicant:
    st.subheader("Applicant Risk Scorecard Report")
    prediction_result = st.session_state.get("current_prediction_result")
    if not prediction_result:
        st.info("Run a prediction above to generate a report.")
    else:
        report = build_applicant_risk_report(
            probability=prediction_result["probability"],
            positive_drivers=prediction_result["positive_drivers"],
            negative_drivers=prediction_result["negative_drivers"],
            guidance=prediction_result["guidance"],
            threshold=prediction_result.get("threshold", DEFAULT_DECISION_THRESHOLD),
            exposure_estimate=prediction_result.get("exposure_estimate"),
            shap_warning=prediction_result.get("shap_warning"),
        )
        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("Default Probability", f"{report['probability']:.2%}")
        metric_b.metric("Decision Threshold", f"{report['threshold']:.0%}")
        metric_c.metric("Risk Band", report["risk_band"])
        recall_threshold = prediction_result.get("recall_policy_threshold")
        screening_rows = [
            {
                "policy": "baseline_threshold_050",
                "threshold": DEFAULT_DECISION_THRESHOLD,
                "screening_result": screening_flag_text(
                    prediction_result["probability"],
                    DEFAULT_DECISION_THRESHOLD,
                ),
            }
        ]
        if recall_threshold is not None:
            screening_rows.append(
                {
                    "policy": prediction_result.get("recall_policy_name"),
                    "threshold": recall_threshold,
                    "screening_result": screening_flag_text(
                        prediction_result["probability"],
                        recall_threshold,
                    ),
                }
            )
        st.markdown("**Screening policy comparison**")
        st.table(pd.DataFrame(screening_rows))
        st.write(report["interpretation"])
        st.markdown("**Top SHAP risk-increasing drivers**")
        for driver in prediction_result["positive_drivers"][:3]:
            st.write(f"- {driver['display_name']}: {driver['shap_value']:+.4f}")
            st.caption(driver["interpretation"])
        st.markdown("**Top SHAP risk-reducing drivers**")
        for driver in prediction_result["negative_drivers"][:3]:
            st.write(f"- {driver['display_name']}: {driver['shap_value']:+.4f}")
            st.caption(driver["interpretation"])
        st.markdown("**Counterfactual guidance**")
        for item in prediction_result["guidance"]:
            st.write(f"- {item}")
        st.info("This is not a production lending decision engine or regulatory scorecard.")
        st.download_button(
            "Download Markdown Report",
            data=report["markdown"],
            file_name="applicant_risk_scorecard_report.md",
            mime="text/markdown",
        )

with tab_governance:
    st.subheader("Leakage Audit")
    leakage_summary = load_json(leakage_path)
    if leakage_summary is None:
        show_warning_if_missing("Leakage audit summary", leakage_path)
    else:
        st.success(leakage_summary.get("conclusion", "Leakage audit completed."))
        st.write(
            "Target shuffle ROC-AUC:",
            round(leakage_summary["target_shuffle_test"]["roc_auc"], 4),
        )
        st.write("Top mutual-information review signals:")
        for item in leakage_summary["suspicious_features"][:10]:
            st.write(f"- {item}")
        st.caption(
            "`PAY_0` to `PAY_6` are historical repayment-status variables before the next-month target."
        )

with tab_governance:
    st.subheader("Advanced Model Comparison")
    st.write(
        "Deep Learning benchmark added to test whether model complexity improves default "
        "detection on structured credit data. The applicant workflow continues to use XGBoost."
    )
    dnn_metrics = load_json(deep_learning_metrics_path)
    dnn_comparison = load_csv(deep_learning_comparison_path)
    dnn_policy = load_json(deep_learning_policy_path)
    dnn_fairness = load_csv(deep_learning_fairness_path)

    if dnn_metrics is None:
        st.info(
            "DNN reports are optional and currently unavailable. Run "
            "`python -m src.deep_learning_benchmark` to generate them."
        )
    elif dnn_metrics.get("status") == "skipped":
        st.warning(dnn_metrics.get("reason", "Deep Learning benchmark was skipped."))
    else:
        st.success("DNN benchmark artifacts are available; XGBoost remains the primary model.")
        if dnn_policy:
            policy_metrics = dnn_policy.get("test_metrics", {})
            columns = st.columns(4)
            columns[0].metric("DNN Threshold", f"{dnn_policy.get('selected_threshold', 0):.2f}")
            columns[1].metric("DNN Recall", f"{policy_metrics.get('recall', 0):.2%}")
            columns[2].metric("DNN Precision", f"{policy_metrics.get('precision', 0):.2%}")
            columns[3].metric("DNN PR-AUC", f"{policy_metrics.get('pr_auc', 0):.4f}")

    if dnn_comparison is not None:
        st.markdown("**ML vs DL held-out comparison**")
        st.dataframe(dnn_comparison, width="stretch")
    if ml_vs_dl_curve_path.exists():
        st.image(
            str(ml_vs_dl_curve_path),
            caption="Logistic Regression vs XGBoost vs DNN precision-recall curve",
        )
    if dnn_fairness is not None:
        with st.expander("DNN fairness diagnostics"):
            st.dataframe(dnn_fairness, width="stretch")
            st.caption("Fairness metrics are diagnostic and are not proof of legal compliance.")
