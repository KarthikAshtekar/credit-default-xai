"""Helpers for UCI dashboard applicant input, prediction, and local explanations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import shap

from src.dataset_adapters import (
    BILL_AMOUNT_COLUMNS,
    ENGINEERED_UCI_COLUMNS,
    PAY_AMOUNT_COLUMNS,
    PAY_STATUS_COLUMNS,
    add_uci_credit_features,
)

USER_INPUT_FIELDS = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    *PAY_STATUS_COLUMNS,
    *BILL_AMOUNT_COLUMNS,
    *PAY_AMOUNT_COLUMNS,
]

EXCLUDED_USER_INPUT_FIELDS = [
    *ENGINEERED_UCI_COLUMNS,
    "Default_Flag",
]

CATEGORICAL_FIELDS = ["SEX", "EDUCATION", "MARRIAGE"]
SENSITIVE_OR_NON_ACTIONABLE_FIELDS = {"SEX", "AGE", "MARRIAGE", "EDUCATION"}

PRESET_NAMES = [
    "Reference cardholder",
    "Low-risk repayment profile",
    "High-delay profile",
    "High-utilization profile",
]

FEATURE_LABELS = {
    "LIMIT_BAL": "Credit Limit",
    "SEX": "Sex",
    "EDUCATION": "Education",
    "MARRIAGE": "Marriage",
    "AGE": "Age",
    "PAY_0": "Most Recent Repayment Status",
    "PAY_2": "Repayment Status 2",
    "PAY_3": "Repayment Status 3",
    "PAY_4": "Repayment Status 4",
    "PAY_5": "Repayment Status 5",
    "PAY_6": "Repayment Status 6",
    "BILL_AMT1": "Most Recent Bill Amount",
    "BILL_AMT2": "Bill Amount 2",
    "BILL_AMT3": "Bill Amount 3",
    "BILL_AMT4": "Bill Amount 4",
    "BILL_AMT5": "Bill Amount 5",
    "BILL_AMT6": "Bill Amount 6",
    "PAY_AMT1": "Most Recent Payment Amount",
    "PAY_AMT2": "Payment Amount 2",
    "PAY_AMT3": "Payment Amount 3",
    "PAY_AMT4": "Payment Amount 4",
    "PAY_AMT5": "Payment Amount 5",
    "PAY_AMT6": "Payment Amount 6",
    "AvgBillToLimitRatio": "Average Bill-to-Limit Ratio",
    "AvgPaymentToBillRatio": "Average Payment-to-Bill Ratio",
    "RecentPaymentDelay": "Recent Payment Delay",
    "MaxPaymentDelay": "Maximum Payment Delay",
    "NumDelayedMonths": "Number of Delayed Months",
    "AvgBillAmount": "Average Bill Amount",
    "AvgPaymentAmount": "Average Payment Amount",
    "PaymentToLimitRatio": "Payment-to-Limit Ratio",
}

SEX_OPTIONS = {1: "Male", 2: "Female"}
EDUCATION_OPTIONS = {1: "Graduate school", 2: "University", 3: "High school", 4: "Other"}
MARRIAGE_OPTIONS = {1: "Married", 2: "Single", 3: "Other"}


def safe_ratio(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    if denominator and denominator != 0:
        return float(numerator) / float(denominator)
    return float(fallback)


def build_defaults(reference_table: pd.DataFrame) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for column in reference_table.columns:
        series = reference_table[column].dropna()
        if pd.api.types.is_numeric_dtype(reference_table[column]):
            defaults[column] = float(series.median()) if not series.empty else 0.0
        else:
            defaults[column] = str(series.mode().iloc[0]) if not series.empty else "Unknown"
    defaults.setdefault("SEX", 2)
    defaults.setdefault("EDUCATION", 2)
    defaults.setdefault("MARRIAGE", 2)
    return defaults


def _numeric(defaults: dict[str, Any], field: str, fallback: float) -> float:
    return float(defaults.get(field, fallback))


def _base_reference(defaults: dict[str, Any]) -> dict[str, Any]:
    reference = {
        "LIMIT_BAL": _numeric(defaults, "LIMIT_BAL", 180000.0),
        "SEX": int(defaults.get("SEX", 2)),
        "EDUCATION": int(defaults.get("EDUCATION", 2)),
        "MARRIAGE": int(defaults.get("MARRIAGE", 2)),
        "AGE": int(_numeric(defaults, "AGE", 35.0)),
    }
    for column in PAY_STATUS_COLUMNS:
        reference[column] = int(_numeric(defaults, column, 0.0))
    for column in BILL_AMOUNT_COLUMNS:
        reference[column] = _numeric(defaults, column, 35000.0)
    for column in PAY_AMOUNT_COLUMNS:
        reference[column] = _numeric(defaults, column, 2000.0)
    return reference


def build_applicant_presets(reference_table: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Return demo-safe presets using UCI credit-card input fields."""

    reference = _base_reference(build_defaults(reference_table))
    low_risk = {
        **reference,
        "LIMIT_BAL": 300000.0,
        "PAY_0": -1,
        "PAY_2": -1,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
    }
    for bill_col in BILL_AMOUNT_COLUMNS:
        low_risk[bill_col] = 25000.0
    for pay_col in PAY_AMOUNT_COLUMNS:
        low_risk[pay_col] = 8000.0

    high_delay = {
        **reference,
        "LIMIT_BAL": 50000.0,
        "PAY_0": 3,
        "PAY_2": 2,
        "PAY_3": 2,
        "PAY_4": 1,
        "PAY_5": 1,
        "PAY_6": 0,
    }
    for bill_col in BILL_AMOUNT_COLUMNS:
        high_delay[bill_col] = 45000.0
    for pay_col in PAY_AMOUNT_COLUMNS:
        high_delay[pay_col] = 500.0

    high_utilization = {
        **reference,
        "LIMIT_BAL": 80000.0,
        "PAY_0": 1,
        "PAY_2": 1,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
    }
    for bill_col in BILL_AMOUNT_COLUMNS:
        high_utilization[bill_col] = 76000.0
    for pay_col in PAY_AMOUNT_COLUMNS:
        high_utilization[pay_col] = 1500.0

    return {
        "Reference cardholder": reference,
        "Low-risk repayment profile": low_risk,
        "High-delay profile": high_delay,
        "High-utilization profile": high_utilization,
    }


def build_applicant_model_row(
    user_input: dict[str, Any],
    reference_table: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    row = {}
    for field in USER_INPUT_FIELDS:
        row[field] = user_input[field]

    row_frame = pd.DataFrame([row])
    for column in USER_INPUT_FIELDS:
        row_frame[column] = pd.to_numeric(row_frame[column], errors="coerce")

    row_frame = add_uci_credit_features(row_frame)
    computed_ratios = {
        column: float(row_frame.iloc[0][column])
        for column in ENGINEERED_UCI_COLUMNS
        if column in row_frame.columns
    }

    defaults = build_defaults(reference_table)
    for column in reference_table.columns:
        if column not in row_frame.columns:
            row_frame[column] = defaults[column]

    row_frame = row_frame[reference_table.columns.tolist()]
    return row_frame, computed_ratios


def humanize_feature_name(transformed_name: str) -> tuple[str, str]:
    if transformed_name.startswith("num__"):
        raw_feature = transformed_name.removeprefix("num__")
        return raw_feature, FEATURE_LABELS.get(raw_feature, raw_feature)

    if transformed_name.startswith("cat__"):
        core_name = transformed_name.removeprefix("cat__")
        for base_name in CATEGORICAL_FIELDS:
            prefix = f"{base_name}_"
            if core_name.startswith(prefix):
                category = core_name[len(prefix) :]
                return base_name, f"{FEATURE_LABELS.get(base_name, base_name)}: {category}"
        return core_name, core_name.replace("_", " ")

    return transformed_name, FEATURE_LABELS.get(transformed_name, transformed_name)


def _relation_to_median(
    raw_feature: str,
    applicant_df: pd.DataFrame,
    reference_table: pd.DataFrame,
) -> str:
    current_value = applicant_df.iloc[0].get(raw_feature)
    if raw_feature not in reference_table.columns or not pd.api.types.is_numeric_dtype(
        reference_table[raw_feature]
    ):
        return "not comparable"
    median_value = float(reference_table[raw_feature].median())
    return "above" if float(current_value) > median_value else "below"


def interpret_driver(
    transformed_name: str,
    shap_value: float,
    applicant_df: pd.DataFrame,
    reference_table: pd.DataFrame,
) -> str:
    raw_feature, readable_name = humanize_feature_name(transformed_name)
    risk_direction = "higher" if shap_value > 0 else "lower"
    relation = _relation_to_median(raw_feature, applicant_df, reference_table)

    if raw_feature in PAY_STATUS_COLUMNS or raw_feature in {
        "RecentPaymentDelay",
        "MaxPaymentDelay",
        "NumDelayedMonths",
    }:
        if shap_value > 0:
            return "Recent repayment delay contributes toward higher predicted default risk in this fitted model."
        return "Timelier recent repayment contributes toward lower predicted default risk in this fitted model."
    if raw_feature.startswith("BillToLimitRatio") or raw_feature == "AvgBillToLimitRatio":
        if shap_value > 0 and relation == "above":
            return "Higher bill-to-limit utilization contributes toward higher predicted risk in this fitted model."
        return (
            "Bill-to-limit utilization contributes toward "
            f"{risk_direction} predicted risk in this fitted model."
        )
    if raw_feature in {"AvgPaymentToBillRatio", "PaymentToLimitRatio"}:
        if shap_value < 0:
            return "Stronger repayment relative to bills contributes toward lower predicted risk in this fitted model."
        return "Low repayment relative to bill or limit contributes toward higher predicted risk in this fitted model."
    if raw_feature in BILL_AMOUNT_COLUMNS or raw_feature == "AvgBillAmount":
        return (
            "Outstanding bill amount contributes toward "
            f"{risk_direction} predicted risk in this fitted model."
        )
    if raw_feature in PAY_AMOUNT_COLUMNS or raw_feature == "AvgPaymentAmount":
        return (
            "Payment amount pattern contributes toward "
            f"{risk_direction} predicted risk in this fitted model."
        )
    if raw_feature == "LIMIT_BAL":
        return (
            f"Credit limit contributes toward {risk_direction} predicted risk in this fitted model."
        )

    return (
        f"For this applicant, the {readable_name.lower()} feature contributes toward "
        f"{risk_direction} predicted risk in this fitted model."
    )


def driver_reason_text(
    transformed_name: str,
    shap_value: float,
    applicant_df: pd.DataFrame,
    reference_table: pd.DataFrame,
) -> str:
    raw_feature, readable_name = humanize_feature_name(transformed_name)
    contribution = "risk-increasing" if shap_value > 0 else "risk-reducing"
    relation = _relation_to_median(raw_feature, applicant_df, reference_table)

    if raw_feature in PAY_STATUS_COLUMNS or raw_feature in {
        "RecentPaymentDelay",
        "MaxPaymentDelay",
        "NumDelayedMonths",
    }:
        return (
            "recent repayment delay is higher"
            if shap_value > 0
            else "recent repayment history is stronger"
        )
    if raw_feature.startswith("BillToLimitRatio") or raw_feature == "AvgBillToLimitRatio":
        if relation == "above" and shap_value > 0:
            return "bill-to-limit utilization is high"
        return f"bill-to-limit utilization is {contribution}"
    if raw_feature in {"AvgPaymentToBillRatio", "PaymentToLimitRatio"}:
        return (
            "repayment relative to bills is weak"
            if shap_value > 0
            else "repayment relative to bills is stronger"
        )
    return f"the fitted model assigns a {contribution} contribution to {readable_name.lower()}"


def compute_local_shap_analysis(
    model_pipeline,
    applicant_df: pd.DataFrame,
    reference_table: pd.DataFrame,
) -> dict[str, Any]:
    preprocessor = model_pipeline.named_steps["preprocessor"]
    estimator = model_pipeline.named_steps["classifier"]
    transformed = preprocessor.transform(applicant_df)
    feature_names = preprocessor.get_feature_names_out()
    feature_frame = pd.DataFrame(
        transformed.toarray() if hasattr(transformed, "toarray") else transformed,
        columns=feature_names,
    )

    explainer = shap.TreeExplainer(estimator)
    raw_shap = explainer.shap_values(feature_frame)
    if isinstance(raw_shap, list):
        shap_row = np.asarray(raw_shap[-1])[0]
    else:
        shap_row = np.asarray(raw_shap)[0]

    contributions = pd.Series(shap_row, index=feature_names).sort_values(
        key=lambda series: series.abs(), ascending=False
    )

    drivers = []
    for transformed_name, shap_value in contributions.items():
        raw_feature, readable_name = humanize_feature_name(transformed_name)
        drivers.append(
            {
                "feature": transformed_name,
                "raw_feature": raw_feature,
                "display_name": readable_name,
                "shap_value": float(shap_value),
                "interpretation": interpret_driver(
                    transformed_name,
                    float(shap_value),
                    applicant_df,
                    reference_table,
                ),
                "reason_text": driver_reason_text(
                    transformed_name,
                    float(shap_value),
                    applicant_df,
                    reference_table,
                ),
            }
        )

    positive_drivers = [driver for driver in drivers if driver["shap_value"] > 0][:5]
    negative_drivers = [driver for driver in drivers if driver["shap_value"] < 0][:5]

    plot_rows = positive_drivers[:3] + negative_drivers[:3]
    plot_df = pd.DataFrame(plot_rows)
    if not plot_df.empty:
        plot_df["direction"] = np.where(
            plot_df["shap_value"] > 0,
            "Increasing Risk",
            "Reducing Risk",
        )

    return {
        "positive_drivers": positive_drivers,
        "negative_drivers": negative_drivers,
        "plot_df": plot_df,
    }


def build_local_shap_figure(plot_df: pd.DataFrame):
    if plot_df.empty:
        return None

    ordered = plot_df.sort_values("shap_value")
    return px.bar(
        ordered,
        x="shap_value",
        y="display_name",
        color="direction",
        orientation="h",
        color_discrete_map={
            "Increasing Risk": "#b22222",
            "Reducing Risk": "#2f6b2f",
        },
        title="Current Cardholder SHAP Contributions",
        labels={"shap_value": "SHAP contribution", "display_name": "Feature"},
    )


def risk_band(probability: float) -> str:
    if probability < 0.30:
        return "Low Risk"
    if probability <= 0.60:
        return "Medium Risk"
    return "High Risk"


def decision_support_recommendation(probability: float) -> str:
    if probability < 0.30:
        return "Low predicted default risk / low review priority"
    if probability <= 0.60:
        return "Manual review recommended"
    return "High predicted default risk / strong review required"


def _select_actionable_positive_drivers(drivers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actionable = [
        driver
        for driver in drivers
        if driver["raw_feature"] not in SENSITIVE_OR_NON_ACTIONABLE_FIELDS
    ]
    return actionable or drivers


def generate_plain_english_explanation(
    probability: float,
    positive_drivers: list[dict[str, Any]],
    negative_drivers: list[dict[str, Any]],
) -> str:
    label = risk_band(probability).lower()
    if probability < 0.30:
        actionable_drivers = _select_actionable_positive_drivers(negative_drivers)[:3]
        lead_in = f"The predicted next-month default risk is {label} mainly because "
    else:
        actionable_drivers = _select_actionable_positive_drivers(positive_drivers)[:3]
        lead_in = f"The predicted next-month default risk is {label} mainly because "

    if not actionable_drivers:
        return f"The predicted next-month default risk is {label} based on the overall credit-card profile."

    reasons = [
        driver.get("reason_text", driver["display_name"].lower()) for driver in actionable_drivers
    ]
    deduped_reasons = list(dict.fromkeys(reasons))[:3]
    return lead_in + ", ".join(deduped_reasons) + "."


def generate_counterfactual_guidance(
    applicant_df: pd.DataFrame,
    positive_drivers: list[dict[str, Any]],
) -> list[str]:
    guidance: list[str] = []
    for driver in _select_actionable_positive_drivers(positive_drivers):
        raw_feature = driver["raw_feature"]
        if raw_feature in PAY_STATUS_COLUMNS or raw_feature in {
            "RecentPaymentDelay",
            "MaxPaymentDelay",
            "NumDelayedMonths",
        }:
            guidance.append(
                "Maintaining timely repayment over recent months may reduce predicted default risk."
            )
        elif raw_feature.startswith("BillToLimitRatio") or raw_feature in {
            "AvgBillToLimitRatio",
            "AvgBillAmount",
        }:
            guidance.append(
                "Lowering revolving utilization or bill balances relative to the credit limit may reduce predicted risk."
            )
        elif raw_feature in {
            "AvgPaymentToBillRatio",
            "PaymentToLimitRatio",
            "AvgPaymentAmount",
            *PAY_AMOUNT_COLUMNS,
        }:
            guidance.append(
                "Increasing repayment amounts relative to outstanding bills may reduce predicted risk."
            )

    deduped_guidance = list(dict.fromkeys(guidance))
    if deduped_guidance:
        return deduped_guidance[:3]

    return [
        "Reducing recent repayment delays, lowering bill-to-limit utilization, and strengthening repayment relative to bill amounts may reduce predicted risk.",
    ]
