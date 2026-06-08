"""Helpers for dashboard applicant input, prediction, and local explanations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import shap

USER_INPUT_FIELDS = [
    "Age",
    "Gender",
    "Nationality",
    "City",
    "EmploymentStatus",
    "AnnualIncome_AED",
    "OtherObligations_AED",
    "BureauScore",
    "LoanType",
    "LoanAmount_AED",
    "LoanTenureMonths",
    "InterestRate_pct",
    "Unemployment_pct",
    "Inflation_pct",
]

EXCLUDED_USER_INPUT_FIELDS = [
    "EMI_AED",
    "LoanToAnnualIncome",
    "DebtToIncomeRatio",
    "EMIToIncomeRatio",
    "LoanBurdenRatio",
    "PaymentStressScore",
    "BehavioralRiskFlag",
    "OnTimePayments_Last12M",
    "MissedPayments_Last12M",
    "MissedEMIs_Last6M",
    "SalaryDropFlag",
    "SpendingSpikeFlag",
    "AvgMonthlyDebit_AED",
    "StdMonthlyDebit_AED",
]

CATEGORICAL_FIELDS = ["Gender", "Nationality", "City", "EmploymentStatus", "LoanType"]
SENSITIVE_OR_NON_ACTIONABLE_FIELDS = {
    "Age",
    "Gender",
    "Nationality",
    "City",
    "Unemployment_pct",
    "Inflation_pct",
    "LoanStartYear",
    "LoanStartMonth",
    "LoanStartQuarter",
}

FEATURE_LABELS = {
    "Age": "Age",
    "Gender": "Gender",
    "Nationality": "Nationality",
    "City": "City",
    "EmploymentStatus": "Employment Status",
    "AnnualIncome_AED": "Annual Income",
    "OtherObligations_AED": "Existing Obligations",
    "BureauScore": "Bureau Score",
    "LoanType": "Loan Type",
    "LoanAmount_AED": "Loan Amount",
    "LoanTenureMonths": "Loan Tenure",
    "InterestRate_pct": "Interest Rate",
    "Unemployment_pct": "Unemployment Rate",
    "Inflation_pct": "Inflation Rate",
    "EMI_AED": "EMI",
    "LoanToAnnualIncome": "Loan-to-Income Ratio",
    "ObligationsToIncome": "Obligations-to-Income Ratio",
    "EMIToIncome": "EMI-to-Income Ratio",
    "DebtToIncomeRatio": "Debt-to-Income Ratio",
    "LoanBurdenRatio": "Loan Burden Ratio",
}


def calculate_emi(principal: float, annual_interest_rate_pct: float, tenure_months: int) -> float:
    principal = float(max(principal, 0.0))
    tenure_months = int(max(tenure_months, 1))
    monthly_rate = float(annual_interest_rate_pct) / 12.0 / 100.0

    if monthly_rate > 0:
        factor = (1 + monthly_rate) ** tenure_months
        return principal * monthly_rate * factor / (factor - 1)
    return principal / tenure_months


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
    return defaults


def build_applicant_model_row(
    user_input: dict[str, Any],
    reference_table: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    defaults = build_defaults(reference_table)
    annual_income = float(user_input["AnnualIncome_AED"])
    other_obligations = float(user_input["OtherObligations_AED"])
    loan_amount = float(user_input["LoanAmount_AED"])
    tenure_months = int(user_input["LoanTenureMonths"])
    interest_rate_pct = float(user_input["InterestRate_pct"])

    emi = calculate_emi(loan_amount, interest_rate_pct, tenure_months)
    fallback_loan_to_income = float(defaults.get("LoanToAnnualIncome", 0.0))
    fallback_emi_to_income = float(defaults.get("EMIToIncome", 0.0))
    fallback_obligations = float(defaults.get("ObligationsToIncome", 0.0))

    computed_ratios = {
        "EMI_AED": float(emi),
        "LoanToAnnualIncome": safe_ratio(loan_amount, annual_income, fallback_loan_to_income),
        "DebtToIncomeRatio": safe_ratio(
            other_obligations + emi * 12.0,
            annual_income,
            fallback_obligations,
        ),
        "EMIToIncomeRatio": safe_ratio(emi * 12.0, annual_income, fallback_emi_to_income),
        "LoanBurdenRatio": safe_ratio(loan_amount, annual_income, fallback_loan_to_income),
    }

    applicant_row = {
        "Age": int(user_input["Age"]),
        "Gender": str(user_input["Gender"]),
        "Nationality": str(user_input["Nationality"]),
        "City": str(user_input["City"]),
        "EmploymentStatus": str(user_input["EmploymentStatus"]),
        "AnnualIncome_AED": annual_income,
        "OtherObligations_AED": other_obligations,
        "BureauScore": float(user_input["BureauScore"]),
        "LoanType": str(user_input["LoanType"]),
        "LoanAmount_AED": loan_amount,
        "LoanTenureMonths": tenure_months,
        "InterestRate_pct": interest_rate_pct,
        "LoanStartYear": float(defaults.get("LoanStartYear", 0.0)),
        "LoanStartMonth": float(defaults.get("LoanStartMonth", 0.0)),
        "LoanStartQuarter": float(defaults.get("LoanStartQuarter", 0.0)),
        "Unemployment_pct": float(user_input["Unemployment_pct"]),
        "Inflation_pct": float(user_input["Inflation_pct"]),
        "EMI_AED": computed_ratios["EMI_AED"],
        "LoanToAnnualIncome": computed_ratios["LoanToAnnualIncome"],
        "ObligationsToIncome": safe_ratio(
            other_obligations,
            annual_income,
            fallback_obligations,
        ),
        "EMIToIncome": safe_ratio(emi, annual_income, fallback_emi_to_income),
    }

    row_frame = pd.DataFrame([applicant_row])
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


def interpret_driver(
    transformed_name: str,
    shap_value: float,
    applicant_df: pd.DataFrame,
    reference_table: pd.DataFrame,
) -> str:
    raw_feature, readable_name = humanize_feature_name(transformed_name)
    direction_text = "higher" if shap_value > 0 else "lower"

    if raw_feature in CATEGORICAL_FIELDS and transformed_name.startswith("cat__"):
        current_value = applicant_df.iloc[0].get(raw_feature, "Unknown")
        return f"Current {FEATURE_LABELS.get(raw_feature, raw_feature).lower()} is {current_value}, which pushes predicted risk {direction_text}."

    current_value = applicant_df.iloc[0].get(raw_feature)
    if raw_feature in reference_table.columns and pd.api.types.is_numeric_dtype(reference_table[raw_feature]):
        median_value = float(reference_table[raw_feature].median())
        relation = "above" if float(current_value) > median_value else "below"
        if raw_feature == "BureauScore":
            return f"Bureau score is {relation} the portfolio median, pushing predicted risk {direction_text}."
        if raw_feature in {"LoanToAnnualIncome", "LoanBurdenRatio"}:
            return f"Loan burden relative to income is {relation} the portfolio median, pushing predicted risk {direction_text}."
        if raw_feature in {"OtherObligations_AED", "ObligationsToIncome"}:
            return f"Existing obligations are {relation} the portfolio median, pushing predicted risk {direction_text}."
        if raw_feature in {"EMI_AED", "EMIToIncome"}:
            return f"Repayment burden is {relation} the portfolio median, pushing predicted risk {direction_text}."
        return f"{readable_name} is {relation} the portfolio median, pushing predicted risk {direction_text}."

    return f"{readable_name} contributes to {direction_text} predicted default risk."


def driver_reason_text(
    transformed_name: str,
    shap_value: float,
    applicant_df: pd.DataFrame,
    reference_table: pd.DataFrame,
) -> str:
    raw_feature, readable_name = humanize_feature_name(transformed_name)

    if raw_feature in CATEGORICAL_FIELDS and transformed_name.startswith("cat__"):
        current_value = applicant_df.iloc[0].get(raw_feature, "Unknown")
        return f"{FEATURE_LABELS.get(raw_feature, raw_feature).lower()} is {current_value}"

    current_value = applicant_df.iloc[0].get(raw_feature)
    if raw_feature in reference_table.columns and pd.api.types.is_numeric_dtype(reference_table[raw_feature]):
        median_value = float(reference_table[raw_feature].median())
        relation = "above" if float(current_value) > median_value else "below"
        if raw_feature == "BureauScore":
            if shap_value > 0:
                return "bureau score is relatively weak" if relation == "below" else "bureau score remains a notable model driver"
            return "bureau score is relatively strong" if relation == "above" else "bureau score helps reduce modeled risk"
        if raw_feature in {"LoanToAnnualIncome", "LoanAmount_AED", "EMI_AED", "EMIToIncome"}:
            if shap_value > 0:
                return (
                    "loan burden is high relative to income"
                    if relation == "above"
                    else "loan burden remains a notable model driver"
                )
            return (
                "loan burden is manageable relative to income"
                if relation == "below"
                else "loan burden helps reduce modeled risk"
            )
        if raw_feature in {"OtherObligations_AED", "ObligationsToIncome"}:
            if shap_value > 0:
                return (
                    "existing obligations are high"
                    if relation == "above"
                    else "existing obligations remain a notable model driver"
                )
            return (
                "existing obligations are manageable"
                if relation == "below"
                else "existing obligations help reduce modeled risk"
            )
        if raw_feature == "InterestRate_pct":
            if shap_value > 0:
                return "interest rate is relatively high" if relation == "above" else "interest rate remains a notable model driver"
            return "interest rate is comparatively moderate" if relation == "below" else "interest rate helps reduce modeled risk"
        if raw_feature == "AnnualIncome_AED":
            if shap_value > 0:
                return "annual income is relatively limited" if relation == "below" else "annual income is a notable model driver"
            return "annual income is relatively strong" if relation == "above" else "income profile helps reduce modeled risk"
        return f"{readable_name.lower()} is a notable model driver"

    return readable_name.lower()


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

    contributions = (
        pd.Series(shap_row, index=feature_names)
        .sort_values(key=lambda series: series.abs(), ascending=False)
    )

    drivers = []
    for transformed_name, shap_value in contributions.items():
        raw_feature, readable_name = humanize_feature_name(transformed_name)
        if transformed_name.startswith("cat__") and applicant_df.iloc[0].get(raw_feature) not in readable_name:
            continue
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
        title="Current Applicant SHAP Contributions",
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
        return "Likely Accept / Low Review Priority"
    if probability <= 0.60:
        return "Manual Review Recommended"
    return "High Risk / Strong Review Required"


def _select_actionable_positive_drivers(drivers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actionable = [
        driver for driver in drivers if driver["raw_feature"] not in SENSITIVE_OR_NON_ACTIONABLE_FIELDS
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
        lead_in = f"Your predicted default risk is {label} mainly because "
    else:
        actionable_drivers = _select_actionable_positive_drivers(positive_drivers)[:3]
        lead_in = f"Your predicted default risk is {label} mainly because "

    if not actionable_drivers:
        return f"Your predicted default risk is {label} based on the overall application-time profile."

    reasons = [driver.get("reason_text", driver["display_name"].lower()) for driver in actionable_drivers]
    deduped_reasons = list(dict.fromkeys(reasons))[:3]
    return lead_in + ", ".join(deduped_reasons) + "."


def generate_counterfactual_guidance(
    applicant_df: pd.DataFrame,
    positive_drivers: list[dict[str, Any]],
) -> list[str]:
    guidance: list[str] = []
    for driver in _select_actionable_positive_drivers(positive_drivers):
        raw_feature = driver["raw_feature"]
        if raw_feature == "BureauScore":
            guidance.append("Improving Bureau Score may reduce predicted risk.")
        elif raw_feature in {"LoanToAnnualIncome", "LoanAmount_AED", "EMI_AED", "EMIToIncome"}:
            guidance.append(
                "Reducing the requested loan amount, extending tenure, or increasing declared income may reduce predicted risk."
            )
        elif raw_feature in {"OtherObligations_AED", "ObligationsToIncome"}:
            guidance.append("Reducing existing obligations may improve affordability and may reduce predicted risk.")
        elif raw_feature == "InterestRate_pct":
            guidance.append("Securing a lower interest rate may reduce predicted risk.")
        elif raw_feature == "AnnualIncome_AED":
            guidance.append("Demonstrating stronger verifiable income may reduce predicted risk.")
        elif raw_feature == "EmploymentStatus":
            guidance.append("Providing stronger employment stability evidence may reduce predicted risk.")

    deduped_guidance = list(dict.fromkeys(guidance))
    if deduped_guidance:
        return deduped_guidance[:3]

    return [
        "Adjusting loan burden, affordability, or credit quality may reduce predicted risk.",
    ]
