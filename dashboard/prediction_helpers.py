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

PRESET_NAMES = [
    "Reference applicant",
    "Low-risk salaried applicant",
    "High-burden applicant",
    "Borderline / medium-risk applicant",
]

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


def _category_default(defaults: dict[str, Any], field: str) -> str:
    return str(defaults.get(field, "Unknown"))


def build_applicant_presets(reference_table: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Return demo-safe presets using only application-time user input fields."""

    defaults = build_defaults(reference_table)
    reference = {
        "Age": int(defaults.get("Age", 35)),
        "Gender": _category_default(defaults, "Gender"),
        "Nationality": _category_default(defaults, "Nationality"),
        "City": _category_default(defaults, "City"),
        "EmploymentStatus": _category_default(defaults, "EmploymentStatus"),
        "AnnualIncome_AED": float(defaults.get("AnnualIncome_AED", 180000.0)),
        "OtherObligations_AED": float(defaults.get("OtherObligations_AED", 20000.0)),
        "BureauScore": float(defaults.get("BureauScore", 650.0)),
        "LoanType": _category_default(defaults, "LoanType"),
        "LoanAmount_AED": float(defaults.get("LoanAmount_AED", 150000.0)),
        "LoanTenureMonths": int(defaults.get("LoanTenureMonths", 36)),
        "InterestRate_pct": float(defaults.get("InterestRate_pct", 10.0)),
        "Unemployment_pct": float(defaults.get("Unemployment_pct", 4.0)),
        "Inflation_pct": float(defaults.get("Inflation_pct", 3.0)),
    }

    low_risk = {
        **reference,
        "Age": 38,
        "EmploymentStatus": "Salaried",
        "AnnualIncome_AED": 420000.0,
        "OtherObligations_AED": 10000.0,
        "BureauScore": 820.0,
        "LoanAmount_AED": 90000.0,
        "LoanTenureMonths": 36,
        "InterestRate_pct": 7.5,
    }
    high_burden = {
        **reference,
        "Age": 29,
        "AnnualIncome_AED": 85000.0,
        "OtherObligations_AED": 45000.0,
        "BureauScore": 430.0,
        "LoanAmount_AED": 260000.0,
        "LoanTenureMonths": 24,
        "InterestRate_pct": 16.5,
    }
    borderline = {
        **reference,
        "Age": 34,
        "AnnualIncome_AED": 185000.0,
        "OtherObligations_AED": 28000.0,
        "BureauScore": 635.0,
        "LoanAmount_AED": 145000.0,
        "LoanTenureMonths": 36,
        "InterestRate_pct": 11.5,
    }

    return {
        "Reference applicant": reference,
        "Low-risk salaried applicant": low_risk,
        "High-burden applicant": high_burden,
        "Borderline / medium-risk applicant": borderline,
    }


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
    risk_direction = "higher" if shap_value > 0 else "lower"

    if raw_feature in CATEGORICAL_FIELDS and transformed_name.startswith("cat__"):
        feature_label = FEATURE_LABELS.get(raw_feature, raw_feature).lower()
        return (
            f"For this applicant, the {feature_label} feature contributes toward "
            f"{risk_direction} predicted risk in this fitted model."
        )

    current_value = applicant_df.iloc[0].get(raw_feature)
    if raw_feature in reference_table.columns and pd.api.types.is_numeric_dtype(
        reference_table[raw_feature]
    ):
        median_value = float(reference_table[raw_feature].median())
        relation = "above" if float(current_value) > median_value else "below"
        if raw_feature == "BureauScore":
            if shap_value > 0 and relation == "below":
                return (
                    "Lower bureau score contributes toward higher predicted risk in this "
                    "fitted model."
                )
            if shap_value < 0 and relation == "above":
                return (
                    "Higher bureau score contributes toward lower predicted risk in this "
                    "fitted model."
                )
            return (
                "For this applicant, the bureau-score feature contributes toward "
                f"{risk_direction} predicted risk in this fitted model."
            )
        if raw_feature in {"LoanToAnnualIncome", "LoanBurdenRatio"}:
            return (
                "For this applicant, the loan-to-income feature contributes toward "
                f"{risk_direction} predicted risk in this fitted model."
            )
        if raw_feature in {"OtherObligations_AED", "ObligationsToIncome"}:
            return (
                "For this applicant, the existing-obligations feature contributes toward "
                f"{risk_direction} predicted risk in this fitted model."
            )
        if raw_feature in {"EMI_AED", "EMIToIncome"}:
            if shap_value > 0 and relation == "above":
                return (
                    "Higher repayment burden contributes toward higher predicted risk in "
                    "this fitted model."
                )
            return (
                "For this applicant, the repayment-burden feature contributes toward "
                f"{risk_direction} predicted risk in this fitted model."
            )
        if raw_feature == "InterestRate_pct" and shap_value < 0:
            return (
                "For this applicant, the interest-rate feature contributes slightly toward "
                "lower predicted risk in this fitted model."
            )
        return (
            f"For this applicant, the {readable_name.lower()} feature contributes toward "
            f"{risk_direction} predicted risk in this fitted model."
        )

    return (
        f"For this fitted model, {readable_name.lower()} contributes toward "
        f"{risk_direction} predicted risk."
    )


def driver_reason_text(
    transformed_name: str,
    shap_value: float,
    applicant_df: pd.DataFrame,
    reference_table: pd.DataFrame,
) -> str:
    raw_feature, readable_name = humanize_feature_name(transformed_name)
    contribution = "risk-increasing" if shap_value > 0 else "risk-reducing"

    if raw_feature in CATEGORICAL_FIELDS and transformed_name.startswith("cat__"):
        feature_label = FEATURE_LABELS.get(raw_feature, raw_feature).lower()
        return (
            f"the fitted model assigns a {contribution} contribution to the {feature_label} feature"
        )

    current_value = applicant_df.iloc[0].get(raw_feature)
    if raw_feature in reference_table.columns and pd.api.types.is_numeric_dtype(
        reference_table[raw_feature]
    ):
        median_value = float(reference_table[raw_feature].median())
        relation = "above" if float(current_value) > median_value else "below"
        if raw_feature == "BureauScore":
            if shap_value > 0 and relation == "below":
                return "lower bureau score contributes toward higher risk in the fitted model"
            if shap_value < 0 and relation == "above":
                return "higher bureau score contributes toward lower risk in the fitted model"
            return f"the fitted model assigns a {contribution} contribution to bureau score"
        if raw_feature in {"LoanToAnnualIncome", "LoanAmount_AED", "EMI_AED", "EMIToIncome"}:
            if shap_value > 0 and relation == "above":
                return "higher repayment burden contributes toward higher risk in the fitted model"
            if shap_value < 0 and relation == "below":
                return "lower repayment burden contributes toward lower risk in the fitted model"
            return (
                f"the fitted model assigns a {contribution} contribution to the "
                "repayment-burden feature"
            )
        if raw_feature in {"OtherObligations_AED", "ObligationsToIncome"}:
            if shap_value > 0 and relation == "above":
                return (
                    "higher existing obligations contribute toward higher risk in the fitted model"
                )
            if shap_value < 0 and relation == "below":
                return "lower existing obligations contribute toward lower risk in the fitted model"
            return (
                f"the fitted model assigns a {contribution} contribution to the "
                "existing-obligations feature"
            )
        if raw_feature == "InterestRate_pct":
            qualifier = "slightly " if shap_value < 0 else ""
            return (
                f"the fitted model assigns a {qualifier}{contribution} contribution to the "
                "interest-rate feature"
            )
        if raw_feature == "AnnualIncome_AED":
            if shap_value > 0 and relation == "below":
                return "lower annual income contributes toward higher risk in the fitted model"
            if shap_value < 0 and relation == "above":
                return "higher annual income contributes toward lower risk in the fitted model"
            return (
                f"the fitted model assigns a {contribution} contribution to the "
                "annual-income feature"
            )
        return (
            f"the fitted model assigns a {contribution} contribution to the "
            f"{readable_name.lower()} feature"
        )

    return (
        f"the fitted model assigns a {contribution} contribution to the "
        f"{readable_name.lower()} feature"
    )


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
        if (
            transformed_name.startswith("cat__")
            and applicant_df.iloc[0].get(raw_feature) not in readable_name
        ):
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
        lead_in = f"Your predicted default risk is {label} mainly because "
    else:
        actionable_drivers = _select_actionable_positive_drivers(positive_drivers)[:3]
        lead_in = f"Your predicted default risk is {label} mainly because "

    if not actionable_drivers:
        return (
            f"Your predicted default risk is {label} based on the overall application-time profile."
        )

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
        if raw_feature == "BureauScore":
            guidance.append("Improving Bureau Score may reduce predicted risk.")
        elif raw_feature in {"LoanToAnnualIncome", "LoanAmount_AED", "EMI_AED", "EMIToIncome"}:
            guidance.append(
                "Reducing the requested loan amount, extending tenure, or increasing declared income may reduce predicted risk."
            )
        elif raw_feature in {"OtherObligations_AED", "ObligationsToIncome"}:
            guidance.append(
                "Reducing existing obligations may improve affordability and may reduce predicted risk."
            )
        elif raw_feature == "InterestRate_pct":
            guidance.append("Securing a lower interest rate may reduce predicted risk.")
        elif raw_feature == "AnnualIncome_AED":
            guidance.append("Demonstrating stronger verifiable income may reduce predicted risk.")
        elif raw_feature == "EmploymentStatus":
            guidance.append(
                "Providing stronger employment stability evidence may reduce predicted risk."
            )

    deduped_guidance = list(dict.fromkeys(guidance))
    if deduped_guidance:
        return deduped_guidance[:3]

    return [
        "Adjusting loan burden, affordability, or credit quality may reduce predicted risk.",
    ]
