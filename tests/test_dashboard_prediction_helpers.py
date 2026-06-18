from __future__ import annotations

import math

import pandas as pd

from dashboard.common import get_application_artifact_paths
from dashboard.prediction_helpers import (
    EXCLUDED_USER_INPUT_FIELDS,
    USER_INPUT_FIELDS,
    build_applicant_model_row,
    build_applicant_presets,
    build_defaults,
    calculate_emi,
    driver_reason_text,
    interpret_driver,
    risk_band,
)


def _reference_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Age": 35,
                "Gender": "Female",
                "Nationality": "UAE",
                "City": "Dubai",
                "EmploymentStatus": "Salaried",
                "AnnualIncome_AED": 240000.0,
                "OtherObligations_AED": 12000.0,
                "BureauScore": 680.0,
                "LoanType": "Personal Loan",
                "LoanAmount_AED": 120000.0,
                "LoanTenureMonths": 24,
                "InterestRate_pct": 12.0,
                "LoanStartYear": 2024,
                "LoanStartMonth": 6,
                "LoanStartQuarter": 2,
                "Unemployment_pct": 4.0,
                "Inflation_pct": 2.5,
                "EMI_AED": 5000.0,
                "LoanToAnnualIncome": 0.5,
                "ObligationsToIncome": 0.05,
                "EMIToIncome": 0.02,
            },
            {
                "Age": 45,
                "Gender": "Male",
                "Nationality": "India",
                "City": "Abu Dhabi",
                "EmploymentStatus": "Self-employed",
                "AnnualIncome_AED": 180000.0,
                "OtherObligations_AED": 18000.0,
                "BureauScore": 620.0,
                "LoanType": "Auto Loan",
                "LoanAmount_AED": 150000.0,
                "LoanTenureMonths": 36,
                "InterestRate_pct": 10.0,
                "LoanStartYear": 2023,
                "LoanStartMonth": 3,
                "LoanStartQuarter": 1,
                "Unemployment_pct": 4.5,
                "Inflation_pct": 3.0,
                "EMI_AED": 4500.0,
                "LoanToAnnualIncome": 0.8,
                "ObligationsToIncome": 0.1,
                "EMIToIncome": 0.025,
            },
        ]
    )


def _sample_user_input(reference_table):
    defaults = build_defaults(reference_table)
    return {
        "Age": 35,
        "Gender": str(defaults["Gender"]),
        "Nationality": str(defaults["Nationality"]),
        "City": str(defaults["City"]),
        "EmploymentStatus": str(defaults["EmploymentStatus"]),
        "AnnualIncome_AED": 240000.0,
        "OtherObligations_AED": 12000.0,
        "BureauScore": 680.0,
        "LoanType": str(defaults["LoanType"]),
        "LoanAmount_AED": 120000.0,
        "LoanTenureMonths": 24,
        "InterestRate_pct": 12.0,
        "Unemployment_pct": 4.0,
        "Inflation_pct": 2.5,
    }


def test_calculate_emi_matches_standard_formula() -> None:
    emi = calculate_emi(100000, 12.0, 12)
    expected = 8884.878867834168
    assert math.isclose(emi, expected, rel_tol=1e-9)


def test_build_applicant_input_row_matches_application_schema() -> None:
    feature_table = _reference_table()
    applicant_df, _ = build_applicant_model_row(_sample_user_input(feature_table), feature_table)
    assert applicant_df.columns.tolist() == feature_table.columns.tolist()
    assert len(applicant_df) == 1


def test_derived_ratios_are_computed_internally() -> None:
    feature_table = _reference_table()
    applicant_df, computed = build_applicant_model_row(
        _sample_user_input(feature_table), feature_table
    )

    assert computed["EMI_AED"] > 0
    assert math.isclose(computed["LoanToAnnualIncome"], 0.5, rel_tol=1e-9)
    assert math.isclose(computed["LoanBurdenRatio"], 0.5, rel_tol=1e-9)
    assert math.isclose(
        applicant_df.iloc[0]["EMI_AED"],
        computed["EMI_AED"],
        rel_tol=1e-9,
    )
    assert math.isclose(
        applicant_df.iloc[0]["LoanToAnnualIncome"],
        computed["LoanToAnnualIncome"],
        rel_tol=1e-9,
    )


def test_dashboard_input_schema_excludes_emi_as_user_input() -> None:
    assert "EMI_AED" not in USER_INPUT_FIELDS
    assert "EMI_AED" in EXCLUDED_USER_INPUT_FIELDS


def test_dashboard_excludes_post_loan_behavioral_fields_from_user_input() -> None:
    disallowed_fields = {
        "OnTimePayments_Last12M",
        "MissedPayments_Last12M",
        "MissedEMIs_Last6M",
        "SalaryDropFlag",
        "SpendingSpikeFlag",
        "AvgMonthlyDebit_AED",
        "StdMonthlyDebit_AED",
    }
    assert disallowed_fields.isdisjoint(USER_INPUT_FIELDS)


def test_applicant_presets_use_only_user_input_fields() -> None:
    feature_table = _reference_table()
    presets = build_applicant_presets(feature_table)

    assert {
        "Low-risk salaried applicant",
        "High-burden applicant",
        "Borderline / medium-risk applicant",
    }.issubset(presets)
    for preset in presets.values():
        assert set(preset) == set(USER_INPUT_FIELDS)
        assert set(preset).isdisjoint(EXCLUDED_USER_INPUT_FIELDS)


def test_risk_band_assignment() -> None:
    assert risk_band(0.29) == "Low Risk"
    assert risk_band(0.30) == "Medium Risk"
    assert risk_band(0.60) == "Medium Risk"
    assert risk_band(0.61) == "High Risk"


def test_driver_interpretation_is_model_specific_and_economically_cautious() -> None:
    feature_table = _reference_table()
    applicant_df, _ = build_applicant_model_row(_sample_user_input(feature_table), feature_table)

    interest_rate_text = interpret_driver(
        "num__InterestRate_pct",
        -0.05,
        applicant_df,
        feature_table,
    )
    loan_to_income_text = interpret_driver(
        "num__LoanToAnnualIncome",
        -0.08,
        applicant_df,
        feature_table,
    )

    assert "interest-rate feature contributes slightly toward lower predicted risk" in (
        interest_rate_text
    )
    assert "loan-to-income feature contributes toward lower predicted risk" in (loan_to_income_text)
    assert "pushing predicted risk" not in interest_rate_text
    assert "fitted model" in interest_rate_text

    interest_rate_reason = driver_reason_text(
        "num__InterestRate_pct",
        -0.05,
        applicant_df,
        feature_table,
    )
    assert "slightly risk-reducing contribution" in interest_rate_reason
    assert "helps reduce modeled risk" not in interest_rate_reason


def test_application_artifact_paths_resolve_to_reports_directory() -> None:
    paths = get_application_artifact_paths()

    assert (
        paths["performance"]
        .as_posix()
        .endswith("reports/model_validation/clean_feature_model_comparison.csv")
    )
    assert (
        paths["shap_summary"]
        .as_posix()
        .endswith(
            "reports/explainability_reports/application_model/xgboost_application_shap_summary.png"
        )
    )
