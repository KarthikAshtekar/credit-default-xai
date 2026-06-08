from __future__ import annotations

import math

from dashboard.common import get_feature_table
from dashboard.prediction_helpers import (
    EXCLUDED_USER_INPUT_FIELDS,
    USER_INPUT_FIELDS,
    build_applicant_model_row,
    build_defaults,
    calculate_emi,
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
    feature_table = get_feature_table()
    applicant_df, _ = build_applicant_model_row(_sample_user_input(feature_table), feature_table)
    assert applicant_df.columns.tolist() == feature_table.columns.tolist()
    assert len(applicant_df) == 1


def test_derived_ratios_are_computed_internally() -> None:
    feature_table = get_feature_table()
    applicant_df, computed = build_applicant_model_row(_sample_user_input(feature_table), feature_table)

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
