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
    driver_reason_text,
    generate_counterfactual_guidance,
    interpret_driver,
    risk_band,
)
from src.dataset_adapters import APPLICATION_PUBLIC_FEATURES, PAY_STATUS_COLUMNS


def _reference_table() -> pd.DataFrame:
    rows = []
    for index in range(2):
        row = {}
        for column in APPLICATION_PUBLIC_FEATURES:
            row[column] = 0.0
        row.update(
            {
                "LIMIT_BAL": 100000.0 + index * 50000,
                "EDUCATION": 2,
                "MARRIAGE": 2,
                "AGE": 35 + index,
                "PAY_0": index,
                "PAY_2": index,
                "PAY_3": 0,
                "PAY_4": 0,
                "PAY_5": 0,
                "PAY_6": 0,
                "BILL_AMT1": 50000.0,
                "BILL_AMT2": 45000.0,
                "BILL_AMT3": 40000.0,
                "BILL_AMT4": 35000.0,
                "BILL_AMT5": 30000.0,
                "BILL_AMT6": 25000.0,
                "PAY_AMT1": 2000.0,
                "PAY_AMT2": 2000.0,
                "PAY_AMT3": 2000.0,
                "PAY_AMT4": 2000.0,
                "PAY_AMT5": 2000.0,
                "PAY_AMT6": 2000.0,
                "AvgBillToLimitRatio": 0.4,
                "AvgPaymentToBillRatio": 0.05,
                "RecentPaymentDelay": index,
                "MaxPaymentDelay": index,
                "NumDelayedMonths": index,
                "AvgBillAmount": 37500.0,
                "AvgPaymentAmount": 2000.0,
                "PaymentToLimitRatio": 0.02,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)[APPLICATION_PUBLIC_FEATURES]


def _sample_user_input(reference_table):
    defaults = build_defaults(reference_table)
    user_input = {field: defaults.get(field, 0.0) for field in USER_INPUT_FIELDS}
    user_input.update(
        {
            "SEX": 2,
            "EDUCATION": 2,
            "MARRIAGE": 2,
            "LIMIT_BAL": 100000.0,
            "AGE": 35,
            "BILL_AMT1": 50000.0,
            "BILL_AMT2": 45000.0,
            "BILL_AMT3": 40000.0,
            "BILL_AMT4": 35000.0,
            "BILL_AMT5": 30000.0,
            "BILL_AMT6": 25000.0,
            "PAY_AMT1": 5000.0,
            "PAY_AMT2": 4000.0,
            "PAY_AMT3": 3000.0,
            "PAY_AMT4": 2000.0,
            "PAY_AMT5": 1000.0,
            "PAY_AMT6": 500.0,
        }
    )
    for column in PAY_STATUS_COLUMNS:
        user_input[column] = 0
    return user_input


def test_build_applicant_input_row_matches_public_application_schema() -> None:
    feature_table = _reference_table()
    applicant_df, _ = build_applicant_model_row(_sample_user_input(feature_table), feature_table)
    assert applicant_df.columns.tolist() == feature_table.columns.tolist()
    assert len(applicant_df) == 1


def test_engineered_uci_features_are_computed_internally() -> None:
    feature_table = _reference_table()
    applicant_df, computed = build_applicant_model_row(
        _sample_user_input(feature_table), feature_table
    )

    assert math.isclose(computed["BillToLimitRatio_1"], 0.5, rel_tol=1e-9)
    assert computed["AvgPaymentToBillRatio"] > 0
    assert applicant_df.iloc[0]["AvgBillToLimitRatio"] == computed["AvgBillToLimitRatio"]


def test_dashboard_input_schema_excludes_engineered_features_as_user_input() -> None:
    assert "AvgBillToLimitRatio" not in USER_INPUT_FIELDS
    assert "AvgBillToLimitRatio" in EXCLUDED_USER_INPUT_FIELDS
    assert "SEX" in USER_INPUT_FIELDS


def test_applicant_presets_use_only_user_input_fields() -> None:
    feature_table = _reference_table()
    presets = build_applicant_presets(feature_table)

    assert {
        "Low-risk repayment profile",
        "High-delay profile",
        "High-utilization profile",
    }.issubset(presets)
    for preset in presets.values():
        assert set(preset) == set(USER_INPUT_FIELDS)
        assert set(preset).isdisjoint(EXCLUDED_USER_INPUT_FIELDS)


def test_risk_band_assignment() -> None:
    assert risk_band(0.29) == "Low Risk"
    assert risk_band(0.30) == "Medium Risk"
    assert risk_band(0.60) == "Medium Risk"
    assert risk_band(0.61) == "High Risk"


def test_driver_interpretation_uses_uci_language() -> None:
    feature_table = _reference_table()
    applicant_df, _ = build_applicant_model_row(_sample_user_input(feature_table), feature_table)

    delay_text = interpret_driver("num__PAY_0", 0.05, applicant_df, feature_table)
    ratio_text = interpret_driver("num__AvgBillToLimitRatio", 0.08, applicant_df, feature_table)

    assert "repayment delay" in delay_text
    assert "bill-to-limit" in ratio_text.lower()
    assert "Bureau" not in delay_text

    reason = driver_reason_text("num__PAY_0", 0.05, applicant_df, feature_table)
    assert "repayment delay" in reason


def test_counterfactual_guidance_uses_repayment_and_utilization_language() -> None:
    guidance = generate_counterfactual_guidance(
        pd.DataFrame(),
        [
            {"raw_feature": "PAY_0"},
            {"raw_feature": "AvgBillToLimitRatio"},
            {"raw_feature": "AvgPaymentToBillRatio"},
        ],
    )

    assert any("timely repayment" in item for item in guidance)
    assert any("utilization" in item for item in guidance)


def test_application_artifact_paths_resolve_to_public_reports() -> None:
    paths = get_application_artifact_paths()

    assert (
        paths["performance"]
        .as_posix()
        .endswith("reports/model_validation/public_credit_model_comparison.csv")
    )
    assert (
        paths["shap_summary"]
        .as_posix()
        .endswith(
            "reports/explainability_reports/application_model/xgboost_public_shap_summary.png"
        )
    )
    assert (
        paths["deep_learning_metrics"]
        .as_posix()
        .endswith("reports/model_validation/deep_learning_metrics.json")
    )
    assert (
        paths["deep_learning_comparison"]
        .as_posix()
        .endswith("reports/model_validation/deep_learning_comparison.csv")
    )
