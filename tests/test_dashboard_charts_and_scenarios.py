from __future__ import annotations

import numpy as np
import pandas as pd

from dashboard.charts import (
    build_fairness_chart,
    build_model_comparison_chart,
    build_pr_curve_chart,
    build_scenario_curve_chart,
    build_threshold_fairness_frontier_chart,
    build_threshold_tradeoff_chart,
    load_report_csv_safely,
)
from dashboard.scenarios import (
    build_summary_applicant_inputs,
    build_target_credit_curve,
    estimate_maximum_advisable_credit_exposure,
    simulate_adjusted_applicant,
    summarize_shortcomings,
)
from dashboard.ui_components import MAIN_APPLICANT_LABELS, friendly_model_name
from src.dataset_adapters import APPLICATION_PUBLIC_FEATURES


class DummyModel:
    def predict_proba(self, X):
        delay = X.get("RecentPaymentDelay", pd.Series([0.0] * len(X))).astype(float)
        utilization = X.get("AvgBillToLimitRatio", pd.Series([0.0] * len(X))).astype(float)
        payment_ratio = X.get("AvgPaymentToBillRatio", pd.Series([0.0] * len(X))).astype(float)
        risk = np.clip(0.1 + 0.08 * delay + 0.35 * utilization - 0.1 * payment_ratio, 0.01, 0.99)
        return np.column_stack([1 - risk, risk])


def _reference_table() -> pd.DataFrame:
    row = {column: 0.0 for column in APPLICATION_PUBLIC_FEATURES}
    row.update(
        {
            "LIMIT_BAL": 100000.0,
            "EDUCATION": 2,
            "MARRIAGE": 2,
            "AGE": 35,
            "PAY_0": 1,
            "PAY_2": 1,
            "PAY_3": 0,
            "PAY_4": 0,
            "PAY_5": 0,
            "PAY_6": 0,
            "BILL_AMT1": 50000.0,
            "BILL_AMT2": 50000.0,
            "BILL_AMT3": 50000.0,
            "BILL_AMT4": 50000.0,
            "BILL_AMT5": 50000.0,
            "BILL_AMT6": 50000.0,
            "PAY_AMT1": 3000.0,
            "PAY_AMT2": 3000.0,
            "PAY_AMT3": 3000.0,
            "PAY_AMT4": 3000.0,
            "PAY_AMT5": 3000.0,
            "PAY_AMT6": 3000.0,
            "AvgBillToLimitRatio": 0.5,
            "AvgPaymentToBillRatio": 0.06,
            "RecentPaymentDelay": 1,
            "MaxPaymentDelay": 1,
            "NumDelayedMonths": 2,
            "AvgBillAmount": 50000.0,
            "AvgPaymentAmount": 3000.0,
            "PaymentToLimitRatio": 0.03,
        }
    )
    return pd.DataFrame([row])[APPLICATION_PUBLIC_FEATURES]


def _preset() -> dict[str, float]:
    return {
        "LIMIT_BAL": 100000.0,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 2,
        "AGE": 35,
        "PAY_0": 1,
        "PAY_2": 1,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": 50000.0,
        "BILL_AMT2": 50000.0,
        "BILL_AMT3": 50000.0,
        "BILL_AMT4": 50000.0,
        "BILL_AMT5": 50000.0,
        "BILL_AMT6": 50000.0,
        "PAY_AMT1": 3000.0,
        "PAY_AMT2": 3000.0,
        "PAY_AMT3": 3000.0,
        "PAY_AMT4": 3000.0,
        "PAY_AMT5": 3000.0,
        "PAY_AMT6": 3000.0,
    }


def test_plotly_chart_builders_return_figures() -> None:
    predictions = pd.DataFrame({"y_true": [0, 0, 1, 1], "y_proba": [0.1, 0.3, 0.6, 0.9]})
    pr_fig = build_pr_curve_chart({"xgboost_public": predictions})
    assert pr_fig is not None
    assert len(pr_fig.data) == 1

    threshold_df = pd.DataFrame(
        {
            "threshold": [0.2, 0.3],
            "precision": [0.5, 0.6],
            "recall": [0.7, 0.6],
            "f2": [0.65, 0.6],
            "candidate_name": ["xgboost_public_baseline_threshold_050"] * 2,
        }
    )
    threshold_fig = build_threshold_tradeoff_chart(
        threshold_df, "xgboost_public_baseline_threshold_050", 0.25
    )
    assert threshold_fig is not None
    assert len(threshold_fig.data) == 3

    comparison_fig = build_model_comparison_chart(
        pd.DataFrame(
            {
                "model_name": ["xgboost_public", "dnn_baseline"],
                "roc_auc": [0.77, 0.76],
                "pr_auc": [0.54, 0.52],
                "recall": [0.34, 0.35],
                "f2": [0.38, 0.38],
            }
        )
    )
    assert comparison_fig is not None

    fairness_fig = build_fairness_chart(
        pd.DataFrame(
            {
                "policy": ["baseline_threshold_050", "recall_optimized"],
                "demographic_parity_difference": [0.02, 0.07],
                "equal_opportunity_difference": [0.01, 0.05],
                "equalized_odds_difference": [0.02, 0.07],
            }
        )
    )
    assert fairness_fig is not None

    frontier_fig = build_threshold_fairness_frontier_chart(
        pd.DataFrame(
            {
                "threshold": [0.25, 0.50],
                "precision": [0.48, 0.66],
                "recall": [0.58, 0.34],
                "demographic_parity_difference": [0.069, 0.022],
                "equalized_odds_difference": [0.072, 0.023],
            }
        )
    )
    assert frontier_fig is not None
    assert len(frontier_fig.data) == 4


def test_chart_builders_handle_missing_data_gracefully() -> None:
    assert build_pr_curve_chart({}) is None
    assert build_threshold_tradeoff_chart(pd.DataFrame()) is None
    assert build_model_comparison_chart(None) is None
    assert build_fairness_chart(None) is None
    assert build_threshold_fairness_frontier_chart(None) is None
    assert build_scenario_curve_chart(pd.DataFrame(), 0.25) is None


def test_dashboard_deep_dive_missing_artifact_guard(tmp_path) -> None:
    assert load_report_csv_safely(tmp_path / "missing_deep_dive.csv") is None
    assert build_threshold_fairness_frontier_chart(pd.DataFrame({"threshold": [0.25]})) is None


def test_scenario_outputs_are_valid() -> None:
    applicant = build_summary_applicant_inputs(_preset(), 100000.0, 1, 1, 50000.0, 3000.0)
    feature_table = _reference_table()
    model = DummyModel()

    exposure = estimate_maximum_advisable_credit_exposure(model, applicant, feature_table, 0.5)
    assert {"max_exposure", "threshold", "probability_at_exposure"}.issubset(exposure)

    curve = build_target_credit_curve(model, applicant, feature_table, points=5)
    assert len(curve) == 5
    assert curve["predicted_default_risk"].between(0, 1).all()

    adjusted = simulate_adjusted_applicant(applicant, 80000.0, 0, 30000.0, 6000.0)
    assert adjusted["LIMIT_BAL"] == 80000.0
    assert adjusted["PAY_0"] == 0

    shortcomings = summarize_shortcomings(applicant, 0.35)
    assert shortcomings
    assert {"shortcoming", "why", "action"}.issubset(shortcomings[0])


def test_main_ui_labels_hide_raw_model_and_feature_names() -> None:
    assert friendly_model_name("xgboost_public_baseline_threshold_050") == "XGBoost baseline"
    assert friendly_model_name("dnn_baseline") == "Deep learning benchmark"
    assert all("PAY_" not in label and "BILL_AMT" not in label for label in MAIN_APPLICANT_LABELS)
