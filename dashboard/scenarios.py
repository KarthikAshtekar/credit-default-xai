"""Scenario simulation helpers for applicant-level dashboard interactions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dashboard.prediction_helpers import (
    BILL_AMOUNT_COLUMNS,
    PAY_AMOUNT_COLUMNS,
    PAY_STATUS_COLUMNS,
    build_applicant_model_row,
)


def build_summary_applicant_inputs(
    preset: dict[str, Any],
    target_credit_amount: float,
    latest_repayment_status: int,
    maximum_recent_delay: int,
    average_bill_amount: float,
    average_payment_amount: float,
) -> dict[str, Any]:
    applicant_inputs = dict(preset)
    applicant_inputs["LIMIT_BAL"] = float(target_credit_amount)
    applicant_inputs["PAY_0"] = int(latest_repayment_status)
    for column in PAY_STATUS_COLUMNS[1:]:
        applicant_inputs[column] = int(maximum_recent_delay)
    for column in BILL_AMOUNT_COLUMNS:
        applicant_inputs[column] = float(average_bill_amount)
    for column in PAY_AMOUNT_COLUMNS:
        applicant_inputs[column] = float(average_payment_amount)
    return applicant_inputs


def average_bill_amount(applicant_inputs: dict[str, Any]) -> float:
    values = [float(applicant_inputs[column]) for column in BILL_AMOUNT_COLUMNS]
    return float(np.mean(values))


def average_payment_amount(applicant_inputs: dict[str, Any]) -> float:
    values = [float(applicant_inputs[column]) for column in PAY_AMOUNT_COLUMNS]
    return float(np.mean(values))


def predict_default_risk(
    model: Any,
    applicant_inputs: dict[str, Any],
    feature_table: pd.DataFrame,
) -> float:
    applicant_df, _ = build_applicant_model_row(applicant_inputs, feature_table)
    return float(model.predict_proba(applicant_df)[:, 1][0])


def estimate_maximum_advisable_credit_exposure(
    model: Any,
    applicant_inputs: dict[str, Any],
    feature_table: pd.DataFrame,
    threshold: float,
) -> dict[str, Any]:
    current_limit = max(float(applicant_inputs["LIMIT_BAL"]), 0.0)
    if current_limit <= 0:
        return {
            "max_exposure": 0.0,
            "threshold": threshold,
            "probability_at_exposure": None,
            "note": "No positive target credit amount was entered for simulation.",
        }

    limits = np.linspace(min(10_000.0, current_limit), current_limit, num=24)
    rows = []
    for candidate_limit in limits:
        scenario = dict(applicant_inputs)
        scenario["LIMIT_BAL"] = float(candidate_limit)
        rows.append(
            {
                "limit": float(candidate_limit),
                "probability": predict_default_risk(model, scenario, feature_table),
            }
        )

    feasible = [row for row in rows if row["probability"] < threshold]
    if feasible:
        selected = max(feasible, key=lambda row: row["limit"])
        return {
            "max_exposure": selected["limit"],
            "threshold": threshold,
            "probability_at_exposure": selected["probability"],
            "note": "Highest simulated target credit amount below the manual-review threshold.",
        }

    lowest_risk = min(rows, key=lambda row: row["probability"])
    return {
        "max_exposure": 0.0,
        "threshold": threshold,
        "probability_at_exposure": lowest_risk["probability"],
        "note": "No simulated target credit amount stayed below the manual-review threshold.",
    }


def build_target_credit_curve(
    model: Any,
    applicant_inputs: dict[str, Any],
    feature_table: pd.DataFrame,
    points: int = 12,
) -> pd.DataFrame:
    current_limit = max(float(applicant_inputs["LIMIT_BAL"]), 1.0)
    lower = max(current_limit * 0.4, 1.0)
    upper = max(current_limit * 1.2, lower)
    rows = []
    for candidate_limit in np.linspace(lower, upper, num=points):
        scenario = dict(applicant_inputs)
        scenario["LIMIT_BAL"] = float(candidate_limit)
        rows.append(
            {
                "target_credit_amount": float(candidate_limit),
                "predicted_default_risk": predict_default_risk(model, scenario, feature_table),
            }
        )
    return pd.DataFrame(rows)


def simulate_adjusted_applicant(
    applicant_inputs: dict[str, Any],
    target_credit_amount: float,
    repayment_delay: int,
    average_bill: float,
    average_payment: float,
) -> dict[str, Any]:
    scenario = dict(applicant_inputs)
    scenario["LIMIT_BAL"] = float(target_credit_amount)
    for column in PAY_STATUS_COLUMNS:
        scenario[column] = int(repayment_delay)
    for column in BILL_AMOUNT_COLUMNS:
        scenario[column] = float(average_bill)
    for column in PAY_AMOUNT_COLUMNS:
        scenario[column] = float(average_payment)
    return scenario


def summarize_shortcomings(
    applicant_inputs: dict[str, Any],
    probability: float,
) -> list[dict[str, str]]:
    limit = max(float(applicant_inputs.get("LIMIT_BAL", 0.0)), 1.0)
    bill = average_bill_amount(applicant_inputs)
    payment = average_payment_amount(applicant_inputs)
    max_delay = max(int(applicant_inputs.get(column, 0)) for column in PAY_STATUS_COLUMNS)
    utilization = bill / limit
    payment_ratio = payment / bill if bill else 0.0

    rows: list[dict[str, str]] = []
    if max_delay > 0:
        rows.append(
            {
                "shortcoming": "Recent repayment delay is high.",
                "why": "Recent delinquency is a strong signal of near-term default risk.",
                "action": "Maintain on-time repayments for future billing cycles before requesting higher exposure.",
            }
        )
    if utilization > 0.45:
        rows.append(
            {
                "shortcoming": "Bill balance is high relative to the target credit amount.",
                "why": "High utilization leaves less repayment capacity buffer.",
                "action": "Reduce outstanding bill balances before increasing the target credit amount.",
            }
        )
    if payment_ratio < 0.08:
        rows.append(
            {
                "shortcoming": "Repayment amount is low relative to bills.",
                "why": "Low repayment intensity can indicate weaker short-term repayment capacity.",
                "action": "Increase regular repayments relative to outstanding bills.",
            }
        )
    if probability >= 0.30:
        rows.append(
            {
                "shortcoming": "Predicted default risk is above the low-risk band.",
                "why": "The score suggests the profile needs manual review before higher exposure.",
                "action": "Lower the requested exposure or improve repayment behavior before applying.",
            }
        )
    if rows:
        return rows[:4]
    return [
        {
            "shortcoming": "No major shortcoming is visible in the current simulation.",
            "why": "The current repayment, utilization, and payment profile is comparatively low risk.",
            "action": "Maintain payment consistency and avoid increasing utilization sharply.",
        }
    ]
