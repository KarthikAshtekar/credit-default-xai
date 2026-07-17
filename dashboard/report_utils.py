"""Reusable applicant risk report helpers for the Streamlit dashboard."""

from __future__ import annotations

from typing import Any

import pandas as pd

from dashboard.prediction_helpers import decision_support_recommendation, risk_band
from src.protected_attributes import sex_group, sex_group_display

DEFAULT_DECISION_THRESHOLD = 0.50


def add_fairness_group_labels(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    """Ensure fairness governance tables have numeric and readable SEX labels."""

    if frame is None:
        return None

    output = frame.copy()
    if "sex_code" not in output.columns:
        source = output["group"] if "group" in output.columns else pd.Series([None] * len(output))
        output["sex_code"] = source.astype(str).str.extract(r"SEX=?\s*(\d+)")[0]
        output["sex_code"] = pd.to_numeric(output["sex_code"], errors="coerce").astype("Int64")

    output["sex_group"] = output["sex_code"].map(lambda value: sex_group(value))
    output["group"] = output["sex_code"].map(lambda value: sex_group_display(value))
    return output


def _driver_line(driver: dict[str, Any]) -> str:
    name = (
        driver.get("display_name") or driver.get("raw_feature") or driver.get("feature", "Feature")
    )
    value = float(driver.get("shap_value", 0.0))
    interpretation = driver.get("interpretation") or driver.get("reason_text") or "Model driver"
    return f"- {name}: SHAP {value:+.4f}. {interpretation}"


def build_applicant_risk_report(
    probability: float,
    positive_drivers: list[dict[str, Any]] | None = None,
    negative_drivers: list[dict[str, Any]] | None = None,
    guidance: list[str] | None = None,
    threshold: float = DEFAULT_DECISION_THRESHOLD,
    exposure_estimate: dict[str, Any] | None = None,
    shap_warning: str | None = None,
) -> dict[str, Any]:
    """Build a stakeholder-facing, non-regulatory applicant risk report."""

    positive_drivers = positive_drivers or []
    negative_drivers = negative_drivers or []
    guidance = guidance or []
    band = risk_band(probability)
    recommendation = decision_support_recommendation(probability)

    if probability >= threshold:
        threshold_text = "above"
    else:
        threshold_text = "below"

    interpretation = (
        f"The model estimates a {probability:.2%} probability of default, which is "
        f"{threshold_text} the {threshold:.0%} review threshold and falls in the {band.lower()} band."
    )

    markdown_lines = [
        "# Applicant Risk Scorecard Report",
        "",
        f"- Predicted default probability: **{probability:.2%}**",
        f"- Model decision threshold: **{threshold:.0%}**",
        f"- Risk band: **{band}**",
        f"- Decision-support recommendation: **{recommendation}**",
        "",
        "## Interpretation",
        interpretation,
        "",
        "## Top Risk-Increasing Drivers",
    ]
    markdown_lines.extend(
        [_driver_line(driver) for driver in positive_drivers[:5]] or ["- Not available."]
    )
    markdown_lines.extend(["", "## Top Risk-Reducing Drivers"])
    markdown_lines.extend(
        [_driver_line(driver) for driver in negative_drivers[:5]] or ["- Not available."]
    )
    markdown_lines.extend(["", "## Counterfactual Guidance"])
    markdown_lines.extend(
        [f"- {item}" for item in guidance[:5]] or ["- Not available for this applicant."]
    )

    if exposure_estimate:
        probability_at_exposure = exposure_estimate.get("probability_at_exposure")
        exposure_probability_text = (
            "not available"
            if probability_at_exposure is None
            else f"{float(probability_at_exposure):.2%}"
        )
        markdown_lines.extend(
            [
                "",
                "## Model-Supported Advisable Credit Exposure",
                (
                    "- Maximum advisable credit exposure tested: "
                    f"**{float(exposure_estimate.get('max_exposure', 0.0)):,.0f}**"
                ),
                f"- Predicted risk at that simulated exposure: **{exposure_probability_text}**",
                f"- Simulation note: {exposure_estimate.get('note', 'Not available.')}",
            ]
        )

    if shap_warning:
        markdown_lines.extend(["", "## Explanation Availability", f"- {shap_warning}"])

    markdown_lines.extend(
        [
            "",
            "## Governance Notes",
            "- Fairness caveat: saved fairness metrics are group-level diagnostics and do not prove complete fairness for every applicant.",
            "- Leakage-safe feature note: this report uses historical UCI repayment, bill, and payment fields available before the next-month default target.",
            "- Feature-policy note: SEX is used for fairness auditing and excluded from active final training features.",
            "- Limitation: this is not a production lending decision engine and is not a calibrated regulatory credit scorecard.",
        ]
    )

    return {
        "probability": probability,
        "threshold": threshold,
        "risk_band": band,
        "recommendation": recommendation,
        "interpretation": interpretation,
        "markdown": "\n".join(markdown_lines),
    }
