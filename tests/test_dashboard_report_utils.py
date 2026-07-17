from __future__ import annotations

import pandas as pd

from dashboard.report_utils import add_fairness_group_labels, build_applicant_risk_report

FORBIDDEN_FAIRNESS_OVERCLAIMS = [
    "the model discriminates",
    "bias is proven",
    "legally discriminatory",
    "causal bias confirmed",
    "fairness guaranteed",
    "bias-free",
]


def test_build_applicant_risk_report_handles_missing_explanations() -> None:
    report = build_applicant_risk_report(
        probability=0.42,
        positive_drivers=None,
        negative_drivers=None,
        guidance=None,
        shap_warning="SHAP unavailable",
    )

    assert report["risk_band"] == "Medium Risk"
    assert "SHAP unavailable" in report["markdown"]
    assert "not a calibrated regulatory credit scorecard" in report["markdown"]
    lower_markdown = report["markdown"].lower()
    for phrase in FORBIDDEN_FAIRNESS_OVERCLAIMS:
        assert phrase not in lower_markdown


def test_build_applicant_risk_report_includes_driver_and_guidance_text() -> None:
    report = build_applicant_risk_report(
        probability=0.72,
        positive_drivers=[
            {
                "display_name": "Loan-to-Income Ratio",
                "shap_value": 0.31,
                "interpretation": "Bill-to-limit utilization is above the portfolio median.",
            }
        ],
        negative_drivers=[
            {
                "display_name": "Bureau Score",
                "shap_value": -0.22,
                "interpretation": "Repayment relative to bills helps reduce modeled risk.",
            }
        ],
        guidance=["Reducing bill-to-limit utilization may reduce predicted risk."],
    )

    assert report["risk_band"] == "High Risk"
    assert "Loan-to-Income Ratio" in report["markdown"]
    assert "Bureau Score" in report["markdown"]
    assert "Reducing bill-to-limit utilization" in report["markdown"]


def test_add_fairness_group_labels_handles_current_and_legacy_inputs() -> None:
    current = add_fairness_group_labels(pd.DataFrame({"sex_code": [1, 2], "metric": [0.1, 0.2]}))
    assert current is not None
    assert current["sex_group"].tolist() == ["Male", "Female"]
    assert current["group"].tolist() == ["Male (SEX=1)", "Female (SEX=2)"]

    legacy = add_fairness_group_labels(
        pd.DataFrame({"group": ["SEX=1", "SEX=2"], "metric": [0.1, 0.2]})
    )
    assert legacy is not None
    assert legacy["sex_code"].tolist() == [1, 2]
    assert legacy["sex_group"].tolist() == ["Male", "Female"]
