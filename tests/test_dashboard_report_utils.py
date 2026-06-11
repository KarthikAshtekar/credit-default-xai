from __future__ import annotations

from dashboard.report_utils import build_applicant_risk_report


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


def test_build_applicant_risk_report_includes_driver_and_guidance_text() -> None:
    report = build_applicant_risk_report(
        probability=0.72,
        positive_drivers=[
            {
                "display_name": "Loan-to-Income Ratio",
                "shap_value": 0.31,
                "interpretation": "Loan burden is above the portfolio median.",
            }
        ],
        negative_drivers=[
            {
                "display_name": "Bureau Score",
                "shap_value": -0.22,
                "interpretation": "Bureau score helps reduce modeled risk.",
            }
        ],
        guidance=["Reducing the requested loan amount may reduce predicted risk."],
    )

    assert report["risk_band"] == "High Risk"
    assert "Loan-to-Income Ratio" in report["markdown"]
    assert "Bureau Score" in report["markdown"]
    assert "Reducing the requested loan amount" in report["markdown"]
