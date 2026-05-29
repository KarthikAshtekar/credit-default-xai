"""Feature engineering helpers for credit default modeling."""

from __future__ import annotations

import pandas as pd

from .utils import safe_divide


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if {"OtherObligations_AED", "AnnualIncome_AED"}.issubset(out.columns):
        out["ObligationsToIncome"] = safe_divide(
            out["OtherObligations_AED"], out["AnnualIncome_AED"]
        )

    if {"EMI_AED", "AnnualIncome_AED"}.issubset(out.columns):
        out["EMIToIncome"] = safe_divide(out["EMI_AED"], out["AnnualIncome_AED"])

    if {"MissedPayments_Last12M", "OnTimePayments_Last12M"}.issubset(out.columns):
        denom = out["MissedPayments_Last12M"] + out["OnTimePayments_Last12M"]
        out["MissedPaymentRate"] = safe_divide(out["MissedPayments_Last12M"], denom)

    return out


def add_behavioral_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "SalaryDropFlag" in out.columns and "SpendingSpikeFlag" in out.columns:
        out["StressSignalCount"] = out["SalaryDropFlag"].astype(float) + out[
            "SpendingSpikeFlag"
        ].astype(float)

    if "PastDefaults" in out.columns and "MissedEMIs_Last6M" in out.columns:
        out["HistoricalRiskScore"] = 2 * out["PastDefaults"].astype(float) + out[
            "MissedEMIs_Last6M"
        ].astype(float)

    return out


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    out = add_ratio_features(df)
    out = add_behavioral_flags(out)
    return out
