"""Dataset adapters for public credit-default data sources."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGET_COL = "Default_Flag"

UCI_DEFAULT_CREDIT_CARD_NAME = "default_credit_card"
UCI_DEFAULT_CREDIT_CARD_DISPLAY_NAME = (
    "UCI Default of Credit Card Clients / Taiwan credit-card default"
)

UCI_COLUMN_MAP = {
    "X1": "LIMIT_BAL",
    "X2": "SEX",
    "X3": "EDUCATION",
    "X4": "MARRIAGE",
    "X5": "AGE",
    "X6": "PAY_0",
    "X7": "PAY_2",
    "X8": "PAY_3",
    "X9": "PAY_4",
    "X10": "PAY_5",
    "X11": "PAY_6",
    "X12": "BILL_AMT1",
    "X13": "BILL_AMT2",
    "X14": "BILL_AMT3",
    "X15": "BILL_AMT4",
    "X16": "BILL_AMT5",
    "X17": "BILL_AMT6",
    "X18": "PAY_AMT1",
    "X19": "PAY_AMT2",
    "X20": "PAY_AMT3",
    "X21": "PAY_AMT4",
    "X22": "PAY_AMT5",
    "X23": "PAY_AMT6",
    "Y": TARGET_COL,
    "DEFAULT_PAYMENT_NEXT_MONTH": TARGET_COL,
    "DEFAULT_PAYMENT_NEXT_MONTH_": TARGET_COL,
    "DEFAULT.PAYMENT.NEXT.MONTH": TARGET_COL,
}

UCI_BASE_COLUMNS = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]

BORROWER_PROFILE_COLUMNS = ["SEX", "EDUCATION", "MARRIAGE", "AGE"]
PAY_STATUS_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_AMOUNT_COLUMNS = [
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
]
PAY_AMOUNT_COLUMNS = [
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]
ENGINEERED_UCI_COLUMNS = [
    "BillToLimitRatio_1",
    "BillToLimitRatio_2",
    "BillToLimitRatio_3",
    "BillToLimitRatio_4",
    "BillToLimitRatio_5",
    "BillToLimitRatio_6",
    "AvgBillToLimitRatio",
    "AvgPaymentToBillRatio",
    "RecentPaymentDelay",
    "MaxPaymentDelay",
    "NumDelayedMonths",
    "AvgBillAmount",
    "AvgPaymentAmount",
    "PaymentToLimitRatio",
]
UCI_ID_COLUMNS = ["ID", "CLIENT_ID"]

PROTECTED_ATTRIBUTE = "SEX"
AUDIT_SENSITIVE_COLUMNS = ["SEX", "AGE", "MARRIAGE", "EDUCATION"]

APPLICATION_PUBLIC_FEATURES = [
    "LIMIT_BAL",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    *PAY_STATUS_COLUMNS,
    *BILL_AMOUNT_COLUMNS,
    *PAY_AMOUNT_COLUMNS,
    *ENGINEERED_UCI_COLUMNS,
]

FULL_PUBLIC_DIAGNOSTIC_FEATURES = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    *PAY_STATUS_COLUMNS,
    *BILL_AMOUNT_COLUMNS,
    *PAY_AMOUNT_COLUMNS,
    *ENGINEERED_UCI_COLUMNS,
]


@dataclass(frozen=True)
class DatasetCoverageBlock:
    block: str
    columns: list[str]
    description: str


def safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    return np.where((b == 0) | (b.isna()), 0, a / b)


def clean_column_key(column: object) -> str:
    key = str(column).strip().replace(".", "_").replace("-", "_").replace(" ", "_")
    while "__" in key:
        key = key.replace("__", "_")
    return key.upper()


def normalize_uci_default_credit_card_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize UCI Taiwan credit-card columns to project-standard names."""

    rename_map = {}
    seen_names: set[str] = set()
    for column in df.columns:
        key = clean_column_key(column)
        candidate = UCI_COLUMN_MAP.get(key, key)
        if candidate in seen_names:
            candidate = f"{candidate}_{len(seen_names)}"
        rename_map[column] = candidate
        seen_names.add(candidate)
    return df.rename(columns=rename_map)


def _coerce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def add_uci_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create UCI-compatible utilization, repayment, and delay features."""

    out = df.copy()
    out = _coerce_numeric_columns(
        out,
        ["LIMIT_BAL", *PAY_STATUS_COLUMNS, *BILL_AMOUNT_COLUMNS, *PAY_AMOUNT_COLUMNS],
    )

    for idx, bill_col in enumerate(BILL_AMOUNT_COLUMNS, start=1):
        out[f"BillToLimitRatio_{idx}"] = safe_divide(out[bill_col], out["LIMIT_BAL"])

    payment_to_bill_ratios = []
    for pay_col, bill_col in zip(PAY_AMOUNT_COLUMNS, BILL_AMOUNT_COLUMNS):
        payment_to_bill_ratios.append(safe_divide(out[pay_col], out[bill_col].abs()))

    out["AvgBillToLimitRatio"] = out[[f"BillToLimitRatio_{idx}" for idx in range(1, 7)]].mean(
        axis=1
    )
    out["AvgPaymentToBillRatio"] = pd.DataFrame(payment_to_bill_ratios).T.mean(axis=1)
    out["RecentPaymentDelay"] = out["PAY_0"]
    out["MaxPaymentDelay"] = out[PAY_STATUS_COLUMNS].max(axis=1)
    out["NumDelayedMonths"] = (out[PAY_STATUS_COLUMNS] > 0).sum(axis=1)
    out["AvgBillAmount"] = out[BILL_AMOUNT_COLUMNS].mean(axis=1)
    out["AvgPaymentAmount"] = out[PAY_AMOUNT_COLUMNS].mean(axis=1)
    out["PaymentToLimitRatio"] = safe_divide(out["AvgPaymentAmount"], out["LIMIT_BAL"])

    out[ENGINEERED_UCI_COLUMNS] = out[ENGINEERED_UCI_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return out


def adapt_uci_default_credit_card(df: pd.DataFrame) -> pd.DataFrame:
    """Adapt UCI Default of Credit Card Clients data to the project schema."""

    out = normalize_uci_default_credit_card_columns(df)
    target_candidates = [
        TARGET_COL,
        "DEFAULT_PAYMENT_NEXT_MONTH",
        "DEFAULT_PAYMENT_NEXT_MONTH_",
        "DEFAULT.PAYMENT.NEXT.MONTH",
        "Y",
    ]
    target_source = next((col for col in target_candidates if col in out.columns), None)
    if target_source is None:
        raise ValueError("UCI default credit-card data is missing the default target column.")
    if target_source != TARGET_COL:
        out = out.rename(columns={target_source: TARGET_COL})

    required = UCI_BASE_COLUMNS + [TARGET_COL]
    missing = [column for column in required if column not in out.columns]
    if missing:
        raise ValueError(
            "UCI default credit-card data is missing required columns: " + ", ".join(missing)
        )

    out = _coerce_numeric_columns(out, required)
    out = out.dropna(subset=[TARGET_COL]).copy()
    out[TARGET_COL] = out[TARGET_COL].astype(int)
    out = add_uci_credit_features(out)
    return out


def get_uci_feature_policy() -> dict[str, object]:
    return {
        "target_column": TARGET_COL,
        "protected_attribute": PROTECTED_ATTRIBUTE,
        "protected_attribute_policy": (
            "SEX is retained in the raw/audit dataframe and excluded from the "
            "application_public training feature set."
        ),
        "audit_sensitive_columns": AUDIT_SENSITIVE_COLUMNS,
        "application_public_features": APPLICATION_PUBLIC_FEATURES,
        "full_public_diagnostic_features": FULL_PUBLIC_DIAGNOSTIC_FEATURES,
        "timing_note": (
            "PAY_0 to PAY_6 are historical repayment-status variables used to predict "
            "next-month default; they are not treated as leakage for this modeling question."
        ),
    }


def dataset_coverage_blocks() -> list[DatasetCoverageBlock]:
    return [
        DatasetCoverageBlock(
            "Borrower Profile",
            BORROWER_PROFILE_COLUMNS,
            "Profile and audit-sensitive fields retained for modeling policy review.",
        ),
        DatasetCoverageBlock(
            "Credit History",
            PAY_STATUS_COLUMNS,
            "Historical repayment-status variables before the next-month target.",
        ),
        DatasetCoverageBlock(
            "Loan / Exposure",
            ["LIMIT_BAL"],
            "Credit limit / exposure amount.",
        ),
        DatasetCoverageBlock(
            "Financial Health",
            [*BILL_AMOUNT_COLUMNS, *PAY_AMOUNT_COLUMNS, *ENGINEERED_UCI_COLUMNS],
            "Historical bill amounts, payments, utilization, and repayment-ratio features.",
        ),
        DatasetCoverageBlock(
            "Target",
            [TARGET_COL],
            "Binary next-month default indicator where 1 means default / bad outcome.",
        ),
    ]
