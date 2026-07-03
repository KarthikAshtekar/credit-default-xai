from __future__ import annotations

import urllib.request

import pandas as pd
import pytest

from src.data_api_loader import load_dataset
from src.data_preprocessing import FEATURE_SET_APPLICATION, TARGET_COL, get_dataset_split
from src.dataset_adapters import (
    APPLICATION_PUBLIC_FEATURES,
    PROTECTED_ATTRIBUTE,
    adapt_uci_default_credit_card,
)


def internet_available(url: str = "https://archive.ics.uci.edu", timeout: int = 3) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except Exception:
        return False


def _synthetic_uci_frame(row_count: int = 40) -> pd.DataFrame:
    rows = []
    for index in range(row_count):
        row = {
            "X1": 20000 + (index * 1000),
            "X2": 1 + (index % 2),
            "X3": 1 + (index % 4),
            "X4": 1 + (index % 3),
            "X5": 22 + (index % 45),
            "Y": index % 2,
        }
        for offset, column in enumerate(["X6", "X7", "X8", "X9", "X10", "X11"]):
            row[column] = (index + offset) % 5 - 2
        for offset, column in enumerate(["X12", "X13", "X14", "X15", "X16", "X17"]):
            row[column] = 1000 + (index * 25) + (offset * 10)
        for offset, column in enumerate(["X18", "X19", "X20", "X21", "X22", "X23"]):
            row[column] = 100 + (index * 5) + offset
        rows.append(row)
    return pd.DataFrame(rows)


def test_uci_adapter_normalizes_target_and_engineered_features() -> None:
    adapted = adapt_uci_default_credit_card(_synthetic_uci_frame())

    assert TARGET_COL in adapted.columns
    assert set(adapted[TARGET_COL].unique()) == {0, 1}
    assert "BillToLimitRatio_1" in adapted.columns
    assert "AvgPaymentToBillRatio" in adapted.columns
    assert PROTECTED_ATTRIBUTE in adapted.columns


def test_application_public_excludes_direct_protected_attribute() -> None:
    adapted = adapt_uci_default_credit_card(_synthetic_uci_frame())
    split = get_dataset_split(
        adapted,
        target_col=TARGET_COL,
        feature_set=FEATURE_SET_APPLICATION,
    )

    assert PROTECTED_ATTRIBUTE not in split.feature_columns
    assert TARGET_COL not in split.feature_columns
    assert set(split.feature_columns).issubset(set(APPLICATION_PUBLIC_FEATURES))


def test_preprocessing_runs_from_uci_adapter_output() -> None:
    adapted = adapt_uci_default_credit_card(_synthetic_uci_frame())
    split = get_dataset_split(adapted, target_col=TARGET_COL, feature_set=FEATURE_SET_APPLICATION)

    assert not split.X_train.empty
    assert not split.X_test.empty
    assert split.feature_columns


def test_url_loader_handles_invalid_urls_gracefully() -> None:
    with pytest.raises(ValueError, match="Unable to load dataset from URL"):
        load_dataset(source="url", url="not-a-valid-url")


def test_uci_loader_works_when_internet_is_available() -> None:
    if not internet_available():
        pytest.skip("Internet is unavailable; skipping live UCI loader test.")

    df, metadata = load_dataset(source="uci", dataset_name="default_credit_card")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert metadata["source"] == "uci"
    assert metadata["target_column"] == TARGET_COL
    assert metadata["primary_project_dataset"] is True
