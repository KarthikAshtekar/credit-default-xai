from __future__ import annotations

import urllib.request

import pandas as pd
import pytest

from src.data_api_loader import DEFAULT_LOCAL_DATASET, load_dataset
from src.data_preprocessing import FEATURE_SET_APPLICATION, TARGET_COL, get_dataset_split


def internet_available(url: str = "https://archive.ics.uci.edu", timeout: int = 3) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except Exception:
        return False


def test_local_loader_works_when_local_dataset_exists() -> None:
    if not DEFAULT_LOCAL_DATASET.exists():
        pytest.skip("Local case-study dataset is not available in this environment.")

    df, metadata = load_dataset(source="local")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert metadata["source"] == "local"
    assert metadata["target_column"] == TARGET_COL


def test_url_loader_handles_invalid_urls_gracefully() -> None:
    with pytest.raises(ValueError, match="Unable to load dataset from URL"):
        load_dataset(source="url", url="not-a-valid-url")


def test_uci_loader_function_exists() -> None:
    assert callable(load_dataset)


def test_preprocessing_runs_from_loader_output() -> None:
    if not DEFAULT_LOCAL_DATASET.exists():
        pytest.skip("Local case-study dataset is not available in this environment.")

    df, _ = load_dataset(source="local")
    split = get_dataset_split(df, target_col=TARGET_COL, feature_set=FEATURE_SET_APPLICATION)
    assert not split.X_train.empty
    assert not split.X_test.empty
    assert split.feature_columns


def test_uci_loader_works_when_internet_is_available() -> None:
    if not internet_available():
        pytest.skip("Internet is unavailable; skipping UCI loader test.")

    df, metadata = load_dataset(source="uci", dataset_name="default_credit_card")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert metadata["source"] == "uci"
    assert metadata["dataset_name"] == "Default of Credit Card Clients"


def test_url_loader_works_when_internet_is_available() -> None:
    if not internet_available("https://raw.githubusercontent.com", timeout=3):
        pytest.skip("Internet is unavailable; skipping public URL loader test.")

    df, metadata = load_dataset(
        source="url",
        url="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
    )
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert metadata["source"] == "url"
