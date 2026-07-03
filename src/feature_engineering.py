"""Feature engineering helpers for credit default modeling."""

from __future__ import annotations

import pandas as pd

from .dataset_adapters import add_uci_credit_features


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    return add_uci_credit_features(df) if "LIMIT_BAL" in df.columns else df.copy()


def add_behavioral_flags(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    out = add_ratio_features(df)
    out = add_behavioral_flags(out)
    return out
