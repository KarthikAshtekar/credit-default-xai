"""Data preprocessing and train/test split utilities."""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .feature_engineering import apply_feature_engineering
from .utils import normalize_target

DROP_COLUMNS_DEFAULT = ["CustomerID", "LoanID"]
TARGET_COL = "Default_Flag"


def enrich_loan_date_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "LoanStartDate" not in out.columns:
        return out

    out["LoanStartDate"] = pd.to_datetime(out["LoanStartDate"], errors="coerce")
    out["LoanStartYear"] = out["LoanStartDate"].dt.year
    out["LoanStartMonth"] = out["LoanStartDate"].dt.month
    out["LoanStartQuarter"] = out["LoanStartDate"].dt.quarter

    reference_date = out["LoanStartDate"].max()
    if pd.notna(reference_date):
        out["LoanAgeDays"] = (reference_date - out["LoanStartDate"]).dt.days

    out = out.drop(columns=["LoanStartDate"])
    return out


def prepare_modeling_table(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    drop_cols: List[str] | None = None,
) -> pd.DataFrame:
    drop_cols = drop_cols or DROP_COLUMNS_DEFAULT

    out = normalize_target(df, target_col=target_col)
    out = enrich_loan_date_features(out)
    out = apply_feature_engineering(out)

    existing_drop = [c for c in drop_cols if c in out.columns]
    out = out.drop(columns=existing_drop)
    return out


def split_features_target(
    df: pd.DataFrame, target_col: str = TARGET_COL
) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )


def train_test_data(
    df: pd.DataFrame, target_col: str = TARGET_COL, test_size: float = 0.2, random_state: int = 42
):
    prepared = prepare_modeling_table(df, target_col=target_col)
    X, y = split_features_target(prepared, target_col=target_col)

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
