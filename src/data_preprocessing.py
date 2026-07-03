"""Data preprocessing, feature-set selection, and split helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .dataset_adapters import (
    APPLICATION_PUBLIC_FEATURES,
    FULL_PUBLIC_DIAGNOSTIC_FEATURES,
    TARGET_COL,
    UCI_ID_COLUMNS,
)
from .feature_engineering import apply_feature_engineering
from .utils import normalize_target

DROP_COLUMNS_DEFAULT = UCI_ID_COLUMNS + ["CustomerID", "LoanID"]

FEATURE_SET_APPLICATION = "application_public"
FEATURE_SET_BEHAVIORAL = "legacy_behavioral"
FEATURE_SET_FULL_DIAGNOSTIC = "full_public_diagnostic"
FEATURE_SET_NAMES = [
    FEATURE_SET_APPLICATION,
    FEATURE_SET_FULL_DIAGNOSTIC,
]

APPLICATION_TIME_FEATURES = APPLICATION_PUBLIC_FEATURES
BEHAVIORAL_MONITORING_ONLY_FEATURES: list[str] = []
POST_OUTCOME_BEHAVIORAL_FEATURES: list[str] = []

DEMOGRAPHIC_FEATURES = ["SEX", "AGE", "MARRIAGE", "EDUCATION"]
FINANCIAL_BURDEN_FEATURES = [
    "AvgBillToLimitRatio",
    "AvgPaymentToBillRatio",
    "PaymentToLimitRatio",
]
BUREAU_FINANCIAL_FEATURES = ["LIMIT_BAL", *FINANCIAL_BURDEN_FEATURES]

FEATURE_POLICY_NOTE = (
    "application_public excludes the direct protected attribute SEX from active training "
    "features while retaining SEX in the audit dataframe for fairness analysis."
)
UCI_TIMING_NOTE = (
    "PAY_0 to PAY_6 are historical repayment-status variables available before the "
    "next-month default target and are not treated as leakage for this modeling question."
)

SplitStrategy = Literal["random", "temporal"]
FeatureSetName = Literal["application_public", "legacy_behavioral", "full_public_diagnostic"]


@dataclass
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_indices: pd.Index
    test_indices: pd.Index
    feature_columns: List[str]


def enrich_loan_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Expand absolute loan-start date parts only.

    Relative age from the dataset max date is intentionally excluded because it introduces
    hindsight about observation timing and can act as leakage.
    """

    out = df.copy()
    if "LoanStartDate" not in out.columns:
        return out

    out["LoanStartDate"] = pd.to_datetime(out["LoanStartDate"], errors="coerce")
    out["LoanStartYear"] = out["LoanStartDate"].dt.year
    out["LoanStartMonth"] = out["LoanStartDate"].dt.month
    out["LoanStartQuarter"] = out["LoanStartDate"].dt.quarter
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


def get_feature_columns(
    prepared_df: pd.DataFrame,
    feature_set: FeatureSetName = FEATURE_SET_FULL_DIAGNOSTIC,
) -> List[str]:
    normalized_feature_set = {
        "application": FEATURE_SET_APPLICATION,
        "behavioral": FEATURE_SET_APPLICATION,
        "full_diagnostic": FEATURE_SET_FULL_DIAGNOSTIC,
    }.get(feature_set, feature_set)

    if normalized_feature_set == FEATURE_SET_APPLICATION:
        candidates = APPLICATION_TIME_FEATURES
    elif normalized_feature_set == FEATURE_SET_BEHAVIORAL:
        candidates = APPLICATION_TIME_FEATURES
    elif normalized_feature_set == FEATURE_SET_FULL_DIAGNOSTIC:
        candidates = FULL_PUBLIC_DIAGNOSTIC_FEATURES
    else:
        raise ValueError(f"Unsupported feature set: {feature_set}")

    drop_cols = set(DROP_COLUMNS_DEFAULT + [TARGET_COL])
    return [c for c in candidates if c in prepared_df.columns and c not in drop_cols]


def select_feature_table(
    prepared_df: pd.DataFrame,
    feature_set: FeatureSetName = FEATURE_SET_FULL_DIAGNOSTIC,
    feature_columns: List[str] | None = None,
) -> pd.DataFrame:
    selected_columns = feature_columns or get_feature_columns(prepared_df, feature_set=feature_set)
    return prepared_df[selected_columns + [TARGET_COL]].copy()


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
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = 0.2,
    random_state: int = 42,
    feature_set: FeatureSetName = FEATURE_SET_FULL_DIAGNOSTIC,
    split_strategy: SplitStrategy = "random",
    feature_columns: List[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split = get_dataset_split(
        df=df,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
        feature_set=feature_set,
        split_strategy=split_strategy,
        feature_columns=feature_columns,
    )
    return split.X_train, split.X_test, split.y_train, split.y_test


def get_dataset_split(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = 0.2,
    random_state: int = 42,
    feature_set: FeatureSetName = FEATURE_SET_FULL_DIAGNOSTIC,
    split_strategy: SplitStrategy = "random",
    feature_columns: List[str] | None = None,
) -> DatasetSplit:
    prepared = prepare_modeling_table(df, target_col=target_col)
    selected = select_feature_table(
        prepared,
        feature_set=feature_set,
        feature_columns=feature_columns,
    )
    X, y = split_features_target(selected, target_col=target_col)

    if split_strategy == "random":
        signature_frame = X.copy()
        signature_frame[target_col] = y
        row_signatures = signature_frame.astype(str).agg("||".join, axis=1)
        groups = pd.Series(pd.factorize(row_signatures)[0], index=X.index, name="group")
        group_targets = (
            pd.DataFrame({"group": groups, target_col: y})
            .drop_duplicates("group")
            .set_index("group")[target_col]
        )
        group_ids = group_targets.index.to_series()
        stratify_groups = group_targets if group_targets.nunique() > 1 else None
        train_groups, test_groups = train_test_split(
            group_ids,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_groups,
        )
        train_mask = groups.isin(train_groups)
        test_mask = groups.isin(test_groups)
        X_train = X.loc[train_mask]
        X_test = X.loc[test_mask]
        y_train = y.loc[train_mask]
        y_test = y.loc[test_mask]
    elif split_strategy == "temporal":
        if "LoanStartDate" not in df.columns:
            raise KeyError(
                "Temporal split requested but the UCI primary dataset has no true application "
                "timestamp. Use the documented stratified random split for final metrics."
            )

        dates = pd.to_datetime(df["LoanStartDate"], errors="coerce")
        valid_mask = dates.notna()
        X = X.loc[valid_mask].copy()
        y = y.loc[valid_mask].copy()
        dates = dates.loc[valid_mask]

        ordering = dates.sort_values(kind="mergesort").index
        cutoff = max(1, int(len(ordering) * (1 - test_size)))
        train_idx = ordering[:cutoff]
        test_idx = ordering[cutoff:]
        if len(test_idx) == 0:
            raise ValueError("Temporal split produced an empty test set.")

        X_train = X.loc[train_idx]
        X_test = X.loc[test_idx]
        y_train = y.loc[train_idx]
        y_test = y.loc[test_idx]
    else:
        raise ValueError(f"Unsupported split strategy: {split_strategy}")

    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_indices=X_train.index,
        test_indices=X_test.index,
        feature_columns=X.columns.tolist(),
    )
