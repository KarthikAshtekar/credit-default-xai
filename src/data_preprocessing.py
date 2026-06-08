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

from .feature_engineering import apply_feature_engineering
from .utils import normalize_target

DROP_COLUMNS_DEFAULT = ["CustomerID", "LoanID"]
TARGET_COL = "Default_Flag"

FEATURE_SET_APPLICATION = "application"
FEATURE_SET_BEHAVIORAL = "behavioral"
FEATURE_SET_FULL_DIAGNOSTIC = "full_diagnostic"
FEATURE_SET_NAMES = [
    FEATURE_SET_APPLICATION,
    FEATURE_SET_BEHAVIORAL,
    FEATURE_SET_FULL_DIAGNOSTIC,
]

APPLICATION_TIME_FEATURES = [
    "Age",
    "Gender",
    "Nationality",
    "City",
    "EmploymentStatus",
    "AnnualIncome_AED",
    "OtherObligations_AED",
    "BureauScore",
    "LoanType",
    "LoanAmount_AED",
    "LoanTenureMonths",
    "InterestRate_pct",
    "LoanStartYear",
    "LoanStartMonth",
    "LoanStartQuarter",
    "Unemployment_pct",
    "Inflation_pct",
    "EMI_AED",
    "LoanToAnnualIncome",
    "ObligationsToIncome",
    "EMIToIncome",
]

BEHAVIORAL_MONITORING_ONLY_FEATURES = [
    "OnTimePayments_Last12M",
    "MissedPayments_Last12M",
    "MissedEMIs_Last6M",
    "PastDefaults",
    "AvgMonthlyDebit_AED",
    "StdMonthlyDebit_AED",
    "SalaryDropFlag",
    "SpendingSpikeFlag",
    "StressSignalCount",
    "HistoricalRiskScore",
    "MissedPaymentRate",
]

POST_OUTCOME_BEHAVIORAL_FEATURES = [
    "OnTimePayments_Last12M",
    "MissedPayments_Last12M",
    "MissedEMIs_Last6M",
    "AvgMonthlyDebit_AED",
    "StdMonthlyDebit_AED",
    "SalaryDropFlag",
    "SpendingSpikeFlag",
    "StressSignalCount",
    "HistoricalRiskScore",
    "MissedPaymentRate",
]

DEMOGRAPHIC_FEATURES = ["Age", "Gender", "Nationality", "City"]
FINANCIAL_BURDEN_FEATURES = [
    "LoanToAnnualIncome",
    "ObligationsToIncome",
    "EMIToIncome",
]
BUREAU_FINANCIAL_FEATURES = [
    "AnnualIncome_AED",
    "OtherObligations_AED",
    "BureauScore",
    "LoanAmount_AED",
    "LoanTenureMonths",
    "InterestRate_pct",
    "Unemployment_pct",
    "Inflation_pct",
    "EMI_AED",
    "LoanToAnnualIncome",
    "ObligationsToIncome",
    "EMIToIncome",
]

PAST_DEFAULTS_ASSUMPTION = (
    "PastDefaults is excluded from the application-time model because the dataset does not "
    "explicitly state whether it only refers to defaults before the current loan."
)

SplitStrategy = Literal["random", "temporal"]
FeatureSetName = Literal["application", "behavioral", "full_diagnostic"]


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
    if feature_set == FEATURE_SET_APPLICATION:
        candidates = APPLICATION_TIME_FEATURES
    elif feature_set == FEATURE_SET_BEHAVIORAL:
        candidates = APPLICATION_TIME_FEATURES + BEHAVIORAL_MONITORING_ONLY_FEATURES
    elif feature_set == FEATURE_SET_FULL_DIAGNOSTIC:
        candidates = [c for c in prepared_df.columns if c != TARGET_COL]
    else:
        raise ValueError(f"Unsupported feature set: {feature_set}")

    return [c for c in candidates if c in prepared_df.columns and c != TARGET_COL]


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
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    elif split_strategy == "temporal":
        if "LoanStartDate" not in df.columns:
            raise KeyError("Temporal split requested but LoanStartDate is missing from the raw dataset.")

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
