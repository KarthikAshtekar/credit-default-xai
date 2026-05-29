"""Shared dashboard utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_preprocessing import TARGET_COL, prepare_modeling_table
from src.train_logistic import run as train_logistic_run
from src.train_xgboost import run as train_xgboost_run
from src.utils import MODELS_DIR, load_dataset_auto, load_model


def load_data_for_dashboard() -> pd.DataFrame:
    df, _ = load_dataset_auto()
    return df


def ensure_model(model_choice: str):
    if model_choice == "Logistic Regression":
        path = MODELS_DIR / "logistic_model.pkl"
        if not path.exists():
            train_logistic_run(path)
    else:
        path = MODELS_DIR / "xgboost_model.pkl"
        if not path.exists():
            train_xgboost_run(path)
    return load_model(path), path


def get_feature_table() -> pd.DataFrame:
    raw = load_data_for_dashboard()
    prepared = prepare_modeling_table(raw, target_col=TARGET_COL)
    return prepared.drop(columns=[TARGET_COL])
