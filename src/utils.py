"""Shared utilities for data loading, paths, and serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"


def ensure_directories() -> None:
    """Create required output directories."""
    for path in [
        DATA_PROCESSED_DIR,
        MODELS_DIR,
        REPORTS_DIR / "figures",
        REPORTS_DIR / "fairness_reports",
        REPORTS_DIR / "explainability_reports",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def detect_dataset_file(data_dir: Optional[Path] = None) -> Path:
    """Auto-detect dataset file from raw data directory.

    Preference order: CSV first, then Excel.
    """
    data_dir = data_dir or DATA_RAW_DIR
    csv_files = sorted(data_dir.glob("*.csv"))
    excel_files = sorted(data_dir.glob("*.xlsx")) + sorted(data_dir.glob("*.xls"))

    candidates = csv_files + excel_files
    if not candidates:
        raise FileNotFoundError(
            f"No CSV/XLSX/XLS files found in raw data directory: {data_dir}"
        )
    return candidates[0]


def load_dataset_auto(data_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, Path]:
    """Load dataset from first detected raw file."""
    dataset_path = detect_dataset_file(data_dir)
    if dataset_path.suffix.lower() == ".csv":
        df = pd.read_csv(dataset_path)
    else:
        df = pd.read_excel(dataset_path)
    return df, dataset_path


def normalize_target(df: pd.DataFrame, target_col: str = "Default_Flag") -> pd.DataFrame:
    """Ensure target column is binary integer where possible."""
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset.")

    out = df.copy()
    out[target_col] = out[target_col].replace({"Yes": 1, "No": 0, "Y": 1, "N": 0})
    out[target_col] = pd.to_numeric(out[target_col], errors="coerce")
    out[target_col] = out[target_col].fillna(0).astype(int)
    return out


def infer_protected_attribute(df: pd.DataFrame) -> str:
    """Pick a default protected attribute for fairness analysis."""
    for col in ["Gender", "Nationality", "City", "EmploymentStatus"]:
        if col in df.columns:
            return col
    raise KeyError(
        "No common protected attribute found. Add one of: "
        "Gender, Nationality, City, EmploymentStatus"
    )


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path):
    return joblib.load(path)


def save_json(payload: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    return np.where((b == 0) | (b.isna()), 0, a / b)
