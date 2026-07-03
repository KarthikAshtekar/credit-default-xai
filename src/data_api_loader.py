"""Lightweight dataset loader for local, URL, and UCI sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from .dataset_adapters import (
    PROTECTED_ATTRIBUTE,
    TARGET_COL,
    UCI_DEFAULT_CREDIT_CARD_DISPLAY_NAME,
    UCI_DEFAULT_CREDIT_CARD_NAME,
    adapt_uci_default_credit_card,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_LOCAL_DATASET = ROOT_DIR / "data" / "raw" / "legacy_local_credit_dataset.xlsx"
PROTECTED_ATTRIBUTE_CANDIDATES = [PROTECTED_ATTRIBUTE, "AGE", "MARRIAGE", "EDUCATION"]
DEFAULT_UCI_DATASET_NAME = UCI_DEFAULT_CREDIT_CARD_NAME

UCI_DATASETS = {
    UCI_DEFAULT_CREDIT_CARD_NAME: {
        "uci_id": 350,
        "display_name": UCI_DEFAULT_CREDIT_CARD_DISPLAY_NAME,
        "notes": "Primary public dataset for the current project pipeline.",
    },
    "south_german_credit": {
        "uci_id": 573,
        "display_name": "South German Credit",
        "notes": "Future-scope public credit risk dataset; not the primary pipeline dataset.",
    },
}
UCI_ALIASES = {
    "default of credit card clients": "default_credit_card",
    "default_credit_card": "default_credit_card",
    "default credit card": "default_credit_card",
    "south german credit": "south_german_credit",
    "south_german_credit": "south_german_credit",
}


def _protected_attributes_available(df: pd.DataFrame) -> list[str]:
    return [col for col in PROTECTED_ATTRIBUTE_CANDIDATES if col in df.columns]


def _project_relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT_DIR.resolve()).as_posix()
    except ValueError:
        return str(path)


def _load_tabular_file(path_or_url: str | Path) -> pd.DataFrame:
    suffix = Path(urlparse(str(path_or_url)).path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path_or_url)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path_or_url)

    csv_error = None
    try:
        return pd.read_csv(path_or_url)
    except Exception as exc:  # pragma: no cover - exercised via fallback path
        csv_error = exc

    try:
        return pd.read_excel(path_or_url)
    except Exception as excel_exc:
        raise ValueError(
            f"Unsupported or unreadable dataset source: {path_or_url}. "
            f"CSV read error: {csv_error}. Excel read error: {excel_exc}."
        ) from excel_exc


def _load_local_dataset(path: str | Path | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset_path = Path(path) if path else DEFAULT_LOCAL_DATASET
    if not dataset_path.exists():
        raise FileNotFoundError(f"Local dataset not found: {dataset_path}")

    df = _load_tabular_file(dataset_path)
    metadata = {
        "dataset_name": "Legacy local credit dataset",
        "source": "local",
        "target_column": TARGET_COL if TARGET_COL in df.columns else None,
        "protected_attributes_available": _protected_attributes_available(df),
        "notes": "Legacy explicit local-file route. Not used as the current final-result source.",
        "path": _project_relative_path(dataset_path),
    }
    return df, metadata


def _load_url_dataset(url: str | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not url:
        raise ValueError("A direct dataset URL is required when source='url'.")

    try:
        df = _load_tabular_file(url)
    except Exception as exc:
        raise ValueError(f"Unable to load dataset from URL: {url}") from exc

    metadata = {
        "dataset_name": Path(urlparse(url).path).name or "remote_dataset",
        "source": "url",
        "target_column": TARGET_COL if TARGET_COL in df.columns else None,
        "protected_attributes_available": _protected_attributes_available(df),
        "notes": "Loaded from a direct public CSV/Excel URL.",
        "url": url,
    }
    return df, metadata


def _normalize_uci_dataset_name(dataset_name: str | None) -> str:
    if not dataset_name:
        return DEFAULT_UCI_DATASET_NAME

    normalized = dataset_name.strip().lower().replace("-", "_")
    normalized = UCI_ALIASES.get(normalized, normalized)
    if normalized not in UCI_DATASETS:
        supported = ", ".join(sorted(UCI_DATASETS))
        raise ValueError(
            f"Unsupported UCI dataset '{dataset_name}'. Supported values: {supported}."
        )
    return normalized


def _load_uci_dataset(dataset_name: str | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    normalized_name = _normalize_uci_dataset_name(dataset_name)
    dataset_config = UCI_DATASETS[normalized_name]

    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise ImportError(
            "ucimlrepo is required for source='uci'. Install it with `pip install ucimlrepo`."
        ) from exc

    dataset = fetch_ucirepo(id=dataset_config["uci_id"])
    features = dataset.data.features
    targets = dataset.data.targets

    if targets is None or getattr(targets, "empty", False):
        df = features.copy()
        target_column = None
    else:
        df = pd.concat([features, targets], axis=1)
        target_column = targets.columns[0]

    if normalized_name == UCI_DEFAULT_CREDIT_CARD_NAME:
        df = adapt_uci_default_credit_card(df)
        target_column = TARGET_COL

    metadata = {
        "dataset_name": dataset_config["display_name"],
        "source": "uci",
        "target_column": target_column,
        "protected_attributes_available": _protected_attributes_available(df),
        "notes": dataset_config["notes"],
        "uci_id": dataset_config["uci_id"],
        "primary_project_dataset": normalized_name == UCI_DEFAULT_CREDIT_CARD_NAME,
    }
    return df, metadata


def load_dataset(
    source: str = "uci",
    dataset_name: str | None = None,
    path: str | Path | None = None,
    url: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a dataset and return both the frame and source metadata."""

    normalized_source = source.strip().lower()
    if normalized_source == "local":
        return _load_local_dataset(path)
    if normalized_source == "url":
        return _load_url_dataset(url)
    if normalized_source == "uci":
        return _load_uci_dataset(dataset_name)

    raise ValueError("Unsupported source. Use one of: local, url, uci.")


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load a dataset from local files, URL, or UCI.")
    parser.add_argument("--source", default="uci", choices=["local", "url", "uci"])
    parser.add_argument("--dataset_name", default=DEFAULT_UCI_DATASET_NAME)
    parser.add_argument("--path", default=None)
    parser.add_argument("--url", default=None)
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    df, metadata = load_dataset(
        source=args.source,
        dataset_name=args.dataset_name,
        path=args.path,
        url=args.url,
    )
    print(json.dumps(metadata, indent=2))
    print(f"Loaded dataset with shape {df.shape}.")


if __name__ == "__main__":
    main()
