from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.external_validation import (
    DATASET_DEFAULT_CREDIT_CARD,
    TARGET_COLUMN,
    build_parser,
    compute_external_fairness_metrics,
    prepare_default_credit_card_dataset,
    run_default_credit_card_validation,
)


def _synthetic_default_credit_card_data(row_count: int = 40) -> pd.DataFrame:
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


def test_prepare_default_credit_card_dataset_maps_uci_columns() -> None:
    df = _synthetic_default_credit_card_data()

    dataset = prepare_default_credit_card_dataset(df)

    assert dataset.name == DATASET_DEFAULT_CREDIT_CARD
    assert dataset.target_column == TARGET_COLUMN
    assert "LIMIT_BAL" in dataset.X.columns
    assert "PAY_AMT6" in dataset.X.columns
    assert set(dataset.y.unique()) == {0, 1}
    assert set(dataset.sensitive_features) == {"SEX", "AGE_GROUP"}


def test_external_fairness_metrics_handle_binary_sensitive_attribute() -> None:
    metrics = compute_external_fairness_metrics(
        y_default_true=pd.Series([0, 1, 0, 1]),
        default_proba=np.array([0.1, 0.8, 0.4, 0.7]),
        sensitive=pd.Series(["group_a", "group_a", "group_b", "group_b"]),
    )

    assert set(metrics) == {
        "demographic_parity_difference",
        "equalized_odds_difference",
        "equal_opportunity_difference",
        "disparate_impact_ratio",
    }


def test_external_validation_writes_outputs_without_model_pickles(tmp_path) -> None:
    df = _synthetic_default_credit_card_data()

    result = run_default_credit_card_validation(
        df,
        metadata={"source": "synthetic"},
        output_dir=tmp_path,
        include_xgboost=False,
    )

    assert result["dataset"] == DATASET_DEFAULT_CREDIT_CARD
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "fairness_metrics.json").exists()
    assert (tmp_path / "fairness_metrics.csv").exists()
    assert (tmp_path / "model_comparison.csv").exists()
    assert (tmp_path / "summary.md").exists()
    assert not list(tmp_path.glob("*.pkl"))


def test_external_validation_cli_help() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--help"])

    assert exc_info.value.code == 0
