from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dataset_adapters import UCI_BASE_COLUMNS
from src.recall_optimization import (
    RULE_A,
    RULE_B,
    build_parser,
    build_recall_optimization_splits_from_frame,
    create_threshold_grid,
    select_preferred_threshold,
    select_threshold_by_rule,
    smote_skip_reason_if_unavailable,
    threshold_metrics,
)


def _synthetic_uci_frame(row_count: int = 120) -> pd.DataFrame:
    rows = []
    for index in range(row_count):
        row = {
            "ID": index + 1,
            "LIMIT_BAL": 20000 + (index * 1000),
            "SEX": 1 + (index % 2),
            "EDUCATION": 1 + (index % 4),
            "MARRIAGE": 1 + (index % 3),
            "AGE": 22 + (index % 45),
            "Default_Flag": index % 2,
        }
        for offset, column in enumerate(["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]):
            row[column] = (index + offset) % 5 - 2
        for offset, column in enumerate(
            ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
        ):
            row[column] = 1000 + (index * 25) + (offset * 10)
        for offset, column in enumerate(
            ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
        ):
            row[column] = 100 + (index * 5) + offset
        rows.append(row)
    return pd.DataFrame(rows)[["ID", *UCI_BASE_COLUMNS, "Default_Flag"]]


def test_threshold_grid_default_values() -> None:
    grid = create_threshold_grid()

    assert grid[0] == 0.10
    assert grid[-1] == 0.70
    assert len(grid) == 13


def test_threshold_metrics_include_confusion_rates_and_f2() -> None:
    metrics = threshold_metrics(
        y_true=np.array([0, 0, 1, 1]),
        y_proba=np.array([0.1, 0.6, 0.4, 0.9]),
        threshold=0.50,
    )

    assert metrics["true_negatives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["true_positives"] == 1
    assert metrics["recall"] == 0.5
    assert metrics["specificity"] == 0.5
    assert metrics["f2"] == pytest.approx(0.5)


def test_rule_a_maximizes_recall_when_precision_floor_is_met() -> None:
    threshold_df = pd.DataFrame(
        [
            {"threshold": 0.10, "precision": 0.40, "recall": 0.95, "f2": 0.80, "expected_cost": 10},
            {"threshold": 0.20, "precision": 0.51, "recall": 0.70, "f2": 0.65, "expected_cost": 20},
            {"threshold": 0.30, "precision": 0.60, "recall": 0.60, "f2": 0.60, "expected_cost": 25},
        ]
    )

    selected = select_threshold_by_rule(threshold_df, RULE_A)

    assert selected is not None
    assert selected["threshold"] == 0.20


def test_preferred_threshold_falls_back_to_f2_when_rule_a_has_no_candidate() -> None:
    threshold_df = pd.DataFrame(
        [
            {"threshold": 0.10, "precision": 0.40, "recall": 0.95, "f2": 0.82, "expected_cost": 10},
            {"threshold": 0.20, "precision": 0.44, "recall": 0.75, "f2": 0.76, "expected_cost": 12},
        ]
    )

    selected, fallback_used, rule = select_preferred_threshold(threshold_df)

    assert selected["threshold"] == 0.10
    assert fallback_used is True
    assert rule == RULE_B


def test_recall_optimization_split_keeps_validation_and_test_separate() -> None:
    splits = build_recall_optimization_splits_from_frame(_synthetic_uci_frame())

    assert set(splits.validation_indices).isdisjoint(set(splits.test_indices))
    assert set(splits.train_indices).isdisjoint(set(splits.validation_indices))
    assert set(splits.train_indices).isdisjoint(set(splits.test_indices))
    assert "SEX" not in splits.X_train_full.columns
    assert "SEX" in _synthetic_uci_frame().columns


def test_smote_skip_helper_reports_missing_optional_dependency(monkeypatch) -> None:
    monkeypatch.setattr("src.recall_optimization._smote_components", lambda: (None, None))

    assert "imbalanced-learn is not installed" in smote_skip_reason_if_unavailable()


def test_recall_optimization_cli_help() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--help"])

    assert exc_info.value.code == 0
