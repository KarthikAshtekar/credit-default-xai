from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import deep_learning_benchmark as dnn
from src.recall_optimization import create_threshold_grid, evaluate_threshold_grid


def test_module_import_and_tensorflow_status_are_safe() -> None:
    tensorflow, reason = dnn.tensorflow_status()
    assert (tensorflow is not None and reason is None) or (
        tensorflow is None and "TensorFlow is unavailable" in reason
    )


def test_skip_behavior_writes_clear_artifacts(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(dnn, "MODEL_VALIDATION_DIR", tmp_path)
    monkeypatch.setattr(dnn, "tensorflow_status", lambda: (None, "TensorFlow is unavailable"))
    result = dnn.run()
    assert result["status"] == "skipped"
    assert "TensorFlow is unavailable" in result["reason"]
    assert (tmp_path / "deep_learning_metrics.json").exists()


def test_architecture_builder_when_tensorflow_is_available() -> None:
    tensorflow, _ = dnn.tensorflow_status()
    if tensorflow is None:
        pytest.skip("TensorFlow is not installed.")
    model = dnn.build_mlp(12, tensorflow)
    dense_units = [layer.units for layer in model.layers if hasattr(layer, "units")]
    assert dense_units == [64, 32, 16, 1]


def test_threshold_grid_metrics_with_synthetic_probabilities() -> None:
    y_true = pd.Series([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.05, 0.25, 0.45, 0.35, 0.65, 0.95])
    frame = evaluate_threshold_grid(
        y_true,
        probabilities,
        create_threshold_grid(),
        candidate_name="dnn_test",
        split_name="validation",
    )
    assert len(frame) == 13
    assert {
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "f2",
        "approval_support_rate",
    }.issubset(frame.columns)


def test_selected_policy_json_schema() -> None:
    selected = pd.Series({"threshold": 0.30, "precision": 0.51, "recall": 0.70})
    payload = dnn.selected_policy_payload(
        "dnn_baseline",
        selected,
        "maximize_recall_precision_050",
        False,
        {"recall": 0.68, "precision": 0.49},
        0.78,
        0.55,
    )
    assert dnn.validate_policy_schema(payload)
    assert payload["selection_split"] == "validation"
    assert payload["evaluation_split"] == "untouched_test"


def test_fairness_function_compatibility() -> None:
    frame = dnn._fairness_rows(
        pd.Series([0, 0, 1, 1]),
        pd.Series([1, 2, 1, 2]),
        np.array([0.1, 0.6, 0.7, 0.8]),
        np.array([0.1, 0.4, 0.6, 0.9]),
        0.4,
    )
    assert len(frame) == 2
    assert "equalized_odds_difference" in frame.columns
