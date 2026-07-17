from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src import fairness_deep_dive as fdd
from src.protected_attributes import SEX_GROUP_LABELS

FORBIDDEN_FAIRNESS_OVERCLAIMS = [
    "the model discriminates",
    "bias is proven",
    "legally discriminatory",
    "causal bias confirmed",
    "fairness guaranteed",
    "bias-free",
]


def _minimal_context() -> fdd.FairnessContext:
    raw = pd.DataFrame(
        {
            "SEX": [1, 1, 2, 2],
            "Default_Flag": [0, 1, 0, 1],
            "score": [0.10, 0.60, 0.70, 0.20],
        }
    )
    split = SimpleNamespace(
        X_train=raw[["score"]].iloc[:2].copy(),
        X_test=raw[["score"]].copy(),
        y_train=raw["Default_Flag"].iloc[:2].copy(),
        y_test=raw["Default_Flag"].copy(),
        train_indices=pd.Index([0, 1]),
        test_indices=pd.Index([0, 1, 2, 3]),
    )
    return fdd.FairnessContext(
        raw=raw,
        split=split,
        sensitive_test=raw["SEX"].copy(),
        y_test=raw["Default_Flag"].copy(),
        xgboost_proba=np.array([0.10, 0.60, 0.70, 0.20]),
        policies=[],
    )


def test_group_metric_computation() -> None:
    metrics = fdd.classification_metrics([0, 1, 0, 1], [0.10, 0.60, 0.70, 0.20], 0.50)
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["approval_support_rate"] == 0.5

    fairness = fdd.fairness_from_predictions(
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [1, 1, 2, 2],
    )
    assert fairness["demographic_parity_difference"] == 0.0
    assert fairness["disparate_impact_ratio"] == 1.0
    assert fairness["false_positive_rate_difference"] == 1.0
    assert fairness["false_negative_rate_difference"] == 1.0


def test_group_confusion_matrix_metrics() -> None:
    frame = fdd.group_confusion_metrics(
        [0, 1, 0, 1],
        [0.10, 0.60, 0.70, 0.20],
        [1, 1, 2, 2],
        0.50,
        "test_policy",
    )

    group_one = frame.loc[frame["group"] == "Male (SEX=1)"].iloc[0]
    group_two = frame.loc[frame["group"] == "Female (SEX=2)"].iloc[0]
    assert group_one["sex_code"] == 1
    assert group_one["sex_group"] == "Male"
    assert group_two["sex_code"] == 2
    assert group_two["sex_group"] == "Female"
    assert group_one["true_positives"] == 1
    assert group_one["true_negatives"] == 1
    assert group_one["accuracy"] == 1.0
    assert group_two["false_positives"] == 1
    assert group_two["false_negatives"] == 1
    assert group_two["false_positive_rate"] == 1.0


def test_calibration_bin_generation() -> None:
    frame = fdd.calibration_bins(
        [0, 1, 1],
        [0.05, 0.15, 0.95],
        [1, 1, 2],
        bins=[0.0, 0.1, 0.2, 1.0],
    )

    first_bin = frame[(frame["group"] == "Male (SEX=1)") & (frame["bin"] == "0.00-0.10")].iloc[0]
    assert first_bin["sex_code"] == 1
    assert first_bin["sex_group"] == "Male"
    assert first_bin["n"] == 1
    assert first_bin["observed_default_rate"] == 0.0
    assert first_bin["calibration_gap"] == pytest.approx(-0.05)


def test_proxy_predictability_fit_and_skip_behavior() -> None:
    raw = pd.DataFrame(
        {
            "SEX": [1, 2] * 10,
            "feature_a": [0.1, 0.9] * 10,
            "feature_b": np.linspace(0, 1, 20),
            "Default_Flag": [0, 1] * 10,
        }
    )
    split = SimpleNamespace(
        X_train=raw[["feature_a", "feature_b"]].iloc[:16].copy(),
        X_test=raw[["feature_a", "feature_b"]].iloc[16:].copy(),
        train_indices=pd.Index(range(16)),
        test_indices=pd.Index(range(16, 20)),
    )
    context = fdd.FairnessContext(
        raw=raw,
        split=split,
        sensitive_test=raw.loc[split.test_indices, "SEX"],
        y_test=raw.loc[split.test_indices, "Default_Flag"],
        xgboost_proba=np.array([0.1, 0.8, 0.2, 0.7]),
        policies=[],
    )

    results, importance = fdd.build_proxy_predictability(context)
    assert set(results["status"]) == {"completed"}
    assert {"proxy_logistic_regression", "proxy_random_forest"} == set(results["model"])
    assert not importance.empty

    raw_one_class = raw.copy()
    raw_one_class["SEX"] = 1
    skipped_context = fdd.FairnessContext(
        raw=raw_one_class,
        split=split,
        sensitive_test=raw_one_class.loc[split.test_indices, "SEX"],
        y_test=raw_one_class.loc[split.test_indices, "Default_Flag"],
        xgboost_proba=np.array([0.1, 0.8, 0.2, 0.7]),
        policies=[],
    )
    skipped, skipped_importance = fdd.build_proxy_predictability(skipped_context)
    assert skipped.iloc[0]["status"] == "skipped"
    assert skipped_importance.empty


def test_threshold_fairness_frontier_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    context = _minimal_context()
    monkeypatch.setattr(fdd, "THRESHOLD_GRID", [0.25, 0.50])

    frontier = fdd.build_threshold_fairness_frontier(context)

    assert frontier["threshold"].tolist() == [0.25, 0.50]
    assert {
        "accuracy",
        "precision",
        "recall",
        "demographic_parity_difference",
        "false_positive_rate_difference",
    }.issubset(frontier.columns)


def test_individual_sex_sensitivity_direct_use_logic(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            proba = X["score"].astype(float).to_numpy()
            return np.column_stack([1 - proba, proba])

    context = _minimal_context()
    monkeypatch.setattr(fdd, "load_model", lambda path: FakeModel())
    monkeypatch.setattr(fdd, "prepare_modeling_table", lambda raw, target_col: raw.copy())
    monkeypatch.setattr(fdd, "get_feature_columns", lambda prepared, feature_set: ["score"])

    sensitivity = fdd.build_individual_sex_sensitivity(context)

    assert sensitivity["absolute_probability_change"].max() == 0.0
    assert {"original_sex_code", "original_sex_group", "flipped_sex_code"}.issubset(
        sensitivity.columns
    )
    assert not sensitivity["baseline_decision_changed"].any()
    assert not sensitivity["recall_policy_decision_changed"].any()


def test_shap_and_nearest_skip_reports_are_written(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fdd, "APPLICATION_FAIRNESS_DIR", tmp_path)

    fdd.write_shap_report(None, "SHAP unavailable in test.")
    fdd.write_nearest_neighbour_report(None, "Nearest-neighbour unavailable in test.")

    assert "SHAP unavailable" in (tmp_path / "shap_driver_comparison_by_sex.md").read_text()
    assert (
        "Nearest-neighbour unavailable"
        in (tmp_path / "nearest_neighbour_individual_fairness.md").read_text()
    )


def test_group_outcome_report_generation(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fdd, "APPLICATION_FAIRNESS_DIR", tmp_path)
    frame = pd.DataFrame(
        [
            {
                "policy": "xgboost_baseline_threshold_050",
                "display_name": "XGBoost baseline threshold 0.50",
                "threshold": 0.50,
                "sex_code": 1,
                "sex_group": "Male",
                "group": "Male (SEX=1)",
                "n": 10,
                "actual_default_rate": 0.2,
                "mean_predicted_default_probability": 0.3,
                "predicted_high_risk_rate": 0.4,
                "predicted_low_risk_rate": 0.6,
                "approval_support_rate": 0.6,
                "demographic_parity_difference": 0.02,
                "disparate_impact_ratio": 0.98,
            },
            {
                "policy": "xgboost_recall_threshold_025",
                "display_name": "XGBoost recall threshold 0.25",
                "threshold": 0.25,
                "sex_code": 1,
                "sex_group": "Male",
                "group": "Male (SEX=1)",
                "n": 10,
                "actual_default_rate": 0.2,
                "mean_predicted_default_probability": 0.3,
                "predicted_high_risk_rate": 0.5,
                "predicted_low_risk_rate": 0.5,
                "approval_support_rate": 0.5,
                "demographic_parity_difference": 0.07,
                "disparate_impact_ratio": 0.91,
            },
        ]
    )

    fdd.write_group_outcome_report(frame)

    assert (tmp_path / "group_outcome_analysis_by_sex.csv").exists()
    report_text = (tmp_path / "group_outcome_analysis_by_sex.md").read_text()
    csv_text = (tmp_path / "group_outcome_analysis_by_sex.csv").read_text()
    assert "sex_code" in csv_text
    assert "sex_group" in csv_text
    assert "Male (SEX=1)" in report_text
    assert "governance diagnostics" in report_text


def test_sex_mapping_is_centralized_and_readable() -> None:
    assert SEX_GROUP_LABELS == {1: "Male", 2: "Female"}
    assert fdd.group_label(1) == "Male (SEX=1)"
    assert fdd.group_label(2) == "Female (SEX=2)"


def test_generated_fairness_reports_avoid_forbidden_overclaims() -> None:
    report_dir = fdd.APPLICATION_FAIRNESS_DIR
    generated_reports = list(report_dir.glob("*.md")) + list(report_dir.glob("*.json"))
    assert generated_reports

    combined_text = "\n".join(
        path.read_text(encoding="utf-8").lower() for path in generated_reports
    )
    for phrase in FORBIDDEN_FAIRNESS_OVERCLAIMS:
        assert phrase not in combined_text
