"""Recall-focused threshold, class-weight, and oversampling experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .data_preprocessing import (
    FEATURE_SET_APPLICATION,
    TARGET_COL,
    build_preprocessor,
    get_dataset_split,
)
from .fairness_metrics import compute_fairness_metrics
from .model_builders import build_xgboost_estimator
from .utils import (
    MODELS_DIR,
    REPORTS_DIR,
    ensure_directories,
    load_dataset_auto,
    save_json,
    save_model,
)

RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.50
VALIDATION_SIZE = 0.25
FN_COST = 5
FP_COST = 1

RULE_A = "maximize_recall_precision_050"
RULE_B = "maximize_f2"
RULE_C = "maximize_recall_precision_045"
RULE_D = "maximize_f1"
RULE_E = "lowest_expected_cost_5_1"
SELECTION_RULES = [RULE_A, RULE_B, RULE_C, RULE_D, RULE_E]

MODEL_VALIDATION_DIR = REPORTS_DIR / "model_validation"
FAIRNESS_OUTPUT_DIR = REPORTS_DIR / "fairness_reports" / "application_model"


@dataclass(frozen=True)
class RecallOptimizationSplits:
    X_inner_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_train_full: pd.DataFrame
    X_test: pd.DataFrame
    y_inner_train: pd.Series
    y_validation: pd.Series
    y_train_full: pd.Series
    y_test: pd.Series
    train_indices: pd.Index
    validation_indices: pd.Index
    test_indices: pd.Index


@dataclass
class CandidateResult:
    candidate_name: str
    family: str
    scale_pos_weight: float | None
    uses_smote: bool
    selected_threshold: float
    selected_rule: str
    fallback_used: bool
    validation_metrics: dict[str, float | int | str | bool | None]
    validation_pr_auc: float
    test_metrics: dict[str, float | int | str | bool | None]
    test_roc_auc: float
    test_pr_auc: float
    test_proba: np.ndarray
    final_model: Any


def create_threshold_grid(
    start: float = 0.10,
    stop: float = 0.70,
    step: float = 0.05,
) -> list[float]:
    values = np.arange(start, stop + step / 2, step)
    return [round(float(value), 2) for value in values]


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    return value


def threshold_metrics(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    y_true_array = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_array, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true_array, y_pred)),
        "precision": float(precision_score(y_true_array, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_array, y_pred, zero_division=0)),
        "specificity": _safe_rate(tn, tn + fp),
        "f1": float(f1_score(y_true_array, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true_array, y_pred, beta=2, zero_division=0)),
        "false_positive_rate": _safe_rate(fp, fp + tn),
        "false_negative_rate": _safe_rate(fn, fn + tp),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "predicted_default_rate": float(np.mean(y_pred == 1)),
        "predicted_non_default_rate": float(np.mean(y_pred == 0)),
        "approval_support_rate": float(np.mean(y_pred == 0)),
        "default_capture_rate": float(recall_score(y_true_array, y_pred, zero_division=0)),
        "expected_cost": int(FN_COST * fn + FP_COST * fp),
    }


def evaluate_threshold_grid(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    thresholds: Sequence[float],
    candidate_name: str,
    split_name: str,
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        row = threshold_metrics(y_true, y_proba, threshold)
        row["candidate_name"] = candidate_name
        row["split"] = split_name
        rows.append(row)
    return pd.DataFrame(rows)


def _sort_for_max(
    df: pd.DataFrame,
    primary: str,
    secondary: list[str] | None = None,
) -> pd.Series | None:
    if df.empty:
        return None
    sort_cols = [primary, *(secondary or []), "threshold"]
    ascending = [False] * (len(sort_cols) - 1) + [True]
    return df.sort_values(sort_cols, ascending=ascending).iloc[0]


def select_threshold_by_rule(
    threshold_df: pd.DataFrame,
    rule: str = RULE_A,
) -> pd.Series | None:
    if rule == RULE_A:
        eligible = threshold_df[threshold_df["precision"] >= 0.50]
        return _sort_for_max(eligible, "recall", ["f2", "precision"])
    if rule == RULE_B:
        return _sort_for_max(threshold_df, "f2", ["recall", "precision"])
    if rule == RULE_C:
        eligible = threshold_df[threshold_df["precision"] >= 0.45]
        return _sort_for_max(eligible, "recall", ["f2", "precision"])
    if rule == RULE_D:
        return _sort_for_max(threshold_df, "f1", ["f2", "recall"])
    if rule == RULE_E:
        if threshold_df.empty:
            return None
        return threshold_df.sort_values(
            ["expected_cost", "recall", "f2", "threshold"],
            ascending=[True, False, False, True],
        ).iloc[0]
    raise ValueError(f"Unsupported threshold selection rule: {rule}")


def select_preferred_threshold(threshold_df: pd.DataFrame) -> tuple[pd.Series, bool, str]:
    preferred = select_threshold_by_rule(threshold_df, RULE_A)
    if preferred is not None:
        return preferred, False, RULE_A
    fallback = select_threshold_by_rule(threshold_df, RULE_B)
    if fallback is None:
        raise ValueError("No threshold rows are available for selection.")
    return fallback, True, RULE_B


def build_recall_optimization_splits_from_frame(
    df_raw: pd.DataFrame,
    validation_size: float = VALIDATION_SIZE,
    random_state: int = RANDOM_STATE,
) -> RecallOptimizationSplits:
    outer_split = get_dataset_split(
        df_raw,
        target_col=TARGET_COL,
        feature_set=FEATURE_SET_APPLICATION,
    )
    train_idx, validation_idx = train_test_split(
        outer_split.X_train.index,
        test_size=validation_size,
        random_state=random_state,
        stratify=outer_split.y_train,
    )
    splits = RecallOptimizationSplits(
        X_inner_train=outer_split.X_train.loc[train_idx],
        X_validation=outer_split.X_train.loc[validation_idx],
        X_train_full=outer_split.X_train,
        X_test=outer_split.X_test,
        y_inner_train=outer_split.y_train.loc[train_idx],
        y_validation=outer_split.y_train.loc[validation_idx],
        y_train_full=outer_split.y_train,
        y_test=outer_split.y_test,
        train_indices=pd.Index(train_idx),
        validation_indices=pd.Index(validation_idx),
        test_indices=outer_split.test_indices,
    )
    return splits


def build_recall_optimization_splits(
    validation_size: float = VALIDATION_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, RecallOptimizationSplits]:
    df_raw, _ = load_dataset_auto()
    splits = build_recall_optimization_splits_from_frame(
        df_raw,
        validation_size=validation_size,
        random_state=random_state,
    )
    return df_raw, splits


def _fit_xgboost_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float,
) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", build_xgboost_estimator(scale_pos_weight=scale_pos_weight)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def _smote_components():
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImblearnPipeline
    except ImportError:
        return None, None
    return SMOTE, ImblearnPipeline


def smote_skip_reason_if_unavailable() -> str | None:
    if _smote_components()[0] is None:
        return "imbalanced-learn is not installed; SMOTE experiment skipped."
    return None


def _fit_smote_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float,
) -> Any:
    smote_cls, pipeline_cls = _smote_components()
    if smote_cls is None or pipeline_cls is None:
        raise ImportError("imbalanced-learn is not installed.")
    pipeline = pipeline_cls(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("smote", smote_cls(random_state=RANDOM_STATE)),
            ("classifier", build_xgboost_estimator(scale_pos_weight=scale_pos_weight)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def _candidate_weight_values(
    y_train: pd.Series,
    max_candidates: int | None = None,
    quick: bool = False,
) -> list[float]:
    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())
    imbalance_ratio = negatives / positives if positives else 1.0
    values = [1, 2, 3, 4, 5, 7, 10, imbalance_ratio]
    if quick:
        values = [1, imbalance_ratio]

    deduped: list[float] = []
    for value in values:
        rounded = round(float(value), 4)
        if rounded not in deduped:
            deduped.append(rounded)
    if max_candidates is not None:
        deduped = deduped[: max(1, max_candidates)]
    return deduped


def _evaluate_probability_metrics(y_true: pd.Series, y_proba: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }


def _selection_rows_for_candidate(
    candidate_name: str,
    validation_thresholds: pd.DataFrame,
    validation_pr_auc: float,
) -> list[dict[str, Any]]:
    rows = []
    for rule in SELECTION_RULES:
        selected = select_threshold_by_rule(validation_thresholds, rule)
        if selected is None:
            rows.append(
                {
                    "candidate_name": candidate_name,
                    "selection_rule": rule,
                    "available": False,
                    "validation_pr_auc": validation_pr_auc,
                }
            )
            continue
        row = selected.to_dict()
        row.update(
            {
                "candidate_name": candidate_name,
                "selection_rule": rule,
                "available": True,
                "validation_pr_auc": validation_pr_auc,
            }
        )
        rows.append(row)
    return rows


def _candidate_result(
    candidate_name: str,
    family: str,
    scale_pos_weight: float | None,
    uses_smote: bool,
    validation_model: Any,
    final_model: Any,
    splits: RecallOptimizationSplits,
    thresholds: Sequence[float],
) -> tuple[CandidateResult, pd.DataFrame, list[dict[str, Any]]]:
    validation_proba = validation_model.predict_proba(splits.X_validation)[:, 1]
    validation_thresholds = evaluate_threshold_grid(
        splits.y_validation,
        validation_proba,
        thresholds,
        candidate_name=candidate_name,
        split_name="validation",
    )
    validation_pr_auc = float(average_precision_score(splits.y_validation, validation_proba))
    selected, fallback_used, selected_rule = select_preferred_threshold(validation_thresholds)

    test_proba = final_model.predict_proba(splits.X_test)[:, 1]
    test_threshold_metrics = threshold_metrics(
        splits.y_test,
        test_proba,
        float(selected["threshold"]),
    )
    test_probability_metrics = _evaluate_probability_metrics(splits.y_test, test_proba)
    test_threshold_metrics.update(test_probability_metrics)

    result = CandidateResult(
        candidate_name=candidate_name,
        family=family,
        scale_pos_weight=scale_pos_weight,
        uses_smote=uses_smote,
        selected_threshold=float(selected["threshold"]),
        selected_rule=selected_rule,
        fallback_used=fallback_used,
        validation_metrics=selected.to_dict(),
        validation_pr_auc=validation_pr_auc,
        test_metrics=test_threshold_metrics,
        test_roc_auc=test_probability_metrics["roc_auc"],
        test_pr_auc=test_probability_metrics["pr_auc"],
        test_proba=test_proba,
        final_model=final_model,
    )
    selection_rows = _selection_rows_for_candidate(
        candidate_name,
        validation_thresholds,
        validation_pr_auc,
    )
    return result, validation_thresholds, selection_rows


def _choose_best_policy(
    candidates: list[CandidateResult],
    include_smote: bool = False,
) -> CandidateResult:
    eligible_pool = [
        candidate for candidate in candidates if include_smote or not candidate.uses_smote
    ]
    rule_a_candidates = [
        candidate
        for candidate in eligible_pool
        if not candidate.fallback_used and candidate.validation_metrics.get("precision", 0) >= 0.50
    ]
    if rule_a_candidates:
        return sorted(
            rule_a_candidates,
            key=lambda candidate: (
                candidate.validation_metrics["recall"],
                candidate.validation_metrics["f2"],
                candidate.validation_pr_auc,
            ),
            reverse=True,
        )[0]
    return sorted(
        eligible_pool,
        key=lambda candidate: (
            candidate.validation_metrics["f2"],
            candidate.validation_metrics["recall"],
            candidate.validation_pr_auc,
        ),
        reverse=True,
    )[0]


def _candidate_summary_row(candidate: CandidateResult) -> dict[str, Any]:
    row = {
        "candidate_name": candidate.candidate_name,
        "family": candidate.family,
        "scale_pos_weight": candidate.scale_pos_weight,
        "uses_smote": candidate.uses_smote,
        "selected_threshold": candidate.selected_threshold,
        "selection_rule": candidate.selected_rule,
        "fallback_used": candidate.fallback_used,
        "validation_precision": candidate.validation_metrics["precision"],
        "validation_recall": candidate.validation_metrics["recall"],
        "validation_f1": candidate.validation_metrics["f1"],
        "validation_f2": candidate.validation_metrics["f2"],
        "validation_pr_auc": candidate.validation_pr_auc,
        "test_precision": candidate.test_metrics["precision"],
        "test_recall": candidate.test_metrics["recall"],
        "test_f1": candidate.test_metrics["f1"],
        "test_f2": candidate.test_metrics["f2"],
        "test_roc_auc": candidate.test_roc_auc,
        "test_pr_auc": candidate.test_pr_auc,
        "test_accuracy": candidate.test_metrics["accuracy"],
        "test_approval_support_rate": candidate.test_metrics["approval_support_rate"],
    }
    return row


def _write_threshold_tuning_report(threshold_df: pd.DataFrame) -> None:
    threshold_df.to_csv(MODEL_VALIDATION_DIR / "threshold_tuning_report.csv", index=False)
    lines = [
        "# Threshold Tuning Report",
        "",
        "Thresholds were evaluated on validation data only, from `0.10` to `0.70` in `0.05` steps.",
        "",
        "`0.50` is a default classification threshold, not a business law. Lowering the threshold usually increases default-class recall, but lowers precision and approval-support rate because more borrowers are flagged as high risk.",
        "",
        "Final threshold choice should use business loss, fairness guardrails, and operational review capacity.",
        "",
        "F1 balances precision and recall equally. F2 gives higher weight to recall and is useful when missing actual defaulters is more costly than flagging some good borrowers for review.",
    ]
    (MODEL_VALIDATION_DIR / "threshold_tuning_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_selection_reports(selection_df: pd.DataFrame, selected: CandidateResult) -> None:
    selection_df.to_csv(MODEL_VALIDATION_DIR / "threshold_selection_summary.csv", index=False)
    lines = [
        "# Threshold Selection Summary",
        "",
        f"Preferred rule: `{RULE_A}`.",
        "Fallback rule: `maximize_f2` if no threshold satisfies precision >= 0.50.",
        "",
        f"Selected policy: `{selected.candidate_name}` at threshold `{selected.selected_threshold:.2f}`.",
        f"Selection rule used: `{selected.selected_rule}`.",
        f"Fallback used: `{selected.fallback_used}`.",
        "",
        "All selection decisions are based on validation data only. Final test metrics are evaluated after selection on the untouched held-out test split.",
    ]
    (MODEL_VALIDATION_DIR / "threshold_selection_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    payload = {
        "selected_candidate": selected.candidate_name,
        "family": selected.family,
        "scale_pos_weight": selected.scale_pos_weight,
        "uses_smote": selected.uses_smote,
        "selected_threshold": selected.selected_threshold,
        "selection_rule": selected.selected_rule,
        "fallback_used": selected.fallback_used,
        "validation_metrics": selected.validation_metrics,
        "validation_pr_auc": selected.validation_pr_auc,
        "test_metrics": selected.test_metrics,
        "test_roc_auc": selected.test_roc_auc,
        "test_pr_auc": selected.test_pr_auc,
        "test_set_policy": "held-out test split was not used for threshold or model selection",
    }
    save_json(_json_safe(payload), MODEL_VALIDATION_DIR / "selected_recall_policy.json")


def _write_class_weight_report(class_weight_df: pd.DataFrame) -> None:
    class_weight_df.to_csv(MODEL_VALIDATION_DIR / "class_weight_tuning_report.csv", index=False)
    lines = [
        "# Class-Weight Tuning Report",
        "",
        "`scale_pos_weight` candidates were trained on inner-train data for validation-based selection. After threshold selection, each setting was refit on the full training split and evaluated once on the untouched test split.",
        "",
        "The computed imbalance ratio candidate is `n_negative / n_positive` from the inner-train split.",
    ]
    (MODEL_VALIDATION_DIR / "class_weight_tuning_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_smote_report(smote_rows: list[dict[str, Any]], skip_reason: str | None) -> None:
    smote_df = pd.DataFrame(smote_rows)
    smote_df.to_csv(MODEL_VALIDATION_DIR / "smote_experiment_report.csv", index=False)
    lines = [
        "# SMOTE Experiment Report",
        "",
        "SMOTE is experimental in tabular credit data. Synthetic borrowers may not always be economically realistic.",
        "",
        "SMOTE is included as a recall-improvement experiment, not as the default production policy.",
        "",
        "SMOTE is applied only on training data inside the training pipeline. It is never applied before splitting and never applied to validation or test data.",
    ]
    if skip_reason:
        lines.extend(["", f"SMOTE skipped: {skip_reason}"])
    (MODEL_VALIDATION_DIR / "smote_experiment_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _plot_precision_recall_curve(
    y_true: pd.Series,
    y_proba: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_precision_recall_comparison(
    y_true: pd.Series,
    curves: list[tuple[str, np.ndarray]],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, proba in curves:
        precision, recall, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        ax.plot(recall, precision, linewidth=2, label=f"{label} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve Comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_precision_recall_artifacts(
    splits: RecallOptimizationSplits,
    baseline: CandidateResult,
    selected: CandidateResult,
) -> None:
    _plot_precision_recall_curve(
        splits.y_test,
        baseline.test_proba,
        "Precision-Recall Curve - Baseline XGBoost",
        MODEL_VALIDATION_DIR / "precision_recall_curve_baseline.png",
    )
    _plot_precision_recall_curve(
        splits.y_test,
        selected.test_proba,
        "Precision-Recall Curve - Recall-Optimized Policy",
        MODEL_VALIDATION_DIR / "precision_recall_curve_recall_optimized.png",
    )
    _plot_precision_recall_comparison(
        splits.y_test,
        [
            ("Baseline", baseline.test_proba),
            ("Recall optimized", selected.test_proba),
        ],
        MODEL_VALIDATION_DIR / "precision_recall_curve_comparison.png",
    )


def _write_recall_summary(
    summary_rows: list[dict[str, Any]],
    selected: CandidateResult,
    smote_skip_reason: str | None,
) -> None:
    summary_df = pd.DataFrame(summary_rows)
    payload = {
        "selected_policy": {
            "candidate_name": selected.candidate_name,
            "selected_threshold": selected.selected_threshold,
            "selection_rule": selected.selected_rule,
            "fallback_used": selected.fallback_used,
            "test_metrics": selected.test_metrics,
            "test_roc_auc": selected.test_roc_auc,
            "test_pr_auc": selected.test_pr_auc,
        },
        "comparisons": summary_rows,
        "smote_skip_reason": smote_skip_reason,
        "recommendation": (
            "For risk-screening use, the recall-optimized threshold/model captures more actual "
            "defaulters at the cost of more false positives. The original 0.50 threshold remains "
            "a conservative baseline, while the recall-optimized policy is more appropriate for "
            "manual-review triage."
        ),
    }
    save_json(_json_safe(payload), MODEL_VALIDATION_DIR / "recall_optimized_summary.json")

    lines = [
        "# Recall-Optimized Summary",
        "",
        "ROC-AUC measures ranking over both classes. PR-AUC is especially useful when default is the minority class. Recall-focused screening should inspect the precision-recall tradeoff, not only ROC-AUC.",
        "",
        "| Candidate | Threshold | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC | PR-AUC | Approval-support rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['candidate_name']} | {row['selected_threshold']:.2f} | "
            f"{row['test_accuracy']:.4f} | {row['test_precision']:.4f} | "
            f"{row['test_recall']:.4f} | {row['test_f1']:.4f} | "
            f"{row['test_f2']:.4f} | {row['test_roc_auc']:.4f} | "
            f"{row['test_pr_auc']:.4f} | {row['test_approval_support_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Recommended wording: For risk-screening use, the recall-optimized threshold/model captures more actual defaulters at the cost of more false positives. The original 0.50 threshold remains a conservative baseline, while the recall-optimized policy is more appropriate for manual-review triage.",
        ]
    )
    if smote_skip_reason:
        lines.extend(["", f"SMOTE note: {smote_skip_reason}"])
    (MODEL_VALIDATION_DIR / "recall_optimized_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _fairness_for_predictions(
    y_default_true: pd.Series,
    default_pred: np.ndarray,
    sensitive: pd.Series,
) -> dict[str, float]:
    approval_true = (1 - np.asarray(y_default_true).astype(int)).astype(int)
    approval_pred = (1 - np.asarray(default_pred).astype(int)).astype(int)
    return compute_fairness_metrics(approval_true, approval_pred, sensitive.astype(str).values)


def _write_fairness_reports(
    df_raw: pd.DataFrame,
    splits: RecallOptimizationSplits,
    baseline: CandidateResult,
    selected: CandidateResult,
) -> None:
    FAIRNESS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sensitive = df_raw.loc[splits.test_indices, "SEX"].astype(str)
    baseline_default_pred = (baseline.test_proba >= DEFAULT_THRESHOLD).astype(int)
    selected_default_pred = (selected.test_proba >= selected.selected_threshold).astype(int)
    baseline_fairness = _fairness_for_predictions(
        splits.y_test,
        baseline_default_pred,
        sensitive,
    )
    selected_fairness = _fairness_for_predictions(
        splits.y_test,
        selected_default_pred,
        sensitive,
    )

    selected_row = {
        "candidate_name": selected.candidate_name,
        "threshold": selected.selected_threshold,
        **selected_fairness,
    }
    pd.DataFrame([selected_row]).to_csv(
        FAIRNESS_OUTPUT_DIR / "recall_optimized_fairness_metrics.csv",
        index=False,
    )
    comparison_rows = [
        {"policy": "baseline_threshold_050", "threshold": DEFAULT_THRESHOLD, **baseline_fairness},
        {
            "policy": "recall_optimized",
            "threshold": selected.selected_threshold,
            **selected_fairness,
        },
    ]
    pd.DataFrame(comparison_rows).to_csv(
        FAIRNESS_OUTPUT_DIR / "threshold_fairness_comparison.csv",
        index=False,
    )

    fairness_note = [
        "# Recall-Optimized Fairness Metrics",
        "",
        "Fairness metrics are computed on favorable outcomes, defined as predicted non-default / lower-risk manual-review support.",
        "",
        "Improving recall can change fairness metrics. Threshold choice is both a business and fairness-governance decision. These metrics are diagnostic and not proof of legal compliance.",
    ]
    (FAIRNESS_OUTPUT_DIR / "recall_optimized_fairness_metrics.md").write_text(
        "\n".join(fairness_note) + "\n", encoding="utf-8"
    )
    comparison_note = [
        "# Threshold Fairness Comparison",
        "",
        "This report compares fairness diagnostics at the baseline `0.50` threshold and at the selected recall-optimized threshold.",
        "",
        "Improving recall can change fairness metrics. Threshold choice is both a business and fairness-governance decision. These metrics are diagnostic and not proof of legal compliance.",
    ]
    (FAIRNESS_OUTPUT_DIR / "threshold_fairness_comparison.md").write_text(
        "\n".join(comparison_note) + "\n", encoding="utf-8"
    )


def run_recall_optimization(
    quick: bool = False,
    max_candidates: int | None = None,
) -> dict[str, Any]:
    ensure_directories()
    MODEL_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    FAIRNESS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df_raw, splits = build_recall_optimization_splits()
    thresholds = [0.30, 0.50, 0.70] if quick else create_threshold_grid()
    class_weight_values = _candidate_weight_values(
        splits.y_inner_train,
        max_candidates=max_candidates,
        quick=quick,
    )

    candidates: list[CandidateResult] = []
    threshold_frames: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []

    for scale_pos_weight in class_weight_values:
        if scale_pos_weight == 1:
            name = "xgboost_public_baseline_threshold_050"
            family = "baseline_xgboost"
        else:
            name = f"xgboost_public_weighted_spw_{str(scale_pos_weight).replace('.', '_')}"
            family = "weighted_xgboost"

        validation_model = _fit_xgboost_pipeline(
            splits.X_inner_train,
            splits.y_inner_train,
            scale_pos_weight=scale_pos_weight,
        )
        final_model = _fit_xgboost_pipeline(
            splits.X_train_full,
            splits.y_train_full,
            scale_pos_weight=scale_pos_weight,
        )
        result, threshold_df, candidate_selection_rows = _candidate_result(
            name,
            family,
            scale_pos_weight,
            False,
            validation_model,
            final_model,
            splits,
            thresholds,
        )
        candidates.append(result)
        threshold_frames.append(threshold_df)
        selection_rows.extend(candidate_selection_rows)

    selected = _choose_best_policy(candidates, include_smote=False)
    save_model(selected.final_model, MODELS_DIR / "xgboost_public_recall_optimized.pkl")
    weighted_candidates = [candidate for candidate in candidates if candidate.scale_pos_weight != 1]
    best_weighted = _choose_best_policy(weighted_candidates, include_smote=False)
    save_model(
        best_weighted.final_model, MODELS_DIR / "xgboost_public_weighted_recall_optimized.pkl"
    )

    smote_rows: list[dict[str, Any]] = []
    smote_candidates: list[CandidateResult] = []
    smote_skip_reason = smote_skip_reason_if_unavailable()
    if smote_skip_reason is not None:
        smote_rows.append({"status": "skipped", "reason": smote_skip_reason})
    else:
        smote_specs = [
            ("smote_xgboost", 1.0),
            ("smote_weighted_xgboost", best_weighted.scale_pos_weight or 1.0),
        ]
        for name, scale_pos_weight in smote_specs:
            try:
                validation_model = _fit_smote_pipeline(
                    splits.X_inner_train,
                    splits.y_inner_train,
                    scale_pos_weight=scale_pos_weight,
                )
                final_model = _fit_smote_pipeline(
                    splits.X_train_full,
                    splits.y_train_full,
                    scale_pos_weight=scale_pos_weight,
                )
                result, threshold_df, candidate_selection_rows = _candidate_result(
                    name,
                    "smote_experiment",
                    scale_pos_weight,
                    True,
                    validation_model,
                    final_model,
                    splits,
                    thresholds,
                )
                smote_candidates.append(result)
                threshold_frames.append(threshold_df)
                selection_rows.extend(candidate_selection_rows)
                smote_rows.append(_candidate_summary_row(result))
            except Exception as exc:
                smote_rows.append({"candidate_name": name, "status": "skipped", "reason": str(exc)})

    all_thresholds = pd.concat(threshold_frames, ignore_index=True)
    _write_threshold_tuning_report(all_thresholds)

    selection_df = pd.DataFrame(selection_rows)
    _write_selection_reports(selection_df, selected)

    class_weight_df = pd.DataFrame([_candidate_summary_row(candidate) for candidate in candidates])
    _write_class_weight_report(class_weight_df)
    _write_smote_report(smote_rows, smote_skip_reason)

    baseline = next(candidate for candidate in candidates if candidate.scale_pos_weight == 1)
    baseline_050_metrics = threshold_metrics(
        splits.y_test,
        baseline.test_proba,
        DEFAULT_THRESHOLD,
    )
    baseline_probability_metrics = _evaluate_probability_metrics(splits.y_test, baseline.test_proba)
    baseline_050_metrics.update(baseline_probability_metrics)
    baseline_050_row = {
        "candidate_name": "current_baseline_threshold_050",
        "selected_threshold": DEFAULT_THRESHOLD,
        "test_accuracy": baseline_050_metrics["accuracy"],
        "test_precision": baseline_050_metrics["precision"],
        "test_recall": baseline_050_metrics["recall"],
        "test_f1": baseline_050_metrics["f1"],
        "test_f2": baseline_050_metrics["f2"],
        "test_roc_auc": baseline_050_metrics["roc_auc"],
        "test_pr_auc": baseline_050_metrics["pr_auc"],
        "test_approval_support_rate": baseline_050_metrics["approval_support_rate"],
        "notes": "Baseline XGBoost at default threshold 0.50.",
    }

    baseline_tuned = baseline
    best_smote = (
        _choose_best_policy(smote_candidates, include_smote=True) if smote_candidates else None
    )
    summary_rows = [
        baseline_050_row,
        {
            **_candidate_summary_row(baseline_tuned),
            "notes": "Baseline XGBoost with validation-tuned threshold.",
        },
        {
            **_candidate_summary_row(best_weighted),
            "notes": "Best class-weighted XGBoost with validation-tuned threshold.",
        },
    ]
    if best_smote is not None:
        summary_rows.append(
            {
                **_candidate_summary_row(best_smote),
                "notes": "Best separated SMOTE experiment; not the default production policy.",
            }
        )

    _write_precision_recall_artifacts(splits, baseline, selected)
    _write_recall_summary(summary_rows, selected, smote_skip_reason)
    _write_fairness_reports(df_raw, splits, baseline, selected)

    result = {
        "selected_policy": {
            "candidate_name": selected.candidate_name,
            "selected_threshold": selected.selected_threshold,
            "selection_rule": selected.selected_rule,
            "fallback_used": selected.fallback_used,
            "validation_metrics": selected.validation_metrics,
            "test_metrics": selected.test_metrics,
            "test_roc_auc": selected.test_roc_auc,
            "test_pr_auc": selected.test_pr_auc,
        },
        "baseline_threshold_050": baseline_050_metrics,
        "best_class_weight": best_weighted.scale_pos_weight,
        "smote_skip_reason": smote_skip_reason,
        "smote_best_candidate": best_smote.candidate_name if best_smote else None,
    }
    return _json_safe(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run validation-based recall optimization for the public UCI model."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a reduced threshold/model grid for smoke testing.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Limit the number of class-weight candidates, mainly for quick local checks.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_recall_optimization(quick=args.quick, max_candidates=args.max_candidates)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
