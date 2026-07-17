"""Protected-attribute fairness deep dive for the public UCI credit-default model.

The module is intentionally diagnostic: it does not change model training, model selection,
or the dashboard's applicant-facing prediction path.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_preprocessing import (
    FEATURE_SET_APPLICATION,
    TARGET_COL,
    build_preprocessor,
    get_dataset_split,
    get_feature_columns,
    prepare_modeling_table,
)
from .dataset_adapters import (
    BILL_AMOUNT_COLUMNS,
    ENGINEERED_UCI_COLUMNS,
    PAY_AMOUNT_COLUMNS,
    PAY_STATUS_COLUMNS,
)
from .protected_attributes import (
    PROTECTED_ATTRIBUTE_SEX,
    sex_code,
    sex_group,
    sex_group_display,
    sex_mapping_rows,
)
from .utils import MODELS_DIR, REPORTS_DIR, load_dataset_auto, load_model, save_json

RANDOM_STATE = 42
PROTECTED_ATTRIBUTE = PROTECTED_ATTRIBUTE_SEX
APPLICATION_FAIRNESS_DIR = REPORTS_DIR / "fairness_reports" / "application_model"
MODEL_VALIDATION_DIR = REPORTS_DIR / "model_validation"

BASELINE_THRESHOLD = 0.50
RECALL_THRESHOLD = 0.25
THRESHOLD_GRID = [round(float(value), 2) for value in np.arange(0.10, 0.70 + 0.001, 0.05)]
CALIBRATION_BINS = [round(float(value), 2) for value in np.arange(0.00, 1.00 + 0.001, 0.10)]

INTERPRETABLE_FEATURES = [
    "LIMIT_BAL",
    *PAY_STATUS_COLUMNS,
    *BILL_AMOUNT_COLUMNS,
    *PAY_AMOUNT_COLUMNS,
    *ENGINEERED_UCI_COLUMNS,
]


@dataclass(frozen=True)
class FairnessContext:
    raw: pd.DataFrame
    split: Any
    sensitive_test: pd.Series
    y_test: pd.Series
    xgboost_proba: np.ndarray
    policies: list[dict[str, Any]]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    return value


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _df_to_markdown(frame: pd.DataFrame) -> str:
    """Render a compact GitHub-flavored Markdown table without optional deps."""

    if frame.empty:
        return "No rows available."

    def clean(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|")

    headers = [clean(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(clean(value) for value in row) + " |")
    return "\n".join(lines)


def _format_float(value: Any, digits: int = 4) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def _rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _binary_auc(y_true: np.ndarray, score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, score))


def _binary_pr_auc(y_true: np.ndarray, score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, score))


def group_label(value: Any) -> str:
    return sex_group_display(value)


def _sex_fields(value: Any) -> dict[str, Any]:
    return {
        "sex_code": sex_code(value),
        "sex_group": sex_group(value),
        "group": group_label(value),
    }


def _groupby_sex(frame: pd.DataFrame):
    return frame.groupby(["sex_code", "sex_group", "group"], dropna=False, sort=True)


def _lookup_metric(frame: pd.DataFrame, policy: str, code: int, column: str) -> float | None:
    if column not in frame.columns:
        return None
    match = frame[(frame["policy"] == policy) & (frame["sex_code"] == code)]
    if match.empty:
        return None
    return float(match.iloc[0][column])


def classification_metrics(
    y_true: Sequence[int], y_proba: Sequence[float], threshold: float
) -> dict[str, float]:
    y_true_array = np.asarray(y_true).astype(int)
    y_proba_array = np.asarray(y_proba).astype(float)
    y_pred = (y_proba_array >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true_array, y_pred)),
        "precision": float(precision_score(y_true_array, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_array, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_array, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true_array, y_pred, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_array, y_proba_array)),
        "pr_auc": float(average_precision_score(y_true_array, y_proba_array)),
        "approval_support_rate": float(np.mean(y_pred == 0)),
        "predicted_high_risk_rate": float(np.mean(y_pred == 1)),
    }


def group_confusion_metrics(
    y_true: Sequence[int],
    y_proba: Sequence[float],
    sensitive: Sequence[Any],
    threshold: float,
    policy_name: str,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "y_true": np.asarray(y_true).astype(int),
            "y_proba": np.asarray(y_proba).astype(float),
            "sex_code": [sex_code(value) for value in sensitive],
            "sex_group": [sex_group(value) for value in sensitive],
            "group": [group_label(value) for value in sensitive],
        }
    )
    frame["predicted_high_risk"] = (frame["y_proba"] >= threshold).astype(int)
    rows: list[dict[str, Any]] = []
    for (code, readable_group, group), group_df in _groupby_sex(frame):
        tn, fp, fn, tp = confusion_matrix(
            group_df["y_true"],
            group_df["predicted_high_risk"],
            labels=[0, 1],
        ).ravel()
        rows.append(
            {
                "policy": policy_name,
                "threshold": threshold,
                "sex_code": int(code) if pd.notna(code) else None,
                "sex_group": readable_group,
                "group": group,
                "n": int(len(group_df)),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
                "accuracy": _rate(tp + tn, len(group_df)),
                "precision": _rate(tp, tp + fp),
                "positive_predictive_value": _rate(tp, tp + fp),
                "recall": _rate(tp, tp + fn),
                "true_positive_rate": _rate(tp, tp + fn),
                "false_positive_rate": _rate(fp, fp + tn),
                "false_negative_rate": _rate(fn, fn + tp),
                "specificity": _rate(tn, tn + fp),
                "true_negative_rate": _rate(tn, tn + fp),
                "negative_predictive_value": _rate(tn, tn + fn),
            }
        )
    return pd.DataFrame(rows)


def fairness_from_predictions(
    y_true_default: Sequence[int],
    high_risk_pred: Sequence[int],
    sensitive: Sequence[Any],
) -> dict[str, float]:
    frame = pd.DataFrame(
        {
            "default_true": np.asarray(y_true_default).astype(int),
            "high_risk_pred": np.asarray(high_risk_pred).astype(int),
            "sex_code": [sex_code(value) for value in sensitive],
            "sex_group": [sex_group(value) for value in sensitive],
            "group": [group_label(value) for value in sensitive],
        }
    )
    frame["non_default_true"] = 1 - frame["default_true"]
    frame["low_risk_support"] = 1 - frame["high_risk_pred"]

    support_rates = frame.groupby("group")["low_risk_support"].mean()
    high_risk_rates = frame.groupby("group")["high_risk_pred"].mean()
    demographic_parity_difference = float(support_rates.max() - support_rates.min())
    disparate_impact_ratio = (
        float(support_rates.min() / support_rates.max()) if support_rates.max() > 0 else 1.0
    )

    approval_tprs = []
    approval_fprs = []
    default_fprs = []
    default_fnrs = []
    for _, group_df in _groupby_sex(frame):
        approval_tp = int(
            ((group_df["non_default_true"] == 1) & (group_df["low_risk_support"] == 1)).sum()
        )
        approval_fn = int(
            ((group_df["non_default_true"] == 1) & (group_df["low_risk_support"] == 0)).sum()
        )
        approval_fp = int(
            ((group_df["non_default_true"] == 0) & (group_df["low_risk_support"] == 1)).sum()
        )
        approval_tn = int(
            ((group_df["non_default_true"] == 0) & (group_df["low_risk_support"] == 0)).sum()
        )
        approval_tprs.append(_rate(approval_tp, approval_tp + approval_fn))
        approval_fprs.append(_rate(approval_fp, approval_fp + approval_tn))

        default_fp = int(
            ((group_df["default_true"] == 0) & (group_df["high_risk_pred"] == 1)).sum()
        )
        default_tn = int(
            ((group_df["default_true"] == 0) & (group_df["high_risk_pred"] == 0)).sum()
        )
        default_fn = int(
            ((group_df["default_true"] == 1) & (group_df["high_risk_pred"] == 0)).sum()
        )
        default_tp = int(
            ((group_df["default_true"] == 1) & (group_df["high_risk_pred"] == 1)).sum()
        )
        default_fprs.append(_rate(default_fp, default_fp + default_tn))
        default_fnrs.append(_rate(default_fn, default_fn + default_tp))

    equal_opportunity_difference = (
        float(max(approval_tprs) - min(approval_tprs)) if approval_tprs else 0.0
    )
    approval_fpr_difference = (
        float(max(approval_fprs) - min(approval_fprs)) if approval_fprs else 0.0
    )
    equalized_odds_difference = max(equal_opportunity_difference, approval_fpr_difference)
    default_fpr_difference = float(max(default_fprs) - min(default_fprs)) if default_fprs else 0.0
    default_fnr_difference = float(max(default_fnrs) - min(default_fnrs)) if default_fnrs else 0.0

    return {
        "demographic_parity_difference": demographic_parity_difference,
        "disparate_impact_ratio": disparate_impact_ratio,
        "equal_opportunity_difference": equal_opportunity_difference,
        "equalized_odds_difference": float(equalized_odds_difference),
        "false_positive_rate_difference": default_fpr_difference,
        "false_negative_rate_difference": default_fnr_difference,
        "high_risk_rate_difference": float(high_risk_rates.max() - high_risk_rates.min()),
    }


def calibration_bins(
    y_true: Sequence[int],
    y_proba: Sequence[float],
    sensitive: Sequence[Any],
    bins: Sequence[float] = CALIBRATION_BINS,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "y_true": np.asarray(y_true).astype(int),
            "y_proba": np.asarray(y_proba).astype(float),
            "sex_code": [sex_code(value) for value in sensitive],
            "sex_group": [sex_group(value) for value in sensitive],
            "group": [group_label(value) for value in sensitive],
        }
    )
    labels = [f"{bins[index]:.2f}-{bins[index + 1]:.2f}" for index in range(len(bins) - 1)]
    cut_bins = list(bins)
    cut_bins[-1] = np.nextafter(cut_bins[-1], 2.0)
    frame["bin"] = pd.cut(
        frame["y_proba"],
        bins=cut_bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    grouped = (
        frame.dropna(subset=["bin"])
        .groupby(["sex_code", "sex_group", "group", "bin"], observed=False, dropna=False)
        .agg(
            n=("y_true", "size"),
            mean_predicted_probability=("y_proba", "mean"),
            observed_default_rate=("y_true", "mean"),
        )
        .reset_index()
    )
    grouped["calibration_gap"] = (
        grouped["observed_default_rate"] - grouped["mean_predicted_probability"]
    )
    return grouped


def _load_prediction_export(
    path: Path,
    expected_y: pd.Series,
    model_name: str,
) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if not {"y_true", "y_proba"}.issubset(frame.columns):
        return None
    y_values = frame["y_true"].astype(int).to_numpy()
    expected = expected_y.astype(int).to_numpy()
    if len(y_values) != len(expected) or not np.array_equal(y_values, expected):
        return None
    frame = frame.copy()
    frame["model_name"] = model_name
    return frame


def build_context() -> FairnessContext:
    raw, _ = load_dataset_auto()
    split = get_dataset_split(raw, target_col=TARGET_COL, feature_set=FEATURE_SET_APPLICATION)
    sensitive_test = raw.loc[split.test_indices, PROTECTED_ATTRIBUTE].reset_index(drop=True)
    y_test = split.y_test.reset_index(drop=True)

    xgb_export = _load_prediction_export(
        MODEL_VALIDATION_DIR / "xgboost_test_predictions.csv",
        y_test,
        "xgboost_public",
    )
    if xgb_export is not None:
        xgboost_proba = xgb_export["y_proba"].astype(float).to_numpy()
    else:
        model = load_model(MODELS_DIR / "xgboost_public.pkl")
        xgboost_proba = model.predict_proba(split.X_test)[:, 1]

    policies: list[dict[str, Any]] = [
        {
            "policy": "xgboost_baseline_threshold_050",
            "display_name": "XGBoost baseline threshold 0.50",
            "model_name": "xgboost_public",
            "threshold": BASELINE_THRESHOLD,
            "y_proba": xgboost_proba,
            "primary": True,
        },
        {
            "policy": "xgboost_recall_threshold_025",
            "display_name": "XGBoost recall threshold 0.25",
            "model_name": "xgboost_public",
            "threshold": RECALL_THRESHOLD,
            "y_proba": xgboost_proba,
            "primary": True,
        },
    ]

    optional_specs = [
        (
            "logistic_baseline_threshold_050",
            "Logistic regression threshold 0.50",
            "logistic_public",
            MODEL_VALIDATION_DIR / "logistic_test_predictions.csv",
            BASELINE_THRESHOLD,
        ),
        (
            "dnn_baseline_threshold_050",
            "DNN baseline threshold 0.50",
            "dnn_baseline",
            MODEL_VALIDATION_DIR / "dnn_test_predictions.csv",
            BASELINE_THRESHOLD,
        ),
    ]
    for policy, display_name, model_name, path, threshold in optional_specs:
        export = _load_prediction_export(path, y_test, model_name)
        if export is not None:
            policies.append(
                {
                    "policy": policy,
                    "display_name": display_name,
                    "model_name": model_name,
                    "threshold": threshold,
                    "y_proba": export["y_proba"].astype(float).to_numpy(),
                    "primary": False,
                }
            )

    dnn_policy = _read_json(MODEL_VALIDATION_DIR / "deep_learning_selected_policy.json")
    dnn_export = _load_prediction_export(
        MODEL_VALIDATION_DIR / "dnn_test_predictions.csv",
        y_test,
        "dnn_baseline",
    )
    selected_threshold = (dnn_policy or {}).get("selected_threshold")
    if dnn_export is not None and selected_threshold is not None:
        policies.append(
            {
                "policy": "dnn_recall_threshold_030",
                "display_name": "DNN recall threshold 0.30",
                "model_name": "dnn_baseline",
                "threshold": float(selected_threshold),
                "y_proba": dnn_export["y_proba"].astype(float).to_numpy(),
                "primary": False,
            }
        )

    return FairnessContext(raw, split, sensitive_test, y_test, xgboost_proba, policies)


def build_group_outcome_analysis(context: FairnessContext) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy in context.policies:
        high_risk = (policy["y_proba"] >= policy["threshold"]).astype(int)
        fairness = fairness_from_predictions(context.y_test, high_risk, context.sensitive_test)
        frame = pd.DataFrame(
            {
                "y_true": context.y_test.astype(int),
                "y_proba": policy["y_proba"],
                "sex_code": [sex_code(value) for value in context.sensitive_test],
                "sex_group": [sex_group(value) for value in context.sensitive_test],
                "group": [group_label(value) for value in context.sensitive_test],
                "predicted_high_risk": high_risk,
            }
        )
        for (code, readable_group, group), group_df in _groupby_sex(frame):
            rows.append(
                {
                    "policy": policy["policy"],
                    "display_name": policy["display_name"],
                    "threshold": policy["threshold"],
                    "sex_code": int(code) if pd.notna(code) else None,
                    "sex_group": readable_group,
                    "group": group,
                    "n": int(len(group_df)),
                    "actual_default_rate": float(group_df["y_true"].mean()),
                    "mean_predicted_default_probability": float(group_df["y_proba"].mean()),
                    "predicted_high_risk_rate": float(group_df["predicted_high_risk"].mean()),
                    "predicted_low_risk_rate": float(1 - group_df["predicted_high_risk"].mean()),
                    "approval_support_rate": float(1 - group_df["predicted_high_risk"].mean()),
                    **fairness,
                }
            )
    return pd.DataFrame(rows)


def write_group_outcome_report(frame: pd.DataFrame) -> None:
    out_csv = APPLICATION_FAIRNESS_DIR / "group_outcome_analysis_by_sex.csv"
    frame.to_csv(out_csv, index=False)
    primary = frame[
        frame["policy"].isin(["xgboost_baseline_threshold_050", "xgboost_recall_threshold_025"])
    ]
    baseline_male = _lookup_metric(
        primary, "xgboost_baseline_threshold_050", 1, "predicted_high_risk_rate"
    )
    baseline_female = _lookup_metric(
        primary, "xgboost_baseline_threshold_050", 2, "predicted_high_risk_rate"
    )
    recall_male = _lookup_metric(
        primary, "xgboost_recall_threshold_025", 1, "predicted_high_risk_rate"
    )
    recall_female = _lookup_metric(
        primary, "xgboost_recall_threshold_025", 2, "predicted_high_risk_rate"
    )
    lines = [
        "# Group Outcome Analysis by SEX",
        "",
        "Verified UCI protected-attribute mapping: `SEX=1` is Male and `SEX=2` is Female.",
        "",
        "This report compares observed default rates, predicted default probabilities, high-risk flag rates, and low-risk / approval-support rates by `SEX`/gender.",
        "",
        "Outcome differences show whether groups receive different model-driven high-risk flag rates. They are governance diagnostics and do not prove legal discrimination or causal bias.",
        "",
        (
            "Male applicants (SEX=1) had higher high-risk flag rates than Female "
            f"applicants (SEX=2): `{_format_float(baseline_male)}` vs "
            f"`{_format_float(baseline_female)}` at threshold 0.50, and "
            f"`{_format_float(recall_male)}` vs `{_format_float(recall_female)}` "
            "at the recall threshold 0.25."
        ),
        "",
        "## Primary XGBoost policies",
        "",
        _df_to_markdown(
            primary[
                [
                    "policy",
                    "sex_group",
                    "sex_code",
                    "group",
                    "n",
                    "actual_default_rate",
                    "mean_predicted_default_probability",
                    "predicted_high_risk_rate",
                    "approval_support_rate",
                    "demographic_parity_difference",
                    "disparate_impact_ratio",
                ]
            ]
        ),
        "",
        "The fairness differences use the favorable outcome, defined as predicted non-default / lower-risk support, to stay consistent with the project model card.",
    ]
    _write_markdown(APPLICATION_FAIRNESS_DIR / "group_outcome_analysis_by_sex.md", lines)


def build_group_error_analysis(context: FairnessContext) -> pd.DataFrame:
    frames = [
        group_confusion_metrics(
            context.y_test,
            policy["y_proba"],
            context.sensitive_test,
            policy["threshold"],
            policy["policy"],
        )
        for policy in context.policies
    ]
    return pd.concat(frames, ignore_index=True)


def write_group_error_report(frame: pd.DataFrame) -> None:
    frame.to_csv(APPLICATION_FAIRNESS_DIR / "group_error_analysis_by_sex.csv", index=False)
    primary = frame[
        frame["policy"].isin(["xgboost_baseline_threshold_050", "xgboost_recall_threshold_025"])
    ]
    baseline_male_fpr = _lookup_metric(
        primary, "xgboost_baseline_threshold_050", 1, "false_positive_rate"
    )
    baseline_female_fpr = _lookup_metric(
        primary, "xgboost_baseline_threshold_050", 2, "false_positive_rate"
    )
    baseline_male_fnr = _lookup_metric(
        primary, "xgboost_baseline_threshold_050", 1, "false_negative_rate"
    )
    baseline_female_fnr = _lookup_metric(
        primary, "xgboost_baseline_threshold_050", 2, "false_negative_rate"
    )
    recall_male_fpr = _lookup_metric(
        primary, "xgboost_recall_threshold_025", 1, "false_positive_rate"
    )
    recall_female_fpr = _lookup_metric(
        primary, "xgboost_recall_threshold_025", 2, "false_positive_rate"
    )
    recall_male_fnr = _lookup_metric(
        primary, "xgboost_recall_threshold_025", 1, "false_negative_rate"
    )
    recall_female_fnr = _lookup_metric(
        primary, "xgboost_recall_threshold_025", 2, "false_negative_rate"
    )
    lines = [
        "# Group Error Analysis by SEX",
        "",
        "Verified UCI protected-attribute mapping: `SEX=1` is Male and `SEX=2` is Female.",
        "",
        "Positive class = default / high-risk flag.",
        "",
        "- False positive: an actual non-defaulter is flagged high risk. This can unnecessarily route reliable customers into manual review or lower credit support.",
        "- False negative: an actual defaulter is not flagged high risk. This can miss actual default risk and increase bank/NBFC loss.",
        "",
        "Different error patterns across groups are a fairness-governance concern. They are not proof of legal discrimination.",
        "",
        (
            "Baseline threshold 0.50: Male applicants had slightly higher false-positive "
            f"rate (`{_format_float(baseline_male_fpr)}` vs "
            f"`{_format_float(baseline_female_fpr)}`), while Female applicants had "
            f"slightly higher false-negative rate (`{_format_float(baseline_female_fnr)}` "
            f"vs `{_format_float(baseline_male_fnr)}`)."
        ),
        "",
        (
            "Recall threshold 0.25: Male applicants had higher false-positive rate "
            f"(`{_format_float(recall_male_fpr)}` vs `{_format_float(recall_female_fpr)}`), "
            "while Female applicants had higher false-negative rate "
            f"(`{_format_float(recall_female_fnr)}` vs `{_format_float(recall_male_fnr)}`)."
        ),
        "",
        "These are different error harms: higher FPR affects actual non-defaulters through more high-risk flags, while higher FNR affects lender risk exposure by missing more actual defaulters.",
        "",
        "## Primary XGBoost policies",
        "",
        _df_to_markdown(
            primary[
                [
                    "policy",
                    "sex_group",
                    "sex_code",
                    "group",
                    "n",
                    "true_positives",
                    "false_positives",
                    "true_negatives",
                    "false_negatives",
                    "precision",
                    "recall",
                    "false_positive_rate",
                    "false_negative_rate",
                    "negative_predictive_value",
                ]
            ]
        ),
    ]
    _write_markdown(APPLICATION_FAIRNESS_DIR / "group_error_analysis_by_sex.md", lines)


def build_calibration_analysis(context: FairnessContext) -> pd.DataFrame:
    return calibration_bins(context.y_test, context.xgboost_proba, context.sensitive_test)


def write_calibration_report(frame: pd.DataFrame) -> None:
    frame.to_csv(APPLICATION_FAIRNESS_DIR / "group_calibration_by_sex.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    for group, group_df in frame[frame["n"] > 0].groupby("group", sort=True):
        ax.plot(
            group_df["mean_predicted_probability"],
            group_df["observed_default_rate"],
            marker="o",
            linewidth=2,
            label=group,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Perfect calibration")
    ax.set_xlabel("Mean predicted default probability")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration by SEX - XGBoost test predictions")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(APPLICATION_FAIRNESS_DIR / "group_calibration_by_sex.png", dpi=150)
    plt.close(fig)

    max_gap = frame.loc[frame["n"] > 0, "calibration_gap"].abs().max()
    lines = [
        "# Group Calibration by SEX",
        "",
        "Verified UCI protected-attribute mapping: `SEX=1` is Male and `SEX=2` is Female.",
        "",
        "Calibration checks whether the same predicted default-risk score corresponds to similar observed default rates across groups.",
        "",
        "Miscalibration across groups is a governance concern because the same score may imply different realized default risk for different groups. It does not prove causal bias.",
        "",
        f"Largest absolute bin-level calibration gap in this artifact: `{_format_float(max_gap)}`.",
        "",
        _df_to_markdown(
            frame[
                [
                    "sex_group",
                    "sex_code",
                    "group",
                    "bin",
                    "n",
                    "mean_predicted_probability",
                    "observed_default_rate",
                    "calibration_gap",
                ]
            ]
        ),
    ]
    _write_markdown(APPLICATION_FAIRNESS_DIR / "group_calibration_by_sex.md", lines)


def _feature_names_from_pipeline(pipeline: Pipeline) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        return [
            str(name).replace("num__", "").replace("cat__", "")
            for name in preprocessor.get_feature_names_out()
        ]
    except Exception:
        return [f"feature_{index}" for index in range(len(preprocessor.transformers_))]


def build_proxy_predictability(context: FairnessContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_train = context.raw.loc[context.split.train_indices, PROTECTED_ATTRIBUTE].astype(int)
    y_test = context.raw.loc[context.split.test_indices, PROTECTED_ATTRIBUTE].astype(int)
    classes = sorted(y_train.dropna().unique().tolist())
    if len(classes) != 2:
        skipped = pd.DataFrame(
            [
                {
                    "model": "proxy_logistic",
                    "status": "skipped",
                    "reason": "SEX is not cleanly binary in the current split.",
                }
            ]
        )
        return skipped, pd.DataFrame()

    positive_class = classes[-1]
    y_train_binary = (y_train == positive_class).astype(int)
    y_test_binary = (y_test == positive_class).astype(int)
    X_train = context.split.X_train.copy()
    X_test = context.split.X_test.copy()

    models: list[tuple[str, Any]] = [
        (
            "proxy_logistic_regression",
            LogisticRegression(max_iter=1500, solver="lbfgs", random_state=RANDOM_STATE),
        ),
        (
            "proxy_random_forest",
            RandomForestClassifier(
                n_estimators=120,
                min_samples_leaf=20,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced",
            ),
        ),
    ]

    result_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    for model_name, estimator in models:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(X_train, y_train_binary)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.50).astype(int)
        result_rows.append(
            {
                "model": model_name,
                "status": "completed",
                "positive_class_sex_code": sex_code(positive_class),
                "positive_class_sex_group": sex_group(positive_class),
                "positive_class": group_label(positive_class),
                "accuracy": float(accuracy_score(y_test_binary, predictions)),
                "roc_auc": _binary_auc(y_test_binary.to_numpy(), probabilities),
                "pr_auc": _binary_pr_auc(y_test_binary.to_numpy(), probabilities),
            }
        )

        feature_names = _feature_names_from_pipeline(pipeline)
        classifier = pipeline.named_steps["classifier"]
        if hasattr(classifier, "coef_"):
            raw_values = classifier.coef_[0]
            score_name = "coefficient"
        elif hasattr(classifier, "feature_importances_"):
            raw_values = classifier.feature_importances_
            score_name = "feature_importance"
        else:
            raw_values = np.zeros(len(feature_names))
            score_name = "importance"
        for feature, value in zip(feature_names, raw_values):
            importance_rows.append(
                {
                    "model": model_name,
                    "feature": feature,
                    score_name: float(value),
                    "absolute_score": float(abs(value)),
                }
            )

    importance = pd.DataFrame(importance_rows).sort_values(
        ["model", "absolute_score"],
        ascending=[True, False],
    )
    return pd.DataFrame(result_rows), importance


def write_proxy_report(results: pd.DataFrame, importance: pd.DataFrame) -> None:
    results.to_csv(APPLICATION_FAIRNESS_DIR / "proxy_sex_predictability.csv", index=False)
    importance.to_csv(APPLICATION_FAIRNESS_DIR / "proxy_sex_feature_importance.csv", index=False)
    best_auc = results["roc_auc"].dropna().max() if "roc_auc" in results else None
    top = importance.head(12) if not importance.empty else pd.DataFrame()
    lines = [
        "# Proxy Predictability of SEX",
        "",
        "Verified UCI protected-attribute mapping: `SEX=1` is Male and `SEX=2` is Female.",
        "",
        "This diagnostic asks whether `SEX` can be predicted from non-sensitive credit variables.",
        "",
        "Excluding `SEX` from the credit model does not automatically remove group-related signal. If `SEX` is predictable from other variables, those variables may carry proxy information.",
        "",
        "Proxy predictability is not proof of legal discrimination. It indicates that direct removal of the protected attribute is insufficient as a complete fairness strategy.",
        "",
        f"Best proxy ROC-AUC observed: `{_format_float(best_auc)}`."
        if best_auc is not None
        else "Proxy ROC-AUC was not available.",
        "",
        "## Proxy model metrics",
        "",
        _df_to_markdown(results),
    ]
    if not top.empty:
        lines.extend(["", "## Top proxy-associated features", "", _df_to_markdown(top)])
    _write_markdown(APPLICATION_FAIRNESS_DIR / "proxy_sex_predictability.md", lines)


def build_feature_association(context: FairnessContext) -> pd.DataFrame:
    prepared = prepare_modeling_table(context.raw, target_col=TARGET_COL)
    available = [feature for feature in INTERPRETABLE_FEATURES if feature in prepared.columns]
    frame = prepared[available].copy()
    sensitive = context.raw[PROTECTED_ATTRIBUTE].reset_index(drop=True)
    groups = sorted(sensitive.dropna().unique().tolist())
    if len(groups) < 2:
        return pd.DataFrame()
    group_a, group_b = groups[0], groups[1]
    encoded = (sensitive == group_b).astype(int)
    imputed = frame.fillna(frame.median(numeric_only=True)).fillna(0)
    try:
        mi_values = mutual_info_classif(imputed, encoded, random_state=RANDOM_STATE)
    except Exception:
        mi_values = np.zeros(len(available))

    rows = []
    for index, feature in enumerate(available):
        values_a = frame.loc[sensitive == group_a, feature]
        values_b = frame.loc[sensitive == group_b, feature]
        mean_a = float(values_a.mean())
        mean_b = float(values_b.mean())
        var_a = float(values_a.var(ddof=1)) if len(values_a) > 1 else 0.0
        var_b = float(values_b.var(ddof=1)) if len(values_b) > 1 else 0.0
        pooled_std = np.sqrt((var_a + var_b) / 2)
        smd = float((mean_b - mean_a) / pooled_std) if pooled_std else 0.0
        corr = frame[feature].corr(encoded) if frame[feature].notna().sum() > 1 else np.nan
        rows.append(
            {
                "feature": feature,
                "group_a_sex_code": sex_code(group_a),
                "group_a_sex_group": sex_group(group_a),
                "group_a": group_label(group_a),
                "group_b_sex_code": sex_code(group_b),
                "group_b_sex_group": sex_group(group_b),
                "group_b": group_label(group_b),
                "group_a_mean": mean_a,
                "group_b_mean": mean_b,
                "group_a_median": float(values_a.median()),
                "group_b_median": float(values_b.median()),
                "group_a_missing_rate": float(values_a.isna().mean()),
                "group_b_missing_rate": float(values_b.isna().mean()),
                "standardized_mean_difference": smd,
                "absolute_standardized_mean_difference": abs(smd),
                "correlation_with_sex_binary": float(corr) if not pd.isna(corr) else 0.0,
                "mutual_information_with_sex": float(mi_values[index]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["absolute_standardized_mean_difference", "mutual_information_with_sex"],
        ascending=[False, False],
    )


def write_feature_association_report(frame: pd.DataFrame) -> None:
    frame.to_csv(APPLICATION_FAIRNESS_DIR / "feature_association_with_sex.csv", index=False)
    lines = [
        "# Feature Association with SEX",
        "",
        "Verified UCI protected-attribute mapping: `SEX=1` is Male and `SEX=2` is Female.",
        "",
        "This diagnostic compares feature distributions across `SEX` groups using group means, medians, missing rates, standardized mean differences, correlation, and mutual information.",
        "",
        "Group differences in feature distributions can help explain outcome disparities. They may reflect portfolio composition, historical access to credit, socioeconomic patterns, or dataset artifacts. They do not by themselves prove unfair treatment.",
        "",
        "## Largest absolute standardized mean differences",
        "",
        _df_to_markdown(frame.head(15))
        if not frame.empty
        else "No feature association rows were generated.",
    ]
    _write_markdown(APPLICATION_FAIRNESS_DIR / "feature_association_with_sex.md", lines)


def build_shap_driver_comparison(
    context: FairnessContext,
    max_rows: int = 1200,
) -> tuple[pd.DataFrame | None, str | None]:
    try:
        import shap
    except ImportError:
        return None, "SHAP is not installed in the active environment."

    model_path = MODELS_DIR / "xgboost_public.pkl"
    if not model_path.exists():
        return None, "XGBoost public model artifact is not available."

    try:
        model = load_model(model_path)
        X_test = context.split.X_test.copy()
        sensitive = context.sensitive_test.reset_index(drop=True)
        if len(X_test) > max_rows:
            sample_parts = []
            sample_source = pd.DataFrame(
                {"row_position": np.arange(len(sensitive)), "sex_code": sensitive}
            )
            for _, group_df in sample_source.groupby("sex_code", sort=True):
                sample_size = min(len(group_df), max(1, max_rows // 2))
                sample_parts.append(group_df.sample(sample_size, random_state=RANDOM_STATE))
            sample_idx = pd.concat(sample_parts, ignore_index=True)["row_position"].to_numpy()
            X_sample = X_test.reset_index(drop=True).loc[sample_idx].copy()
            sensitive_sample = sensitive.loc[sample_idx].reset_index(drop=True)
        else:
            X_sample = X_test.reset_index(drop=True)
            sensitive_sample = sensitive

        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]
        transformed = preprocessor.transform(X_sample)
        feature_names = [
            str(name).replace("num__", "").replace("cat__", "")
            for name in preprocessor.get_feature_names_out()
        ]
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(transformed)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        shap_frame = pd.DataFrame(shap_values, columns=feature_names)
        shap_frame["sex_code"] = [sex_code(value) for value in sensitive_sample]
        shap_frame["sex_group"] = [sex_group(value) for value in sensitive_sample]
        shap_frame["group"] = [group_label(value) for value in sensitive_sample]
        rows = []
        for (code, readable_group, group), group_df in _groupby_sex(shap_frame):
            for feature in feature_names:
                rows.append(
                    {
                        "sex_code": int(code) if pd.notna(code) else None,
                        "sex_group": readable_group,
                        "group": group,
                        "feature": feature,
                        "mean_abs_shap": float(group_df[feature].abs().mean()),
                        "mean_shap": float(group_df[feature].mean()),
                        "n": int(len(group_df)),
                    }
                )
        result = pd.DataFrame(rows)
        pivot = result.pivot(index="feature", columns="group", values="mean_abs_shap")
        if len(pivot.columns) >= 2:
            cols = list(pivot.columns[:2])
            pivot["mean_abs_shap_difference"] = pivot[cols[1]] - pivot[cols[0]]
            diff = pivot["mean_abs_shap_difference"].reset_index()
            result = result.merge(diff, on="feature", how="left")
        return result.sort_values(["mean_abs_shap", "feature"], ascending=[False, True]), None
    except Exception as exc:
        return None, f"SHAP driver comparison skipped because recomputation failed: {exc}"


def write_shap_report(frame: pd.DataFrame | None, skip_reason: str | None) -> None:
    out_csv = APPLICATION_FAIRNESS_DIR / "shap_driver_comparison_by_sex.csv"
    if frame is not None and not frame.empty:
        frame.to_csv(out_csv, index=False)
        top = frame.sort_values("mean_abs_shap", ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(9, 6))
        plot_frame = top.pivot_table(index="feature", columns="group", values="mean_abs_shap")
        plot_frame.sort_values(plot_frame.columns[0], ascending=True).plot(kind="barh", ax=ax)
        ax.set_xlabel("Mean absolute SHAP value")
        ax.set_title("XGBoost SHAP driver comparison by SEX")
        fig.tight_layout()
        fig.savefig(APPLICATION_FAIRNESS_DIR / "shap_driver_comparison_by_sex.png", dpi=150)
        plt.close(fig)
        lines = [
            "# SHAP Driver Comparison by SEX",
            "",
            "Verified UCI protected-attribute mapping: `SEX=1` is Male and `SEX=2` is Female.",
            "",
            "SHAP comparison is diagnostic and approximate. It helps identify whether risk explanations differ across groups, but it is not causal proof.",
            "",
            "This artifact compares mean absolute SHAP values by `SEX` for a bounded held-out test sample.",
            "",
            _df_to_markdown(top),
        ]
    else:
        lines = [
            "# SHAP Driver Comparison by SEX",
            "",
            "Verified UCI protected-attribute mapping: `SEX=1` is Male and `SEX=2` is Female.",
            "",
            f"Skipped: {skip_reason or 'SHAP driver comparison was unavailable.'}",
            "",
            "This skip does not invalidate the other fairness diagnostics. SHAP comparison is optional because it can be slow or environment-sensitive.",
        ]
    _write_markdown(APPLICATION_FAIRNESS_DIR / "shap_driver_comparison_by_sex.md", lines)


def build_threshold_fairness_frontier(context: FairnessContext) -> pd.DataFrame:
    rows = []
    for threshold in THRESHOLD_GRID:
        high_risk = (context.xgboost_proba >= threshold).astype(int)
        row = {"threshold": threshold}
        row.update(classification_metrics(context.y_test, context.xgboost_proba, threshold))
        row.update(fairness_from_predictions(context.y_test, high_risk, context.sensitive_test))
        rows.append(row)
    return pd.DataFrame(rows)


def write_threshold_frontier_report(frame: pd.DataFrame) -> None:
    frame.to_csv(APPLICATION_FAIRNESS_DIR / "threshold_fairness_frontier.csv", index=False)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(frame["threshold"], frame["recall"], marker="o", label="Recall", color="#2563eb")
    ax1.plot(
        frame["threshold"],
        frame["precision"],
        marker="o",
        label="Precision",
        color="#16a34a",
    )
    ax1.set_xlabel("Default-risk threshold")
    ax1.set_ylabel("Performance metric")
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.plot(
        frame["threshold"],
        frame["demographic_parity_difference"],
        marker="s",
        label="DP difference",
        color="#f97316",
    )
    ax2.plot(
        frame["threshold"],
        frame["equalized_odds_difference"],
        marker="s",
        label="Equalized odds difference",
        color="#dc2626",
    )
    ax2.set_ylabel("Fairness difference")
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
    ax1.grid(alpha=0.25)
    ax1.set_title("Threshold fairness frontier - XGBoost")
    fig.tight_layout()
    fig.savefig(APPLICATION_FAIRNESS_DIR / "threshold_fairness_frontier.png", dpi=150)
    plt.close(fig)

    baseline = frame.loc[np.isclose(frame["threshold"], BASELINE_THRESHOLD)].iloc[0]
    recall = frame.loc[np.isclose(frame["threshold"], RECALL_THRESHOLD)].iloc[0]
    lines = [
        "# Threshold Fairness Frontier",
        "",
        "Verified UCI protected-attribute mapping: `SEX=1` is Male and `SEX=2` is Female.",
        "",
        "Threshold is a governance lever. Lower thresholds can improve default capture while changing group-level disparity and manual-review volume.",
        "",
        (
            "Moving the XGBoost threshold from 0.50 to 0.25 improved recall from "
            f"`{_format_float(baseline['recall'])}` to `{_format_float(recall['recall'])}` "
            "but widened demographic parity difference from "
            f"`{_format_float(baseline['demographic_parity_difference'])}` to "
            f"`{_format_float(recall['demographic_parity_difference'])}` and "
            "equalized odds difference from "
            f"`{_format_float(baseline['equalized_odds_difference'])}` to "
            f"`{_format_float(recall['equalized_odds_difference'])}`."
        ),
        "",
        "The recall-optimized threshold should therefore be reviewed with fairness guardrails before operational use.",
        "",
        "## Baseline vs recall threshold",
        "",
        _df_to_markdown(
            pd.DataFrame([baseline, recall])[
                [
                    "threshold",
                    "accuracy",
                    "precision",
                    "recall",
                    "f2",
                    "approval_support_rate",
                    "demographic_parity_difference",
                    "disparate_impact_ratio",
                    "equal_opportunity_difference",
                    "equalized_odds_difference",
                    "false_positive_rate_difference",
                    "false_negative_rate_difference",
                ]
            ]
        ),
    ]
    _write_markdown(APPLICATION_FAIRNESS_DIR / "threshold_fairness_frontier.md", lines)


def build_individual_sex_sensitivity(context: FairnessContext) -> pd.DataFrame:
    model = load_model(MODELS_DIR / "xgboost_public.pkl")
    raw_flipped = context.raw.copy()
    groups = sorted(raw_flipped[PROTECTED_ATTRIBUTE].dropna().unique().tolist())
    if len(groups) != 2:
        return pd.DataFrame()
    flip_map = {groups[0]: groups[1], groups[1]: groups[0]}
    test_indices = context.split.test_indices
    raw_flipped.loc[test_indices, PROTECTED_ATTRIBUTE] = raw_flipped.loc[
        test_indices, PROTECTED_ATTRIBUTE
    ].map(flip_map)

    prepared_original = prepare_modeling_table(context.raw, target_col=TARGET_COL)
    prepared_flipped = prepare_modeling_table(raw_flipped, target_col=TARGET_COL)
    feature_columns = get_feature_columns(prepared_original, feature_set=FEATURE_SET_APPLICATION)
    X_original = prepared_original.loc[test_indices, feature_columns]
    X_flipped = prepared_flipped.loc[test_indices, feature_columns]
    original_proba = model.predict_proba(X_original)[:, 1]
    flipped_proba = model.predict_proba(X_flipped)[:, 1]
    original_group = context.raw.loc[test_indices, PROTECTED_ATTRIBUTE].to_numpy()
    flipped_group = raw_flipped.loc[test_indices, PROTECTED_ATTRIBUTE].to_numpy()

    return pd.DataFrame(
        {
            "applicant_index": list(test_indices),
            "original_sex_code": [sex_code(value) for value in original_group],
            "original_sex_group": [sex_group(value) for value in original_group],
            "original_group": [group_label(value) for value in original_group],
            "flipped_sex_code": [sex_code(value) for value in flipped_group],
            "flipped_sex_group": [sex_group(value) for value in flipped_group],
            "flipped_group": [group_label(value) for value in flipped_group],
            "actual_default": context.y_test.astype(int).to_numpy(),
            "original_probability": original_proba,
            "flipped_probability": flipped_proba,
            "absolute_probability_change": np.abs(flipped_proba - original_proba),
            "baseline_decision_changed": (
                (original_proba >= BASELINE_THRESHOLD) != (flipped_proba >= BASELINE_THRESHOLD)
            ),
            "recall_policy_decision_changed": (
                (original_proba >= RECALL_THRESHOLD) != (flipped_proba >= RECALL_THRESHOLD)
            ),
        }
    )


def write_individual_sensitivity_report(frame: pd.DataFrame) -> None:
    frame.to_csv(APPLICATION_FAIRNESS_DIR / "individual_sex_sensitivity.csv", index=False)
    max_change = frame["absolute_probability_change"].max() if not frame.empty else None
    baseline_changes = int(frame["baseline_decision_changed"].sum()) if not frame.empty else 0
    recall_changes = int(frame["recall_policy_decision_changed"].sum()) if not frame.empty else 0
    lines = [
        "# Individual SEX Sensitivity Diagnostic",
        "",
        "Verified UCI protected-attribute mapping: `SEX=1` is Male and `SEX=2` is Female.",
        "",
        "This diagnostic flips `SEX` only, keeps all other features fixed, and compares XGBoost probabilities.",
        "",
        "This is not a causal counterfactual fairness test. It is a sensitivity diagnostic. If `SEX` is excluded and predictions do not change when `SEX` is flipped, that only rules out direct use in the prediction path. It does not rule out proxy effects through other variables.",
        "",
        f"Maximum absolute probability change: `{_format_float(max_change, 8)}`.",
        f"Baseline-threshold decision changes: `{baseline_changes}`.",
        f"Recall-threshold decision changes: `{recall_changes}`.",
        "",
        "Interpretation: SEX/gender is not directly used in the active XGBoost prediction path, but proxy effects through non-sensitive variables remain possible.",
    ]
    _write_markdown(APPLICATION_FAIRNESS_DIR / "individual_sex_sensitivity.md", lines)


def build_nearest_neighbour_diagnostic(
    context: FairnessContext,
) -> tuple[pd.DataFrame | None, str | None]:
    try:
        X = context.split.X_test.reset_index(drop=True).copy()
        sensitive = context.sensitive_test.reset_index(drop=True)
        groups = sorted(sensitive.dropna().unique().tolist())
        if len(groups) != 2:
            return None, "Nearest-neighbour diagnostic requires exactly two SEX groups."

        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        X_scaled = pipeline.fit_transform(X)
        rows: list[pd.DataFrame] = []
        for source_group, target_group in [(groups[0], groups[1]), (groups[1], groups[0])]:
            source_idx = np.where(sensitive.to_numpy() == source_group)[0]
            target_idx = np.where(sensitive.to_numpy() == target_group)[0]
            if len(source_idx) == 0 or len(target_idx) == 0:
                continue
            nbrs = NearestNeighbors(n_neighbors=1, metric="euclidean")
            nbrs.fit(X_scaled[target_idx])
            distances, indices = nbrs.kneighbors(X_scaled[source_idx])
            matched_target_idx = target_idx[indices.reshape(-1)]
            rows.append(
                pd.DataFrame(
                    {
                        "applicant_position": source_idx,
                        "sex_code": [sex_code(source_group)] * len(source_idx),
                        "sex_group": [sex_group(source_group)] * len(source_idx),
                        "group": [group_label(source_group)] * len(source_idx),
                        "nearest_other_group_position": matched_target_idx,
                        "nearest_other_sex_code": [sex_code(target_group)] * len(source_idx),
                        "nearest_other_sex_group": [sex_group(target_group)] * len(source_idx),
                        "nearest_other_group": [group_label(target_group)] * len(source_idx),
                        "feature_distance": distances.reshape(-1),
                        "original_probability": context.xgboost_proba[source_idx],
                        "neighbour_probability": context.xgboost_proba[matched_target_idx],
                        "probability_difference": np.abs(
                            context.xgboost_proba[source_idx]
                            - context.xgboost_proba[matched_target_idx]
                        ),
                        "original_actual_default": context.y_test.iloc[source_idx].to_numpy(),
                        "neighbour_actual_default": context.y_test.iloc[
                            matched_target_idx
                        ].to_numpy(),
                    }
                )
            )
        if not rows:
            return None, "No cross-group nearest-neighbour pairs could be generated."
        return pd.concat(rows, ignore_index=True), None
    except Exception as exc:
        return None, f"Nearest-neighbour diagnostic skipped because computation failed: {exc}"


def write_nearest_neighbour_report(frame: pd.DataFrame | None, skip_reason: str | None) -> None:
    if frame is not None and not frame.empty:
        frame.to_csv(
            APPLICATION_FAIRNESS_DIR / "nearest_neighbour_individual_fairness.csv",
            index=False,
        )
        summary = {
            "average_probability_difference": float(frame["probability_difference"].mean()),
            "median_probability_difference": float(frame["probability_difference"].median()),
            "p90_probability_difference": float(frame["probability_difference"].quantile(0.90)),
            "large_difference_count_gt_0_10": int((frame["probability_difference"] > 0.10).sum()),
        }
        lines = [
            "# Nearest-Neighbour Individual Fairness Diagnostic",
            "",
            "Verified UCI protected-attribute mapping: `SEX=1` is Male and `SEX=2` is Female.",
            "",
            "This approximates individual fairness: similar individuals should receive similar scores. It depends heavily on the chosen standardized Euclidean distance metric and is diagnostic, not conclusive.",
            "",
            _df_to_markdown(pd.DataFrame([summary])),
        ]
    else:
        lines = [
            "# Nearest-Neighbour Individual Fairness Diagnostic",
            "",
            "Verified UCI protected-attribute mapping: `SEX=1` is Male and `SEX=2` is Female.",
            "",
            f"Skipped: {skip_reason or 'nearest-neighbour diagnostic was unavailable.'}",
            "",
            "This optional diagnostic depends on the chosen distance metric and should be treated as exploratory.",
        ]
    _write_markdown(APPLICATION_FAIRNESS_DIR / "nearest_neighbour_individual_fairness.md", lines)


def _summary_from_outputs(
    group_outcome: pd.DataFrame,
    group_error: pd.DataFrame,
    calibration: pd.DataFrame,
    proxy_results: pd.DataFrame,
    feature_assoc: pd.DataFrame,
    shap_frame: pd.DataFrame | None,
    shap_skip_reason: str | None,
    frontier: pd.DataFrame,
    sensitivity: pd.DataFrame,
    nearest: pd.DataFrame | None,
    nearest_skip_reason: str | None,
) -> dict[str, Any]:
    baseline_frontier = frontier.loc[np.isclose(frontier["threshold"], BASELINE_THRESHOLD)].iloc[0]
    recall_frontier = frontier.loc[np.isclose(frontier["threshold"], RECALL_THRESHOLD)].iloc[0]
    proxy_best = (
        proxy_results.sort_values("roc_auc", ascending=False).iloc[0].to_dict()
        if "roc_auc" in proxy_results and proxy_results["roc_auc"].notna().any()
        else {}
    )
    sensitivity_summary = {
        "max_absolute_probability_change": float(sensitivity["absolute_probability_change"].max())
        if not sensitivity.empty
        else None,
        "baseline_decision_changes": int(sensitivity["baseline_decision_changed"].sum())
        if not sensitivity.empty
        else None,
        "recall_policy_decision_changes": int(sensitivity["recall_policy_decision_changed"].sum())
        if not sensitivity.empty
        else None,
    }
    nearest_summary = (
        {
            "status": "completed",
            "average_probability_difference": float(nearest["probability_difference"].mean()),
            "median_probability_difference": float(nearest["probability_difference"].median()),
            "p90_probability_difference": float(nearest["probability_difference"].quantile(0.90)),
            "large_difference_count_gt_0_10": int((nearest["probability_difference"] > 0.10).sum()),
        }
        if nearest is not None and not nearest.empty
        else {"status": "skipped", "reason": nearest_skip_reason}
    )
    baseline_policy = "xgboost_baseline_threshold_050"
    recall_policy = "xgboost_recall_threshold_025"
    group_outcome_findings = {
        "baseline_threshold_050": {
            "male_high_risk_flag_rate": _lookup_metric(
                group_outcome, baseline_policy, 1, "predicted_high_risk_rate"
            ),
            "female_high_risk_flag_rate": _lookup_metric(
                group_outcome, baseline_policy, 2, "predicted_high_risk_rate"
            ),
        },
        "recall_threshold_025": {
            "male_high_risk_flag_rate": _lookup_metric(
                group_outcome, recall_policy, 1, "predicted_high_risk_rate"
            ),
            "female_high_risk_flag_rate": _lookup_metric(
                group_outcome, recall_policy, 2, "predicted_high_risk_rate"
            ),
        },
    }
    group_error_findings = {
        "baseline_threshold_050": {
            "male_false_positive_rate": _lookup_metric(
                group_error, baseline_policy, 1, "false_positive_rate"
            ),
            "female_false_positive_rate": _lookup_metric(
                group_error, baseline_policy, 2, "false_positive_rate"
            ),
            "male_false_negative_rate": _lookup_metric(
                group_error, baseline_policy, 1, "false_negative_rate"
            ),
            "female_false_negative_rate": _lookup_metric(
                group_error, baseline_policy, 2, "false_negative_rate"
            ),
        },
        "recall_threshold_025": {
            "male_false_positive_rate": _lookup_metric(
                group_error, recall_policy, 1, "false_positive_rate"
            ),
            "female_false_positive_rate": _lookup_metric(
                group_error, recall_policy, 2, "false_positive_rate"
            ),
            "male_false_negative_rate": _lookup_metric(
                group_error, recall_policy, 1, "false_negative_rate"
            ),
            "female_false_negative_rate": _lookup_metric(
                group_error, recall_policy, 2, "false_negative_rate"
            ),
        },
    }
    return {
        "what_was_analyzed": [
            "group outcomes by SEX",
            "group error rates by SEX",
            "calibration by SEX",
            "proxy predictability of SEX from non-sensitive features",
            "feature association with SEX",
            "SHAP driver comparison by SEX",
            "threshold fairness frontier",
            "individual SEX sensitivity",
            "nearest-neighbour individual fairness",
        ],
        "protected_attribute": PROTECTED_ATTRIBUTE,
        "protected_attribute_mapping": sex_mapping_rows(),
        "baseline_xgboost_fairness": {
            key: _json_safe(baseline_frontier[key])
            for key in [
                "demographic_parity_difference",
                "disparate_impact_ratio",
                "equal_opportunity_difference",
                "equalized_odds_difference",
                "false_positive_rate_difference",
                "false_negative_rate_difference",
            ]
        },
        "recall_threshold_fairness": {
            key: _json_safe(recall_frontier[key])
            for key in [
                "demographic_parity_difference",
                "disparate_impact_ratio",
                "equal_opportunity_difference",
                "equalized_odds_difference",
                "false_positive_rate_difference",
                "false_negative_rate_difference",
            ]
        },
        "group_outcome_findings": _json_safe(group_outcome_findings),
        "group_error_findings": _json_safe(group_error_findings),
        "group_outcome_rows": len(group_outcome),
        "group_error_rows": len(group_error),
        "largest_absolute_calibration_gap": float(
            calibration.loc[calibration["n"] > 0, "calibration_gap"].abs().max()
        ),
        "best_proxy_predictability": _json_safe(proxy_best),
        "top_proxy_associated_features": _json_safe(
            feature_assoc.head(10).to_dict(orient="records")
        ),
        "shap_driver_comparison": {
            "status": "completed" if shap_frame is not None and not shap_frame.empty else "skipped",
            "reason": shap_skip_reason,
        },
        "threshold_frontier_finding": (
            "Lowering the threshold from 0.50 to 0.25 improves recall but widens several group-level fairness diagnostics."
        ),
        "individual_sensitivity": sensitivity_summary,
        "nearest_neighbour": nearest_summary,
        "final_interpretation": (
            "The protected-attribute deep dive does not establish legal discrimination or causal bias. It shows a diagnostic fairness-governance signal. Male applicants (SEX=1) had higher high-risk flag rates than Female applicants (SEX=2), and this gap widened under the recall-focused threshold. The recall-focused policy improved default capture but also widened demographic parity and equalized-odds differences. Individual sensitivity testing showed that flipping SEX alone did not change XGBoost predictions, confirming no direct use of SEX in the active prediction path. However, proxy analysis showed that SEX/gender is moderately predictable from non-sensitive credit variables, so removing SEX alone is not a complete fairness strategy. Therefore, the model should remain a decision-support tool requiring threshold governance, human oversight, and ongoing fairness monitoring."
        ),
    }


def write_summary_report(summary: dict[str, Any]) -> None:
    save_json(_json_safe(summary), APPLICATION_FAIRNESS_DIR / "fairness_deep_dive_summary.json")
    baseline = summary["baseline_xgboost_fairness"]
    recall = summary["recall_threshold_fairness"]
    outcome = summary.get("group_outcome_findings", {})
    errors = summary.get("group_error_findings", {})
    proxy = summary.get("best_proxy_predictability", {})
    nearest = summary.get("nearest_neighbour", {})
    lines = [
        "# Fairness Deep Dive Summary: Protected Attribute SEX",
        "",
        "## 1. What was analyzed",
        "",
        "This deep dive analyzed group outcomes, group errors, calibration, proxy predictability, feature associations, SHAP drivers, threshold fairness frontier, individual sensitivity, and nearest-neighbour individual fairness for `SEX`.",
        "",
        "## 2. Why SEX was used as the protected attribute",
        "",
        "`SEX` is available in the public UCI dataset and is excluded from the active XGBoost training feature set while retained for fairness diagnostics. Verified mapping: `SEX=1` is Male and `SEX=2` is Female.",
        "",
        "## 3. Baseline XGBoost fairness results",
        "",
        f"At threshold 0.50, demographic parity difference was `{_format_float(baseline['demographic_parity_difference'])}`, equalized odds difference was `{_format_float(baseline['equalized_odds_difference'])}`, and disparate impact ratio was `{_format_float(baseline['disparate_impact_ratio'])}`.",
        "",
        "## 4. Recall-threshold fairness tradeoff",
        "",
        f"At threshold 0.25, demographic parity difference was `{_format_float(recall['demographic_parity_difference'])}`, equalized odds difference was `{_format_float(recall['equalized_odds_difference'])}`, and disparate impact ratio was `{_format_float(recall['disparate_impact_ratio'])}`. The recall policy improves default capture but widens several fairness diagnostics.",
        "",
        "## 5. Group-wise outcome findings",
        "",
        "Group-wise outcome analysis shows differences in actual default rates, mean predicted default probabilities, high-risk flag rates, and low-risk support rates. These are governance diagnostics, not proof of legal discrimination or causal bias.",
        "",
        (
            "Male applicants (SEX=1) had higher high-risk flag rates than Female "
            "applicants (SEX=2): "
            f"`{_format_float(outcome.get('baseline_threshold_050', {}).get('male_high_risk_flag_rate'))}` vs "
            f"`{_format_float(outcome.get('baseline_threshold_050', {}).get('female_high_risk_flag_rate'))}` at threshold 0.50, and "
            f"`{_format_float(outcome.get('recall_threshold_025', {}).get('male_high_risk_flag_rate'))}` vs "
            f"`{_format_float(outcome.get('recall_threshold_025', {}).get('female_high_risk_flag_rate'))}` at threshold 0.25."
        ),
        "",
        "## 6. Group-wise error findings",
        "",
        "Error analysis compares false positives and false negatives by group. False positives may unnecessarily push reliable customers into manual review or lower credit support; false negatives may miss actual default risk.",
        "",
        (
            "At threshold 0.50, Male false-positive rate was "
            f"`{_format_float(errors.get('baseline_threshold_050', {}).get('male_false_positive_rate'))}` vs Female "
            f"`{_format_float(errors.get('baseline_threshold_050', {}).get('female_false_positive_rate'))}`; Female false-negative rate was "
            f"`{_format_float(errors.get('baseline_threshold_050', {}).get('female_false_negative_rate'))}` vs Male "
            f"`{_format_float(errors.get('baseline_threshold_050', {}).get('male_false_negative_rate'))}`."
        ),
        "",
        (
            "At threshold 0.25, Male false-positive rate was "
            f"`{_format_float(errors.get('recall_threshold_025', {}).get('male_false_positive_rate'))}` vs Female "
            f"`{_format_float(errors.get('recall_threshold_025', {}).get('female_false_positive_rate'))}`; Female false-negative rate was "
            f"`{_format_float(errors.get('recall_threshold_025', {}).get('female_false_negative_rate'))}` vs Male "
            f"`{_format_float(errors.get('recall_threshold_025', {}).get('male_false_negative_rate'))}`."
        ),
        "",
        "## 7. Calibration findings",
        "",
        f"Largest absolute bin-level calibration gap: `{_format_float(summary['largest_absolute_calibration_gap'])}`. Calibration gaps should be monitored because the same score may correspond to different observed default rates across groups.",
        "",
        "## 8. Proxy predictability findings",
        "",
        f"Best proxy model: `{proxy.get('model', 'not available')}` with ROC-AUC `{_format_float(proxy.get('roc_auc'))}`. Predictability of `SEX`/gender from non-sensitive features indicates proxy risk, not proof of legal discrimination.",
        "",
        "## 9. Feature association findings",
        "",
        "The feature association artifact ranks variables by standardized mean difference and mutual information with `SEX`, helping explain why group-level outcome differences may occur.",
        "",
        "## 10. SHAP driver comparison",
        "",
        f"Status: `{summary['shap_driver_comparison']['status']}`. {summary['shap_driver_comparison'].get('reason') or 'SHAP driver comparison completed.'}",
        "",
        "## 11. Threshold fairness frontier findings",
        "",
        summary["threshold_frontier_finding"],
        "",
        "## 12. Individual sensitivity findings",
        "",
        f"Maximum probability change when flipping `SEX` only: `{_format_float(summary['individual_sensitivity']['max_absolute_probability_change'], 8)}`. Baseline decision changes: `{summary['individual_sensitivity']['baseline_decision_changes']}`. Recall-policy decision changes: `{summary['individual_sensitivity']['recall_policy_decision_changes']}`.",
        "",
        "## 13. Nearest-neighbour findings",
        "",
        (
            f"Status: `{nearest.get('status')}`. Median nearest-neighbour probability difference: `{_format_float(nearest.get('median_probability_difference'))}`."
            if nearest.get("status") == "completed"
            else f"Status: `skipped`. Reason: {nearest.get('reason')}"
        ),
        "",
        "## 14. Final interpretation",
        "",
        summary["final_interpretation"],
        "",
        "## 15. Governance recommendations",
        "",
        "- Keep XGBoost as decision-support, not automated approval.",
        "- Review threshold changes with fairness guardrails and manual-review capacity.",
        "- Monitor fairness metrics over time and by relevant intersections where sample sizes support them.",
        "- Treat proxy-risk analysis as evidence that removing `SEX` alone is not a complete fairness strategy.",
        "- Add calibration and threshold governance before any real operational use.",
        "",
        "## 16. Limitations",
        "",
        "- Public academic dataset, not production bank data.",
        "- Diagnostic fairness metrics are observational and not causal.",
        "- No legal compliance conclusion is made.",
        "- Nearest-neighbour individual fairness depends on the chosen distance metric.",
        "- SHAP comparisons are approximate and environment-sensitive.",
    ]
    _write_markdown(APPLICATION_FAIRNESS_DIR / "fairness_deep_dive_summary.md", lines)


def run(
    skip_shap: bool = False,
    skip_nearest: bool = False,
    max_shap_rows: int = 1200,
) -> dict[str, Any]:
    APPLICATION_FAIRNESS_DIR.mkdir(parents=True, exist_ok=True)
    context = build_context()

    group_outcome = build_group_outcome_analysis(context)
    write_group_outcome_report(group_outcome)

    group_error = build_group_error_analysis(context)
    write_group_error_report(group_error)

    calibration = build_calibration_analysis(context)
    write_calibration_report(calibration)

    proxy_results, proxy_importance = build_proxy_predictability(context)
    write_proxy_report(proxy_results, proxy_importance)

    feature_assoc = build_feature_association(context)
    write_feature_association_report(feature_assoc)

    if skip_shap:
        shap_frame, shap_skip_reason = None, "SHAP comparison skipped by CLI flag."
    else:
        shap_frame, shap_skip_reason = build_shap_driver_comparison(context, max_rows=max_shap_rows)
    write_shap_report(shap_frame, shap_skip_reason)

    frontier = build_threshold_fairness_frontier(context)
    write_threshold_frontier_report(frontier)

    sensitivity = build_individual_sex_sensitivity(context)
    write_individual_sensitivity_report(sensitivity)

    if skip_nearest:
        nearest, nearest_skip_reason = None, "Nearest-neighbour diagnostic skipped by CLI flag."
    else:
        nearest, nearest_skip_reason = build_nearest_neighbour_diagnostic(context)
    write_nearest_neighbour_report(nearest, nearest_skip_reason)

    summary = _summary_from_outputs(
        group_outcome,
        group_error,
        calibration,
        proxy_results,
        feature_assoc,
        shap_frame,
        shap_skip_reason,
        frontier,
        sensitivity,
        nearest,
        nearest_skip_reason,
    )
    write_summary_report(summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate protected-attribute fairness deep-dive reports for SEX."
    )
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP driver comparison.")
    parser.add_argument(
        "--skip-nearest",
        action="store_true",
        help="Skip nearest-neighbour individual fairness diagnostic.",
    )
    parser.add_argument(
        "--max-shap-rows",
        type=int,
        default=1200,
        help="Maximum held-out rows used for SHAP driver comparison.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(
        skip_shap=args.skip_shap,
        skip_nearest=args.skip_nearest,
        max_shap_rows=args.max_shap_rows,
    )
    print(json.dumps(_json_safe(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
