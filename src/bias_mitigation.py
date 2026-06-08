"""Bias mitigation methods and fairness-performance tradeoff analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.base import clone

from .data_preprocessing import (
    FEATURE_SET_APPLICATION,
    FEATURE_SET_BEHAVIORAL,
    FEATURE_SET_FULL_DIAGNOSTIC,
    TARGET_COL,
    get_dataset_split,
)
from .evaluate_models import evaluate_classification
from .fairness_metrics import compute_fairness_metrics
from .utils import (
    MODELS_DIR,
    REPORTS_DIR,
    infer_protected_attribute,
    load_dataset_auto,
    load_model,
    save_json,
)


def _manual_reweighing_weights(y: pd.Series, sensitive: pd.Series) -> np.ndarray:
    """Kamiran-style reweighing fallback if AIF360 API differs across versions."""
    df = pd.DataFrame({"y": y.astype(int), "s": sensitive.astype(str)})
    n = len(df)

    p_y = df["y"].value_counts(normalize=True)
    p_s = df["s"].value_counts(normalize=True)
    p_sy = df.groupby(["s", "y"]).size() / n

    weights = np.ones(n)
    for idx, row in df.iterrows():
        sy = p_sy.loc[(row["s"], row["y"])]
        weights[idx] = (p_s.loc[row["s"]] * p_y.loc[row["y"]]) / sy if sy > 0 else 1.0
    return weights


def _aif360_reweighing_weights(y: pd.Series, sensitive: pd.Series):
    try:
        from aif360.algorithms.preprocessing import Reweighing
        from aif360.datasets import BinaryLabelDataset
    except ImportError:
        return None

    tmp = pd.DataFrame(
        {
            "label": y.reset_index(drop=True).astype(int),
            "sensitive": sensitive.reset_index(drop=True).astype(str),
        }
    )
    # Map protected groups into a numeric binary partition required by AIF360.
    groups = sorted(tmp["sensitive"].dropna().unique().tolist())
    if len(groups) < 2:
        return np.ones(len(tmp))
    privileged_group = groups[0]
    tmp["sensitive_bin"] = (tmp["sensitive"] == privileged_group).astype(int)

    dataset = BinaryLabelDataset(
        favorable_label=0,
        unfavorable_label=1,
        df=tmp[["label", "sensitive_bin"]],
        label_names=["label"],
        protected_attribute_names=["sensitive_bin"],
    )

    rw = Reweighing(
        unprivileged_groups=[{"sensitive_bin": 0}],
        privileged_groups=[{"sensitive_bin": 1}],
    )
    dataset_rw = rw.fit_transform(dataset)
    weights = np.asarray(dataset_rw.instance_weights)
    return weights


def apply_reweighing(model, X_train, y_train, sensitive_train):
    sample_weights = _aif360_reweighing_weights(y_train, sensitive_train)
    if sample_weights is None:
        sample_weights = _manual_reweighing_weights(
            y_train.reset_index(drop=True), sensitive_train.reset_index(drop=True)
        )
    weighted_model = clone(model)
    weighted_model.fit(X_train, y_train, classifier__sample_weight=sample_weights)
    return weighted_model


def apply_threshold_optimizer(model, X_train, y_train, sensitive_train):
    # Fit post-processor on probability scores from the base classifier.
    threshold_opt = ThresholdOptimizer(
        estimator=model,
        constraints="equalized_odds",
        prefit=False,
        predict_method="predict_proba",
    )
    threshold_opt.fit(X_train, y_train, sensitive_features=sensitive_train)
    return threshold_opt


def _model_context(model_path: Path) -> tuple[str, Path]:
    stem = model_path.stem
    if "behavioral" in stem:
        return FEATURE_SET_BEHAVIORAL, REPORTS_DIR / "fairness_reports" / "behavioral_model"
    if "full_diagnostic" in stem:
        return FEATURE_SET_FULL_DIAGNOSTIC, REPORTS_DIR / "fairness_reports" / "full_diagnostic"
    return FEATURE_SET_APPLICATION, REPORTS_DIR / "fairness_reports" / "application_model"


def _approval_fairness(y_true_default, y_pred_default, sensitive) -> Dict[str, float]:
    approval_true = 1 - np.asarray(y_true_default).astype(int)
    approval_pred = 1 - np.asarray(y_pred_default).astype(int)
    return compute_fairness_metrics(approval_true, approval_pred, sensitive)


def run(model_path: Path | None = None) -> Dict:
    model_path = model_path or (
        MODELS_DIR / "xgboost_application.pkl"
        if (MODELS_DIR / "xgboost_application.pkl").exists()
        else MODELS_DIR / "logistic_application.pkl"
    )
    feature_set, report_dir = _model_context(model_path)
    report_dir.mkdir(parents=True, exist_ok=True)

    df_raw, _ = load_dataset_auto()
    protected_col = infer_protected_attribute(df_raw)
    split = get_dataset_split(df_raw, target_col=TARGET_COL, feature_set=feature_set)
    X_train, X_test = split.X_train, split.X_test
    y_train, y_test = split.y_train, split.y_test
    s_train = df_raw.loc[split.train_indices, protected_col].astype(str)
    s_test = df_raw.loc[split.test_indices, protected_col].astype(str)

    base_model = load_model(model_path)

    # Baseline
    y_pred_base = base_model.predict(X_test)
    y_proba_base = base_model.predict_proba(X_test)[:, 1]
    perf_base = evaluate_classification(y_test, y_pred_base, y_proba_base)
    fair_base = _approval_fairness(y_test.values, y_pred_base, s_test.values)

    # Reweighing
    rw_model = apply_reweighing(base_model, X_train, y_train, s_train)
    y_pred_rw = rw_model.predict(X_test)
    y_proba_rw = rw_model.predict_proba(X_test)[:, 1]
    perf_rw = evaluate_classification(y_test, y_pred_rw, y_proba_rw)
    fair_rw = _approval_fairness(y_test.values, y_pred_rw, s_test.values)

    # Post-processing with Fairlearn
    postproc = apply_threshold_optimizer(base_model, X_train, y_train, s_train)
    y_pred_post = postproc.predict(X_test, sensitive_features=s_test)
    y_proba_post = y_pred_post.astype(float)
    perf_post = evaluate_classification(y_test, y_pred_post, y_proba_post)
    fair_post = _approval_fairness(y_test.values, y_pred_post, s_test.values)

    summary = {
        "model": str(model_path),
        "protected_attribute": protected_col,
        "baseline": {"performance": perf_base, "fairness": fair_base},
        "reweighing": {"performance": perf_rw, "fairness": fair_rw},
        "fairlearn_postprocessing": {"performance": perf_post, "fairness": fair_post},
    }

    save_json(
        summary,
        report_dir / f"{model_path.stem}_bias_mitigation_summary.json",
    )

    rows = []
    for method, block in summary.items():
        if method in {"model", "protected_attribute"}:
            continue
        row = {"method": method}
        row.update({f"perf_{k}": v for k, v in block["performance"].items()})
        row.update({f"fair_{k}": v for k, v in block["fairness"].items()})
        rows.append(row)

    pd.DataFrame(rows).to_csv(
        report_dir / f"{model_path.stem}_fairness_accuracy_tradeoff.csv",
        index=False,
    )

    return summary


if __name__ == "__main__":
    result = run()
    print("Bias mitigation analysis completed.")
    print(result)
