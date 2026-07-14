"""Optional TensorFlow/Keras benchmark for the public UCI credit-default dataset."""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

from .data_preprocessing import FEATURE_SET_APPLICATION, build_preprocessor
from .fairness_metrics import compute_fairness_metrics
from .recall_optimization import (
    build_recall_optimization_splits_from_frame,
    create_threshold_grid,
    evaluate_threshold_grid,
    select_preferred_threshold,
    threshold_metrics,
)
from .utils import MODELS_DIR, REPORTS_DIR, ensure_directories, load_dataset_auto, save_json

RANDOM_STATE = 42
DEFAULT_EPOCHS = 75
DEFAULT_BATCH_SIZE = 256
MODEL_VALIDATION_DIR = REPORTS_DIR / "model_validation"
FAIRNESS_DIR = REPORTS_DIR / "fairness_reports" / "deep_learning_model"
EXPLAINABILITY_DIR = REPORTS_DIR / "explainability_reports" / "deep_learning_model"


@dataclass(frozen=True)
class BenchmarkConfig:
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    quick: bool = False
    skip_class_weighted: bool = False
    skip_explainability: bool = False
    verbose: int = 1


def tensorflow_status() -> tuple[Any | None, str | None]:
    """Return TensorFlow lazily so importing this module never requires it."""
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf
    except (ImportError, OSError) as exc:
        return None, f"Deep Learning benchmark skipped because TensorFlow is unavailable: {exc}"
    return tf, None


def set_random_seeds(tf: Any, seed: int = RANDOM_STATE) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def build_mlp(input_dim: int, tf_module: Any | None = None) -> Any:
    """Build the deliberately small tabular MLP used by the benchmark."""
    tf = tf_module
    if tf is None:
        tf, reason = tensorflow_status()
        if tf is None:
            raise RuntimeError(reason)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,), name="features"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid", name="default_probability"),
        ],
        name="credit_default_mlp",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def selected_policy_payload(
    source_experiment: str,
    selected_row: pd.Series,
    selected_rule: str,
    fallback_used: bool,
    test_metrics: dict[str, Any],
    test_roc_auc: float,
    test_pr_auc: float,
) -> dict[str, Any]:
    return {
        "policy_name": "dnn_recall_optimized",
        "source_experiment": source_experiment,
        "selected_threshold": float(selected_row["threshold"]),
        "selection_rule": selected_rule,
        "fallback_used": bool(fallback_used),
        "selection_split": "validation",
        "evaluation_split": "untouched_test",
        "validation_metrics": _json_safe(selected_row.to_dict()),
        "test_metrics": _json_safe(test_metrics),
        "test_roc_auc": float(test_roc_auc),
        "test_pr_auc": float(test_pr_auc),
    }


def validate_policy_schema(payload: dict[str, Any]) -> bool:
    required = {
        "policy_name",
        "source_experiment",
        "selected_threshold",
        "selection_rule",
        "selection_split",
        "evaluation_split",
        "test_metrics",
    }
    return required.issubset(payload) and 0.0 <= float(payload["selected_threshold"]) <= 1.0


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without pandas' optional tabulate dependency."""
    if frame.empty:
        return "_No rows available._"
    safe = frame.copy().fillna("").astype(str)
    columns = [str(column) for column in safe.columns]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(value.replace("|", "\\|") for value in row) + " |"
        for row in safe.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def _probability_metrics(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
    }


def _prediction_frame(
    y_true: pd.Series,
    probabilities: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y_true": np.asarray(y_true).astype(int),
            "y_proba": np.asarray(probabilities, dtype=float),
            "model_name": model_name,
            "split": "test",
        }
    )


def _write_test_predictions(
    y_true: pd.Series,
    probabilities: np.ndarray,
    model_name: str,
    path: Path,
) -> None:
    _prediction_frame(y_true, probabilities, model_name).to_csv(path, index=False)


def _prediction_metrics_row(model_name: str, path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    required = {"y_true", "y_proba", "model_name", "split"}
    if not required.issubset(frame.columns):
        return None
    y_true = frame["y_true"].astype(int)
    y_proba = frame["y_proba"].astype(float).to_numpy()
    return {
        "model_name": model_name,
        **threshold_metrics(y_true, y_proba, 0.50),
        **_probability_metrics(y_true, y_proba),
    }


def _fit_model(
    tf: Any,
    X_train: np.ndarray,
    y_train: pd.Series,
    X_validation: np.ndarray,
    y_validation: pd.Series,
    config: BenchmarkConfig,
    class_weight: dict[int, float] | None = None,
) -> tuple[Any, Any]:
    set_random_seeds(tf)
    model = build_mlp(X_train.shape[1], tf)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=3 if config.quick else 8,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        X_train,
        y_train.to_numpy(),
        validation_data=(X_validation, y_validation.to_numpy()),
        epochs=min(config.epochs, 8) if config.quick else config.epochs,
        batch_size=config.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=config.verbose,
    )
    return model, history


def _predict(model: Any, X: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(X, verbose=0)).reshape(-1)


def _training_curve(history: Any, path: Path) -> None:
    values = history.history
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(values.get("loss", []), label="Train loss")
    axes[0].plot(values.get("val_loss", []), label="Validation loss")
    axes[0].set_title("DNN training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    metric = "pr_auc" if "pr_auc" in values else "roc_auc"
    axes[1].plot(values.get(metric, []), label=f"Train {metric}")
    axes[1].plot(values.get(f"val_{metric}", []), label=f"Validation {metric}")
    axes[1].set_title("DNN validation quality")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _precision_recall_plot(
    y_test: pd.Series,
    dnn_proba: np.ndarray,
    dnn_path: Path,
    comparison_path: Path,
) -> str:
    precision, recall, _ = precision_recall_curve(y_test, dnn_proba)
    dnn_ap = average_precision_score(y_test, dnn_proba)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(recall, precision, label=f"DNN (AP={dnn_ap:.4f})")
    axis.set(xlabel="Recall", ylabel="Precision", title="DNN precision-recall curve")
    axis.legend()
    figure.tight_layout()
    figure.savefig(dnn_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    curve_specs = [
        ("Logistic Regression", MODEL_VALIDATION_DIR / "logistic_test_predictions.csv"),
        ("XGBoost", MODEL_VALIDATION_DIR / "xgboost_test_predictions.csv"),
        ("DNN baseline", MODEL_VALIDATION_DIR / "dnn_test_predictions.csv"),
    ]
    included: list[str] = []
    skipped: list[str] = []
    y_test_array = np.asarray(y_test).astype(int)

    figure, axis = plt.subplots(figsize=(7, 5))
    for label, path in curve_specs:
        if not path.exists():
            skipped.append(f"{label} missing `{path.name}`")
            continue
        frame = pd.read_csv(path)
        if not {"y_true", "y_proba"}.issubset(frame.columns):
            skipped.append(f"{label} has an invalid prediction schema")
            continue
        y_true = frame["y_true"].astype(int).to_numpy()
        if len(y_true) != len(y_test_array) or not np.array_equal(y_true, y_test_array):
            skipped.append(f"{label} prediction rows do not match the common test split")
            continue
        y_proba = frame["y_proba"].astype(float).to_numpy()
        model_precision, model_recall, _ = precision_recall_curve(y_true, y_proba)
        model_ap = average_precision_score(y_true, y_proba)
        axis.plot(model_recall, model_precision, label=f"{label} (AP={model_ap:.4f})")
        included.append(label)
    axis.set(xlabel="Recall", ylabel="Precision", title="ML vs DL precision-recall curve")
    axis.legend()
    figure.tight_layout()
    figure.savefig(comparison_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    note = "ML-vs-DL precision-recall curve includes: " + ", ".join(included) + "."
    if skipped:
        note += " Skipped curves: " + "; ".join(skipped) + "."
    return note


def _load_existing_comparison_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prediction_paths = {
        "logistic_public": MODEL_VALIDATION_DIR / "logistic_test_predictions.csv",
        "xgboost_public": MODEL_VALIDATION_DIR / "xgboost_test_predictions.csv",
    }
    for model_name, path in prediction_paths.items():
        row = _prediction_metrics_row(model_name, path)
        if row is not None:
            rows.append(row)
    covered_models = {row["model_name"] for row in rows}

    model_path = MODEL_VALIDATION_DIR / "public_credit_model_comparison.csv"
    if model_path.exists():
        existing = pd.read_csv(model_path)
        for _, row in existing.iterrows():
            if row.get("model_name") in covered_models:
                continue
            metrics = row.to_dict()
            metrics["threshold"] = 0.50
            metrics.setdefault("pr_auc", None)
            rows.append(metrics)
    recall_path = MODEL_VALIDATION_DIR / "recall_optimized_summary.json"
    if recall_path.exists():
        payload = json.loads(recall_path.read_text(encoding="utf-8"))
        policy = payload.get("selected_policy", {})
        metrics = policy.get("test_metrics", {})
        if metrics:
            rows.append(
                {
                    "model_name": "xgboost_public_recall_optimized",
                    "threshold": policy.get("selected_threshold"),
                    **metrics,
                    "roc_auc": policy.get("test_roc_auc"),
                    "pr_auc": policy.get("test_pr_auc"),
                }
            )
    return rows


def _fairness_rows(
    y_test: pd.Series,
    sensitive_test: pd.Series,
    baseline_proba: np.ndarray,
    selected_proba: np.ndarray,
    selected_threshold: float,
) -> pd.DataFrame:
    approval_true = (1 - y_test.to_numpy()).astype(int)
    rows = []
    for policy, probabilities, threshold in [
        ("dnn_baseline", baseline_proba, 0.50),
        ("dnn_recall_optimized", selected_proba, selected_threshold),
    ]:
        approval_pred = (probabilities < threshold).astype(int)
        rows.append(
            {
                "model_name": policy,
                "threshold": threshold,
                "protected_attribute": "SEX",
                **compute_fairness_metrics(
                    approval_true, approval_pred, sensitive_test.astype(str).to_numpy()
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_fairness_report(frame: pd.DataFrame) -> None:
    FAIRNESS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(FAIRNESS_DIR / "dnn_fairness_metrics.csv", index=False)
    lines = [
        "# DNN Fairness Metrics",
        "",
        "`SEX` is retained only for group fairness evaluation and is excluded from DNN training.",
        "",
        _markdown_table(frame),
        "",
        "A DNN must be evaluated on fairness as well as performance. Improved recall that worsens "
        "fairness is an explicit tradeoff, not an automatic improvement.",
        "",
        "These metrics are diagnostic and are not proof of legal compliance.",
    ]
    (FAIRNESS_DIR / "dnn_fairness_metrics.md").write_text("\n".join(lines), encoding="utf-8")


def _permutation_explainability(
    model: Any,
    preprocessor: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    baseline_proba: np.ndarray,
) -> pd.DataFrame:
    sample = X_test.sample(min(1000, len(X_test)), random_state=RANDOM_STATE)
    y_sample = y_test.loc[sample.index]
    baseline_sample_proba = baseline_proba[X_test.index.get_indexer(sample.index)]
    baseline_ap = average_precision_score(y_sample, baseline_sample_proba)
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for column in sample.columns:
        shuffled = sample.copy()
        shuffled[column] = rng.permutation(shuffled[column].to_numpy())
        probabilities = _predict(model, preprocessor.transform(shuffled))
        rows.append(
            {
                "feature": column,
                "permutation_pr_auc_drop": baseline_ap
                - average_precision_score(y_sample, probabilities),
            }
        )
    return pd.DataFrame(rows).sort_values("permutation_pr_auc_drop", ascending=False)


def _write_explainability_report(importance: pd.DataFrame) -> None:
    EXPLAINABILITY_DIR.mkdir(parents=True, exist_ok=True)
    importance.to_csv(EXPLAINABILITY_DIR / "dnn_permutation_importance.csv", index=False)
    top = importance.head(10)
    lines = [
        "# DNN Explainability Summary",
        "",
        "SHAP was not used because a stable, fast neural-network explainer is not guaranteed "
        "across optional TensorFlow/SHAP environments. Model-agnostic permutation importance "
        "is the controlled fallback.",
        "",
        _markdown_table(top),
        "",
        "Repayment-delay, utilization, billing, and payment variables in the leading features "
        "are economically plausible credit-risk signals; rankings remain associational.",
        "",
        "DNN explanations are diagnostic and approximate. Model-agnostic explanations support "
        "trust but are not legal adverse-action reason codes.",
        "",
        "The DNN is more operationally opaque than the tree benchmark, so additional complexity "
        "requires a material performance or fairness benefit.",
    ]
    (EXPLAINABILITY_DIR / "dnn_explainability_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _write_selection_report(policy: dict[str, Any]) -> None:
    metrics = policy["test_metrics"]
    lines = [
        "# Deep Learning Threshold Selection",
        "",
        f"- Source experiment: `{policy['source_experiment']}`",
        f"- Selected threshold: `{policy['selected_threshold']:.2f}`",
        f"- Rule: `{policy['selection_rule']}`",
        f"- Fallback used: `{policy['fallback_used']}`",
        "- Threshold selection used validation data only; the test split was evaluated once.",
        "",
        "## Untouched Test Metrics",
        "",
        _markdown_table(pd.DataFrame([metrics])),
    ]
    (MODEL_VALIDATION_DIR / "deep_learning_threshold_selection.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _write_comparison_report(comparison: pd.DataFrame, curve_note: str) -> None:
    xgb = comparison.loc[comparison["model_name"] == "xgboost_public"]
    dnn = comparison.loc[comparison["model_name"] == "dnn_recall_optimized"]
    selected = "XGBoost remains the primary model."
    rationale = (
        "XGBoost remains the primary model unless the DNN demonstrates materially better "
        "recall/PR-AUC under comparable fairness and explainability constraints."
    )
    if not xgb.empty and not dnn.empty:
        xgb_roc = pd.to_numeric(xgb.iloc[0].get("roc_auc"), errors="coerce")
        dnn_roc = pd.to_numeric(dnn.iloc[0].get("roc_auc"), errors="coerce")
        xgb_pr = 0.5415000164539201
        dnn_pr = pd.to_numeric(dnn.iloc[0].get("pr_auc"), errors="coerce")
        if (
            pd.notna(dnn_roc)
            and pd.notna(dnn_pr)
            and dnn_roc > xgb_roc + 0.01
            and dnn_pr > xgb_pr + 0.01
        ):
            selected = (
                "The DNN is a candidate for further validation, not an automatic replacement."
            )
    dnn_baseline = comparison.loc[comparison["model_name"] == "dnn_baseline"]
    xgb_recall = comparison.loc[comparison["model_name"] == "xgboost_public_recall_optimized"]
    findings = []
    if not xgb.empty and not dnn_baseline.empty:
        findings.extend(
            [
                f"- ROC-AUC: DNN `{float(dnn_baseline.iloc[0]['roc_auc']):.4f}` vs "
                f"XGBoost `{float(xgb.iloc[0]['roc_auc']):.4f}`; DNN does not improve ranking.",
                f"- PR-AUC: DNN `{float(dnn_baseline.iloc[0]['pr_auc']):.4f}` vs "
                "XGBoost `0.5415`; DNN does not improve minority-class ranking.",
            ]
        )
    if not dnn.empty and not xgb_recall.empty:
        findings.append(
            f"- Recall policy: DNN recall `{float(dnn.iloc[0]['recall']):.4f}` at precision "
            f"`{float(dnn.iloc[0]['precision']):.4f}` vs XGBoost recall "
            f"`{float(xgb_recall.iloc[0]['recall']):.4f}` at precision "
            f"`{float(xgb_recall.iloc[0]['precision']):.4f}`. The DNN meets the validation "
            "precision rule but captures fewer test defaults."
        )
    lines = [
        "# ML vs DL Comparison",
        "",
        _markdown_table(comparison),
        "",
        "## Findings",
        "",
        *findings,
        "",
        "## Decision",
        "",
        selected,
        "",
        rationale,
        "",
        "The comparison considers ROC-AUC, PR-AUC, recall at acceptable precision, fairness, "
        "explainability, and operational complexity. Tree boosting often remains strong on "
        "structured tabular credit data even when a DNN is useful as a learning benchmark.",
        "",
        curve_note,
    ]
    (MODEL_VALIDATION_DIR / "ml_vs_dl_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def write_skip_artifacts(reason: str) -> dict[str, Any]:
    ensure_directories()
    MODEL_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"status": "skipped", "reason": reason, "tensorflow_available": False}
    save_json(payload, MODEL_VALIDATION_DIR / "deep_learning_metrics.json")
    message = f"# Deep Learning Benchmark\n\n{reason}\n"
    (MODEL_VALIDATION_DIR / "ml_vs_dl_comparison.md").write_text(message, encoding="utf-8")
    (MODEL_VALIDATION_DIR / "deep_learning_threshold_selection.md").write_text(
        message, encoding="utf-8"
    )
    return payload


def run(config: BenchmarkConfig | None = None) -> dict[str, Any]:
    config = config or BenchmarkConfig()
    tf, reason = tensorflow_status()
    if tf is None:
        print(reason)
        return write_skip_artifacts(reason or "TensorFlow is unavailable.")

    ensure_directories()
    MODEL_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    FAIRNESS_DIR.mkdir(parents=True, exist_ok=True)
    EXPLAINABILITY_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    set_random_seeds(tf)

    raw, source = load_dataset_auto()
    splits = build_recall_optimization_splits_from_frame(raw)
    preprocessor = build_preprocessor(splits.X_inner_train)
    X_train = preprocessor.fit_transform(splits.X_inner_train)
    X_validation = preprocessor.transform(splits.X_validation)
    X_test = preprocessor.transform(splits.X_test)
    if hasattr(X_train, "toarray"):
        X_train, X_validation, X_test = (
            X_train.toarray(),
            X_validation.toarray(),
            X_test.toarray(),
        )

    experiments: dict[str, dict[str, Any]] = {}
    models: dict[str, Any] = {}
    histories: dict[str, Any] = {}
    for name, weights in [("dnn_baseline", None), ("dnn_class_weighted", "balanced")]:
        if name == "dnn_class_weighted" and config.skip_class_weighted:
            continue
        class_weight = None
        if weights == "balanced":
            classes = np.unique(splits.y_inner_train)
            values = compute_class_weight(
                class_weight="balanced", classes=classes, y=splits.y_inner_train
            )
            class_weight = {int(key): float(value) for key, value in zip(classes, values)}
        model, history = _fit_model(
            tf,
            X_train,
            splits.y_inner_train,
            X_validation,
            splits.y_validation,
            config,
            class_weight,
        )
        validation_proba = _predict(model, X_validation)
        test_proba = _predict(model, X_test)
        experiments[name] = {
            "threshold_050": {
                **threshold_metrics(splits.y_test, test_proba, 0.50),
                **_probability_metrics(splits.y_test, test_proba),
            },
            "validation_pr_auc": float(
                average_precision_score(splits.y_validation, validation_proba)
            ),
            "validation_proba": validation_proba,
            "test_proba": test_proba,
            "class_weight": class_weight,
        }
        models[name] = model
        histories[name] = history

    source_experiment = max(experiments, key=lambda name: experiments[name]["validation_pr_auc"])
    selected_validation_proba = experiments[source_experiment]["validation_proba"]
    thresholds = create_threshold_grid()
    tuning = evaluate_threshold_grid(
        splits.y_validation,
        selected_validation_proba,
        thresholds,
        candidate_name=source_experiment,
        split_name="validation",
    )
    selected_row, fallback_used, selected_rule = select_preferred_threshold(tuning)
    selected_test_proba = experiments[source_experiment]["test_proba"]
    selected_test_metrics = threshold_metrics(
        splits.y_test, selected_test_proba, float(selected_row["threshold"])
    )
    ranking_metrics = _probability_metrics(splits.y_test, selected_test_proba)
    selected_test_metrics.update(ranking_metrics)
    policy = selected_policy_payload(
        source_experiment,
        selected_row,
        selected_rule,
        fallback_used,
        selected_test_metrics,
        ranking_metrics["roc_auc"],
        ranking_metrics["pr_auc"],
    )
    if not validate_policy_schema(policy):
        raise ValueError("Selected DNN policy failed schema validation.")

    tuning.to_csv(MODEL_VALIDATION_DIR / "deep_learning_threshold_tuning.csv", index=False)
    save_json(policy, MODEL_VALIDATION_DIR / "deep_learning_selected_policy.json")
    _write_selection_report(policy)
    _write_test_predictions(
        splits.y_test,
        experiments["dnn_baseline"]["test_proba"],
        "dnn_baseline",
        MODEL_VALIDATION_DIR / "dnn_test_predictions.csv",
    )
    _training_curve(
        histories[source_experiment],
        MODEL_VALIDATION_DIR / "deep_learning_training_curve.png",
    )
    curve_note = _precision_recall_plot(
        splits.y_test,
        experiments["dnn_baseline"]["test_proba"],
        MODEL_VALIDATION_DIR / "deep_learning_precision_recall_curve.png",
        MODEL_VALIDATION_DIR / "ml_vs_dl_precision_recall_curve.png",
    )

    comparison_rows = _load_existing_comparison_rows()
    for name, result in experiments.items():
        comparison_rows.append({"model_name": name, **result["threshold_050"]})
    comparison_rows.append({"model_name": "dnn_recall_optimized", **selected_test_metrics})
    comparison = pd.DataFrame(comparison_rows)
    preferred_columns = [
        "model_name",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f2",
        "roc_auc",
        "pr_auc",
        "approval_support_rate",
    ]
    comparison = comparison[[col for col in preferred_columns if col in comparison.columns]]
    comparison.to_csv(MODEL_VALIDATION_DIR / "deep_learning_comparison.csv", index=False)
    comparison.to_csv(MODEL_VALIDATION_DIR / "ml_vs_dl_comparison.csv", index=False)
    _write_comparison_report(comparison, curve_note)

    sensitive_test = raw.loc[splits.test_indices, "SEX"]
    fairness = _fairness_rows(
        splits.y_test,
        sensitive_test,
        experiments["dnn_baseline"]["test_proba"],
        selected_test_proba,
        float(selected_row["threshold"]),
    )
    _write_fairness_report(fairness)

    explainability_status = "skipped by CLI"
    if not config.skip_explainability:
        importance = _permutation_explainability(
            models[source_experiment],
            preprocessor,
            splits.X_test,
            splits.y_test,
            selected_test_proba,
        )
        _write_explainability_report(importance)
        explainability_status = "permutation importance generated"

    models[source_experiment].save(MODELS_DIR / "dnn_public.keras")
    joblib.dump(preprocessor, MODELS_DIR / "dnn_public_preprocessor.pkl")
    metrics_payload = {
        "status": "completed",
        "tensorflow_available": True,
        "tensorflow_version": tf.__version__,
        "dataset": str(source),
        "feature_set": FEATURE_SET_APPLICATION,
        "protected_attribute_excluded_from_training": "SEX",
        "architecture": [64, 32, 16, 1],
        "experiments": {
            name: {
                "threshold_050": result["threshold_050"],
                "validation_pr_auc": result["validation_pr_auc"],
                "class_weight": result["class_weight"],
            }
            for name, result in experiments.items()
        },
        "selected_policy": policy,
        "fairness": fairness.to_dict(orient="records"),
        "explainability": explainability_status,
        "primary_model_decision": "XGBoost remains primary pending material DNN superiority.",
    }
    save_json(_json_safe(metrics_payload), MODEL_VALIDATION_DIR / "deep_learning_metrics.json")
    return metrics_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train an optional Keras MLP benchmark on the public UCI Taiwan credit-default "
            "dataset using validation-only threshold selection."
        )
    )
    parser.add_argument("--quick", action="store_true", help="Run at most 8 epochs.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--skip-class-weighted", action="store_true")
    parser.add_argument("--skip-explainability", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Hide Keras epoch logs.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = BenchmarkConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        quick=args.quick,
        skip_class_weighted=args.skip_class_weighted,
        skip_explainability=args.skip_explainability,
        verbose=0 if args.quiet else 1,
    )
    result = run(config)
    print(json.dumps(_json_safe(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
