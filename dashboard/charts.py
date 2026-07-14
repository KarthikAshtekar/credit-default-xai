"""Plotly chart builders for dashboard governance and scenario views."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import average_precision_score, precision_recall_curve

from dashboard.ui_components import friendly_model_name

PLOT_TEMPLATE = "plotly_dark"
PALETTE = ["#38bdf8", "#22c55e", "#f59e0b", "#f97316", "#a78bfa", "#e879f9"]


def load_report_csv_safely(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        return None


def _style_figure(fig: go.Figure, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.55)",
        font={"color": "#e5e7eb", "size": 13},
        margin={"l": 36, "r": 20, "t": 52 if title else 24, "b": 42},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.16)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.16)", zeroline=False)
    return fig


def build_pr_curve_chart(prediction_frames: dict[str, pd.DataFrame]) -> go.Figure | None:
    fig = go.Figure()
    added = 0
    for index, (label, frame) in enumerate(prediction_frames.items()):
        if frame is None or not {"y_true", "y_proba"}.issubset(frame.columns):
            continue
        y_true = frame["y_true"].astype(int)
        y_proba = frame["y_proba"].astype(float)
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=f"{friendly_model_name(label)} AP {ap:.3f}",
                line={"color": PALETTE[index % len(PALETTE)], "width": 2.5},
                hovertemplate="Recall %{x:.2f}<br>Precision %{y:.2f}<extra></extra>",
            )
        )
        added += 1

    if not added:
        return None
    fig.update_xaxes(title="Recall", range=[0, 1])
    fig.update_yaxes(title="Precision", range=[0, 1])
    return _style_figure(fig, "Precision-recall comparison")


def build_threshold_tradeoff_chart(
    threshold_df: pd.DataFrame | None,
    candidate_name: str | None = None,
    selected_threshold: float | None = None,
) -> go.Figure | None:
    if threshold_df is None or threshold_df.empty:
        return None
    required = {"threshold", "precision", "recall", "f2"}
    if not required.issubset(threshold_df.columns):
        return None

    frame = threshold_df.copy()
    if candidate_name and "candidate_name" in frame.columns:
        frame = frame[frame["candidate_name"] == candidate_name]
    if frame.empty:
        return None

    frame = frame.sort_values("threshold")
    fig = go.Figure()
    for index, metric in enumerate(["precision", "recall", "f2"]):
        fig.add_trace(
            go.Scatter(
                x=frame["threshold"],
                y=frame[metric],
                mode="lines+markers",
                name=metric.upper() if metric == "f2" else metric.title(),
                line={"color": PALETTE[index], "width": 2.4},
                hovertemplate="Threshold %{x:.2f}<br>%{y:.2f}<extra></extra>",
            )
        )
    if selected_threshold is not None:
        fig.add_vline(
            x=selected_threshold,
            line_dash="dash",
            line_color="#f59e0b",
            annotation_text=f"Selected {selected_threshold:.2f}",
            annotation_position="top",
        )
    fig.update_xaxes(title="Default-risk threshold", range=[0.08, 0.72])
    fig.update_yaxes(title="Metric value", range=[0, 1])
    return _style_figure(fig, "Threshold tradeoff")


def build_model_comparison_chart(comparison_df: pd.DataFrame | None) -> go.Figure | None:
    if comparison_df is None or comparison_df.empty or "model_name" not in comparison_df.columns:
        return None
    metrics = [
        metric for metric in ["roc_auc", "pr_auc", "recall", "f2"] if metric in comparison_df
    ]
    if not metrics:
        return None

    preferred = [
        "logistic_public",
        "xgboost_public",
        "xgboost_public_recall_optimized",
        "dnn_baseline",
        "dnn_recall_optimized",
    ]
    frame = comparison_df[comparison_df["model_name"].isin(preferred)].copy()
    if frame.empty:
        frame = comparison_df.copy()
    frame["model_label"] = frame["model_name"].map(friendly_model_name)

    fig = go.Figure()
    for index, metric in enumerate(metrics):
        fig.add_trace(
            go.Bar(
                x=frame["model_label"],
                y=pd.to_numeric(frame[metric], errors="coerce"),
                name=metric.upper().replace("_", "-"),
                marker_color=PALETTE[index % len(PALETTE)],
                hovertemplate="%{x}<br>%{y:.3f}<extra></extra>",
            )
        )
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Score", range=[0, 1])
    fig.update_xaxes(title=None)
    return _style_figure(fig, "Model benchmark comparison")


def build_fairness_chart(fairness_df: pd.DataFrame | None) -> go.Figure | None:
    if fairness_df is None or fairness_df.empty:
        return None
    metrics = [
        metric
        for metric in [
            "demographic_parity_difference",
            "equal_opportunity_difference",
            "equalized_odds_difference",
        ]
        if metric in fairness_df.columns
    ]
    policy_column = "policy" if "policy" in fairness_df.columns else "model_name"
    if not metrics or policy_column not in fairness_df.columns:
        return None

    frame = fairness_df.copy()
    frame["policy_label"] = frame[policy_column].map(friendly_model_name)
    fig = go.Figure()
    for index, metric in enumerate(metrics):
        fig.add_trace(
            go.Bar(
                x=frame["policy_label"],
                y=pd.to_numeric(frame[metric], errors="coerce"),
                name=metric.replace("_", " ").title(),
                marker_color=PALETTE[index % len(PALETTE)],
                hovertemplate="%{x}<br>%{y:.3f}<extra></extra>",
            )
        )
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Difference")
    fig.update_xaxes(title=None)
    return _style_figure(fig, "Fairness diagnostics by policy")


def build_scenario_curve_chart(
    scenario_curve: pd.DataFrame | None,
    threshold: float,
) -> go.Figure | None:
    if scenario_curve is None or scenario_curve.empty:
        return None
    if not {"target_credit_amount", "predicted_default_risk"}.issubset(scenario_curve.columns):
        return None

    frame = scenario_curve.sort_values("target_credit_amount")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["target_credit_amount"],
            y=frame["predicted_default_risk"],
            mode="lines+markers",
            name="Predicted default risk",
            line={"color": "#38bdf8", "width": 2.6},
            hovertemplate="Target credit %{x:,.0f}<br>Risk %{y:.1%}<extra></extra>",
        )
    )
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text=f"Review threshold {threshold:.0%}",
        annotation_position="top left",
    )
    fig.update_xaxes(title="Target credit amount")
    fig.update_yaxes(title="Predicted default risk", tickformat=".0%", range=[0, 1])
    return _style_figure(fig, "Scenario risk curve")
