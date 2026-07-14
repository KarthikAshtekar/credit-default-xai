"""Reusable UI helpers for the Streamlit dashboard."""

from __future__ import annotations

from html import escape

import streamlit as st

MODEL_LABELS = {
    "logistic_public": "Logistic regression",
    "xgboost_public": "XGBoost baseline",
    "xgboost_full_public_diagnostic": "XGBoost diagnostic",
    "xgboost_public_recall_optimized": "Recall-optimized XGBoost screening",
    "xgboost_public_baseline_threshold_050": "XGBoost baseline",
    "current_baseline_threshold_050": "XGBoost baseline",
    "baseline_threshold_050": "XGBoost baseline",
    "recall_optimized": "Recall-optimized XGBoost screening",
    "dnn_baseline": "Deep learning benchmark",
    "dnn_class_weighted": "Class-weighted deep learning benchmark",
    "dnn_recall_optimized": "Deep learning recall policy",
}

MAIN_APPLICANT_LABELS = [
    "Target credit amount",
    "Latest repayment status",
    "Maximum recent delay",
    "Average bill amount",
    "Average payment amount",
]

TONE_COLORS = {
    "neutral": "#38bdf8",
    "good": "#22c55e",
    "watch": "#f59e0b",
    "risk": "#f97316",
    "high": "#ef4444",
}


def apply_dark_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --risk-bg: #080d18;
            --risk-panel: #101827;
            --risk-panel-2: #142033;
            --risk-border: rgba(148, 163, 184, 0.22);
            --risk-text: #e5e7eb;
            --risk-muted: #9ca3af;
            --risk-accent: #38bdf8;
        }
        .stApp {
            background: radial-gradient(circle at top left, #111827 0, #080d18 38%);
            color: var(--risk-text);
        }
        [data-testid="stSidebar"] {
            background: #0b1220;
            border-right: 1px solid var(--risk-border);
        }
        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }
        h1, h2, h3,
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3 {
            letter-spacing: 0;
            color: #f8fafc !important;
        }
        p, label, [data-testid="stMarkdownContainer"] p, [data-testid="stCaptionContainer"] {
            color: #cbd5e1 !important;
        }
        [data-testid="stTabs"] button {
            color: #cbd5e1 !important;
            border-radius: 8px 8px 0 0;
            padding: 0.65rem 1rem;
        }
        [data-testid="stTabs"] button[aria-selected="true"] {
            color: #ffffff !important;
            background: rgba(56, 189, 248, 0.12);
            border-bottom-color: var(--risk-accent);
        }
        div[data-testid="stExpander"] {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid var(--risk-border);
            border-radius: 8px;
        }
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] summary p,
        div[data-testid="stExpander"] label {
            color: #f1f5f9 !important;
        }
        div[data-testid="stAlert"] {
            border-radius: 8px;
        }
        .risk-hero {
            border: 1px solid var(--risk-border);
            border-radius: 8px;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(17, 24, 39, 0.90));
            padding: 1.25rem 1.35rem;
            margin-bottom: 1rem;
        }
        .risk-hero-title {
            font-size: 1.78rem;
            line-height: 1.15;
            font-weight: 700;
            margin: 0 0 0.35rem 0;
            color: #f8fafc !important;
        }
        .risk-hero-subtitle {
            color: var(--risk-muted) !important;
            margin: 0;
            font-size: 0.98rem;
        }
        .risk-card {
            border: 1px solid var(--risk-border);
            border-radius: 8px;
            background: rgba(15, 23, 42, 0.78);
            padding: 1rem;
            min-height: 114px;
            box-shadow: 0 16px 36px rgba(0,0,0,0.16);
        }
        .risk-card * {
            color: inherit;
        }
        .risk-card-label {
            color: #a8b3c7 !important;
            font-size: 0.82rem;
            line-height: 1.2;
            margin-bottom: 0.55rem;
        }
        .risk-card-value {
            color: #f8fafc !important;
            font-size: 1.55rem;
            font-weight: 700;
            line-height: 1.12;
            word-break: break-word;
        }
        .risk-card-detail {
            color: #d1d9e6 !important;
            font-size: 0.82rem;
            margin-top: 0.55rem;
        }
        .risk-panel {
            border: 1px solid var(--risk-border);
            border-radius: 8px;
            background: rgba(15, 23, 42, 0.70);
            padding: 1rem;
            margin: 0.75rem 0;
        }
        .risk-pill {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            border: 1px solid var(--risk-border);
            background: rgba(56, 189, 248, 0.10);
            color: #bae6fd;
            padding: 0.2rem 0.65rem;
            font-size: 0.82rem;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }
        .risk-muted {
            color: #a8b3c7 !important;
        }
        .risk-card.accent-neutral { border-top: 3px solid #38bdf8; }
        .risk-card.accent-good { border-top: 3px solid #22c55e; }
        .risk-card.accent-watch { border-top: 3px solid #f59e0b; }
        .risk-card.accent-risk { border-top: 3px solid #f97316; }
        .risk-card.accent-high { border-top: 3px solid #ef4444; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def friendly_model_name(name: object) -> str:
    text = str(name)
    return MODEL_LABELS.get(text, text.replace("_", " ").title())


def format_percent(value: object, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}%}"
    except (TypeError, ValueError):
        return "Not available"


def format_currency(value: object) -> str:
    try:
        return f"{float(value):,.0f}"
    except (TypeError, ValueError):
        return "Not available"


def risk_tone(risk_band: str) -> str:
    normalized = risk_band.lower()
    if "low" in normalized:
        return "good"
    if "medium" in normalized:
        return "watch"
    return "high"


def render_hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="risk-hero">
            <div class="risk-hero-title">{escape(title)}</div>
            <p class="risk-hero-subtitle">{escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(
    label: str,
    value: str,
    detail: str = "",
    tone: str = "neutral",
) -> None:
    safe_tone = tone if tone in TONE_COLORS else "neutral"
    detail_html = f'<div class="risk-card-detail">{escape(detail)}</div>' if detail else ""
    st.markdown(
        f"""
        <div class="risk-card accent-{safe_tone}">
            <div class="risk-card-label">{escape(label)}</div>
            <div class="risk-card-value">{escape(value)}</div>
            {detail_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_panel(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="risk-panel">
            <strong>{escape(title)}</strong>
            <div class="risk-muted">{escape(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pills(labels: list[str]) -> None:
    pills = "".join(f'<span class="risk-pill">{escape(label)}</span>' for label in labels)
    st.markdown(pills, unsafe_allow_html=True)
