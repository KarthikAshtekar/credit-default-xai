"""Streamlit entry point for Explainable and Fair Credit Default Risk app."""

from __future__ import annotations

import streamlit as st
from counterfactual_dashboard import render_counterfactual_page
from fairness_dashboard import render_fairness_page
from prediction import render_prediction_page
from shap_dashboard import render_shap_page

st.set_page_config(
    page_title="Credit Default XAI & Fairness",
    page_icon="📊",
    layout="wide",
)

st.title("Explainable and Fair Credit Default Risk Prediction")
st.caption("Performance + Explainability + Fairness + Actionable Counterfactuals")

pages = {
    "Credit Risk Prediction": render_prediction_page,
    "SHAP Explanations": render_shap_page,
    "Fairness Dashboard": render_fairness_page,
    "Counterfactual Dashboard": render_counterfactual_page,
}

choice = st.sidebar.radio("Navigate", list(pages.keys()))
pages[choice]()
