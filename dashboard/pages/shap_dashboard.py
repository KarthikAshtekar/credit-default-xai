from __future__ import annotations

from pathlib import Path

import streamlit as st

from common import ensure_model
from src.shap_explainer import generate_shap_artifacts


def render_shap_page() -> None:
    st.subheader("SHAP Explanations")

    model_choice = st.selectbox("Model for SHAP", ["XGBoost", "Logistic Regression"])
    _, model_path = ensure_model(model_choice)

    if st.button("Generate SHAP Plots"):
        artifacts = generate_shap_artifacts(model_path)
        st.success("SHAP artifacts generated.")
        st.image(artifacts["summary_plot"], caption="Global SHAP Summary")
        st.image(artifacts["local_plot"], caption="Local SHAP Waterfall")
