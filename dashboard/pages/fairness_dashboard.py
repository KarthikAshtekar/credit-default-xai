from __future__ import annotations

import pandas as pd
import streamlit as st

from common import ensure_model
from src.fairness_metrics import run as fairness_run


def render_fairness_page() -> None:
    st.subheader("Fairness Dashboard")

    model_choice = st.selectbox("Model for Fairness", ["XGBoost", "Logistic Regression"])
    _, model_path = ensure_model(model_choice)

    if st.button("Compute Fairness Metrics"):
        result = fairness_run(model_path)
        metrics = result["fairness_metrics"]

        st.write("Protected Attribute:", result["protected_attribute"])
        st.dataframe(pd.DataFrame([metrics]))

        plot_df = pd.DataFrame(
            {
                "Metric": list(metrics.keys()),
                "Value": list(metrics.values()),
            }
        )
        st.bar_chart(plot_df.set_index("Metric"))
