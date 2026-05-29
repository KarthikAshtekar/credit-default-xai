from __future__ import annotations

import json

import streamlit as st
from common import ensure_model, get_feature_table

from src.counterfactuals import generate_counterfactual


def render_counterfactual_page() -> None:
    st.subheader("Counterfactual Dashboard")

    model_choice = st.selectbox("Model for Counterfactuals", ["XGBoost", "Logistic Regression"])
    _, model_path = ensure_model(model_choice)

    X = get_feature_table()
    row_idx = st.number_input("Select customer row index", min_value=0, max_value=max(0, len(X) - 1), value=0)

    if st.button("Generate Counterfactuals"):
        query = X.iloc[[int(row_idx)]].copy()
        out = generate_counterfactual(model_path, query_instance=query, total_CFs=3)

        st.success("Counterfactuals generated.")
        st.write("Saved file:", out["counterfactual_file"])

        with open(out["counterfactual_file"], "r", encoding="utf-8") as f:
            payload = json.load(f)

        st.json(payload)
