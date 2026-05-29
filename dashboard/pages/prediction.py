from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from common import ensure_model, get_feature_table


def render_prediction_page() -> None:
    st.subheader("Credit Risk Prediction")

    model_choice = st.selectbox("Model", ["Logistic Regression", "XGBoost"])
    model, model_path = ensure_model(model_choice)

    X = get_feature_table()
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    st.markdown("Adjust customer profile inputs:")

    user_input = {}
    c1, c2 = st.columns(2)

    for i, col in enumerate(numeric_cols):
        default_val = float(np.nanmedian(X[col])) if np.issubdtype(X[col].dtype, np.number) else 0.0
        if i % 2 == 0:
            user_input[col] = c1.number_input(col, value=default_val)
        else:
            user_input[col] = c2.number_input(col, value=default_val)

    for col in categorical_cols:
        options = X[col].dropna().astype(str).unique().tolist()
        if not options:
            options = ["Unknown"]
        user_input[col] = st.selectbox(col, options=options)

    input_df = pd.DataFrame([user_input])

    if st.button("Predict Risk"):
        proba = float(model.predict_proba(input_df)[:, 1][0])
        pred = int(proba >= 0.5)
        risk = "High" if pred == 1 else "Low"

        st.metric("Default Probability", f"{proba:.2%}")
        st.metric("Risk Category", risk)
        st.caption(f"Prediction generated using `{model_path.name}`")
