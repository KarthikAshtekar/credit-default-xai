"""DiCE counterfactual explanation generation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from .data_preprocessing import TARGET_COL, prepare_modeling_table, split_features_target
from .utils import MODELS_DIR, REPORTS_DIR, ensure_directories, load_dataset_auto, load_model


def generate_counterfactual(
    model_path: Path,
    query_instance: pd.DataFrame | None = None,
    total_CFs: int = 3,
) -> Dict:
    import dice_ml

    ensure_directories()
    df_raw, _ = load_dataset_auto()
    df = prepare_modeling_table(df_raw, target_col=TARGET_COL)

    X, y = split_features_target(df, target_col=TARGET_COL)
    full = X.copy()
    full[TARGET_COL] = y.values

    model_pipeline = load_model(model_path)

    continuous_features = X.select_dtypes(include=["number"]).columns.tolist()

    data_dice = dice_ml.Data(
        dataframe=full,
        continuous_features=continuous_features,
        outcome_name=TARGET_COL,
    )

    model_dice = dice_ml.Model(model=model_pipeline, backend="sklearn")
    dice = dice_ml.Dice(data_dice, model_dice, method="random")

    if query_instance is None:
        query_instance = X.iloc[[0]].copy()

    cf = dice.generate_counterfactuals(
        query_instance,
        total_CFs=total_CFs,
        desired_class="opposite",
    )

    out_path = REPORTS_DIR / "explainability_reports" / f"{model_path.stem}_counterfactuals.json"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(cf.to_json())

    return {
        "model": str(model_path),
        "counterfactual_file": str(out_path),
        "total_counterfactuals": total_CFs,
    }


def run() -> Dict:
    xgb_path = MODELS_DIR / "xgboost_model.pkl"
    log_path = MODELS_DIR / "logistic_model.pkl"

    if xgb_path.exists():
        return generate_counterfactual(xgb_path)
    if log_path.exists():
        return generate_counterfactual(log_path)

    raise FileNotFoundError("No trained model found in models/. Train a model first.")


if __name__ == "__main__":
    result = run()
    print("Counterfactual explanations generated.")
    print(result)
