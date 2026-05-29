from pathlib import Path


def test_repository_structure_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    required = [
        root / "src",
        root / "dashboard",
        root / "notebooks",
        root / "data" / "raw",
        root / ".github" / "workflows" / "ci.yml",
    ]
    for path in required:
        assert path.exists(), f"Missing required path: {path}"


def test_all_required_notebooks_present() -> None:
    root = Path(__file__).resolve().parents[1]
    nb_dir = root / "notebooks"
    expected = [
        "01_data_understanding.ipynb",
        "02_data_cleaning.ipynb",
        "03_feature_engineering.ipynb",
        "04_logistic_regression.ipynb",
        "05_xgboost_model.ipynb",
        "06_model_comparison.ipynb",
        "07_shap_analysis.ipynb",
        "08_lime_analysis.ipynb",
        "09_counterfactuals.ipynb",
        "10_fairness_analysis.ipynb",
        "11_bias_mitigation.ipynb",
        "12_final_results.ipynb",
        "13_fairness_vs_accuracy_tradeoff.ipynb",
    ]
    missing = [name for name in expected if not (nb_dir / name).exists()]
    assert not missing, f"Missing notebooks: {missing}"
