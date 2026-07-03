from __future__ import annotations

from argparse import Namespace

from src.run_pipeline import PipelineStep, build_steps, run_steps


def test_pipeline_default_order_includes_required_artifact_stages() -> None:
    steps = build_steps(
        Namespace(
            skip_explainability=False,
            skip_counterfactuals=False,
            skip_mitigation=False,
        )
    )

    assert [step.name for step in steps] == [
        "Write dataset coverage audit",
        "Train logistic models",
        "Train XGBoost models",
        "Evaluate model variants",
        "Run leakage audit",
        "Generate SHAP artifacts",
        "Generate LIME artifacts",
        "Compute fairness metrics",
        "Run bias mitigation experiments",
        "Generate counterfactuals",
    ]


def test_pipeline_skip_flags_remove_optional_stages() -> None:
    steps = build_steps(
        Namespace(
            skip_explainability=True,
            skip_counterfactuals=True,
            skip_mitigation=True,
        )
    )

    names = [step.name for step in steps]
    assert "Write dataset coverage audit" in names
    assert "Generate SHAP artifacts" not in names
    assert "Generate LIME artifacts" not in names
    assert "Run bias mitigation experiments" not in names
    assert "Generate counterfactuals" not in names
    assert names[-1] == "Compute fairness metrics"


def test_run_steps_returns_nonzero_on_failure() -> None:
    def fail():
        raise RuntimeError("boom")

    result = run_steps([PipelineStep("Failing step", fail)])

    assert result == 1


def test_run_steps_returns_zero_when_all_steps_pass() -> None:
    calls = []

    def ok():
        calls.append("ok")

    result = run_steps([PipelineStep("Passing step", ok)])

    assert result == 0
    assert calls == ["ok"]
