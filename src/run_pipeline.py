"""One-command orchestration for regenerating project artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from . import (
    bias_mitigation,
    counterfactuals,
    data_audit,
    evaluate_models,
    fairness_metrics,
    leakage_audit,
    lime_explainer,
    shap_explainer,
    train_logistic,
    train_xgboost,
)


@dataclass(frozen=True)
class PipelineStep:
    name: str
    runner: Callable[[], Any]
    group: str = "required"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate credit-default-xai model, validation, explainability, and fairness artifacts."
    )
    parser.add_argument(
        "--skip-explainability",
        action="store_true",
        help="Skip SHAP and LIME artifact generation.",
    )
    parser.add_argument(
        "--skip-counterfactuals",
        action="store_true",
        help="Skip DiCE counterfactual generation.",
    )
    parser.add_argument(
        "--skip-mitigation",
        action="store_true",
        help="Skip fairness mitigation experiments.",
    )
    return parser


def build_steps(args: argparse.Namespace) -> list[PipelineStep]:
    steps = [
        PipelineStep("Write dataset coverage audit", data_audit.run),
        PipelineStep("Train logistic models", train_logistic.run),
        PipelineStep("Train XGBoost models", train_xgboost.run),
        PipelineStep("Evaluate model variants", evaluate_models.run),
        PipelineStep("Run leakage audit", leakage_audit.run),
    ]

    if not args.skip_explainability:
        steps.extend(
            [
                PipelineStep("Generate SHAP artifacts", shap_explainer.run, group="explainability"),
                PipelineStep("Generate LIME artifacts", lime_explainer.run, group="explainability"),
            ]
        )

    steps.append(PipelineStep("Compute fairness metrics", fairness_metrics.run))

    if not args.skip_mitigation:
        steps.append(
            PipelineStep("Run bias mitigation experiments", bias_mitigation.run, group="mitigation")
        )

    if not args.skip_counterfactuals:
        steps.append(
            PipelineStep("Generate counterfactuals", counterfactuals.run, group="counterfactuals")
        )

    return steps


def run_steps(steps: Sequence[PipelineStep]) -> int:
    for index, step in enumerate(steps, start=1):
        print(f"[{index}/{len(steps)}] {step.name}...")
        try:
            step.runner()
        except Exception as exc:
            print(f"FAILED: {step.name}: {exc}")
            return 1
        print(f"OK: {step.name}")
    print("Pipeline completed successfully.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_steps(build_steps(args))


if __name__ == "__main__":
    raise SystemExit(main())
