# Contributing Guide

Thanks for contributing to **Explainable and Fair Credit Default Risk Prediction**.

## Branching Rules
- Never commit directly to `main`.
- Create feature branches: `feat/<short-name>`, `fix/<short-name>`, or `docs/<short-name>`.
- Open Pull Requests early as Draft.

## Development Setup
1. Create environment and install dependencies:
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install
```
2. Run local checks:
```bash
pre-commit run --all-files
python -m compileall src dashboard
pytest -q
```

## Coding Standards
- Keep modules small and reusable.
- Add type hints for public functions.
- Write docstrings for non-trivial logic.
- Prefer deterministic random seeds for experiments.

## Notebook Standards
- Keep notebook outputs minimal before commit.
- Move reusable logic into `src/`.
- Use notebooks for analysis/storytelling, not core business logic.
- Core CI excludes notebooks from Ruff checks. Use `nbqa ruff notebooks` separately when notebook linting is needed.

## PR Requirements
- Clear title and description
- Linked issue (if applicable)
- Summary of fairness/performance impact
- Evidence of checks passing

## Data and Ethics Guardrails
- Do not commit PII or sensitive personal data snapshots.
- Document fairness changes and tradeoffs in PRs.
- Avoid using protected attributes for direct adverse decisions without governance review.

## Review Expectations
- At least 1 approving review
- CI must pass
- No unresolved requested changes
