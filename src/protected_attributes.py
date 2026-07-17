"""Shared protected-attribute labels used for audit/reporting surfaces."""

from __future__ import annotations

from typing import Any

PROTECTED_ATTRIBUTE_SEX = "SEX"

SEX_GROUP_LABELS: dict[int, str] = {
    1: "Male",
    2: "Female",
}


def sex_code(value: Any) -> int | None:
    """Return the integer SEX code when it can be parsed."""

    try:
        if value is None:
            return None
        if value != value:  # NaN-safe without importing pandas/numpy.
            return None
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None


def sex_group(value: Any) -> str:
    """Return the readable group name for a SEX code."""

    code = sex_code(value)
    if code is None:
        return "Unknown"
    return SEX_GROUP_LABELS.get(code, "Unknown")


def sex_group_display(value: Any) -> str:
    """Return a governance-facing label that preserves code and readable group."""

    code = sex_code(value)
    group = sex_group(value)
    if code is None:
        return f"{group} (SEX={value})"
    return f"{group} (SEX={code})"


def sex_mapping_rows() -> list[dict[str, Any]]:
    """Return mapping rows for reports/tests."""

    return [
        {
            "sex_code": code,
            "sex_group": group,
            "group": sex_group_display(code),
        }
        for code, group in SEX_GROUP_LABELS.items()
    ]
