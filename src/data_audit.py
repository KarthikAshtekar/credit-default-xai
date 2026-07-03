"""Dataset coverage reporting for the primary UCI dataset."""

from __future__ import annotations

import pandas as pd

from .dataset_adapters import (
    UCI_DEFAULT_CREDIT_CARD_DISPLAY_NAME,
    dataset_coverage_blocks,
)
from .utils import REPORTS_DIR, ensure_directories, load_dataset_auto


def run() -> pd.DataFrame:
    ensure_directories()
    df, _ = load_dataset_auto()
    rows = []
    for block in dataset_coverage_blocks():
        available = [column for column in block.columns if column in df.columns]
        rows.append(
            {
                "block": block.block,
                "columns": ", ".join(block.columns),
                "available_columns": ", ".join(available),
                "available_count": len(available),
                "expected_count": len(block.columns),
                "description": block.description,
            }
        )

    output = pd.DataFrame(rows)
    out_dir = REPORTS_DIR / "data_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_dir / "five_block_dataset_mapping.csv", index=False)

    lines = [
        "# Five-Block Dataset Mapping",
        "",
        f"Primary dataset: {UCI_DEFAULT_CREDIT_CARD_DISPLAY_NAME}",
        "",
        "| Block | Columns | Coverage |",
        "| --- | --- | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['block']} | `{row['columns']}` | "
            f"{row['available_count']}/{row['expected_count']} |"
        )
    lines.append("")
    lines.append(
        "`Default_Flag` is the project-standard target name and equals 1 for next-month default."
    )
    (out_dir / "five_block_dataset_mapping.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return output


if __name__ == "__main__":
    print(run().to_string(index=False))
