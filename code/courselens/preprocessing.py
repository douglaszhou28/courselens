from pathlib import Path
import pandas as pd
from .config import COLUMN_MAP, NUMERIC_COLUMNS, IDENTIFIER_COLUMNS


def ensure_output_dirs(output_dir: str | Path) -> dict[str, Path]:
    out = Path(output_dir)
    paths = {"root": out, "tables": out / "tables", "figures": out / "figures"}
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def data_quality_markdown(df: pd.DataFrame, raw_columns: list[str], schema: dict) -> str:
    formulas = schema.get("workbook", {}).get("formula_cells", {}).get("课程成绩", [])
    lines = [
        "# Data Quality Report",
        "",
        f"Input workbook: `{schema.get('workbook', {}).get('path', '')}`",
        f"Rows: **{len(df)}**",
        f"Columns: **{df.shape[1]}**",
        "",
        "## Column mapping",
        "",
        "| Original column | Internal name |",
        "|---|---|",
    ]
    for c in raw_columns:
        lines.append(f"| {c} | {COLUMN_MAP.get(c, c)} |")
    lines += ["", "## Missing values", "", "| Column | Missing |", "|---|---:|"]
    for c, n in df.isna().sum().items():
        lines.append(f"| {c} | {int(n)} |")
    lines += ["", "## Numeric ranges", "", "| Column | Min | Max |", "|---|---:|---:|"]
    for c in NUMERIC_COLUMNS:
        if c in df and df[c].notna().any():
            lines.append(f"| {c} | {df[c].min():.3g} | {df[c].max():.3g} |")
    lines += ["", "## Formula-derived cells", ""]
    if formulas:
        by_col = {}
        for f in formulas:
            by_col[f.get("column", "unknown")] = by_col.get(f.get("column", "unknown"), 0) + 1
        for col, count in by_col.items():
            lines.append(f"- `{col}` contains {count} formula cells; treat as derived, not independent raw input.")
    else:
        lines.append("- No formula cells detected in the course sheet.")
    lines += [
        "", "## Privacy-sensitive fields", "",
        "The following fields are treated as identifiers and removed from default exports:",
        "", *(f"- `{c}`" for c in IDENTIFIER_COLUMNS),
        "", "## Modeling missing-value policy", "",
        "Missing numeric questionnaire values are retained and handled by median imputation inside sklearn pipelines, with imputation performed within cross-validation to avoid leakage.",
    ]
    return "\n".join(lines) + "\n"
