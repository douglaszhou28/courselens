import pandas as pd
from .config import IDENTIFIER_COLUMNS


def add_anonymous_course_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "course_id", [f"course_{i:03d}" for i in range(1, len(out) + 1)])
    return out


def remove_personal_fields(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in IDENTIFIER_COLUMNS if c in df.columns], errors="ignore")


def safe_export_frame(df: pd.DataFrame, include_identifiers: bool = False) -> pd.DataFrame:
    out = add_anonymous_course_id(df) if "course_id" not in df.columns else df.copy()
    return out if include_identifiers else remove_personal_fields(out)
