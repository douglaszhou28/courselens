#!/usr/bin/env python
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from courselens.data_loading import inspect_workbook, load_course_data, schema_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="demo/example_transcript_courses_edited.xlsx")
    args = ap.parse_args()
    path = Path(args.input)
    if not path.is_absolute():
        path = ROOT.parent / path
    info = inspect_workbook(path)
    df = load_course_data(path)
    schema = schema_summary(df, info)
    print(f"Workbook: {path}")
    print(f"Sheets: {', '.join(info['sheets'])}")
    print(f"Rows: {schema['rows']}; columns: {schema['columns']}")
    print("Missing values:")
    for k, v in schema["missing"].items():
        if v:
            print(f"  {k}: {v}")
    formulas = info.get("formula_cells", {}).get("课程成绩", [])
    print(f"Formula cells in course sheet: {len(formulas)}")

if __name__ == "__main__":
    main()
