#!/usr/bin/env python
from pathlib import Path
import argparse
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

from courselens.config import FEATURE_SETS
from courselens.data_loading import inspect_workbook, load_course_data, schema_summary
from courselens.descriptive_analysis import write_descriptive_report
from courselens.feature_engineering import add_derived_features, feature_dictionary
from courselens.interpretation import write_factor_report
from courselens.modeling import write_model_report
from courselens.preprocessing import data_quality_markdown, ensure_output_dirs
from courselens.privacy import safe_export_frame
from courselens.reporting import write_full_report
from courselens.statistical_analysis import write_statistical_report


def resolve_path(p):
    path = Path(p)
    return path if path.is_absolute() else ROOT.parent / path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="demo/example_transcript_courses_edited.xlsx")
    ap.add_argument("--output-dir", default="code/outputs")
    ap.add_argument("--target", default="grade_point", choices=["grade_point", "grade"])
    ap.add_argument("--include-grade-target", action="store_true")
    ap.add_argument("--include-identifiers", action="store_true")
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--export-safe-dataset", action="store_true", default=True)
    args = ap.parse_args()

    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)
    paths = ensure_output_dirs(output_dir)

    workbook_info = inspect_workbook(input_path)
    raw_df = load_course_data(input_path)
    df = add_derived_features(raw_df)
    schema = schema_summary(df, workbook_info)

    (output_dir / "data_quality_report.md").write_text(data_quality_markdown(df, list(raw_df.columns), schema), encoding="utf-8")
    feature_dictionary().to_csv(paths["tables"] / "feature_dictionary.csv", index=False)
    if args.export_safe_dataset:
        safe_export_frame(df, include_identifiers=args.include_identifiers).to_csv(paths["tables"] / "safe_modeling_dataset.csv", index=False)

    descriptive = write_descriptive_report(df, output_dir)
    all_features = sorted(set(sum(FEATURE_SETS.values(), [])))
    all_features = [f for f in all_features if f in df.columns]
    stats = write_statistical_report(df, output_dir, all_features)
    modeling = write_model_report(df, output_dir, target=args.target, random_state=args.random_seed)
    factors = write_factor_report(df, stats["correlations"], output_dir, random_state=args.random_seed)
    write_full_report(output_dir, schema, descriptive, stats, modeling, factors)

    if args.include_grade_target and args.target != "grade":
        write_model_report(df, output_dir, target="grade", random_state=args.random_seed)

    print(f"Analysis complete. Report: {output_dir / 'full_analysis_report.md'}")

if __name__ == "__main__":
    main()
