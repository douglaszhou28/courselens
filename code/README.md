# CourseLens Analysis Prototype

This directory contains the first reproducible analytical core for CourseLens.

Run the full analysis:

```bash
python code/scripts/run_full_analysis.py \
  --input demo/example_transcript_courses_edited.xlsx \
  --output-dir code/outputs
```

Inspect the workbook only:

```bash
python code/scripts/inspect_workbook.py --input demo/example_transcript_courses_edited.xlsx
```

The default outputs are privacy-safe: course names and raw course selection IDs are excluded from exported datasets and reports unless `--include-identifiers` is provided.
