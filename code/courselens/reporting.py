from pathlib import Path
import json
import pandas as pd
from .markdown import df_to_markdown


def write_full_report(output_dir: Path, schema: dict, descriptive: dict, stats: dict, modeling: dict, factors: dict) -> None:
    summary = descriptive["summary"]
    metrics = modeling["metrics"]
    factor_summary = factors["factor_summary"]
    corr = stats["correlations"]
    best = metrics.sort_values("mae").head(8) if len(metrics) else pd.DataFrame()
    baseline = metrics[metrics["model"] == "dummy_mean"].sort_values("mae").head(1) if len(metrics) else pd.DataFrame()
    top_factors = factor_summary.head(8) if len(factor_summary) else pd.DataFrame()
    top_corr = corr[corr["target"] == "grade_point"].head(10) if len(corr) else pd.DataFrame()

    lines = [
        "# CourseLens Full Analysis Report",
        "",
        "## 1. Executive summary",
        "",
        f"This analysis covers **{summary['course_count']}** course-level rows and **{summary['total_credits']:.1f}** credits from the edited transcript workbook.",
        f"The credit-weighted GPA is **{summary['credit_weighted_gpa']:.3f}** and the credit-weighted numeric grade is **{summary['credit_weighted_grade']:.2f}**.",
        "The modeling results should be read as exploratory evidence about within-transcript course variation, not as a causal or population-level model.",
        "",
        "## 2. Dataset description",
        "",
        f"Rows: **{schema['rows']}**  ",
        f"Columns after canonical naming: **{schema['columns']}**",
        "",
        "The source workbook is manually edited and already excludes courses that are not suitable for numeric GPA analysis. Python recomputes all summaries rather than trusting the workbook summary sheet.",
        "",
        "## 3. Privacy/de-identification note",
        "",
        "Raw course names and course selection IDs are treated as identifiers. Default exported datasets use anonymous course IDs and exclude raw identifiers.",
        "",
        "## 4. Data quality and missingness",
        "",
        "See `data_quality_report.md` for full details. Numeric questionnaire missing values are retained and imputed inside model cross-validation.",
        "",
        "## 5. Feature engineering",
        "",
        "Derived features include term index, credit-weighted GPA contribution, engagement mean, study-effort mean, and learning-experience mean. The final prediction models keep the original questionnaire items; averages are added only as extra candidate features. Ridge/ElasticNet/Lasso are used to reduce instability from correlated raw items and average features. `expected_grade` and `expectation_gap` are reported descriptively only and are excluded from all final prediction models to avoid target leakage.",
        "",
        "## 6. Descriptive GPA analysis",
        "",
        f"- Mean grade: **{summary['mean_grade']:.2f}**",
        f"- Median grade: **{summary['median_grade']:.2f}**",
        f"- Mean grade point: **{summary['mean_grade_point']:.3f}**",
        f"- Credit-weighted GPA: **{summary['credit_weighted_gpa']:.3f}**",
        f"- Mean expectation gap (`expected_grade - grade`): **{summary['mean_expectation_gap']:.2f}**",
        "",
        "Figures: `figures/grade_distribution.png`, `figures/gpa_by_term.png`, `figures/feature_distributions.png`.",
        "",
        "## 7. Semester trends",
        "",
        df_to_markdown(descriptive["term_summary"]),
        "",
        "## 8. Correlation analysis",
        "",
        "Top exploratory associations with GPA points:",
        "",
        df_to_markdown(top_corr[["feature", "中文含义 / 原始题项", "n", "spearman_r", "pearson_r"]]) if len(top_corr) else "No correlations available.",
        "",
        "## 9. Statistical models",
        "",
        "Small OLS models are written to `tables/ols_models.csv`. Coefficients and p-values are exploratory because the sample size is small and several predictors are self-reported.",
        "",
        "## 10. ML model comparison",
        "",
        df_to_markdown(best) if len(best) else "No model metrics available.",
        "",
        "Baseline row:",
        "",
        df_to_markdown(baseline) if len(baseline) else "No baseline available.",
        "",
        "## 11. Important factors associated with GPA",
        "",
        df_to_markdown(top_factors[["feature", "中文含义 / 原始题项", "factor_group", "ridge_standardized_coef", "permutation_importance_mean", "spearman_r"]]) if len(top_factors) else "No factor summary available.",
        "",
        "Interpret factors as stronger when correlation direction, Ridge coefficient direction, and permutation importance agree. Inconsistent signals should be considered unstable.",
        "",
        "## 12. Limitations",
        "",
        "- One student only; findings are within-person/course-level patterns.",
        "- Only about 54 course rows, which is small for ML.",
        "- Reflection variables are self-reported and may have been filled after outcomes were known.",
        "- `expected_grade` and `expectation_gap` are excluded from prediction models because they are outcome-adjacent leakage features.",
        "- Original questionnaire items are correlated with each other and with derived averages; regularized models reduce but do not eliminate this interpretability problem.",
        "- One effort column contains missing values and is imputed for modeling.",
        "- The overall effort field is formula-derived, not an independent survey answer.",
        "- Correlation and feature importance are not causal evidence.",
        "- Grades and GPA points are high and compressed, which limits predictive signal.",
        "",
        "## 13. Recommended next data to collect",
        "",
        "- Pre-course expectations and baseline interest before grades are known.",
        "- Weekly effort/time logs rather than retrospective effort only.",
        "- Assessment structure: exam, paper, project, presentation, participation share.",
        "- Course type/category and workload estimates.",
        "- More students or more terms if the goal becomes general prediction.",
        "",
        "## 14. Next development steps",
        "",
        "- Stabilize cleaning rules so the edited workbook can be regenerated from raw transcript and questionnaire data.",
        "- Add SHAP only after the modeling feature set is stable and more data is available.",
        "- Build a questionnaire schema so future terms can be appended consistently.",
    ]
    (output_dir / "full_analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    machine = {
        "rows": schema["rows"],
        "course_count": summary["course_count"],
        "total_credits": summary["total_credits"],
        "credit_weighted_gpa": summary["credit_weighted_gpa"],
        "best_model_by_mae": best.head(1).to_dict(orient="records") if len(best) else [],
        "top_factors": top_factors.head(10).to_dict(orient="records") if len(top_factors) else [],
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(machine, ensure_ascii=False, indent=2), encoding="utf-8")
