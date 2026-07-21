from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .config import PALETTE
from .markdown import df_to_markdown

sns.set_theme(style="whitegrid", font="Arial", palette=PALETTE)


def academic_summary(df: pd.DataFrame) -> dict:
    total_credits = df["credits"].sum()
    return {
        "course_count": int(len(df)),
        "total_credits": float(total_credits),
        "mean_grade": float(df["grade"].mean()),
        "median_grade": float(df["grade"].median()),
        "mean_grade_point": float(df["grade_point"].mean()),
        "credit_weighted_gpa": float((df["grade_point"] * df["credits"]).sum() / total_credits),
        "credit_weighted_grade": float((df["grade"] * df["credits"]).sum() / total_credits),
        "mean_expectation_gap": float(df["expectation_gap"].mean()),
    }


def term_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["term_index", "academic_year", "term_number"], dropna=False)
    rows = []
    for keys, x in g:
        credits = x["credits"].sum()
        rows.append({
            "term_index": keys[0], "academic_year": keys[1], "term_number": keys[2],
            "course_count": len(x), "credits": credits,
            "mean_grade": x["grade"].mean(),
            "mean_grade_point": x["grade_point"].mean(),
            "credit_weighted_gpa": (x["grade_point"] * x["credits"]).sum() / credits,
        })
    return pd.DataFrame(rows).sort_values("term_index")


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.select_dtypes(include="number").columns if c not in ["term_number"]]
    return df[cols].describe().T


def write_descriptive_report(df: pd.DataFrame, output_dir: Path) -> dict:
    tables = output_dir / "tables"
    figures = output_dir / "figures"
    summary = academic_summary(df)
    terms = term_summary(df)
    stats = descriptive_stats(df)
    terms.to_csv(tables / "term_summary.csv", index=False)
    stats.to_csv(tables / "descriptive_stats.csv")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(df["grade"], bins=10, kde=True, color=PALETTE[0], ax=ax)
    ax.set_title("Grade distribution")
    ax.set_xlabel("Grade")
    ax.set_ylabel("Course count")
    fig.tight_layout()
    fig.savefig(figures / "grade_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.lineplot(data=terms, x="term_index", y="credit_weighted_gpa", marker="o", color=PALETTE[1], ax=ax)
    ax.set_title("Credit-weighted GPA by term")
    ax.set_xlabel("Term index")
    ax.set_ylabel("Credit-weighted GPA")
    ax.set_ylim(max(0, terms["credit_weighted_gpa"].min() - 0.4), min(5.2, terms["credit_weighted_gpa"].max() + 0.2))
    fig.tight_layout()
    fig.savefig(figures / "gpa_by_term.png", dpi=180)
    plt.close(fig)

    feature_cols = ["offline_attendance", "class_participation", "ongoing_study_effort", "exam_effort", "perceived_mastery", "interest", "course_design_gain", "teaching_helpfulness", "difficulty"]
    long = df[feature_cols].melt(var_name="feature", value_name="score")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.boxplot(data=long, y="feature", x="score", color=PALETTE[5], ax=ax)
    ax.set_title("Questionnaire feature distributions")
    ax.set_xlabel("Score")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(figures / "feature_distributions.png", dpi=180)
    plt.close(fig)

    lines = ["# Descriptive Summary", "", "## Academic outcomes", ""]
    for k, v in summary.items():
        lines.append(f"- **{k}**: {v:.3f}" if isinstance(v, float) else f"- **{k}**: {v}")
    lines += ["", "## Semester trend", "", df_to_markdown(terms), "", "## Notes", "", "This is a course-level description of one transcript. It should be interpreted as within-student course variation, not population-level evidence."]
    (output_dir / "descriptive_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"summary": summary, "term_summary": terms, "descriptive_stats": stats}
