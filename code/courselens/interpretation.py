from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from .config import FEATURE_SETS, PALETTE
from .modeling import fit_primary_ridge, PRIMARY_FEATURE_SET
from .markdown import df_to_markdown
from .feature_engineering import add_feature_labels

FACTOR_GROUPS = {
    "credits": "credit/course-load effects",
    "term_index": "semester/time effects",
    "offline_attendance": "effort and engagement",
    "class_participation": "effort and engagement",
    "ongoing_study_effort": "effort and engagement",
    "exam_effort": "effort and engagement",
    "overall_effort_composite": "effort and engagement",
    "engagement_mean": "effort and engagement",
    "study_effort_mean": "effort and engagement",
    "interest": "course experience",
    "course_design_gain": "course experience",
    "teaching_helpfulness": "course experience",
    "learning_experience_mean": "course experience",
    "perceived_mastery": "perceived mastery",
    "difficulty": "difficulty",
    "expected_grade": "expectations/calibration",
    "expectation_gap": "expectations/calibration",
}


def ridge_coefficients(model, features):
    reg = model.named_steps["regressor"]
    coefs = reg.coef_[:len(features)]
    return pd.DataFrame({"feature": features, "ridge_standardized_coef": coefs})


def permutation_table(model, data, features, target="grade_point", random_state=42):
    result = permutation_importance(model, data[features], data[target], n_repeats=50, random_state=random_state, scoring="neg_mean_absolute_error")
    return pd.DataFrame({"feature": features, "permutation_importance_mean": result.importances_mean, "permutation_importance_std": result.importances_std})


def factor_summary(correlations: pd.DataFrame, coef: pd.DataFrame, perm: pd.DataFrame) -> pd.DataFrame:
    corr = correlations[correlations["target"] == "grade_point"][["feature", "spearman_r", "pearson_r"]]
    out = coef.merge(perm, on="feature", how="outer").merge(corr, on="feature", how="left")
    out["factor_group"] = out["feature"].map(FACTOR_GROUPS).fillna("other")
    out["abs_ridge_coef"] = out["ridge_standardized_coef"].abs()
    out = out.sort_values(["permutation_importance_mean", "abs_ridge_coef"], ascending=False)
    return add_feature_labels(out)


def write_factor_report(df: pd.DataFrame, correlations: pd.DataFrame, output_dir: Path, random_state=42) -> dict:
    tables = output_dir / "tables"
    figures = output_dir / "figures"
    model, features, data = fit_primary_ridge(df, feature_set=PRIMARY_FEATURE_SET, random_state=random_state)
    coef = ridge_coefficients(model, features)
    perm = permutation_table(model, data, features, random_state=random_state)
    summary = factor_summary(correlations, coef, perm)
    coef.to_csv(tables / "ridge_coefficients.csv", index=False)
    perm.to_csv(tables / "permutation_importance.csv", index=False)
    summary.to_csv(tables / "factor_summary.csv", index=False)

    plot = summary.sort_values("permutation_importance_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=plot, y="feature", x="permutation_importance_mean", color=PALETTE[3], ax=ax)
    ax.set_title("Permutation importance: compact Ridge model")
    ax.set_xlabel("Mean increase in MAE when permuted")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(figures / "feature_importance.png", dpi=180)
    plt.close(fig)

    lines = ["# Factor Analysis Report", "", "The table below combines three exploratory signals: Spearman correlation with GPA, standardized Ridge coefficient, and permutation importance in the `raw_plus_averages` Ridge model. This model keeps the original questionnaire items and includes averages as additional derived features; Ridge regularization is used because these features are correlated.", "", df_to_markdown(summary), "", "## Interpretation rule", "", "Prioritize factors that have consistent direction across correlation and coefficients and non-trivial permutation importance. Do not interpret these as causal effects. When raw items and their averages disagree, prefer the raw-item interpretation and treat the average as a summary index."]
    (output_dir / "factor_analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"factor_summary": summary, "coefficients": coef, "permutation": perm}
