from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
from .config import PALETTE
from .markdown import df_to_markdown
from .feature_engineering import add_feature_labels


def correlation_table(df: pd.DataFrame, features: list[str], targets: list[str]) -> pd.DataFrame:
    rows = []
    for target in targets:
        for feature in features:
            x = df[feature]
            y = df[target]
            ok = x.notna() & y.notna()
            if ok.sum() < 3 or x[ok].nunique() < 2 or y[ok].nunique() < 2:
                continue
            pr, pp = pearsonr(x[ok], y[ok])
            sr, sp = spearmanr(x[ok], y[ok])
            rows.append({"target": target, "feature": feature, "n": int(ok.sum()), "pearson_r": pr, "pearson_p": pp, "spearman_r": sr, "spearman_p": sp})
    if not rows:
        return pd.DataFrame(columns=["target", "feature", "中文含义 / 原始题项", "n", "pearson_r", "pearson_p", "spearman_r", "spearman_p"])
    out = pd.DataFrame(rows)
    out["abs_spearman_r"] = out["spearman_r"].abs()
    out = out.sort_values(["target", "abs_spearman_r"], ascending=[True, False]).drop(columns=["abs_spearman_r"])
    return add_feature_labels(out)


def fit_ols_models(df: pd.DataFrame) -> pd.DataFrame:
    specs = {
        "effort_experience": ["study_effort_mean", "learning_experience_mean", "difficulty", "term_index"],
        "self_assessment": ["expected_grade", "expectation_gap", "difficulty", "term_index"],
        "compact": ["study_effort_mean", "perceived_mastery", "interest", "difficulty", "term_index"],
    }
    rows = []
    for name, features in specs.items():
        data = df[["grade_point"] + features].dropna()
        if len(data) <= len(features) + 2:
            continue
        model = sm.OLS(data["grade_point"], sm.add_constant(data[features])).fit()
        for term, coef in model.params.items():
            rows.append({"model": name, "term": term, "coef": coef, "std_err": model.bse[term], "p_value": model.pvalues[term], "r2": model.rsquared, "adj_r2": model.rsquared_adj, "n": int(model.nobs)})
    return pd.DataFrame(rows)


def write_statistical_report(df: pd.DataFrame, output_dir: Path, features: list[str]) -> dict:
    tables = output_dir / "tables"
    figures = output_dir / "figures"
    corr = correlation_table(df, features, ["grade_point", "grade"])
    corr.to_csv(tables / "correlations.csv", index=False)
    ols = fit_ols_models(df)
    ols.to_csv(tables / "ols_models.csv", index=False)

    heat_cols = ["grade_point", "grade"] + features
    heat = df[heat_cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(heat, cmap="vlag", center=0, linewidths=.5, linecolor="white", ax=ax)
    ax.set_title("Spearman correlation heatmap")
    fig.tight_layout()
    fig.savefig(figures / "correlation_heatmap.png", dpi=180)
    plt.close(fig)

    top = corr[corr["target"] == "grade_point"].copy().head(12)
    lines = [
        "# Correlation and Statistical Analysis", "",
        "## Strongest Spearman associations with grade point", "",
        df_to_markdown(top[["feature", "中文含义 / 原始题项", "n", "spearman_r", "spearman_p", "pearson_r"]]), "",
        "## OLS model coefficients", "",
        df_to_markdown(ols) if len(ols) else "No OLS models were fit.", "",
        "## Interpretation cautions", "",
        "These associations are exploratory. The dataset contains one student's course-level records, and self-reported variables may have been recorded after outcomes were known.",
    ]
    (output_dir / "correlation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"correlations": corr, "ols": ols}
