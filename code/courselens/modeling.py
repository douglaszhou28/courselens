from pathlib import Path
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut, RepeatedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .config import FEATURE_SETS, PALETTE
from .markdown import df_to_markdown

PRIMARY_FEATURE_SET = "raw_plus_averages"


def _linear_pipeline(features, regressor):
    prep = ColumnTransformer([("num", Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scaler", StandardScaler())]), features)])
    return Pipeline([("preprocessor", prep), ("regressor", regressor)])


def _tree_pipeline(features, regressor):
    prep = ColumnTransformer([("num", SimpleImputer(strategy="median", add_indicator=True), features)])
    return Pipeline([("preprocessor", prep), ("regressor", regressor)])


def model_specs(features, random_state=42):
    specs = {
        "dummy_mean": _tree_pipeline(features, DummyRegressor(strategy="mean")),
        "linear": _linear_pipeline(features, LinearRegression()),
        "ridge": _linear_pipeline(features, Ridge(alpha=10.0)),
        "elastic_net": _linear_pipeline(features, ElasticNet(alpha=0.05, l1_ratio=0.3, random_state=random_state, max_iter=10000)),
        "lasso": _linear_pipeline(features, Lasso(alpha=0.03, random_state=random_state, max_iter=10000)),
        "random_forest_shallow": _tree_pipeline(features, RandomForestRegressor(n_estimators=200, max_depth=3, min_samples_leaf=5, random_state=random_state)),
        "gradient_boosting_conservative": _tree_pipeline(features, GradientBoostingRegressor(n_estimators=60, max_depth=2, learning_rate=0.05, min_samples_leaf=5, random_state=random_state)),
    }
    return specs


def _metrics(y, pred):
    return {"mae": mean_absolute_error(y, pred), "rmse": float(np.sqrt(mean_squared_error(y, pred))), "r2": r2_score(y, pred)}


def repeated_cv_metrics(model, X, y, cv):
    rows = []
    for train_idx, test_idx in cv.split(X, y):
        fitted = clone(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fitted.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = fitted.predict(X.iloc[test_idx])
        rows.append(_metrics(y.iloc[test_idx], pred))
    return {
        "mae": float(np.mean([r["mae"] for r in rows])),
        "rmse": float(np.mean([r["rmse"] for r in rows])),
        "r2": float(np.mean([r["r2"] for r in rows if np.isfinite(r["r2"])])) if any(np.isfinite(r["r2"]) for r in rows) else np.nan,
    }


def evaluate_models(df: pd.DataFrame, target="grade_point", feature_sets=None, random_state=42) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_sets = feature_sets or FEATURE_SETS
    metric_rows, pred_frames = [], []
    for set_name, features in feature_sets.items():
        features = [f for f in features if f in df.columns]
        data = df[[target] + features].dropna(subset=[target]).copy()
        X, y = data[features], data[target]
        if len(data) < 10:
            continue
        cv = RepeatedKFold(n_splits=min(5, len(data)), n_repeats=5, random_state=random_state)
        loo = LeaveOneOut()
        for model_name, model in model_specs(features, random_state).items():
            m = repeated_cv_metrics(model, X, y, cv)
            metric_rows.append({"target": target, "feature_set": set_name, "model": model_name, "cv": "RepeatedKFold(5x5)", "n_rows": len(data), "n_features": len(features), **m})
            if set_name == PRIMARY_FEATURE_SET and model_name in ["dummy_mean", "ridge", "random_forest_shallow"]:
                pred_cv = KFold(n_splits=min(5, len(data)), shuffle=True, random_state=random_state)
                preds = cross_val_predict(model, X, y, cv=pred_cv, n_jobs=None)
                pred_frames.append(pd.DataFrame({"target": target, "feature_set": set_name, "model": model_name, "actual": y.to_numpy(), "predicted": preds, "residual": y.to_numpy() - preds}))
            if model_name in ["dummy_mean", "ridge"] and set_name == PRIMARY_FEATURE_SET:
                loo_preds = cross_val_predict(model, X, y, cv=loo, n_jobs=None)
                metric_rows.append({"target": target, "feature_set": set_name, "model": model_name, "cv": "LeaveOneOut", "n_rows": len(data), "n_features": len(features), **_metrics(y, loo_preds)})
    preds_out = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    return pd.DataFrame(metric_rows), preds_out


def fit_primary_ridge(df: pd.DataFrame, target="grade_point", feature_set=PRIMARY_FEATURE_SET, random_state=42):
    features = [f for f in FEATURE_SETS[feature_set] if f in df.columns]
    data = df[[target] + features].dropna(subset=[target]).copy()
    model = _linear_pipeline(features, Ridge(alpha=10.0))
    model.fit(data[features], data[target])
    return model, features, data


def write_model_report(df: pd.DataFrame, output_dir: Path, target="grade_point", random_state=42) -> dict:
    tables = output_dir / "tables"
    figures = output_dir / "figures"
    metrics, preds = evaluate_models(df, target=target, random_state=random_state)
    metrics.to_csv(tables / "model_metrics.csv", index=False)
    preds.to_csv(tables / "model_predictions.csv", index=False)

    if len(preds):
        ridge = preds[preds["model"] == "ridge"]
        fig, ax = plt.subplots(figsize=(5.5, 5))
        sns.scatterplot(data=ridge, x="actual", y="predicted", color=PALETTE[0], s=55, ax=ax)
        lo, hi = min(ridge["actual"].min(), ridge["predicted"].min()), max(ridge["actual"].max(), ridge["predicted"].max())
        ax.plot([lo, hi], [lo, hi], color="#444444", linewidth=1.5, linestyle="--")
        ax.set_title("Predicted vs actual GPA points: Ridge")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        fig.tight_layout()
        fig.savefig(figures / "predicted_vs_actual.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.histplot(ridge["residual"], bins=10, kde=True, color=PALETTE[2], ax=ax)
        ax.axvline(0, color="#444444", linewidth=1.5)
        ax.set_title("Ridge residuals")
        ax.set_xlabel("Actual - predicted")
        fig.tight_layout()
        fig.savefig(figures / "residuals.png", dpi=180)
        plt.close(fig)

    best = metrics.sort_values("mae").head(10) if len(metrics) else pd.DataFrame()
    baseline = metrics[metrics["model"] == "dummy_mean"].sort_values("mae").head(1) if len(metrics) else pd.DataFrame()
    lines = [
        "# Model Report", "", f"Target: `{target}`", "",
        "## Best models by cross-validated MAE", "", df_to_markdown(best), "",
        "## Baseline", "", df_to_markdown(baseline) if len(baseline) else "No baseline results.", "",
        "## Caution", "",
        "The primary feature set is `raw_plus_averages`: it keeps all original questionnaire items and adds average/composite features as extra candidates. Ridge/ElasticNet/Lasso are included specifically because the original items and average features are correlated.",
    ]
    (output_dir / "model_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"metrics": metrics, "predictions": preds}
