from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from courselens.modeling import evaluate_models


def test_modeling_returns_baseline_and_ridge():
    rng = np.random.default_rng(42)
    n = 20
    df = pd.DataFrame({
        "grade_point": 4 + rng.normal(0, 0.2, n),
        "credits": rng.choice([1, 2, 3, 4], n),
        "term_index": np.tile([1, 2, 3, 4], 5),
        "offline_attendance": rng.integers(1, 6, n),
        "class_participation": rng.integers(1, 6, n),
        "ongoing_study_effort": rng.integers(1, 6, n),
        "exam_effort": rng.integers(1, 6, n),
        "perceived_mastery": rng.integers(1, 6, n),
        "interest": rng.integers(1, 6, n),
        "course_design_gain": rng.integers(1, 6, n),
        "teaching_helpfulness": rng.integers(1, 6, n),
        "difficulty": rng.integers(1, 6, n),
        "engagement_mean": rng.normal(3, 0.5, n),
        "study_effort_mean": rng.normal(3, 0.5, n),
        "learning_experience_mean": rng.normal(3, 0.5, n),
    })
    features = ["credits", "term_index", "offline_attendance", "class_participation", "ongoing_study_effort", "exam_effort", "perceived_mastery", "interest", "course_design_gain", "teaching_helpfulness", "difficulty", "engagement_mean", "study_effort_mean", "learning_experience_mean"]
    metrics, preds = evaluate_models(df, feature_sets={"raw_plus_averages": features})
    assert {"dummy_mean", "ridge"}.issubset(set(metrics["model"]))
    assert len(preds) > 0
