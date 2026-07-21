from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from courselens.feature_engineering import add_derived_features


def test_term_and_gap_features():
    df = pd.DataFrame({
        "course_selection_id": ["(2023-2024-1)-A-1-1", "(2023-2024-2)-B-1-1"],
        "credits": [3, 2],
        "grade": [90, 95],
        "grade_point": [4.5, 5.0],
        "expected_grade": [92, 94],
        "offline_attendance": [4, 5],
        "class_participation": [5, 5],
        "ongoing_study_effort": [3, 4],
        "exam_effort": [4, 5],
        "interest": [5, 4],
        "course_design_gain": [4, 4],
        "teaching_helpfulness": [5, 4],
    })
    out = add_derived_features(df)
    assert out["term_index"].tolist() == [1, 2]
    assert (out["expectation_gap"] == out["expected_grade"] - out["grade"]).all()
    assert "study_effort_mean" in out.columns
