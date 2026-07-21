from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from courselens.privacy import safe_export_frame


def test_safe_export_removes_identifiers():
    df = pd.DataFrame({"course_name": ["课程A"], "course_selection_id": ["id"], "notes": [""], "grade": [90]})
    safe = safe_export_frame(df)
    assert "course_id" in safe.columns
    assert "course_name" not in safe.columns
    assert "course_selection_id" not in safe.columns
    assert "notes" not in safe.columns
