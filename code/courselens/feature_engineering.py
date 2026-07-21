import re
import numpy as np
import pandas as pd
from .config import COLUMN_MAP

TERM_RE = re.compile(r"\((\d{4}-\d{4})-(\d)\)-([^-]+)")

FEATURE_LABELS = {
    "grade_point": "绩点",
    "grade": "成绩",
    "credits": "学分",
    "term_index": "学期顺序（由选课课号解析）",
    "offline_attendance": "我出席了这门课绝大多数的线下课堂。",
    "class_participation": "我认真参与了这门课绝大多数的课堂，不论以线下、线上还是看录播的方式。",
    "ongoing_study_effort": "在非考试周期间，我也会持续投入时间学习这门课程（不包括听课，但包括认真完成作业）。",
    "exam_effort": "为了考试或最终考核，我在这门课上投入了（客观上相对于其他课）充足的时间。",
    "overall_effort_composite": "总体而言，我在这门课程上的投入程度较高。（Excel 公式合成项）",
    "perceived_mastery": "在课程结束时，我认为自己已经掌握了这门课程的大部分核心知识或技能。",
    "interest": "我对这门课程实际教授的内容感兴趣。",
    "course_design_gain": "我在课程设计（如教材、作业、考试、项目等，不包括教师授课）中收获满满。",
    "teaching_helpfulness": "任课老师的授课方式能够帮助我理解课程内容。",
    "difficulty": "我认为这门课程的知识难度较高，具有挑战性。",
    "expected_grade": "根据这门课的实际掌握情况，我认为自己应得的成绩是多少？（仅描述，不进预测模型）",
    "expectation_gap": "自评应得成绩 - 实际成绩（仅描述，不进预测模型）",
    "actual_above_expected": "实际成绩 - 自评应得成绩（仅描述，不进预测模型）",
    "engagement_mean": "课堂参与均值 = 线下出席 + 课堂参与 的平均值",
    "study_effort_mean": "学习投入均值 = 非考试周学习投入 + 考试/最终考核投入 的平均值",
    "learning_experience_mean": "课程体验均值 = 兴趣 + 课程设计收获 + 教师帮助理解 的平均值",
    "credit_weighted_grade_point": "学分加权绩点贡献 = 学分 × 绩点",
    "credit_weighted_grade": "学分加权成绩贡献 = 学分 × 成绩",
    "academic_year": "学年（由选课课号解析）",
    "term_number": "学期编号（由选课课号解析）",
    "course_code_prefix": "课程代码前缀（由选课课号解析，默认不用于隐私安全导出解释）",
}

FEATURE_DESCRIPTIONS = {
    "grade_point": "Primary target: transcript grade point / GPA points",
    "grade": "Secondary target: numeric course grade",
    "credits": "Course credit weight",
    "term_index": "Ordered semester index parsed from course selection ID",
    "engagement_mean": "Mean of offline attendance and class participation",
    "study_effort_mean": "Mean of ongoing study effort and exam/final effort",
    "learning_experience_mean": "Mean of interest, course design gain, and teaching helpfulness",
    "expectation_gap": "Expected/deserved grade minus actual grade; excluded from prediction models",
    "actual_above_expected": "Actual grade minus expected/deserved grade; excluded from prediction models",
}


def parse_terms(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parsed = out["course_selection_id"].astype(str).str.extract(TERM_RE)
    out["academic_year"] = parsed[0]
    out["term_number"] = pd.to_numeric(parsed[1], errors="coerce")
    ordered = out[["academic_year", "term_number"]].drop_duplicates().sort_values(["academic_year", "term_number"])
    mapping = {(r.academic_year, r.term_number): i + 1 for i, r in enumerate(ordered.itertuples(index=False))}
    out["term_index"] = [mapping.get((a, t), np.nan) for a, t in zip(out["academic_year"], out["term_number"])]
    out["course_code_prefix"] = parsed[2]
    return out


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = parse_terms(df)
    out["credit_weighted_grade_point"] = out["credits"] * out["grade_point"]
    out["credit_weighted_grade"] = out["credits"] * out["grade"]
    out["expectation_gap"] = out["expected_grade"] - out["grade"]
    out["actual_above_expected"] = out["grade"] - out["expected_grade"]
    out["engagement_mean"] = out[["offline_attendance", "class_participation"]].mean(axis=1)
    out["study_effort_mean"] = out[["ongoing_study_effort", "exam_effort"]].mean(axis=1)
    out["learning_experience_mean"] = out[["interest", "course_design_gain", "teaching_helpfulness"]].mean(axis=1)
    return out


def feature_dictionary() -> pd.DataFrame:
    reverse = {v: k for k, v in COLUMN_MAP.items()}
    features = sorted(set(FEATURE_LABELS) | set(reverse))
    return pd.DataFrame([
        {
            "feature": f,
            "中文含义 / 原始题项": FEATURE_LABELS.get(f, reverse.get(f, "")),
            "原始Excel列名": reverse.get(f, "派生变量"),
            "说明": FEATURE_DESCRIPTIONS.get(f, ""),
        }
        for f in features
    ])


def add_feature_labels(df: pd.DataFrame, feature_col: str = "feature") -> pd.DataFrame:
    out = df.copy()
    if feature_col in out.columns:
        out.insert(out.columns.get_loc(feature_col) + 1, "中文含义 / 原始题项", out[feature_col].map(FEATURE_LABELS).fillna(""))
    return out
