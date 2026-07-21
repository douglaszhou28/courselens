from pathlib import Path

COURSE_SHEET = "课程成绩"
SUMMARY_SHEET = "汇总"

COLUMN_MAP = {
    "课程名称": "course_name",
    "选课课号": "course_selection_id",
    "成绩": "grade",
    "学分": "credits",
    "绩点": "grade_point",
    "备注": "notes",
    "我出席了这门课绝大多数的线下课堂。": "offline_attendance",
    "我认真参与了这门课绝大多数的课堂，不论以线下、线上还是看录播的方式。": "class_participation",
    "在非考试周期间，我也会持续投入时间学习这门课程（不包括听课，但包括认真完成作业）。": "ongoing_study_effort",
    "为了考试或最终考核，我在这门课上投入了（客观上相对于其他课）充足的时间。": "exam_effort",
    "总体而言，我在这门课程上的投入程度较高。": "overall_effort_composite",
    "在课程结束时，我认为自己已经掌握了这门课程的大部分核心知识或技能。": "perceived_mastery",
    "我对这门课程实际教授的内容感兴趣。": "interest",
    "我在课程设计（如教材、作业、考试、项目等，不包括教师授课）中收获满满。": "course_design_gain",
    "任课老师的授课方式能够帮助我理解课程内容。": "teaching_helpfulness",
    "我认为这门课程的知识难度较高，具有挑战性。": "difficulty",
    "根据这门课的实际掌握情况，我认为自己应得的成绩是多少？": "expected_grade",
}

REQUIRED_COLUMNS = [
    "course_name", "course_selection_id", "grade", "credits", "grade_point",
    "offline_attendance", "class_participation", "ongoing_study_effort",
    "exam_effort", "overall_effort_composite", "perceived_mastery", "interest",
    "course_design_gain", "teaching_helpfulness", "difficulty", "expected_grade",
]

NUMERIC_COLUMNS = [
    "grade", "credits", "grade_point", "offline_attendance", "class_participation",
    "ongoing_study_effort", "exam_effort", "overall_effort_composite",
    "perceived_mastery", "interest", "course_design_gain", "teaching_helpfulness",
    "difficulty", "expected_grade",
]

IDENTIFIER_COLUMNS = ["course_name", "course_selection_id", "notes"]
TARGETS = ["grade_point", "grade"]

COMPONENT_FEATURES = [
    "credits", "term_index", "offline_attendance", "class_participation",
    "ongoing_study_effort", "exam_effort", "perceived_mastery", "interest",
    "course_design_gain", "teaching_helpfulness", "difficulty",
]
AVERAGE_FEATURES = [
    "credits", "term_index", "engagement_mean", "study_effort_mean",
    "learning_experience_mean", "perceived_mastery", "difficulty",
]
RAW_PLUS_AVERAGES_FEATURES = [
    "credits", "term_index", "offline_attendance", "class_participation",
    "ongoing_study_effort", "exam_effort", "perceived_mastery", "interest",
    "course_design_gain", "teaching_helpfulness", "difficulty",
    "engagement_mean", "study_effort_mean", "learning_experience_mean",
]
PRE_OUTCOME_FEATURES = [
    "credits", "term_index", "offline_attendance", "class_participation",
    "ongoing_study_effort", "exam_effort", "interest", "course_design_gain",
    "teaching_helpfulness", "difficulty",
]
COMPOSITE_FEATURES = [
    "credits", "term_index", "offline_attendance", "overall_effort_composite",
    "perceived_mastery", "interest", "course_design_gain", "teaching_helpfulness",
    "difficulty",
]

LEAKAGE_COLUMNS = ["expected_grade", "expectation_gap", "actual_above_expected"]

FEATURE_SETS = {
    "component_raw_items": COMPONENT_FEATURES,
    "raw_plus_averages": RAW_PLUS_AVERAGES_FEATURES,
    "average_only_diagnostic": AVERAGE_FEATURES,
    "pre_outcome_raw_items": PRE_OUTCOME_FEATURES,
    "composite_effort_diagnostic": COMPOSITE_FEATURES,
}

PALETTE = ["#2F6BFF", "#14A085", "#D97706", "#7C3AED", "#DC2626", "#0891B2"]
DEFAULT_INPUT = Path("demo/example_transcript_courses_edited.xlsx")
