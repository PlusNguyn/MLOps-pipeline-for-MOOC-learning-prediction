import pandas as pd
from pathlib import Path

JOIN_KEYS = ["code_module", "code_presentation", "id_student"]


def load_oulad(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)

    student_info = pd.read_csv(data_dir / "studentInfo.csv")
    student_vle = pd.read_csv(data_dir / "studentVle.csv")
    student_assessment = pd.read_csv(data_dir / "studentAssessment.csv")
    assessments = pd.read_csv(data_dir / "assessments.csv")

    # Aggregate clickstream at the student-module-presentation level.
    clicks_agg = student_vle.groupby(JOIN_KEYS).agg(
        num_clicks=("sum_click", "sum"),
        days_active=("date", "nunique")
    ).reset_index()

    assessment_with_context = student_assessment.merge(
        assessments[
            ["id_assessment", "code_module", "code_presentation"]
        ],
        on="id_assessment",
        how="left",
    )

    # Aggregate assessment scores with module/presentation preserved.
    score_agg = assessment_with_context.groupby(JOIN_KEYS).agg(
        avg_score=("score", "mean"),
        num_submissions=("id_assessment", "count")
    ).reset_index()

    df = student_info.merge(clicks_agg, on=JOIN_KEYS, how="left")
    df = df.merge(score_agg, on=JOIN_KEYS, how="left")

    return df
