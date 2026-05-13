from pathlib import Path

import pandas as pd

SYNTHETIC_DIRNAME = "synthetic"
CURRENT_DIRNAME = "current"
TABLE_FILE_MAP = {
    "students": "studentInfo.csv",
    "assessments": "studentAssessment.csv",
    "vle": "studentVle.csv",
}
DEDUP_KEYS = {
    "students": ["code_module", "code_presentation", "id_student"],
    "assessments": ["id_assessment", "id_student", "date_submitted"],
    "vle": ["code_module", "code_presentation", "id_student", "id_site", "date"],
}


def _merge_synthetic_table(
    table_name: str,
    base_df: pd.DataFrame,
    synthetic_dir: Path,
) -> pd.DataFrame:
    synthetic_path = synthetic_dir / TABLE_FILE_MAP[table_name]
    if not synthetic_path.exists():
        return base_df

    synthetic_df = pd.read_csv(synthetic_path)
    if synthetic_df.empty:
        return base_df

    merged = pd.concat([base_df, synthetic_df], ignore_index=True, sort=False)
    return merged.drop_duplicates(subset=DEDUP_KEYS[table_name], keep="last")


def load_oulad(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    synthetic_dir = data_dir.parent / SYNTHETIC_DIRNAME / CURRENT_DIRNAME

    students = pd.read_csv(data_dir / "studentInfo.csv")
    assessments = pd.read_csv(data_dir / "studentAssessment.csv")
    vle = pd.read_csv(data_dir / "studentVle.csv")

    return {
        "students": _merge_synthetic_table("students", students, synthetic_dir),
        "assessments": _merge_synthetic_table("assessments", assessments, synthetic_dir),
        "vle": _merge_synthetic_table("vle", vle, synthetic_dir),
        "assess_def": pd.read_csv(data_dir / "assessments.csv"),
    }
