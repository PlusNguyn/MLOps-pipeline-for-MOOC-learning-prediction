from pathlib import Path

import pandas as pd


def load_oulad(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)

    return {
        "students": pd.read_csv(data_dir / "studentInfo.csv"),
        "assessments": pd.read_csv(data_dir / "studentAssessment.csv"),
        "vle": pd.read_csv(data_dir / "studentVle.csv"),
        "assess_def": pd.read_csv(data_dir / "assessments.csv"),
    }
