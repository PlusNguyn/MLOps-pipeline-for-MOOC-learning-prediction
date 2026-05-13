from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
SEED = 42


@dataclass(frozen=True)
class SyntheticBatch:
    students: pd.DataFrame
    assessments: pd.DataFrame
    vle: pd.DataFrame

    def to_payload(self) -> dict[str, list[dict[str, object]]]:
        return {
            "students": self.students.to_dict(orient="records"),
            "assessments": self.assessments.to_dict(orient="records"),
            "vle": self.vle.to_dict(orient="records"),
        }

    def row_counts(self) -> dict[str, int]:
        return {
            "students": int(len(self.students)),
            "assessments": int(len(self.assessments)),
            "vle": int(len(self.vle)),
        }


@lru_cache(maxsize=1)
def _load_reference_tables() -> dict[str, pd.DataFrame]:
    return {
        "students": pd.read_csv(RAW_DATA_DIR / "studentInfo.csv"),
        "assessments": pd.read_csv(RAW_DATA_DIR / "studentAssessment.csv"),
        "vle": pd.read_csv(RAW_DATA_DIR / "studentVle.csv"),
        "assess_def": pd.read_csv(RAW_DATA_DIR / "assessments.csv"),
    }


def _resolve_target_date(target_date: str | date | None) -> date:
    if target_date is None:
        return datetime.utcnow().date()
    if isinstance(target_date, date):
        return target_date
    return datetime.strptime(target_date, "%Y-%m-%d").date()


def _build_rng(target_date: date, batch_size: int) -> np.random.Generator:
    seed = SEED + target_date.toordinal() + (batch_size * 17)
    return np.random.default_rng(seed)


def _success_probability(row: pd.Series, drift: float, rng: np.random.Generator) -> float:
    education_bonus = {
        "Post Graduate Qualification": 0.08,
        "HE Qualification": 0.05,
        "A Level or equivalent": 0.02,
        "Lower Than A Level": -0.03,
        "No Formal quals": -0.08,
    }.get(str(row["highest_education"]), 0.0)

    age_bonus = {
        "0-35": 0.0,
        "35-55": 0.03,
        "55<=": 0.01,
    }.get(str(row["age_band"]), 0.0)

    disability_penalty = -0.03 if str(row["disability"]) == "Y" else 0.0
    attempts_penalty = -0.05 * min(int(row["num_of_prev_attempts"]), 3)
    credits_bonus = 0.04 if int(row["studied_credits"]) <= 90 else -0.02
    noise = float(rng.normal(0.0, 0.035))

    probability = 0.58 + drift + education_bonus + age_bonus + credits_bonus
    probability += disability_penalty + attempts_penalty + noise
    return float(np.clip(probability, 0.32, 0.8))


def _sample_final_result(success_probability: float, rng: np.random.Generator) -> str:
    if rng.random() < success_probability:
        return "Distinction" if rng.random() < 0.16 else "Pass"
    return "Withdrawn" if rng.random() < 0.38 else "Fail"


def _score_center(final_result: str) -> float:
    return {
        "Distinction": 86.0,
        "Pass": 67.0,
        "Fail": 38.0,
        "Withdrawn": 24.0,
    }[final_result]


def _generate_assessments(
    student_row: pd.Series,
    final_result: str,
    rng: np.random.Generator,
    assess_def: pd.DataFrame,
) -> pd.DataFrame:
    pool = assess_def[
        (assess_def["code_module"] == student_row["code_module"]) &
        (assess_def["code_presentation"] == student_row["code_presentation"])
    ]
    if pool.empty:
        pool = assess_def[assess_def["code_module"] == student_row["code_module"]]
    if pool.empty:
        pool = assess_def

    submission_count = int(rng.integers(3, 7))
    sampled = pool.sample(
        n=submission_count,
        replace=len(pool) < submission_count,
        random_state=int(rng.integers(0, 1_000_000)),
    ).reset_index(drop=True)

    center = _score_center(final_result)
    score_sigma = 8.0 if final_result in {"Pass", "Distinction"} else 12.0
    late_rate = 0.08 if final_result in {"Pass", "Distinction"} else 0.24
    if final_result == "Withdrawn":
        late_rate = 0.35

    rows: list[dict[str, object]] = []
    for _, assessment in sampled.iterrows():
        score = float(np.clip(rng.normal(center, score_sigma), 0, 100))
        submitted_delta = int(rng.integers(-3, 8))
        assessment_date = assessment.get("date")
        if pd.isna(assessment_date):
            assessment_date = int(rng.integers(10, 220))
        date_submitted = int(max(int(float(assessment_date)) + submitted_delta, -20))
        rows.append(
            {
                "id_assessment": int(assessment["id_assessment"]),
                "id_student": int(student_row["id_student"]),
                "date_submitted": date_submitted,
                "is_banked": int(rng.random() < late_rate),
                "score": round(score, 2),
            }
        )

    return pd.DataFrame(rows)


def _generate_vle(
    student_row: pd.Series,
    final_result: str,
    rng: np.random.Generator,
    vle_reference: pd.DataFrame,
) -> pd.DataFrame:
    pool = vle_reference[
        (vle_reference["code_module"] == student_row["code_module"]) &
        (vle_reference["code_presentation"] == student_row["code_presentation"])
    ]
    if pool.empty:
        pool = vle_reference[vle_reference["code_module"] == student_row["code_module"]]
    if pool.empty:
        pool = vle_reference

    days_active = int(rng.integers(10, 28))
    if final_result == "Distinction":
        days_active += 6
    elif final_result == "Withdrawn":
        days_active = max(days_active - 6, 4)

    base_clicks = {
        "Distinction": rng.integers(35, 90),
        "Pass": rng.integers(18, 55),
        "Fail": rng.integers(8, 28),
        "Withdrawn": rng.integers(3, 18),
    }[final_result]

    start_day = int(rng.integers(-15, 20))
    day_offsets = np.sort(rng.choice(np.arange(days_active + 10), size=days_active, replace=False))
    records: list[dict[str, object]] = []

    sampled_sites = pool.sample(
        n=days_active,
        replace=len(pool) < days_active,
        random_state=int(rng.integers(0, 1_000_000)),
    ).reset_index(drop=True)

    for offset, (_, site_row) in zip(day_offsets, sampled_sites.iterrows(), strict=False):
        clicks = int(max(1, round(rng.normal(base_clicks, max(base_clicks * 0.25, 2.0)))))
        if final_result == "Withdrawn" and offset > day_offsets.mean():
            clicks = max(1, int(clicks * 0.6))

        records.append(
            {
                "code_module": student_row["code_module"],
                "code_presentation": student_row["code_presentation"],
                "id_student": int(student_row["id_student"]),
                "id_site": int(site_row["id_site"]),
                "date": int(start_day + offset),
                "sum_click": clicks,
            }
        )

    return pd.DataFrame(records)


def generate_synthetic_oulad_batch(
    target_date: str | date | None = None,
    batch_size: int = 64,
) -> SyntheticBatch:
    target_date = _resolve_target_date(target_date)
    batch_size = max(int(batch_size), 1)
    rng = _build_rng(target_date, batch_size)
    reference = _load_reference_tables()

    students_reference = reference["students"]
    assess_reference = reference["assessments"]
    vle_reference = reference["vle"]
    assess_def = reference["assess_def"]

    templates = students_reference.sample(
        n=batch_size,
        replace=True,
        random_state=int(rng.integers(0, 1_000_000)),
    ).reset_index(drop=True)

    max_student_id = int(students_reference["id_student"].max())
    daily_offset = (target_date - date(2026, 1, 1)).days
    drift = float(np.clip(daily_offset / 365.0, -0.03, 0.07))

    synthetic_students: list[dict[str, object]] = []
    synthetic_assessments: list[pd.DataFrame] = []
    synthetic_vle: list[pd.DataFrame] = []

    for idx, (_, template) in enumerate(templates.iterrows(), start=1):
        new_student = template.copy()
        new_student["id_student"] = max_student_id + (daily_offset * 10_000) + idx

        success_probability = _success_probability(new_student, drift=drift, rng=rng)
        final_result = _sample_final_result(success_probability, rng=rng)
        new_student["final_result"] = final_result

        prev_attempts = int(new_student["num_of_prev_attempts"])
        if final_result in {"Fail", "Withdrawn"} and rng.random() < 0.25:
            new_student["num_of_prev_attempts"] = min(prev_attempts + 1, 6)

        synthetic_students.append(new_student.to_dict())
        synthetic_assessments.append(
            _generate_assessments(new_student, final_result, rng=rng, assess_def=assess_def)
        )
        synthetic_vle.append(
            _generate_vle(new_student, final_result, rng=rng, vle_reference=vle_reference)
        )

    students_df = pd.DataFrame(synthetic_students)[students_reference.columns]
    assessments_df = pd.concat(synthetic_assessments, ignore_index=True)
    assessments_df = assessments_df[assess_reference.columns]
    vle_df = pd.concat(synthetic_vle, ignore_index=True)
    vle_df = vle_df[vle_reference.columns]

    return SyntheticBatch(
        students=students_df,
        assessments=assessments_df,
        vle=vle_df,
    )
