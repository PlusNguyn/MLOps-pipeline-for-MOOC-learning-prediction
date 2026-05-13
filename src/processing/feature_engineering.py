from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

SEED = 42

CATEGORICAL_COLUMNS = [
    "code_module",
    "code_presentation",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability",
]

FEATURE_COLUMNS = [
    "code_module",
    "code_presentation",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "num_of_prev_attempts",
    "studied_credits",
    "disability",
    "total_clicks",
    "active_days",
    "avg_daily_clicks",
    "max_clicks_day",
    "engagement_span",
    "avg_score",
    "min_score",
    "submission_count",
    "late_submissions",
    "weighted_avg",
]

TARGET_COLUMN = "label"
LABEL_MAP = {
    "Withdrawn": 0,
    "Fail": 0,
    "Pass": 1,
    "Distinction": 1,
}

VLE_FILL_ZERO_COLUMNS = [
    "total_clicks",
    "active_days",
    "avg_daily_clicks",
    "max_clicks_day",
    "last_activity",
    "first_activity",
    "engagement_span",
]

DEFAULT_MODEL_FEATURES = {
    "total_clicks": "num_clicks",
    "active_days": "days_active",
    "avg_score": "avg_score",
    "min_score": "avg_score",
    "weighted_avg": "avg_score",
    "studied_credits": "studied_credits",
}


def build_feature_row(
    num_clicks: float,
    days_active: float,
    avg_score: float,
    studied_credits: float,
) -> dict[str, float]:
    safe_days = max(float(days_active), 1.0)
    safe_credits = max(float(studied_credits), 1.0)

    engagement_score = (
        0.5 * (float(num_clicks) / 1000.0) +
        0.25 * (float(days_active) / 30.0) +
        0.25 * (float(avg_score) / 100.0)
    )

    return {
        "num_clicks": float(num_clicks),
        "days_active": float(days_active),
        "avg_score": float(avg_score),
        "studied_credits": float(studied_credits),
        "engagement_score": engagement_score,
        "consistency": float(days_active) / safe_credits,
        "avg_daily_clicks": float(num_clicks) / safe_days,
    }


def _weighted_average(group: pd.DataFrame) -> float:
    weights = group["weight"].fillna(1.0)
    weight_sum = float(weights.sum())
    if weight_sum == 0:
        return float(group["score"].mean())
    return float((group["score"] * weights).sum() / weight_sum)


def _build_vle_features(vle: pd.DataFrame) -> pd.DataFrame:
    vle_features = vle.groupby("id_student").agg(
        total_clicks=("sum_click", "sum"),
        active_days=("date", "nunique"),
        avg_daily_clicks=("sum_click", "mean"),
        max_clicks_day=("sum_click", "max"),
        last_activity=("date", "max"),
        first_activity=("date", "min"),
    ).reset_index()

    vle_features["engagement_span"] = (
        vle_features["last_activity"] - vle_features["first_activity"]
    )

    return vle_features


def _build_assessment_features(
    assessments: pd.DataFrame,
    assess_def: pd.DataFrame,
) -> pd.DataFrame:
    assess_merged = assessments.merge(
        assess_def[["id_assessment", "weight"]],
        on="id_assessment",
        how="left",
    )

    assess_features = assess_merged.groupby("id_student").agg(
        avg_score=("score", "mean"),
        min_score=("score", "min"),
        submission_count=("id_assessment", "count"),
        late_submissions=("is_banked", "sum"),
    ).reset_index()

    assess_merged["weight"] = assess_merged["weight"].fillna(1.0)
    assess_merged["weighted_score"] = assess_merged["score"] * assess_merged["weight"]
    weight_sum = assess_merged.groupby("id_student")["weight"].sum().replace(0, np.nan)
    weighted_score_sum = assess_merged.groupby("id_student")["weighted_score"].sum()
    weighted_avg = (
        weighted_score_sum.div(weight_sum)
        .fillna(assess_merged.groupby("id_student")["score"].mean())
        .reset_index(name="weighted_avg")
    )

    return assess_features.merge(
        weighted_avg,
        on="id_student",
        how="left",
    )


def engineer_features(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    students = tables["students"].copy()
    assessments = tables["assessments"].copy()
    vle = tables["vle"].copy()
    assess_def = tables["assess_def"].copy()

    students[TARGET_COLUMN] = students["final_result"].map(LABEL_MAP)
    students = students.dropna(subset=[TARGET_COLUMN]).copy()
    students[TARGET_COLUMN] = students[TARGET_COLUMN].astype(int)

    vle_features = _build_vle_features(vle)
    assess_features = _build_assessment_features(assessments, assess_def)

    df = students[
        [
            "id_student",
            "code_module",
            "code_presentation",
            "gender",
            "region",
            "highest_education",
            "imd_band",
            "age_band",
            "num_of_prev_attempts",
            "studied_credits",
            "disability",
            TARGET_COLUMN,
        ]
    ].copy()

    df = df.merge(vle_features, on="id_student", how="left")
    df = df.merge(assess_features, on="id_student", how="left")

    df[VLE_FILL_ZERO_COLUMNS] = df[VLE_FILL_ZERO_COLUMNS].fillna(0)

    for column in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column].astype(str))

    numeric_feature_columns = [col for col in FEATURE_COLUMNS if col not in CATEGORICAL_COLUMNS]
    df[numeric_feature_columns] = df[numeric_feature_columns].replace([np.inf, -np.inf], np.nan)
    df[numeric_feature_columns] = df[numeric_feature_columns].fillna(
        df[numeric_feature_columns].median(numeric_only=True)
    )

    return df


def build_inference_frame(
    payload: dict[str, float | int | None],
    medians: dict[str, float],
) -> pd.DataFrame:
    row = {feature: float(medians.get(feature, 0.0)) for feature in FEATURE_COLUMNS}

    for model_feature, payload_key in DEFAULT_MODEL_FEATURES.items():
        value = payload.get(payload_key)
        if value is not None:
            row[model_feature] = float(value)

    num_clicks = payload.get("num_clicks")
    days_active = payload.get("days_active")
    if num_clicks is not None and days_active is not None:
        safe_days = max(float(days_active), 1.0)
        row["avg_daily_clicks"] = float(num_clicks) / safe_days
        row["max_clicks_day"] = max(row["max_clicks_day"], row["avg_daily_clicks"])
        row["engagement_span"] = max(float(days_active) - 1.0, 0.0)

    for feature in FEATURE_COLUMNS:
        direct_value = payload.get(feature)
        if direct_value is not None:
            row[feature] = float(direct_value)

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def save_feature_metadata(
    output_path: str | Path,
    medians: dict[str, float] | None = None,
) -> dict[str, object]:
    payload = {"features": FEATURE_COLUMNS, "target": TARGET_COLUMN}
    if medians is not None:
        payload["medians"] = medians

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
