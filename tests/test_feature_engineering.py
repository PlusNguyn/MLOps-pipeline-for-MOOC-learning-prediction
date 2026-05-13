import pandas as pd
import pytest

from src.processing.feature_engineering import (
    FEATURE_COLUMNS,
    build_feature_row,
    build_inference_frame,
    engineer_features,
)


def test_build_feature_row():
    row = build_feature_row(
        num_clicks=500,
        days_active=15,
        avg_score=80,
        studied_credits=5,
    )

    assert row["num_clicks"] == 500.0
    assert row["days_active"] == 15.0
    assert row["avg_score"] == 80.0
    assert row["consistency"] == 3.0
    assert row["avg_daily_clicks"] == pytest.approx(500.0 / 15.0)


def test_engineer_features_builds_notebook_style_dataset():
    tables = {
        "students": pd.DataFrame(
            [
                {
                    "id_student": 1,
                    "code_module": "AAA",
                    "code_presentation": "2013J",
                    "gender": "M",
                    "region": "East",
                    "highest_education": "HE Qualification",
                    "imd_band": "80-90%",
                    "age_band": "35-55",
                    "num_of_prev_attempts": 0,
                    "studied_credits": 60,
                    "disability": "N",
                    "final_result": "Pass",
                },
                {
                    "id_student": 2,
                    "code_module": "BBB",
                    "code_presentation": "2014B",
                    "gender": "F",
                    "region": "Scotland",
                    "highest_education": "A Level",
                    "imd_band": "20-30%",
                    "age_band": "0-35",
                    "num_of_prev_attempts": 1,
                    "studied_credits": 30,
                    "disability": "Y",
                    "final_result": "Withdrawn",
                },
            ]
        ),
        "vle": pd.DataFrame(
            [
                {"id_student": 1, "date": 1, "sum_click": 10},
                {"id_student": 1, "date": 2, "sum_click": 30},
                {"id_student": 2, "date": 4, "sum_click": 5},
            ]
        ),
        "assessments": pd.DataFrame(
            [
                {"id_assessment": 10, "id_student": 1, "is_banked": 0, "score": 80},
                {"id_assessment": 11, "id_student": 1, "is_banked": 1, "score": 90},
                {"id_assessment": 12, "id_student": 2, "is_banked": 0, "score": 40},
            ]
        ),
        "assess_def": pd.DataFrame(
            [
                {"id_assessment": 10, "weight": 20},
                {"id_assessment": 11, "weight": 80},
                {"id_assessment": 12, "weight": 50},
            ]
        ),
    }

    result = engineer_features(tables)

    assert set(FEATURE_COLUMNS).issubset(result.columns)
    assert result["label"].tolist() == [1, 0]
    assert result.loc[result["id_student"] == 1, "weighted_avg"].iloc[0] == pytest.approx(88.0)
    assert result.loc[result["id_student"] == 1, "engagement_span"].iloc[0] == 1


def test_build_inference_frame_fills_missing_values_from_medians():
    medians = {feature: 1.0 for feature in FEATURE_COLUMNS}
    frame = build_inference_frame(
        payload={
            "num_clicks": 120,
            "days_active": 6,
            "avg_score": 75,
            "studied_credits": 60,
        },
        medians=medians,
    )

    assert list(frame.columns) == FEATURE_COLUMNS
    assert frame.loc[0, "total_clicks"] == 120.0
    assert frame.loc[0, "active_days"] == 6.0
    assert frame.loc[0, "avg_daily_clicks"] == pytest.approx(20.0)
    assert frame.loc[0, "code_module"] == 1.0
