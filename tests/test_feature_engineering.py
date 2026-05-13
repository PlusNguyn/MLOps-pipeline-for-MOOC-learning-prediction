import pandas as pd
import pytest

from src.processing.feature_engineering import (
    build_feature_row,
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
    assert row["engagement_score"] == pytest.approx(
        0.4 * (500.0 / 1000.0) + 0.3 * (15.0 / 30.0) + 0.3 * (80.0 / 100.0)
    )


def test_engineer_features_handles_missing_values_and_label_mapping():
    df = pd.DataFrame(
        [
            {
                "avg_score": None,
                "num_clicks": None,
                "days_active": None,
                "studied_credits": None,
                "final_result": "Pass",
            },
            {
                "avg_score": 95,
                "num_clicks": 300,
                "days_active": 18,
                "studied_credits": 6,
                "final_result": "Distinction",
            },
            {
                "avg_score": 50,
                "num_clicks": 10,
                "days_active": 1,
                "studied_credits": 0,
                "final_result": "Withdrawn",
            },
            {
                "avg_score": 70,
                "num_clicks": 20,
                "days_active": 2,
                "studied_credits": 1,
                "final_result": "Unknown",
            },
        ]
    )

    result = engineer_features(df)

    assert "engagement_score" in result.columns
    assert "consistency" in result.columns
    assert "label" in result.columns
    assert result["label"].isin([0, 1, 2]).all()
    assert result.shape[0] == 3
    assert result.loc[result["final_result"] == "Distinction", "label"].iloc[0] == 2
    assert result.loc[result["final_result"] == "Withdrawn", "label"].iloc[0] == 0
    assert result.loc[result["final_result"] == "Pass", "label"].iloc[0] == 1
