import pandas as pd
import numpy as np

ENGAGEMENT_NUM_CLICKS_SCALE = 1000.0
ENGAGEMENT_DAYS_ACTIVE_SCALE = 30.0


def build_feature_row(
    num_clicks: float,
    days_active: float,
    avg_score: float,
    studied_credits: float,
) -> dict:
    safe_credits = max(float(studied_credits), 1.0)

    engagement_score = (
        0.4 * (float(num_clicks) / ENGAGEMENT_NUM_CLICKS_SCALE) +
        0.3 * (float(days_active) / ENGAGEMENT_DAYS_ACTIVE_SCALE) +
        0.3 * (float(avg_score) / 100.0)
    )

    return {
        "num_clicks": float(num_clicks),
        "days_active": float(days_active),
        "avg_score": float(avg_score),
        "engagement_score": engagement_score,
        "consistency": float(days_active) / safe_credits,
    }


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # =========================
    # Handle Missing Values
    # =========================

    df["avg_score"] = df["avg_score"].fillna(
        df["avg_score"].median()
    )

    df["num_clicks"] = df["num_clicks"].fillna(0)

    df["days_active"] = df["days_active"].fillna(0)

    df["studied_credits"] = df["studied_credits"].fillna(1)

    # =========================
    # Feature Engineering
    # =========================

    df["engagement_score"] = (
        0.4 * (df["num_clicks"] / ENGAGEMENT_NUM_CLICKS_SCALE) +
        0.3 * (df["days_active"] / ENGAGEMENT_DAYS_ACTIVE_SCALE) +
        0.3 * (df["avg_score"] / 100)
    )

    # Learning consistency
    df["consistency"] = (
        df["days_active"] / df["studied_credits"]
    )

    # =========================
    # Handle Infinite Values
    # =========================

    df.replace([np.inf, -np.inf], 0, inplace=True)

    # =========================
    # Label Encoding
    # =========================

    label_map = {
        "Withdrawn": 0,
        "Fail": 0,
        "Pass": 1,
        "Distinction": 2
    }

    df["label"] = df["final_result"].map(label_map)

    # Remove rows with unknown labels
    df = df.dropna(subset=["label"])

    # Convert label to integer
    df["label"] = df["label"].astype(int)

    return df
