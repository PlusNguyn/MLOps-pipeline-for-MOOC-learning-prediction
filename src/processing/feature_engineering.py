# src/processing/feature_engineering.py

import pandas as pd
import numpy as np


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

    max_clicks = max(df["num_clicks"].max(), 1)
    max_days = max(df["days_active"].max(), 1)

    df["engagement_score"] = (
        0.4 * (df["num_clicks"] / max_clicks) +
        0.3 * (df["days_active"] / max_days) +
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