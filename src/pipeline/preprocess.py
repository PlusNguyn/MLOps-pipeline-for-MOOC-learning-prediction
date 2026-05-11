import pandas as pd

from pathlib import Path

from src.ingestion.load_oulad import load_oulad

from src.processing.feature_engineering import (
    engineer_features
)


# =========================
# Paths
# =========================

RAW_DATA_PATH = Path("data/raw")

PROCESSED_DATA_PATH = Path(
    "data/processed/train.csv"
)


# =========================
# Preprocess Pipeline
# =========================

def preprocess():

    print("=" * 50)
    print("LOADING DATA")
    print("=" * 50)

    # Load raw dataset
    df = load_oulad(
        RAW_DATA_PATH
    )

    print(f"Raw Shape: {df.shape}")

    # =========================
    # Feature Engineering
    # =========================

    processed_df = engineer_features(df)

    print(
        f"Processed Shape: {processed_df.shape}"
    )

    # =========================
    # Create Directory
    # =========================

    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # =========================
    # Save Processed Data
    # =========================

    processed_df.to_csv(
        PROCESSED_DATA_PATH,
        index=False
    )

    print("=" * 50)

    print(
        f"Saved to: {PROCESSED_DATA_PATH}"
    )

    print("=" * 50)

    return processed_df