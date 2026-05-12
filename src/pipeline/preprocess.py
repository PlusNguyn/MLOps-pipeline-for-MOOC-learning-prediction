import pandas as pd

from pathlib import Path

from src.ingestion.load_oulad import load_oulad

from src.processing.feature_engineering import (
    engineer_features
)

from src.utils.database import (
    init_db,
    save_dataframe
)


# =========================
# Paths
# =========================

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = BASE_DIR / "data" / "raw"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "train.csv"


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

    # =========================
    # Save Processed Data to Postgres
    # =========================

    init_db()
    save_dataframe(
        table_name="processed_train_data",
        df=processed_df,
        if_exists="replace"
    )

    print("=" * 50)

    print(
        f"Saved to: {PROCESSED_DATA_PATH}"
    )

    print("=" * 50)

    return processed_df