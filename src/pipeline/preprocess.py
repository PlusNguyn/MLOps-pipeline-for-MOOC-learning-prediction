from pathlib import Path

import pandas as pd

from src.ingestion.load_oulad import load_oulad
from src.processing.feature_engineering import engineer_features, save_feature_metadata
from src.utils.database import init_db, save_dataframe


BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = BASE_DIR / "data" / "raw"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "train.csv"
FEATURE_METADATA_PATH = BASE_DIR / "data" / "processed" / "feature_metadata.json"


def _safe_path_display(path: Path) -> str:
    return ascii(str(path))


def preprocess():
    print("=" * 50)
    print("LOADING OULAD TABLES")
    print("=" * 50)

    tables = load_oulad(RAW_DATA_PATH)
    for name, frame in tables.items():
        print(f"{name}: {frame.shape}")

    processed_df = engineer_features(tables)
    print(f"Processed Shape: {processed_df.shape}")

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(PROCESSED_DATA_PATH, index=False)
    save_feature_metadata(FEATURE_METADATA_PATH)

    init_db()
    save_dataframe(
        table_name="processed_train_data",
        df=processed_df,
        if_exists="replace",
    )

    print("=" * 50)
    print(f"Saved processed data to: {_safe_path_display(PROCESSED_DATA_PATH)}")
    print(f"Saved feature metadata to: {_safe_path_display(FEATURE_METADATA_PATH)}")
    print("=" * 50)

    return processed_df
