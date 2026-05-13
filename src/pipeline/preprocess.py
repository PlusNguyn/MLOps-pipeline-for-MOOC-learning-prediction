from pathlib import Path

import pandas as pd

from src.ingestion.load_oulad import load_oulad
from src.processing.feature_engineering import engineer_features, save_feature_metadata
from src.utils.database import (
    init_db,
    save_dataframe,
    save_feature_metadata as save_feature_metadata_to_db,
)


BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = BASE_DIR / "data" / "raw"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "train.csv"
FEATURE_METADATA_PATH = BASE_DIR / "data" / "processed" / "feature_metadata.json"
SYNTHETIC_CURRENT_PATH = BASE_DIR / "data" / "synthetic" / "current"


def _safe_path_display(path: Path) -> str:
    return ascii(str(path))


def preprocess():
    print("=" * 50)
    print("LOADING OULAD TABLES")
    print("=" * 50)

    tables = load_oulad(RAW_DATA_PATH)
    for name, frame in tables.items():
        print(f"{name}: {frame.shape}")
    if SYNTHETIC_CURRENT_PATH.exists():
        print(f"Synthetic merge source: {ascii(str(SYNTHETIC_CURRENT_PATH))}")

    processed_df = engineer_features(tables)
    print(f"Processed Shape: {processed_df.shape}")

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(PROCESSED_DATA_PATH, index=False)
    metadata = save_feature_metadata(FEATURE_METADATA_PATH)

    init_db()
    save_dataframe(
        table_name="raw_students",
        df=tables["students"],
        if_exists="replace",
    )
    save_dataframe(
        table_name="raw_assessments",
        df=tables["assessments"],
        if_exists="replace",
    )
    save_dataframe(
        table_name="raw_vle",
        df=tables["vle"],
        if_exists="replace",
    )
    save_dataframe(
        table_name="raw_assess_def",
        df=tables["assess_def"],
        if_exists="replace",
    )

    if SYNTHETIC_CURRENT_PATH.exists():
        for filename, table_name in [
            ("studentInfo.csv", "synthetic_students"),
            ("studentAssessment.csv", "synthetic_assessments"),
            ("studentVle.csv", "synthetic_vle"),
        ]:
            csv_path = SYNTHETIC_CURRENT_PATH / filename
            if csv_path.exists():
                save_dataframe(
                    table_name=table_name,
                    df=pd.read_csv(csv_path),
                    if_exists="replace",
                )

    save_dataframe(
        table_name="processed_train_data",
        df=processed_df,
        if_exists="replace",
    )
    save_feature_metadata_to_db(metadata)

    print("=" * 50)
    print(f"Saved processed data to: {_safe_path_display(PROCESSED_DATA_PATH)}")
    print(f"Saved feature metadata to: {_safe_path_display(FEATURE_METADATA_PATH)}")
    print("=" * 50)

    return processed_df
