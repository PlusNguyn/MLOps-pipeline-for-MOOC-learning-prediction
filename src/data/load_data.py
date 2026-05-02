import pandas as pd
import os
from pathlib import Path

def load_raw_data(file_path: str) -> pd.DataFrame:
    """Load raw dataset from CSV."""
    return pd.read_csv(file_path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset: handle missing values, data types."""
    # Drop rows with missing target
    df = df.dropna(subset=['certified'])
    
    # Fill missing values for numerical columns with median
    num_cols = ['grade', 'nevents', 'ndays_act', 'nplay_video', 'nchapters', 'nforum_posts', 'age']
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # For categorical, fill with mode or 'unknown'
    cat_cols = ['institute', 'course_id', 'final_cc_cname_DI', 'LoE_DI', 'gender']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')
    
    # Convert dates if needed, but for now, keep as is or parse
    # Assuming start_time_DI and last_event_DI are strings, convert to datetime
    df['start_time_DI'] = pd.to_datetime(df['start_time_DI'], errors='coerce')
    df['last_event_DI'] = pd.to_datetime(df['last_event_DI'], errors='coerce')
    
    # Convert certified to int
    df['certified'] = df['certified'].astype(int)
    
    return df

if __name__ == "__main__":
    raw_path = Path(__file__).parent.parent.parent / "data" / "raw" / "big_student_clear_third_version.csv"
    processed_path = Path(__file__).parent.parent.parent / "data" / "processed" / "cleaned_data.csv"
    
    df = load_raw_data(str(raw_path))
    df_clean = clean_data(df)
    df_clean.to_csv(processed_path, index=False)
    print(f"Cleaned data saved to {processed_path}")