import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path
from datetime import datetime

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for the model."""
    # Encode categorical variables
    cat_cols = ['institute', 'course_id', 'final_cc_cname_DI', 'LoE_DI', 'gender']
    le = LabelEncoder()
    for col in cat_cols:
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
    
    # Create time-based features
    df['duration_days'] = (df['last_event_DI'] - df['start_time_DI']).dt.days
    df['duration_days'] = df['duration_days'].fillna(0)
    
    # Interaction features
    df['activity_per_day'] = df['nevents'] / (df['ndays_act'] + 1)  # avoid div by zero
    df['video_per_chapter'] = df['nplay_video'] / (df['nchapters'] + 1)
    
    # Select features
    features = [
        'year', 'semester', 'viewed', 'explored', 'grade', 'nevents', 'ndays_act', 
        'nplay_video', 'nchapters', 'nforum_posts', 'incomplete_flag', 'age', 'duration_days',
        'activity_per_day', 'video_per_chapter'
    ] + [col + '_encoded' for col in cat_cols]
    
    target = 'certified'
    
    # Keep only relevant columns
    df_feat = df[features + [target] + ['userid_DI']].copy()
    
    # Add event_timestamp for Feast
    df_feat['event_timestamp'] = datetime.now()
    
    # Scale numerical features
    num_features = ['grade', 'nevents', 'ndays_act', 'nplay_video', 'nchapters', 'nforum_posts', 'age', 'duration_days', 'activity_per_day', 'video_per_chapter']
    scaler = StandardScaler()
    df_feat[num_features] = scaler.fit_transform(df_feat[num_features])
    
    return df_feat

if __name__ == "__main__":
    processed_path = Path(__file__).parent.parent.parent / "data" / "processed" / "cleaned_data.csv"
    features_path = Path(__file__).parent.parent.parent / "data" / "processed" / "features.csv"
    
    df = pd.read_csv(processed_path)
    df_feat = engineer_features(df)
    df_feat.to_csv(features_path, index=False)
    print(f"Features engineered and saved to {features_path}")