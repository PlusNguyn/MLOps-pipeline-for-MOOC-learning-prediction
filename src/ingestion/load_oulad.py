# src/ingestion/load_oulad.py
import pandas as pd

def load_oulad(data_dir: str) -> pd.DataFrame:
    student_info = pd.read_csv(f"{data_dir}/studentInfo.csv")
    student_vle  = pd.read_csv(f"{data_dir}/studentVle.csv")
    assessments  = pd.read_csv(f"{data_dir}/studentAssessment.csv")
    
    # Aggregate clickstream
    clicks_agg = student_vle.groupby("id_student").agg(
        num_clicks=("sum_click", "sum"),
        days_active=("date", "nunique")
    ).reset_index()
    
    # Aggregate scores
    score_agg = assessments.groupby("id_student").agg(
        avg_score=("score", "mean"),
        num_submissions=("id_assessment", "count")
    ).reset_index()
    
    df = student_info.merge(clicks_agg, on="id_student", how="left")
    df = df.merge(score_agg, on="id_student", how="left")
    return df