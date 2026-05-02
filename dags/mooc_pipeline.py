from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.load_data import clean_data, load_raw_data
from features.feature_engineering import engineer_features
from models.train import train_model
import pandas as pd
from pathlib import Path

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'mooc_ml_pipeline',
    default_args=default_args,
    description='MLOps pipeline for MOOC certification prediction',
    schedule_interval=timedelta(days=1),
    catchup=False,
)

def load_and_clean():
    raw_path = Path(__file__).parent.parent / "data" / "raw" / "big_student_clear_third_version.csv"
    processed_path = Path(__file__).parent.parent / "data" / "processed" / "cleaned_data.csv"
    
    df = load_raw_data(str(raw_path))
    df_clean = clean_data(df)
    df_clean.to_csv(processed_path, index=False)

def feature_eng():
    processed_path = Path(__file__).parent.parent / "data" / "processed" / "cleaned_data.csv"
    features_path = Path(__file__).parent.parent / "data" / "processed" / "features.csv"
    
    df = pd.read_csv(processed_path)
    df_feat = engineer_features(df)
    df_feat.to_csv(features_path, index=False)

def train():
    features_path = Path(__file__).parent.parent / "data" / "processed" / "features.csv"
    model_path = Path(__file__).parent.parent / "models" / "lgb_model.txt"
    
    train_model(str(features_path), str(model_path))

task_load = PythonOperator(
    task_id='load_and_clean_data',
    python_callable=load_and_clean,
    dag=dag,
)

task_features = PythonOperator(
    task_id='feature_engineering',
    python_callable=feature_eng,
    dag=dag,
)

task_train = PythonOperator(
    task_id='train_model',
    python_callable=train,
    dag=dag,
)

task_load >> task_features >> task_train