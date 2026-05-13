from datetime import datetime, timedelta
from pathlib import Path
import sys
import seaborn as sns
import matplotlib.pyplot as plt

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path("/opt/airflow/project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.pipeline.preprocess import preprocess
from src.training.optuna_tune import tune_pipeline
from src.training.train import train_pipeline


default_args = {
    "owner": "admin",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="student_learning_pipeline",
    description="Preprocess OULAD data, compare baseline models, then run Optuna tuning.",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=3),
    tags=["mlops", "oulad", "student-risk", "mlflow", "optuna"],
    doc_md="""
    ### EduPulse Student At-Risk Pipeline

    DAG này chạy theo đúng flow mới của project:

    1. `preprocess_oulad_dataset`: gộp các bảng OULAD và tạo feature theo notebook.
    2. `train_baseline_models`: train nhiều model baseline, log metric + chart vào MLflow.
    3. `tune_xgboost_with_optuna`: tune XGBoost bằng Optuna và chỉ promote nếu model tốt hơn.

    Sau khi preprocess xong, hai nhánh training sẽ chạy song song để rút ngắn thời gian DAG.
    """,
) as dag:
    preprocess_task = PythonOperator(
        task_id="preprocess_oulad_dataset",
        python_callable=preprocess,
        execution_timeout=timedelta(minutes=45),
        doc_md="Tạo tập `data/processed/train.csv` và feature metadata từ các bảng OULAD gốc.",
    )

    train_task = PythonOperator(
        task_id="train_baseline_models",
        python_callable=train_pipeline,
        execution_timeout=timedelta(hours=1),
        doc_md="Train XGBoost + LightGBM baseline, log metrics, charts và register best model.",
    )

    tune_task = PythonOperator(
        task_id="tune_xgboost_with_optuna",
        python_callable=tune_pipeline,
        execution_timeout=timedelta(hours=2),
        doc_md="Chạy Optuna tuning cho XGBoost và chỉ cập nhật Production nếu metric tốt hơn.",
    )

    preprocess_task >> [train_task, tune_task]
