from datetime import datetime, timedelta
from pathlib import Path
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path("/opt/airflow/project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.pipeline.preprocess import preprocess
from src.training.optuna_tune import tune_pipeline
from src.training.train import (
    compare_baseline_pipeline,
    train_lgbm_pipeline,
    train_xgboost_pipeline,
)


def pipeline_done():
    print("Student learning pipeline completed successfully.")


default_args = {
    "owner": "admin",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="student_learning_pipeline",
    description="Preprocess OULAD data, train baselines, compare models, tune XGBoost, then mark completion.",
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
    2. `train_lgbm_baseline`: huấn luyện mô hình baseline LightGBM.
    3. `train_xgboost_baseline`: huấn luyện mô hình baseline XGBoost.
    4. `compare_baseline_models`: so sánh hai baseline và log kết quả.
    5. `tune_xgboost`: tune XGBoost bằng Optuna.
    6. `done`: đánh dấu hoàn thành pipeline.
    """,
) as dag:
    preprocess_task = PythonOperator(
        task_id="preprocess_oulad_dataset",
        python_callable=preprocess,
        execution_timeout=timedelta(minutes=45),
        doc_md="Tạo tập `data/processed/train.csv` và feature metadata từ các bảng OULAD gốc.",
    )

    train_lgbm_task = PythonOperator(
        task_id="train_lgbm_baseline",
        python_callable=train_lgbm_pipeline,
        execution_timeout=timedelta(hours=1),
        doc_md="Huấn luyện baseline LightGBM và log metrics vào MLflow.",
    )

    train_xgboost_task = PythonOperator(
        task_id="train_xgboost_baseline",
        python_callable=train_xgboost_pipeline,
        execution_timeout=timedelta(hours=1),
        doc_md="Huấn luyện baseline XGBoost và log metrics vào MLflow.",
    )

    compare_task = PythonOperator(
        task_id="compare_baseline_models",
        python_callable=compare_baseline_pipeline,
        execution_timeout=timedelta(hours=1),
        doc_md="So sánh performance giữa LightGBM và XGBoost baseline.",
    )

    tune_task = PythonOperator(
        task_id="tune_xgboost",
        python_callable=tune_pipeline,
        execution_timeout=timedelta(hours=2),
        doc_md="Tune XGBoost bằng Optuna và log kết quả tốt nhất.",
    )

    done_task = PythonOperator(
        task_id="done",
        python_callable=pipeline_done,
        execution_timeout=timedelta(minutes=10),
        doc_md="Đánh dấu pipeline đã hoàn thành.",
    )

    preprocess_task >> [train_lgbm_task, train_xgboost_task]
    [train_lgbm_task, train_xgboost_task] >> compare_task
    compare_task >> tune_task >> done_task
