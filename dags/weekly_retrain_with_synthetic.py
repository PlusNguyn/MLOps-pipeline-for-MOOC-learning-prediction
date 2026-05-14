from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path("/opt/airflow/project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.ingestion.synthetic_api import archive_and_reset_synthetic_data
from src.pipeline.preprocess import preprocess
from src.training.optuna_tune import tune_pipeline
from src.training.train import (
    compare_baseline_pipeline,
    train_lgbm_pipeline,
    train_xgboost_pipeline,
)


def pipeline_done():
    print("Weekly retraining with synthetic data completed successfully.")


def archive_synthetic_window(**context):
    archive_dir = archive_and_reset_synthetic_data(run_label=f"{context['ds_nodash']}_weekly_window")
    print({"archive_dir": archive_dir})


default_args = {
    "owner": "admin",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="student_learning_weekly_retrain",
    description="Merge daily collected synthetic data during preprocess, then retrain and evaluate weekly.",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=4),
    tags=["mlops", "oulad", "synthetic", "weekly", "retrain"],
) as dag:
    preprocess_task = PythonOperator(
        task_id="preprocess_with_synthetic_window",
        python_callable=preprocess,
        execution_timeout=timedelta(minutes=45),
    )

    train_lgbm_task = PythonOperator(
        task_id="train_lgbm_baseline",
        python_callable=train_lgbm_pipeline,
        execution_timeout=timedelta(hours=1),
    )

    train_xgboost_task = PythonOperator(
        task_id="train_xgboost_baseline",
        python_callable=train_xgboost_pipeline,
        execution_timeout=timedelta(hours=1),
    )

    compare_task = PythonOperator(
        task_id="compare_baseline_models",
        python_callable=compare_baseline_pipeline,
        execution_timeout=timedelta(hours=1),
    )

    tune_task = PythonOperator(
        task_id="tune_xgboost",
        python_callable=tune_pipeline,
        execution_timeout=timedelta(hours=2),
    )

    archive_task = PythonOperator(
        task_id="archive_synthetic_window",
        python_callable=archive_synthetic_window,
        execution_timeout=timedelta(minutes=15),
    )

    done_task = PythonOperator(
        task_id="done",
        python_callable=pipeline_done,
        execution_timeout=timedelta(minutes=10),
    )

    preprocess_task >> [train_lgbm_task, train_xgboost_task]
    [train_lgbm_task, train_xgboost_task] >> compare_task
    compare_task >> tune_task >> archive_task >> done_task
