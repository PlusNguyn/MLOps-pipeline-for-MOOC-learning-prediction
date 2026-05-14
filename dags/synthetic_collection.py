from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path("/opt/airflow/project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.ingestion.synthetic_api import collect_synthetic_data


def collect_daily_synthetic_data(**context):
    target_date = context["ds"]
    summary = collect_synthetic_data(target_date=target_date, batch_size=96)
    print(summary)


with DAG(
    dag_id="synthetic_oulad_daily_collection",
    description="Collect daily synthetic OULAD-like records from FastAPI for later weekly retraining.",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "admin",
        "depends_on_past": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["mlops", "synthetic", "fastapi", "daily"],
) as dag:
    collect_task = PythonOperator(
        task_id="collect_synthetic_daily_batch",
        python_callable=collect_daily_synthetic_data,
        execution_timeout=timedelta(minutes=15),
    )

    collect_task
