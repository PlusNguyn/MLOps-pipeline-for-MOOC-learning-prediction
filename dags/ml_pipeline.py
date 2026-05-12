# dags/ml_pipeline.py

import sys
sys.path.append(
    "/opt/airflow/project"
)

from airflow import DAG

from airflow.operators.python import (
    PythonOperator
)

from datetime import datetime

from src.pipeline.preprocess import (
    preprocess
)

from src.training.train import (
    train_pipeline
)

from src.training.optuna_tune import (
    tune_pipeline
)


default_args = {
    "owner": "admin"
}


with DAG(

    dag_id="student_learning_pipeline",

    default_args=default_args,

    start_date=datetime(2025, 1, 1),

    schedule="@daily",

    catchup=False

) as dag:

    preprocess_task = PythonOperator(
        task_id="preprocess",
        python_callable=preprocess
    )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_pipeline
    )

    tune_task = PythonOperator(
        task_id="optuna_tuning",
        python_callable=tune_pipeline
    )

    preprocess_task >> train_task >> tune_task