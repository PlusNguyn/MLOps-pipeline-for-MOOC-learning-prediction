# src/training/optuna_tune.py

import optuna
import mlflow
import mlflow.sklearn
import pandas as pd
import os

from pathlib import Path

from xgboost import XGBClassifier
from mlflow.tracking import MlflowClient

from sklearn.metrics import (
    f1_score
)

from sklearn.model_selection import (
    train_test_split
)


# =========================
# Config
# =========================

FEATURES = [
    "num_clicks",
    "days_active",
    "avg_score",
    "engagement_score",
    "consistency"
]

PROCESSED_DATA_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "train.csv"
)
MLFLOW_MODEL_NAME = os.getenv(
    "MLFLOW_MODEL_NAME",
    "student_performance_model"
)


# =========================
# Core Optuna Function
# =========================

def tune_model(df: pd.DataFrame):

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://localhost:5000"
    )

    print(f"MLflow Tracking URI: {tracking_uri}")

    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(
        "oulad_optuna_tuning"
    )

    # =========================
    # Features & Labels
    # =========================

    X = df[FEATURES]

    y = df["label"]

    # =========================
    # Train/Test Split
    # =========================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # =========================
    # Optuna Objective
    # =========================

    def objective(trial):

        params = {

            "n_estimators": trial.suggest_int(
                "n_estimators",
                50,
                300
            ),

            "max_depth": trial.suggest_int(
                "max_depth",
                3,
                10
            ),

            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.3
            ),

            "subsample": trial.suggest_float(
                "subsample",
                0.5,
                1.0
            ),

            "random_state": 42,
            "objective": "multi:softmax",
            "num_class": 3,
            "eval_metric": "mlogloss"
        }

        with mlflow.start_run(
            nested=True
        ):

            model = XGBClassifier(
                **params
            )

            model.fit(
                X_train,
                y_train
            )

            preds = model.predict(
                X_test
            )

            f1 = f1_score(
                y_test,
                preds,
                average="weighted"
            )

            mlflow.log_params(
                params
            )

            mlflow.log_metric(
                "f1_score",
                f1
            )

        return f1

    # =========================
    # Run Optuna
    # =========================

    with mlflow.start_run(run_name="optuna_tuning"):

        study = optuna.create_study(
            direction="maximize"
        )

        study.optimize(
            objective,
            n_trials=20
        )

    # =========================
    # Best Results
    # =========================

    print("=" * 50)

    print("BEST PARAMETERS")

    print(
        study.best_params
    )

    print(
        f"BEST F1: {study.best_value:.4f}"
    )

    print("=" * 50)

    # =========================
    # Train Final Best Model
    # =========================

    best_params = {
        **study.best_params,
        "random_state": 42,
        "objective": "multi:softmax",
        "num_class": 3,
        "eval_metric": "mlogloss"
    }

    best_model = XGBClassifier(
        **best_params
    )

    best_model.fit(
        X_train,
        y_train
    )

    # =========================
    # Register Best Model
    # =========================

    with mlflow.start_run(
        run_name="optuna_best_model"
    ):

        logged_model = mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="best_model"
        )

        result = mlflow.register_model(
            logged_model.model_uri,
            MLFLOW_MODEL_NAME
        )

        mlflow.log_params(
            best_params
        )

        mlflow.log_metric(
            "best_f1_score",
            study.best_value
        )

        client = MlflowClient()

        client.transition_model_version_stage(
            name=MLFLOW_MODEL_NAME,
            version=result.version,
            stage="Production"
        )

    return best_model


# =========================
# Airflow Pipeline Function
# =========================

def tune_pipeline():

    print("=" * 50)

    print(
        "LOADING PROCESSED DATA FOR OPTUNA"
    )

    print("=" * 50)

    df = pd.read_csv(
        PROCESSED_DATA_PATH
    )

    print(
        f"Dataset Shape: {df.shape}"
    )

    tune_model(df)


# =========================
# Standalone Execution
# =========================

if __name__ == "__main__":

    tune_pipeline()
