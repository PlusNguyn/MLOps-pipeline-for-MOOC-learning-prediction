# src/training/train.py
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from pathlib import Path

from xgboost import XGBClassifier

from mlflow.tracking import MlflowClient

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
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

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = BASE_DIR / "data" / "raw"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "train.csv"

MODEL_OUTPUT_PATH = BASE_DIR / "models" / "xgboost_model.pkl"
MLFLOW_MODEL_NAME = os.getenv(
    "MLFLOW_MODEL_NAME",
    "student_performance_model"
)


# =========================
# Core Training Function
# =========================

def train(df: pd.DataFrame):

    # =========================
    # MLflow Setup
    # =========================

   
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://localhost:5000"
    )

    print(f"MLflow Tracking URI: {tracking_uri}")

    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(
        "oulad_learning_prediction"
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
    # Start MLflow Run
    # =========================

    with mlflow.start_run():

        params = {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
            "random_state": 42,
            "objective": "multi:softmax",
            "num_class": 3,
            "eval_metric": "mlogloss"
        }

        model = XGBClassifier(
            **params
        )

        # =========================
        # Training
        # =========================

        model.fit(
            X_train,
            y_train
        )

        # =========================
        # Prediction
        # =========================

        preds = model.predict(
            X_test
        )

        # =========================
        # Metrics
        # =========================

        acc = accuracy_score(
            y_test,
            preds
        )

        f1 = f1_score(
            y_test,
            preds,
            average="weighted"
        )

        # =========================
        # Logging Metrics
        # =========================

        mlflow.log_params(
            params
        )

        mlflow.log_metric(
            "accuracy",
            acc
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        # =========================
        # Save Local Model
        # =========================

        MODEL_OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(
            model,
            MODEL_OUTPUT_PATH
        )

        # Log + Register Model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MLFLOW_MODEL_NAME
        )

        client = MlflowClient()

        latest = client.get_latest_versions(
            name=MLFLOW_MODEL_NAME
        )[-1]

        client.transition_model_version_stage(
            name=MLFLOW_MODEL_NAME,
            version=latest.version,
            stage="Production",
            archive_existing_versions=True
        )

        # =========================
        # Console Output
        # =========================

        print("=" * 50)

        print(
            f"Accuracy : {acc:.4f}"
        )

        print(
            f"F1 Score : {f1:.4f}"
        )

        print("=" * 50)

        print(
            classification_report(
                y_test,
                preds
            )
        )

        print("=" * 50)

        print(
            f"Model saved to: {MODEL_OUTPUT_PATH}"
        )

        print("=" * 50)

    return model


# =========================
# Airflow Pipeline Function
# =========================

def train_pipeline():

    print("=" * 50)

    print(
        "LOADING PROCESSED DATA"
    )

    print("=" * 50)

    df = pd.read_csv(
        PROCESSED_DATA_PATH
    )

    print(
        f"Dataset Shape: {df.shape}"
    )

    train(df)


# =========================
# Standalone Execution
# =========================

if __name__ == "__main__":

    train_pipeline()
