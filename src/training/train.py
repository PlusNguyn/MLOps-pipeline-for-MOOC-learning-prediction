# src/training/train.py

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

from sklearn.model_selection import train_test_split


FEATURES = [
    "num_clicks",
    "days_active",
    "avg_score",
    "engagement_score",
    "consistency"
]


def train(df: pd.DataFrame):

    # =========================
    # MLflow Setup
    # =========================

    mlflow.set_tracking_uri("http://localhost:5000")

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
            "random_state": 42
        }

        model = XGBClassifier(
            **params
        )

        # =========================
        # Training
        # =========================

        model.fit(X_train, y_train)

        # =========================
        # Prediction
        # =========================

        preds = model.predict(X_test)

        # =========================
        # Metrics
        # =========================

        acc = accuracy_score(y_test, preds)

        f1 = f1_score(
            y_test,
            preds,
            average="weighted"
        )

        # =========================
        # Logging
        # =========================

        mlflow.log_params(params)

        mlflow.log_metric(
            "accuracy",
            acc
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        # Log model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model"
        )

        # Save local model
        joblib.dump(
            model,
            "models/model.pkl"
        )

        # =========================
        # Console Output
        # =========================

        print("=" * 50)

        print(f"Accuracy : {acc:.4f}")

        print(f"F1 Score : {f1:.4f}")

        print("=" * 50)

        print(
            classification_report(
                y_test,
                preds
            )
        )

    return model
