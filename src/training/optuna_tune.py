import optuna
import mlflow
import mlflow.sklearn

import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


FEATURES = [
    "num_clicks",
    "days_active",
    "avg_score",
    "engagement_score",
    "consistency"
]


def objective(trial, X_train, X_test, y_train, y_test):

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

        "random_state": 42
    }

    with mlflow.start_run(nested=True):

        model = XGBClassifier(**params)

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        f1 = f1_score(
            y_test,
            preds,
            average="weighted"
        )

        mlflow.log_params(params)

        mlflow.log_metric("f1_score", f1)

    return f1


def tune_model(df: pd.DataFrame):

    mlflow.set_tracking_uri(
        "http://localhost:5000"
    )

    mlflow.set_experiment(
        "oulad_optuna_tuning"
    )

    X = df[FEATURES]

    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    with mlflow.start_run(run_name="optuna_tuning"):

        study = optuna.create_study(
            direction="maximize"
        )

        study.optimize(
            lambda trial: objective(
                trial,
                X_train,
                X_test,
                y_train,
                y_test
            ),
            n_trials=20
        )

        best_params = study.best_params

        best_model = XGBClassifier(
            **best_params,
            random_state=42
        )

        best_model.fit(X_train, y_train)

        mlflow.sklearn.log_model(
            best_model,
            name="best_model"
        )

        best_score = study.best_value

        mlflow.log_params(best_params)

        mlflow.log_metric(
            "best_f1_score",
            best_score
        )

        print("=" * 50)

        print("BEST PARAMETERS")

        print(best_params)

        print(f"BEST F1: {best_score:.4f}")