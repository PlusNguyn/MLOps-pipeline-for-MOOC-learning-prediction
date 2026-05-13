import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.training.common import (
    LOCAL_FEATURES_PATH,
    LOCAL_MEDIANS_PATH,
    LOCAL_METRICS_PATH,
    PROCESSED_DATA_PATH,
    SEED,
    configure_mlflow,
    log_model_diagnostics,
    register_model,
    save_local_artifacts,
    split_training_data,
)


def _load_existing_best_roc_auc() -> float | None:
    metrics_path = Path(LOCAL_METRICS_PATH)
    if not metrics_path.exists():
        return None

    try:
        metrics = pd.read_json(metrics_path, typ="series")
    except ValueError:
        return None

    if "roc_auc" not in metrics:
        return None

    return float(metrics["roc_auc"])


def _evaluate_model(model, X_test, y_test) -> dict[str, float]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
    }


def tune_model(df: pd.DataFrame):
    configure_mlflow("oulad_student_at_risk_optuna")
    X_train, X_test, y_train, y_test = split_training_data(df)
    medians = X_train.median().to_dict()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "eval_metric": "logloss",
            "random_state": SEED,
            "n_jobs": -1,
        }

        model = XGBClassifier(**params)
        score = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        ).mean()

        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("cv_roc_auc", float(score))

        return float(score)

    with mlflow.start_run(run_name="optuna_xgboost_tuning") as run:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=40)

        best_params = {
            **study.best_params,
            "eval_metric": "logloss",
            "random_state": SEED,
            "n_jobs": -1,
        }
        best_model = XGBClassifier(**best_params)

        best_model.fit(X_train, y_train)

        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "best_cv_roc_auc": float(study.best_value),
        }

        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.log_dict(study.best_params, "reports/best_params.json")
        log_model_diagnostics(best_model, X_test, y_test, prefix="Optuna XGBoost")
        existing_best_roc_auc = _load_existing_best_roc_auc()
        should_promote = (
            existing_best_roc_auc is None or
            metrics["roc_auc"] >= existing_best_roc_auc
        )

        mlflow.log_metric("existing_best_roc_auc", existing_best_roc_auc or 0.0)
        mlflow.log_param("promoted_to_production", should_promote)
        mlflow.log_param("tuned_model", "xgboost")

        save_local_artifacts(best_model, medians, metrics)
        mlflow.log_artifact(str(LOCAL_FEATURES_PATH), artifact_path="artifacts")
        mlflow.log_artifact(str(LOCAL_MEDIANS_PATH), artifact_path="artifacts")
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        if should_promote:
            register_model(run.info.run_id, "model")

    print("=" * 50)
    print("OPTUNA BEST PARAMETERS FOR XGBOOST")
    print(study.best_params)
    print(f"Best CV ROC-AUC: {study.best_value:.4f}")
    print("=" * 50)

    return best_model


def tune_pipeline():
    print("=" * 50)
    print("LOADING PROCESSED DATA FOR OPTUNA")
    print("=" * 50)

    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"Dataset Shape: {df.shape}")
    tune_model(df)


if __name__ == "__main__":
    tune_pipeline()
