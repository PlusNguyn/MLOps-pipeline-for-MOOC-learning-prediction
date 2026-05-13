import mlflow
import mlflow.sklearn
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

from src.processing.feature_engineering import FEATURE_COLUMNS
from src.training.common import (
    LOCAL_FEATURES_PATH,
    LOCAL_MEDIANS_PATH,
    PROCESSED_DATA_PATH,
    configure_mlflow,
    log_dataframe_artifact,
    log_dataset_overview,
    log_model_diagnostics,
    register_model,
    save_local_artifacts,
    split_training_data,
)


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


def _log_model_run(model_name: str, model, df: pd.DataFrame) -> dict[str, float]:
    configure_mlflow("oulad_student_at_risk_training")
    X_train, X_test, y_train, y_test = split_training_data(df)

    model.fit(X_train, y_train)
    metrics = _evaluate_model(model, X_test, y_test)

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        log_model_diagnostics(model, X_test, y_test, prefix=model_name.title())

    print("=" * 50)
    print(f"Trained {model_name} baseline model")
    print(metrics)
    print("=" * 50)

    return metrics


def train_lgbm(df: pd.DataFrame) -> dict[str, float]:
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
    )
    return _log_model_run("lightgbm", model, df)


def train_xgboost(df: pd.DataFrame) -> dict[str, float]:
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=1e-4,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    return _log_model_run("xgboost", model, df)


def compare_baseline_models(df: pd.DataFrame) -> pd.DataFrame:
    configure_mlflow("oulad_student_at_risk_training")
    X_train, X_test, y_train, y_test = split_training_data(df)
    models = {
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=1e-4,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
        ),
    }

    comparison_rows = []
    best_name = None
    best_model = None
    best_metrics = None

    with mlflow.start_run(run_name="baseline_model_comparison") as run:
        mlflow.log_param("feature_count", len(FEATURE_COLUMNS))
        mlflow.log_param("models_compared", ",".join(models.keys()))

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            metrics = _evaluate_model(model, X_test, y_test)

            with mlflow.start_run(run_name=model_name, nested=True):
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics)
                log_model_diagnostics(model, X_test, y_test, prefix=model_name.title())

            comparison_rows.append({"model": model_name, **metrics})

            if best_metrics is None or metrics["roc_auc"] > best_metrics["roc_auc"]:
                best_name = model_name
                best_model = model
                best_metrics = metrics

        comparison_df = pd.DataFrame(comparison_rows).sort_values("roc_auc", ascending=False)
        log_dataframe_artifact(comparison_df, "baseline_model_comparison.csv")
        mlflow.log_param("best_model_name", comparison_df.iloc[0]["model"])

        if best_model is not None:
            save_local_artifacts(best_model, X_train.median().to_dict(), best_metrics)
            mlflow.log_artifact(str(LOCAL_FEATURES_PATH), artifact_path="artifacts")
            mlflow.log_artifact(str(LOCAL_MEDIANS_PATH), artifact_path="artifacts")
            mlflow.sklearn.log_model(best_model, artifact_path="model")
            register_model(run.info.run_id, "model")

    print("=" * 50)
    print("BASELINE MODEL COMPARISON")
    print(comparison_df.to_string(index=False))
    print("=" * 50)

    return comparison_df


def train_lgbm_pipeline():
    print("=" * 50)
    print("LOADING PROCESSED DATA FOR LGBM")
    print("=" * 50)
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"Dataset Shape: {df.shape}")
    train_lgbm(df)


def train_xgboost_pipeline():
    print("=" * 50)
    print("LOADING PROCESSED DATA FOR XGBOOST")
    print("=" * 50)
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"Dataset Shape: {df.shape}")
    train_xgboost(df)


def compare_baseline_pipeline():
    print("=" * 50)
    print("LOADING PROCESSED DATA FOR BASELINE COMPARISON")
    print("=" * 50)
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"Dataset Shape: {df.shape}")
    compare_baseline_models(df)


def train(df: pd.DataFrame):
    configure_mlflow("oulad_student_at_risk_training")
    X_train, X_test, y_train, y_test = split_training_data(df)
    medians = X_train.median().to_dict()

    models = {
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=1e-4,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
        ),
    }

    comparison_rows = []
    best_name = None
    best_model = None
    best_metrics = None

    with mlflow.start_run(run_name="baseline_model_comparison") as run:
        mlflow.log_param("feature_count", len(FEATURE_COLUMNS))
        mlflow.log_param("models_compared", ",".join(models.keys()))
        mlflow.log_text("\n".join(FEATURE_COLUMNS), "reports/features.txt")
        log_dataset_overview(df)

        for model_name, model in models.items():
            with mlflow.start_run(run_name=model_name, nested=True):
                model.fit(X_train, y_train)
                metrics = _evaluate_model(model, X_test, y_test)
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics)
                log_model_diagnostics(model, X_test, y_test, prefix=model_name.title())

                comparison_rows.append({"model": model_name, **metrics})

                if best_metrics is None or metrics["roc_auc"] > best_metrics["roc_auc"]:
                    best_name = model_name
                    best_model = model
                    best_metrics = metrics

        comparison_df = pd.DataFrame(comparison_rows).sort_values("roc_auc", ascending=False)
        log_dataframe_artifact(comparison_df, "baseline_model_comparison.csv")
        mlflow.log_metrics({f"best_{k}": v for k, v in best_metrics.items()})
        mlflow.log_param("best_model_name", best_name)

        save_local_artifacts(best_model, medians, best_metrics)
        mlflow.log_artifact(str(LOCAL_FEATURES_PATH), artifact_path="artifacts")
        mlflow.log_artifact(str(LOCAL_MEDIANS_PATH), artifact_path="artifacts")

        mlflow.sklearn.log_model(best_model, artifact_path="model")
        register_model(run.info.run_id, "model")

    print("=" * 50)
    print("BASELINE MODEL COMPARISON")
    print(comparison_df.to_string(index=False))
    print("=" * 50)
    print(f"Best model: {best_name}")
    print(f"Saved local model artifacts to: {ascii(str(LOCAL_FEATURES_PATH.parent))}")
    print("=" * 50)

    return best_model


def train_pipeline():
    print("=" * 50)
    print("LOADING PROCESSED DATA")
    print("=" * 50)

    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"Dataset Shape: {df.shape}")
    train(df)


if __name__ == "__main__":
    train_pipeline()
