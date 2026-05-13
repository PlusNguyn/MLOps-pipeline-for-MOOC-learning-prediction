from __future__ import annotations
import json
import os
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
from mlflow.tracking import MlflowClient
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split

from src.processing.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN
from src.utils.database import init_db, save_model_features, save_model_medians

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "train.csv"
MODEL_DIR = BASE_DIR / "models"
LOCAL_MODEL_PATH = MODEL_DIR / "student_at_risk_model.joblib"
LOCAL_MEDIANS_PATH = MODEL_DIR / "medians.json"
LOCAL_FEATURES_PATH = MODEL_DIR / "features.json"
LOCAL_METRICS_PATH = MODEL_DIR / "model_metrics.json"
SEED = 42

MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "student_at_risk_model")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def configure_mlflow(experiment_name: str) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        client.create_experiment(experiment_name)
    elif experiment.lifecycle_stage == "deleted":
        try:
            client.restore_experiment(experiment.experiment_id)
        except AttributeError:
            # Older MLflow versions may not support restore_experiment.
            pass

    mlflow.set_experiment(experiment_name)


def split_training_data(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=SEED,
    )


def ensure_model_dir() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def save_local_artifacts(
    model,
    medians: dict[str, float],
    metrics: dict[str, float],
) -> None:
    ensure_model_dir()
    joblib.dump(model, LOCAL_MODEL_PATH)
    LOCAL_MEDIANS_PATH.write_text(json.dumps(medians, indent=2), encoding="utf-8")
    LOCAL_FEATURES_PATH.write_text(json.dumps(FEATURE_COLUMNS, indent=2), encoding="utf-8")
    LOCAL_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    init_db()
    # Save to database
    save_model_features(FEATURE_COLUMNS)
    save_model_medians(medians)


def register_model(run_id: str, artifact_path: str) -> None:
    model_uri = f"runs:/{run_id}/{artifact_path}"
    result = mlflow.register_model(model_uri=model_uri, name=MLFLOW_MODEL_NAME)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    try:
        latest_versions = client.get_latest_versions(name=MLFLOW_MODEL_NAME)
    except Exception:
        latest_versions = []

    for version in latest_versions:
        if str(version.version) != str(result.version) and version.current_stage == "Production":
            client.transition_model_version_stage(
                name=MLFLOW_MODEL_NAME,
                version=version.version,
                stage="Archived",
            )

    client.transition_model_version_stage(
        name=MLFLOW_MODEL_NAME,
        version=result.version,
        stage="Production",
        archive_existing_versions=True,
    )


def log_dataframe_artifact(df: pd.DataFrame, artifact_file: str) -> None:
    ensure_model_dir()
    output_path = MODEL_DIR / artifact_file
    df.to_csv(output_path, index=False)
    mlflow.log_artifact(str(output_path), artifact_path="reports")


def log_model_diagnostics(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    prefix: str,
) -> None:
    y_pred = model.predict(X_test)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["At-Risk", "Success"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title(f"{prefix} Confusion Matrix")
    fig.tight_layout()
    mlflow.log_figure(fig, f"plots/{prefix.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.close(fig)

    if hasattr(model, "predict_proba"):
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(
            y_test,
            model.predict_proba(X_test)[:, 1],
            ax=ax,
        )
        ax.set_title(f"{prefix} ROC Curve")
        fig.tight_layout()
        mlflow.log_figure(fig, f"plots/{prefix.lower().replace(' ', '_')}_roc_curve.png")
        plt.close(fig)

    feature_importances = getattr(model, "feature_importances_", None)
    if feature_importances is not None:
        importance_df = pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "importance": feature_importances,
            }
        ).sort_values("importance", ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(9, 6))
        sns.barplot(
            data=importance_df,
            x="importance",
            y="feature",
            hue="feature",
            dodge=False,
            legend=False,
            ax=ax,
            palette="viridis",
        )
        ax.set_title(f"{prefix} Feature Importance")
        ax.set_xlabel("Importance")
        ax.set_ylabel("")
        fig.tight_layout()
        mlflow.log_figure(fig, f"plots/{prefix.lower().replace(' ', '_')}_feature_importance.png")
        plt.close(fig)


def log_dataset_overview(df: pd.DataFrame) -> None:
    target_counts = df[TARGET_COLUMN].map({0: "At-Risk", 1: "Success"}).value_counts()

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(
        x=target_counts.index,
        y=target_counts.values,
        hue=target_counts.index,
        dodge=False,
        legend=False,
        ax=ax,
        palette=["#ef4444", "#22c55e"],
    )
    ax.set_title("Target Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Students")
    fig.tight_layout()
    mlflow.log_figure(fig, "plots/target_distribution.png")
    plt.close(fig)

    demographics = ["gender", "highest_education", "imd_band", "age_band", "disability"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, column in zip(axes.flatten(), demographics, strict=False):
        chart_df = df.groupby([column, TARGET_COLUMN]).size().unstack(fill_value=0)
        chart_df.columns = ["At-Risk", "Success"]
        chart_df.plot(kind="bar", ax=ax, color=["#ef4444", "#22c55e"], rot=25)
        ax.set_title(column)
        ax.set_xlabel("")
        ax.legend(fontsize=8)

    axes[-1, -1].axis("off")
    fig.suptitle("Demographics vs Outcome", fontsize=14)
    fig.tight_layout()
    mlflow.log_figure(fig, "plots/demographics_vs_outcome.png")
    plt.close(fig)
