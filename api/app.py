import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError

from src.processing.feature_engineering import build_feature_row, build_inference_frame
from src.training.common import LOCAL_FEATURES_PATH, LOCAL_MEDIANS_PATH, MLFLOW_MODEL_NAME
from src.utils.database import PredictionLog, SessionLocal, init_db

app = FastAPI(title="Learning Prediction API")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model = None
feature_names: list[str] = []
feature_medians: dict[str, float] = {}


def load_model_from_registry():
    return mlflow.sklearn.load_model(f"models:/{MLFLOW_MODEL_NAME}/Production")


def load_local_artifacts() -> tuple[list[str], dict[str, float]]:
    features = json.loads(Path(LOCAL_FEATURES_PATH).read_text(encoding="utf-8"))
    medians = json.loads(Path(LOCAL_MEDIANS_PATH).read_text(encoding="utf-8"))
    return features, medians


@app.on_event("startup")
def load_model():
    global model, feature_names, feature_medians

    init_db()

    try:
        model = load_model_from_registry()
        feature_names, feature_medians = load_local_artifacts()
        print("=" * 50)
        print("MODEL LOADED")
        print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
        print("=" * 50)
    except Exception as exc:
        print("=" * 50)
        print(f"FAILED TO LOAD MODEL: {exc}")
        print("=" * 50)


class StudentData(BaseModel):
    num_clicks: int | None = None
    days_active: int | None = None
    avg_score: float | None = None
    studied_credits: int | None = None


@app.get("/")
def root():
    return {"message": "Learning Prediction API Running"}


@app.get("/health")
def health():
    return {"model_loaded": model is not None}


@app.post("/predict")
def predict(data: StudentData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    payload = data.model_dump()
    if all(value is None for value in payload.values()):
        raise HTTPException(status_code=400, detail="At least one feature is required")

    legacy_metrics = build_feature_row(
        num_clicks=payload.get("num_clicks") or 0,
        days_active=payload.get("days_active") or 0,
        avg_score=payload.get("avg_score") or 0,
        studied_credits=payload.get("studied_credits") or 1,
    )
    input_df = build_inference_frame(payload=payload, medians=feature_medians)
    input_df = input_df[feature_names]

    prediction = int(model.predict(input_df)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_df)[0][1])

    label_map = {
        0: "At-Risk",
        1: "Success",
    }

    result = {
        "prediction": prediction,
        "level": label_map[prediction],
        "risk_probability": None if probability is None else round(1 - probability, 4),
        "success_probability": None if probability is None else round(probability, 4),
        "engagement_score": round(legacy_metrics["engagement_score"], 4),
        "consistency": round(legacy_metrics["consistency"], 4),
    }

    try:
        with SessionLocal() as session:
            log = PredictionLog(
                num_clicks=int(payload.get("num_clicks") or 0),
                days_active=int(payload.get("days_active") or 0),
                avg_score=float(payload.get("avg_score") or 0),
                studied_credits=int(payload.get("studied_credits") or 0),
                engagement_score=legacy_metrics["engagement_score"],
                consistency=legacy_metrics["consistency"],
                prediction=prediction,
                prediction_label=label_map[prediction],
                model_name=MLFLOW_MODEL_NAME,
                model_stage="Production",
            )
            session.add(log)
            session.commit()
    except SQLAlchemyError as exc:
        print(f"Failed to save prediction log: {exc}")

    return result


@app.post("/reload-model")
def reload_model():
    global model, feature_names, feature_medians

    try:
        model = load_model_from_registry()
        feature_names, feature_medians = load_local_artifacts()
        return {"message": "Model reloaded", "model_loaded": True}
    except Exception as exc:
        model = None
        raise HTTPException(status_code=500, detail=str(exc))
