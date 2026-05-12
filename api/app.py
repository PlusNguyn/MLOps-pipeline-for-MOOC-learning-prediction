from fastapi import FastAPI, HTTPException
import os
import mlflow
import mlflow.pyfunc
from pydantic import BaseModel
import pandas as pd

from src.processing.feature_engineering import build_feature_row


# =========================
# FastAPI App
# =========================

app = FastAPI(
    title="Learning Prediction API"
)


# =========================
# MLflow Setup
# =========================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000"
)
MLFLOW_MODEL_NAME = os.getenv(
    "MLFLOW_MODEL_NAME",
    "student_performance_model"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# =========================
# Load Model
# =========================

model = None


def load_model_from_registry():
    return mlflow.pyfunc.load_model(
        f"models:/{MLFLOW_MODEL_NAME}/Production"
    )


@app.on_event("startup")
def load_model():

    global model

    try:

        model = load_model_from_registry()

        print("=" * 50)
        print("MODEL LOADED")
        print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
        print("=" * 50)

    except Exception as e:

        print("=" * 50)
        print(f"FAILED TO LOAD MODEL: {e}")
        print("=" * 50)


# =========================
# Request Schema
# =========================

class StudentData(BaseModel):

    num_clicks: int
    days_active: int
    avg_score: float
    studied_credits: int


# =========================
# Root Route
# =========================

@app.get("/")
def root():

    return {
        "message": "Learning Prediction API Running"
    }


# =========================
# Health Check
# =========================

@app.get("/health")
def health():

    return {
        "model_loaded": model is not None
    }


# =========================
# Prediction Route
# =========================

@app.post("/predict")
def predict(data: StudentData):

    if model is None:

        raise HTTPException(
            status_code=500,
            detail="Model not loaded"
        )

    # =========================
    # Feature Engineering
    # =========================

    features = build_feature_row(
        num_clicks=data.num_clicks,
        days_active=data.days_active,
        avg_score=data.avg_score,
        studied_credits=data.studied_credits,
    )

    # =========================
    # Create DataFrame
    # =========================

    input_df = pd.DataFrame([features])

    # =========================
    # Prediction
    # =========================

    prediction = model.predict(input_df)[0]

    label_map = {
        0: "Low",
        1: "Medium",
        2: "High"
    }

    return {
        "prediction": int(prediction),
        "level": label_map[int(prediction)],
        "engagement_score": round(
            features["engagement_score"],
            4
        ),
        "consistency": round(
            features["consistency"],
            4
        )
    }

@app.post("/reload-model")
def reload_model():

    global model

    try:
        model = load_model_from_registry()

        return {
            "message": "Model reloaded",
            "model_loaded": True
        }

    except Exception as e:
        model = None

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
