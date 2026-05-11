from fastapi import FastAPI
from pydantic import BaseModel

import joblib
import pandas as pd


# =========================
# Load Model
# =========================

model = joblib.load("models/model.pkl")


# =========================
# FastAPI App
# =========================

app = FastAPI(
    title="MOOC Learning Prediction API"
)


# =========================
# Request Schema
# =========================

class StudentData(BaseModel):

    num_clicks: int
    days_active: int
    avg_score: float
    engagement_score: float
    consistency: float


# =========================
# Home Route
# =========================

@app.get("/")
def home():

    return {
        "message": "MOOC Prediction API Running"
    }


# =========================
# Prediction Route
# =========================

@app.post("/predict")
def predict(data: StudentData):

    input_df = pd.DataFrame([{
        "num_clicks": data.num_clicks,
        "days_active": data.days_active,
        "avg_score": data.avg_score,
        "engagement_score": data.engagement_score,
        "consistency": data.consistency
    }])

    prediction = model.predict(input_df)[0]

    label_map = {
        0: "Low",
        1: "Medium",
        2: "High"
    }

    return {
        "prediction": int(prediction),
        "level": label_map[int(prediction)]
    }