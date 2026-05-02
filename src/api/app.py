from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import lightgbm as lgb
import pandas as pd
import numpy as np
from pathlib import Path
import joblib  # For scaler if needed, but for simplicity, assume features are preprocessed

app = FastAPI(title="MOOC Certification Prediction API")

# Load model
model_path = Path(__file__).parent.parent.parent / "models" / "lgb_model.txt"
model = lgb.Booster(model_file=str(model_path))

# Assume feature names from training
# In real, load from somewhere
feature_names = [
    'year', 'semester', 'viewed', 'explored', 'grade', 'nevents', 'ndays_act', 
    'nplay_video', 'nchapters', 'nforum_posts', 'incomplete_flag', 'age', 'duration_days',
    'activity_per_day', 'video_per_chapter', 'institute_encoded', 'course_id_encoded',
    'final_cc_cname_DI_encoded', 'LoE_DI_encoded', 'gender_encoded'
]

class PredictionRequest(BaseModel):
    features: dict  # Expect a dict with feature names

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Convert to DataFrame
        df = pd.DataFrame([request.features])
        
        # Ensure all features are present
        for feat in feature_names:
            if feat not in df.columns:
                df[feat] = 0  # or raise error
        
        df = df[feature_names]
        
        # Predict
        preds = model.predict(df.values)
        prob = preds[0]
        prediction = int(prob > 0.5)
        
        return {"prediction": prediction, "probability": prob}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def root():
    return {"message": "MOOC Certification Prediction API"}