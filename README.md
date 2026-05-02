# MOOC Certification Prediction MLOps Pipeline

This project implements a complete MLOps pipeline for predicting MOOC course certification using machine learning.

## Project Structure

```
.
├── data/
│   ├── raw/
│   │   └── big_student_clear_third_version.csv
│   └── processed/
│       ├── cleaned_data.csv
│       └── features.csv
├── src/
│   ├── data/
│   │   └── load_data.py
│   ├── features/
│   │   └── feature_engineering.py
│   ├── models/
│   │   └── train.py
│   └── api/
│       └── app.py
├── configs/
├── dags/
│   └── mooc_pipeline.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── feature_store/
│   ├── feature_store.yaml
│   └── features.py
├── models/
│   └── lgb_model.txt
├── requirements.txt
├── dvc.yaml
└── README.md
```

## Setup and Run Instructions

### Prerequisites
- Docker and Docker Compose
- Python 3.9+
- Git

### 1. Clone and Setup
```bash
git clone <repo>
cd mooc-mlops
pip install -r requirements.txt
```

### 2. Initialize DVC
```bash
dvc init
dvc add data/raw/big_student_clear_third_version.csv
```

### 3. Run the Pipeline with Docker Compose
```bash
cd docker
docker-compose up --build
```

This will start:
- PostgreSQL database
- MLflow tracking server
- Airflow webserver and scheduler
- FastAPI prediction service

### 4. Access Services
- Airflow UI: http://localhost:8080 (admin/admin)
- MLflow UI: http://localhost:5000
- API: http://localhost:8000

### 5. Run Pipeline Manually (Alternative)
```bash
# Load and clean data
python src/data/load_data.py

# Feature engineering
python src/features/feature_engineering.py

# Train model
python src/models/train.py
```

### 6. API Usage
Send POST request to `/predict` with JSON:
```json
{
  "features": {
    "year": 2012,
    "semester": 1,
    "viewed": 1,
    "explored": 0,
    "grade": 0.0,
    "nevents": 10,
    "ndays_act": 5,
    "nplay_video": 20,
    "nchapters": 5,
    "nforum_posts": 0,
    "incomplete_flag": 1,
    "age": 25,
    "duration_days": 30,
    "activity_per_day": 2.0,
    "video_per_chapter": 4.0,
    "institute_encoded": 0,
    "course_id_encoded": 0,
    "final_cc_cname_DI_encoded": 0,
    "LoE_DI_encoded": 0,
    "gender_encoded": 0
  }
}
```

## Pipeline Architecture
1. Raw data ingestion with DVC
2. Data cleaning and preprocessing
3. Feature engineering
4. Model training with Optuna hyperparameter tuning
5. Model logging to MLflow
6. Feature serving with Feast
7. Orchestration with Airflow
8. Prediction API with FastAPI
9. Containerization with Docker