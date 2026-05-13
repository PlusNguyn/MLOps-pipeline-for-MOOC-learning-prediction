import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
