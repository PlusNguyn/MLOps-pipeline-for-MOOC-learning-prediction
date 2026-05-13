import os
from datetime import datetime

import pandas as pd
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. Please define it in your environment or .env file."
    )

engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    num_clicks = Column(Integer, nullable=False)
    days_active = Column(Integer, nullable=False)
    avg_score = Column(Float, nullable=False)
    studied_credits = Column(Integer, nullable=False)
    engagement_score = Column(Float, nullable=False)
    consistency = Column(Float, nullable=False)
    prediction = Column(Integer, nullable=False)
    prediction_label = Column(String(32), nullable=False)
    model_name = Column(String(128), nullable=True)
    model_stage = Column(String(64), nullable=True)


class ModelFeatures(Base):
    __tablename__ = "model_features"

    id = Column(Integer, primary_key=True, index=True)
    features = Column(String, nullable=False)  # JSON string of feature list
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class ModelMedians(Base):
    __tablename__ = "model_medians"

    id = Column(Integer, primary_key=True, index=True)
    medians = Column(String, nullable=False)  # JSON string of medians dict
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class FeatureMetadata(Base):
    __tablename__ = "feature_metadata"

    id = Column(Integer, primary_key=True, index=True)
    metadata_json = Column("metadata", String, nullable=False)  # JSON string of feature metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def save_dataframe(table_name: str, df: pd.DataFrame, if_exists: str = "replace") -> None:
    df.to_sql(
        table_name,
        con=engine,
        if_exists=if_exists,
        index=False,
        method="multi",
        chunksize=500,
    )


def save_model_features(features: list[str]) -> None:
    import json
    session = SessionLocal()
    try:
        features_json = json.dumps(features)
        db_features = ModelFeatures(features=features_json)
        session.add(db_features)
        session.commit()
    finally:
        session.close()


def save_model_medians(medians: dict[str, float]) -> None:
    import json
    session = SessionLocal()
    try:
        medians_json = json.dumps(medians)
        db_medians = ModelMedians(medians=medians_json)
        session.add(db_medians)
        session.commit()
    finally:
        session.close()


def save_feature_metadata(metadata: dict[str, object]) -> None:
    import json
    session = SessionLocal()
    try:
        metadata_json = json.dumps(metadata)
        db_metadata = FeatureMetadata(metadata_json=metadata_json)
        session.add(db_metadata)
        session.commit()
    finally:
        session.close()
