import asyncio

from httpx import ASGITransport, AsyncClient

from api import app as api_app


class FakeModel:
    def predict(self, X):
        return [1]

    def predict_proba(self, X):
        return [[0.15, 0.85]]


class DummySession:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def add(self, obj):
        self.added = True

    def commit(self):
        self.committed = True


def test_root_endpoint(monkeypatch):
    monkeypatch.setattr(api_app, "init_db", lambda: None)
    monkeypatch.setattr(api_app, "load_model_from_registry", lambda: FakeModel())

    async def request_root():
        transport = ASGITransport(app=api_app.app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get("/")

    response = asyncio.run(request_root())

    assert response.status_code == 200
    assert response.json() == {"message": "Learning Prediction API Running"}


def test_health_endpoint_when_model_loaded(monkeypatch):
    monkeypatch.setattr(api_app, "model", FakeModel())

    async def request_health():
        transport = ASGITransport(app=api_app.app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get("/health")

    response = asyncio.run(request_health())

    assert response.status_code == 200
    assert response.json() == {"model_loaded": True}


def test_predict_endpoint_returns_prediction_and_logs(monkeypatch):
    monkeypatch.setattr(api_app, "model", FakeModel())
    monkeypatch.setattr(api_app, "SessionLocal", lambda: DummySession())
    monkeypatch.setattr(
        api_app,
        "feature_names",
        [
            "code_module",
            "code_presentation",
            "gender",
            "region",
            "highest_education",
            "imd_band",
            "age_band",
            "num_of_prev_attempts",
            "studied_credits",
            "disability",
            "total_clicks",
            "active_days",
            "avg_daily_clicks",
            "max_clicks_day",
            "engagement_span",
            "avg_score",
            "min_score",
            "submission_count",
            "late_submissions",
            "weighted_avg",
        ],
    )
    monkeypatch.setattr(api_app, "feature_medians", {name: 0.0 for name in api_app.feature_names})

    payload = {
        "num_clicks": 100,
        "days_active": 20,
        "avg_score": 85.0,
        "studied_credits": 5,
    }

    async def request_predict():
        transport = ASGITransport(app=api_app.app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post("/predict", json=payload)

    response = asyncio.run(request_predict())

    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1
    assert data["level"] == "Success"
    assert data["success_probability"] == 0.85
    assert "engagement_score" in data
    assert "consistency" in data
