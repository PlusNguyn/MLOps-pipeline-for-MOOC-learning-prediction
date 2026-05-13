from api import app as api_app


def test_load_model_from_registry_uses_mlflow(monkeypatch):
    loaded = {"path": None}

    def fake_load_model(path):
        loaded["path"] = path
        return "fake-model"

    monkeypatch.setattr(api_app.mlflow.sklearn, "load_model", fake_load_model)

    model = api_app.load_model_from_registry()

    assert model == "fake-model"
    assert loaded["path"] == f"models:/{api_app.MLFLOW_MODEL_NAME}/Production"
