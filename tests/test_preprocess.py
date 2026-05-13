import pandas as pd
import pytest

from src.pipeline import preprocess as preprocess_module


def test_preprocess_pipeline_monkeypatch(monkeypatch):
    sample_df = pd.DataFrame(
        [
            {
                "avg_score": 90,
                "num_clicks": 200,
                "days_active": 12,
                "studied_credits": 3,
                "final_result": "Pass",
            }
        ]
    )

    called = {
        "load_oulad": False,
        "init_db": False,
        "save_dataframe": False,
        "to_csv": False,
    }

    def fake_load_oulad(path):
        called["load_oulad"] = True
        return sample_df

    def fake_init_db():
        called["init_db"] = True

    def fake_save_dataframe(table_name, df, if_exists="replace"):
        called["save_dataframe"] = True
        assert table_name == "processed_train_data"
        assert "engagement_score" in df.columns

    def fake_to_csv(self, *args, **kwargs):
        called["to_csv"] = True

    monkeypatch.setattr(preprocess_module, "load_oulad", fake_load_oulad)
    monkeypatch.setattr(preprocess_module, "init_db", fake_init_db)
    monkeypatch.setattr(preprocess_module, "save_dataframe", fake_save_dataframe)
    monkeypatch.setattr(preprocess_module.pd.DataFrame, "to_csv", fake_to_csv)

    result_df = preprocess_module.preprocess()

    assert called["load_oulad"]
    assert called["init_db"]
    assert called["save_dataframe"]
    assert called["to_csv"]
    assert result_df["label"].iloc[0] == 1
    assert result_df["engagement_score"].iloc[0] >= 0
    assert result_df["consistency"].iloc[0] == pytest.approx(4.0)
