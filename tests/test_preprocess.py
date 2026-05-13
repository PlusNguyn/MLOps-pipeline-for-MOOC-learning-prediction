import pandas as pd

from src.pipeline import preprocess as preprocess_module


def test_preprocess_pipeline_monkeypatch(monkeypatch):
    sample_tables = {
        "students": pd.DataFrame(
            [
                {
                    "id_student": 1,
                    "code_module": "AAA",
                    "code_presentation": "2013J",
                    "gender": "M",
                    "region": "East",
                    "highest_education": "HE Qualification",
                    "imd_band": "80-90%",
                    "age_band": "35-55",
                    "num_of_prev_attempts": 0,
                    "studied_credits": 60,
                    "disability": "N",
                    "final_result": "Pass",
                }
            ]
        ),
        "vle": pd.DataFrame([{"id_student": 1, "date": 2, "sum_click": 30}]),
        "assessments": pd.DataFrame([{"id_assessment": 10, "id_student": 1, "is_banked": 0, "score": 85}]),
        "assess_def": pd.DataFrame([{"id_assessment": 10, "weight": 100}]),
    }

    called = {
        "load_oulad": False,
        "init_db": False,
        "save_dataframe": False,
        "to_csv": False,
        "save_feature_metadata": False,
    }

    def fake_load_oulad(path):
        called["load_oulad"] = True
        return sample_tables

    def fake_init_db():
        called["init_db"] = True

    def fake_save_dataframe(table_name, df, if_exists="replace"):
        called["save_dataframe"] = True
        assert table_name == "processed_train_data"
        assert "total_clicks" in df.columns

    def fake_to_csv(self, *args, **kwargs):
        called["to_csv"] = True

    def fake_save_feature_metadata(*args, **kwargs):
        called["save_feature_metadata"] = True

    monkeypatch.setattr(preprocess_module, "load_oulad", fake_load_oulad)
    monkeypatch.setattr(preprocess_module, "init_db", fake_init_db)
    monkeypatch.setattr(preprocess_module, "save_dataframe", fake_save_dataframe)
    monkeypatch.setattr(preprocess_module, "save_feature_metadata", fake_save_feature_metadata)
    monkeypatch.setattr(preprocess_module.pd.DataFrame, "to_csv", fake_to_csv, raising=False)

    result_df = preprocess_module.preprocess()

    assert called["load_oulad"]
    assert called["init_db"]
    assert called["save_dataframe"]
    assert called["to_csv"]
    assert called["save_feature_metadata"]
    assert result_df["label"].iloc[0] == 1
    assert result_df["total_clicks"].iloc[0] == 30
