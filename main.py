import pandas as pd

from src.pipeline.preprocess import preprocess

from src.training.train import train

from src.training.optuna_tune import tune_model


def main():

    print("=" * 50)
    print("STARTING MLOPS PIPELINE")
    print("=" * 50)

    # =========================
    # Preprocess Dataset
    # =========================

    processed_df = preprocess()

    # =========================
    # Baseline Training
    # =========================

    train(processed_df)

    # =========================
    # Hyperparameter Tuning
    # =========================

    tune_model(processed_df)

    print("=" * 50)
    print("PIPELINE COMPLETED")
    print("=" * 50)


if __name__ == "__main__":

    main()

