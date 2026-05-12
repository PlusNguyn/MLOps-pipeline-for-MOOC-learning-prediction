# main.py

from src.pipeline.preprocess import (
    preprocess
)

from src.training.train import (
    train_pipeline
)

from src.training.optuna_tune import (
    tune_pipeline
)


def main():

    print("=" * 50)
    print("STARTING MLOPS PIPELINE")
    print("=" * 50)

    # =========================
    # Preprocess
    # =========================

    preprocess()

    # =========================
    # Train Baseline Model
    # =========================

    train_pipeline()

    # =========================
    # Hyperparameter Tuning
    # =========================

    tune_pipeline()

    print("=" * 50)
    print("PIPELINE COMPLETED")
    print("=" * 50)


if __name__ == "__main__":

    main()