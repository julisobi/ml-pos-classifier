"""Train file.

This module provides methods for training FastText model.
"""

import yaml
import logging
import os

from pos_classifier.config.logging_config import setup_logging
from pos_classifier.config.config import (
    FASTTEXT_MODEL_PATH,
    PARAMS_PATH,
    TRAIN_DATA_PATH,
    FASTTEXT_TRAIN_FILE,
    MODEL_DIR,
)
from pos_classifier.model.fasttext_wrapper import FastTextModelWrapper
from pos_classifier.data.data_loader import load_data
from pos_classifier.data.preprocessing import prepare_data_for_fasttext, preprocess_data

setup_logging()

logger = logging.getLogger(__name__)


def load_params(yaml_path=PARAMS_PATH):
    """Load model parameters from a YAML configuration file.

    Parameters
    ----------
    yaml_path : str
        Path to the YAML file containing model parameters.

    Returns
    -------
    dict
        Dictionary of parameters to be used for training.

    """
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    return config["parameters"]


def main():
    """Load data and train FastText model."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    params = load_params()
    params.update(
        {"input": str(FASTTEXT_TRAIN_FILE), "model_location": str(FASTTEXT_MODEL_PATH)}
    )

    logger.info("Loading and preprocessing training data...")
    df = load_data(TRAIN_DATA_PATH)
    train_df = preprocess_data(df)

    logger.info("Saving FastText formatted training data...")
    prepare_data_for_fasttext(train_df, FASTTEXT_TRAIN_FILE)

    logger.info("Training FastText model...")
    model = FastTextModelWrapper(params)
    model.train()

    logger.info(f"Model saved to {FASTTEXT_MODEL_PATH}")


if __name__ == "__main__":
    main()
