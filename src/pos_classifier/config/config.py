"""Config file.

This module provides all project configurations.
"""

import os

from pathlib import Path
from datetime import datetime

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

# Source directory
SOURCE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
TRAIN_DATA_PATH = DATA_DIR / "Training_Data.csv"
QUERY_VAL_DATA_PATH = DATA_DIR / "Query_and_Validation_Data.csv"

# Processed data
FASTTEXT_TRAIN_FILE = DATA_DIR / "fasttext_train.txt"
FASTTEXT_TEST_FILE = DATA_DIR / "fasttext_test.txt"

# Model paths
MODEL_DIR = BASE_DIR / "artifacts"
FASTTEXT_MODEL_PATH = MODEL_DIR / "fasttext_model.bin"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

# Output paths
OUTPUT_DIR = BASE_DIR / "outputs"
PREDICTION_PATH = OUTPUT_DIR / "predictions.csv"

# Logging paths
LOG_DIR = BASE_DIR / "logs"
LOG_PATH = LOG_DIR / "app.log"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Config paths
CONFIG_DIR = SOURCE_DIR / "config"
PARAMS_PATH = CONFIG_DIR / "params.yaml"

# Monitoring paths
APP_DIR = BASE_DIR / "app"
MONITORING_DIR = APP_DIR / "monitoring"
MONITORING_PATH = MONITORING_DIR / "monitor.json"

# Experiments
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000/"
MLFLOW_EXPERIMENT_NAME = "POS Classification"
EXPERIMENT_DIR = BASE_DIR / "experiments"
EXPERIMENT_MODEL_PATH = EXPERIMENT_DIR / "experiment_models"


def get_prediction_output_path() -> Path:
    """Generate a timestamped file path for saving batch prediction results.

    Returns:
        Path: A Path object pointing to the output CSV file within the OUTPUT_DIR.

    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUT_DIR / f"predictions_{timestamp}.csv"
