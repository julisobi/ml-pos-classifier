from pathlib import Path
from datetime import datetime

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
TRAIN_DATA_PATH = DATA_DIR / "Training_Data.csv"
QUERY_VAL_DATA_PATH = DATA_DIR / "Query_and_Validation_Data.csv"

# Processed data
FASTTEXT_TRAIN_FILE = DATA_DIR / "fasttext_train.txt"
FASTTEXT_TEST_FILE = DATA_DIR / "fasttext_test.txt"

# Model paths
MODEL_DIR = BASE_DIR / "models"
FASTTEXT_MODEL_PATH = MODEL_DIR / "fasttext_model.bin"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

# Output paths
OUTPUT_DIR = BASE_DIR / "outputs"
PREDICTION_PATH = OUTPUT_DIR / "predictions.csv"


def get_prediction_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUT_DIR / f"predictions_{timestamp}.csv"
