"""Predict file.

This module provides predicting and evaluating of the FastText model.
"""

import fasttext
import joblib

from src.pos_classifier.config.config import (
    FASTTEXT_MODEL_PATH,
    LABEL_ENCODER_PATH,
)


def load_model() -> fasttext.FastText._FastText:
    """Load the FastText model from FASTTEXT_MODEL_PATH.

    Returns
    -------
    fasttext.FastText._FastText
        Loaded FastText model

    """
    return fasttext.load_model(str(FASTTEXT_MODEL_PATH))


def load_label_encoder() -> joblib:
    """Load the LabelEncoder from LABEL_ENCODER_PATH.

    Returns
    -------
    LabelEncoder
        Loaded label encoder instance

    """
    return joblib.load(LABEL_ENCODER_PATH)


def predict_label(text) -> tuple[list[str], list[float]]:
    """Predict label and probability for a given text input using FastText.

    Parameters
    ----------
    text : str
        Product description

    Returns
    -------
    tuple[list[str], list[float]]
        Predicted label(s) and associated probability(ies)

    """
    model = load_model()
    predicted_label, prob = model.predict(text)
    return predicted_label, prob


def evaluate_model(test_path: str) -> dict:
    """Evaluate FastText model on validation data file.

    Parameters
    ----------
    test_path : str
        Path to .txt validation data in FastText format

    Returns
    -------
    dict
        FastText test results with precision, recall, and number of samples

    """
    model = load_model()
    results = model.test(test_path)
    return results


def decode_fasttext_label(predicted_label: list[str]) -> str:
    """Decode FastText label string back to original category.

    Parameters
    ----------
    predicted_label : list[str]
        FastText label (e.g., '__label__3')

    Returns
    -------
    str
        Original category label

    """
    label_encoder = load_label_encoder()
    label_id = predicted_label[0].replace("__label__", "")
    predicted_class = label_encoder.inverse_transform([int(label_id)])[0]
    return predicted_class
