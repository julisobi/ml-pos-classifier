import joblib
import os

from pos_classifier.config.config import LABEL_ENCODER_PATH


def load_label_encoder() -> joblib:
    """
    Load the LabelEncoder from LABEL_ENCODER_PATH.

    Returns
    -------
    LabelEncoder
        Loaded label encoder instance

    Raises
    ------
    FileNotFoundError
        If the encoder file does not exist.

    ValueError
        If the file is not a .pkl file.
    """
    if not os.path.exists(LABEL_ENCODER_PATH):
        raise FileNotFoundError(f"Label encoder file not found at: {LABEL_ENCODER_PATH}")

    if not str(LABEL_ENCODER_PATH).endswith(".pkl"):
        raise ValueError("Label encoder file must be a .pkl file.")

    return joblib.load(LABEL_ENCODER_PATH)


def decode_fasttext_label(predicted_label: list[str]) -> str:
    """
    Decode FastText label string back to original category.

    Parameters
    ----------
    predicted_label : list[str]
        FastText label (e.g.: '__label__3')

    Returns
    -------
    str
        Original category label
    """
    label_encoder = load_label_encoder()
    label_id = predicted_label[0].replace("__label__", "")
    predicted_class = label_encoder.inverse_transform([int(label_id)])[0]
    return predicted_class
