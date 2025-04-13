"""Train file.

This module provides testing of the FastText model.
"""

import fasttext
import logging

from src.pos_classifier.config.config import (
    FASTTEXT_TRAIN_FILE,
    FASTTEXT_MODEL_PATH,
)

logger = logging.getLogger(__name__)


def train(ft_train: str, epoch: int, lr: float, word_ngrams: int, verbose: int) -> None:
    """Train a FastText classifier and save the model to FASTTEXT_MODEL_PATH.

    Parameters
    ----------
    ft_train : str
        Path to FastText training file (.txt)
    epoch : int
        Number of training epochs
    lr : float
        Learning rate
    word_ngrams : int
        Maximum length of word n-grams
    verbose : int
        Verbosity level

    """
    logger.info("Training FastText model...")
    model = fasttext.train_supervised(
        input=ft_train, epoch=epoch, lr=lr, wordNgrams=word_ngrams, verbose=verbose
    )

    output_dir = FASTTEXT_MODEL_PATH.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving trained model to {FASTTEXT_MODEL_PATH}...")
    model.save_model(str(FASTTEXT_MODEL_PATH))
    logger.info("Model training completed and saved.")


if __name__ == "__main__":
    train(ft_train=str(FASTTEXT_TRAIN_FILE), epoch=25, lr=0.1, word_ngrams=2, verbose=2)
