import fasttext

from src.pos_classifier.config.config import (
    FASTTEXT_TRAIN_FILE,
    FASTTEXT_MODEL_PATH,
)


def train(ft_train: str, epoch: int, lr: float, word_ngrams: int, verbose: int, min_count: int,) -> None:
    """
    Train a FastText classifier and save the model to FASTTEXT_MODEL_PATH.

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
    min_count : int
        Minimal number of word occurrences
    """
    print("Training FastText model...")
    model = fasttext.train_supervised(
        input=ft_train,
        epoch=epoch,
        lr=lr,
        wordNgrams=word_ngrams,
        verbose=verbose,
        minCount=min_count,
    )

    output_dir = FASTTEXT_MODEL_PATH.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving trained model to {FASTTEXT_MODEL_PATH}...")
    model.save_model(str(FASTTEXT_MODEL_PATH))
    print("Model training completed and saved.")


if __name__ == "__main__":
    train(
        ft_train=str(FASTTEXT_TRAIN_FILE),
        epoch=25,
        lr=0.1,
        word_ngrams=2,
        verbose=2,
        min_count=1,
    )
