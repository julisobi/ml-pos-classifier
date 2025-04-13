import string
import logging
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import nltk

from src.pos_classifier.data.data_loader import load_data
from src.pos_classifier.config.config import (
    TRAIN_DATA_PATH,
    FASTTEXT_TRAIN_FILE,
    FASTTEXT_TEST_FILE,
    LABEL_ENCODER_PATH,
)

nltk.download('stopwords')

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean the text by lowering case, removing punctuation and stopwords.

    Parameters
    ----------
    text : str
        Raw text

    Returns
    -------
    cleaned_text : str
        Cleaned text
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    cleaned_text = ' '.join([word for word in text.split() if word not in stop_words])
    return cleaned_text


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess training data and encode labels.
    LabelEncoder is trained and saved inside the function.

    Parameters
    ----------
    df : pd.DataFrame
        Raw training DataFrame

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame with encoded labels
    """
    df['product_description'] = df['product_description'].apply(clean_text)
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['category'])
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into training and test sets for experimentation.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame
    test_size : float
        Proportion of the dataset to include in the test split
    random_state : int
        Random seed

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Train and test DataFrames
    """
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=random_state)
    return train_df, test_df


def prepare_data_for_fasttext(df: pd.DataFrame, output_path: str) -> None:
    """
    Save data in FastText format.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'product_description' and 'label'
    output_path : str
        Output .txt file path for FastText
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            if pd.notna(row['category']):
                label = f"__label__{row['label']}"
                text = row['product_description'].replace("\n", " ").strip()
                f.write(f"{label} {text}\n")


def main():
    """Main script to preprocess and save training and validation data for FastText."""
    logger.info("Loading and preprocessing training data...")
    df = load_data(TRAIN_DATA_PATH)
    df = preprocess_data(df)

    logger.info("Splitting the data into train and test sets...")
    train_df, test_df = split_data(df)

    logger.info("Saving FastText formatted training data...")
    prepare_data_for_fasttext(train_df, FASTTEXT_TRAIN_FILE)

    logger.info("Saving FastText formatted test data...")
    prepare_data_for_fasttext(test_df, FASTTEXT_TEST_FILE)


if __name__ == '__main__':
    main()
