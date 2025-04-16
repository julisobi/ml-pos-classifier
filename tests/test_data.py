"""Test data file.

This file provides tests for data module in pos classifier package.
"""

import pytest
import pandas as pd

from pos_classifier.data.data_loader import load_data
from pos_classifier.data.preprocessing import clean_text, preprocess_data, split_data
from pos_classifier.data.postprocessing import decode_fasttext_label


@pytest.fixture
def sample_csv(tmp_path):
    """Fixture that creates a temporary CSV file for testing."""
    content = """Product Description,Category
    Apple juice,Beverages
    Toothpaste,Household & Personal Care
    """
    file_path = tmp_path / "sample.csv"
    file_path.write_text(content)
    return str(file_path)


@pytest.fixture
def sample_dataframe():
    """Fixture returning a sample dataframe for testing."""
    return pd.DataFrame(
        {
            "product_description": [
                "Apple JUICE!!",
                "Toothpaste...",
                "product3",
                "product4",
                "product5",
            ],
            "category": [
                "Beverages",
                "Household & Personal Care",
                "Fresh & Perishable Items",
                "Specialty & Miscellaneous",
                "Dry Goods & Pantry Staples",
            ],
            "label": [0, 3, 0, 3, 0],
        }
    )


def test_load_data(sample_csv):
    """Test that load_data correctly loads data and renames column names."""
    df = load_data(sample_csv)
    assert isinstance(df, pd.DataFrame)
    assert "product_description" in df.columns
    assert "category" in df.columns
    assert len(df) == 2


@pytest.mark.parametrize(
    "text,expected",
    [
        ("This. is. a. TEST!", "test"),
        ("Health Probiotic Supplement-Capsules", "health probiotic supplementcapsules"),
        ("ProbiotiC! Supplement, Capsules.", "probiotic supplement capsules"),
    ],
)
def test_clean_text(text, expected):
    """Test that clean_text correctly normalizes input strings."""
    assert clean_text(text) == expected


def test_preprocess_data(sample_dataframe):
    """Test that preprocess_data adds a label column."""
    df = preprocess_data(sample_dataframe)
    assert "label" in df.columns
    assert pd.api.types.is_integer_dtype(df["label"])
    assert df["label"].between(0, 4).all()


def test_split_data(sample_dataframe):
    """Test that split_data correctly splits the data into train and test sets."""
    train_df, test_df = split_data(sample_dataframe, test_size=0.4)
    assert len(train_df) == 3
    assert len(test_df) == 2


def test_decode_fasttext_label():
    """Test that decode_fasttext_label correctly maps FastText labels to their corresponding product categories."""
    assert decode_fasttext_label(["__label__0"]) == "Beverages"
    assert decode_fasttext_label(["__label__1"]) == "Dry Goods & Pantry Staples"
    assert decode_fasttext_label(["__label__2"]) == "Fresh & Perishable Items"
    assert decode_fasttext_label(["__label__3"]) == "Household & Personal Care"
    assert decode_fasttext_label(["__label__4"]) == "Specialty & Miscellaneous"
