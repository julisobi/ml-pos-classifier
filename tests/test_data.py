"""Test data file.

This file provides tests for data module in pos classifier package.
"""

import pytest

from pos_classifier.data.preprocessing import clean_text
from pos_classifier.data.postprocessing import decode_fasttext_label


@pytest.mark.parametrize(
    "text,expected",
    [
        ("This. is. a. TEST!", "test"),
        ("Health Probiotic Supplement-Capsules", "health probiotic supplementcapsules"),
        ("ProbiotiC! Supplement, Capsules.", "probiotic supplement capsules"),
    ],
)
def test_clean_text(text, expected):
    """Test that the `clean_text` function correctly normalizes input strings.

    Args:
        text (str): Raw input text to be cleaned.
        expected (str): Expected normalized output.

    Asserts:
        The output of `clean_text` matches the expected cleaned version.

    """
    assert clean_text(text) == expected


def test_decode_fasttext_label():
    """Test that `decode_fasttext_label` correctly maps FastText labels to their corresponding human-readable product categories.

    Asserts:
        Each known FastText label maps to the correct category name.
    """
    assert decode_fasttext_label(["__label__0"]) == "Beverages"
    assert decode_fasttext_label(["__label__1"]) == "Dry Goods & Pantry Staples"
    assert decode_fasttext_label(["__label__2"]) == "Fresh & Perishable Items"
    assert decode_fasttext_label(["__label__3"]) == "Household & Personal Care"
    assert decode_fasttext_label(["__label__4"]) == "Specialty & Miscellaneous"
