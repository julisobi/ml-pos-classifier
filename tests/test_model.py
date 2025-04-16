"""Test model file.

This file provides tests for FastTextModelWrapper class in pos classifier package.
"""

import pytest
from unittest.mock import MagicMock, patch

from pos_classifier.model.fasttext_wrapper import FastTextModelWrapper


@pytest.fixture
def default_params(tmp_path):
    """Fixture for default FastText training parameters."""
    return {
        "input": str(tmp_path / "train.txt"),
        "model_location": str(tmp_path / "model.bin"),
        "epoch": 10,
        "lr": 0.1,
    }


def test_init_params(default_params):
    """Test initialization stores parameters and sets model to None."""
    model = FastTextModelWrapper(default_params)
    assert model.params == default_params
    assert model.model is None


def test_load_model_missing_path():
    """Test that error is raised if model_location is missing in params."""
    model = FastTextModelWrapper(params={})
    with pytest.raises(ValueError, match="Model location is not specified"):
        model.load_model()


@patch("fasttext.load_model")
def test_load_model_success(mock_load, default_params):
    """Test that load_model successfully loads model when path is provided."""
    mock_model = MagicMock()
    mock_load.return_value = mock_model

    model = FastTextModelWrapper(default_params)
    model.load_model()

    mock_load.assert_called_once_with(default_params["model_location"])
    assert model.model == mock_model


def test_clear_model(default_params):
    """Test that clear_model sets internal model to None."""
    model = FastTextModelWrapper(default_params)
    model.model = "model"
    model.clear_model()
    assert model.model is None


@patch("fasttext.train_supervised")
def test_train_model_saves_and_returns_path(mock_train, default_params):
    """Test training a model and saving to the specified location."""
    mock_model = MagicMock()
    mock_train.return_value = mock_model

    model = FastTextModelWrapper(default_params)
    path = model.train()

    mock_model.save_model.assert_called_once_with(default_params["model_location"])
    assert path == default_params["model_location"]


def test_predict_model_not_loaded(default_params):
    """Test that error is raised when predicting without a loaded model."""
    model = FastTextModelWrapper(default_params)
    with pytest.raises(ValueError, match="Model is not loaded"):
        model.predict("product")


@patch("pos_classifier.model.fasttext_wrapper.clean_text", return_value="cleaned")
def test_predict_returns_predictions(mock_clean, default_params):
    """Test that predict returns label and probability for clean input."""
    model = FastTextModelWrapper(default_params)
    mock_model = MagicMock()
    mock_model.predict.return_value = (["__label__1"], [0.95])
    model.model = mock_model

    labels, probs = model.predict("Some dirty text", k=1, threshold=0.0)

    mock_clean.assert_called_once_with("Some dirty text")
    mock_model.predict.assert_called_once_with("cleaned", k=1, threshold=0.0)
    assert labels == ["__label__1"]
    assert probs == [0.95]


def test_evaluate_raises_if_model_not_loaded(default_params):
    """Test that error is raised when evaluating without a loaded model."""
    model = FastTextModelWrapper(default_params)
    with pytest.raises(ValueError, match="Model is not loaded"):
        model.evaluate("test.txt")


def test_evaluate_returns_metrics(default_params):
    """Test that evaluate returns precision, recall and F1-score."""
    model = FastTextModelWrapper(default_params)
    mock_model = MagicMock()
    mock_model.test.return_value = (100, 0.5, 0.5)
    model.model = mock_model

    result = model.evaluate("test.txt")

    mock_model.test.assert_called_once_with("test.txt", k=-1, threshold=0.65)
    assert result == {
        "num_test": 100,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }
