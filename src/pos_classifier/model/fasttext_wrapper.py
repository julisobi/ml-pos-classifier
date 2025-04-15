"""FastText Wrapper class.

This module provides FastText Model Wrapper.
"""

import fasttext
from mlflow.pyfunc import PythonModel

from pos_classifier.data.preprocessing import clean_text


class FastTextModelWrapper(PythonModel):
    """A wrapper class for training, testing, and predicting with a FastText model."""

    def __init__(self, params):
        """Initialize the FastTextModelWrapper with training and model parameters.

        Parameters
        ----------
        params : dict
            Dictionary of training and model parameters. Must include keys like 'input' and 'model_location'.

        """
        self.params = params
        self.model = None

    def load_model(self):
        """Load a pre-trained FastText model from the specified location in the configuration."""
        if "model_location" not in self.params:
            raise ValueError("Model location is not specified in the configuration.")
        self.model = fasttext.load_model(self.params["model_location"])

    def clear_model(self):
        """Clear the current model from memory."""
        self.model = None

    def train(self):
        """Train a FastText model using the parameters provided in `self.params`.

        Returns
        -------
        str
            Path where the trained FastText model is saved.

        """
        fasttext_params = {
            key: value
            for key, value in self.params.items()
            if key not in ["model_location", "test_input"]
        }

        self.model = fasttext.train_supervised(**fasttext_params)
        self.model.save_model(self.params["model_location"])
        return self.params["model_location"]

    def predict(self, text: str, threshold: float = 0.0, k: int = 1) -> tuple:
        """Predict the label(s) for a given input text using the trained FastText model.

        Parameters
        ----------
        text : str
            The input text to classify.
        threshold : float, optional
            The probability threshold to filter predictions. Defaults to 0.0.
        k : int, optional
            The number of top predictions to return. Defaults to 1.

        Returns
        -------
        tuple
            A tuple containing:
            - List of predicted labels
            - List of corresponding prediction probabilities

        """
        if not self.model:
            raise ValueError("Model is not loaded. Please train or load a model first.")

        text = clean_text(text)
        return self.model.predict(text, k=k, threshold=threshold)

    def evaluate(self, test_file: str, threshold: float = 0.65) -> dict:
        """Evaluate the model's performance on a labeled test dataset.

        Parameters
        ----------
        test_file : str
            Path to the test file.
        threshold : float, optional
            Probability threshold to consider predictions as valid. Default is 0.65.

        Returns
        -------
        dict
            Dictionary containing the following evaluation metrics:
            - 'num_test' : int
                Number of test samples.
            - 'precision' : float
                Precision score of the model.
            - 'recall' : float
                Recall score of the model.
            - 'f1' : float
                F1-score of the model.

        """
        if not self.model:
            raise ValueError("Model is not loaded. Please train or load a model first.")

        results = self.model.test(test_file, k=-1, threshold=threshold)
        precision = results[1]
        recall = results[2]

        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)

        return {
            "num_test": int(results[0]),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1_score),
        }
