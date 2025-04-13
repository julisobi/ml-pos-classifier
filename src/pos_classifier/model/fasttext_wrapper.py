"""FastText Wrapper class.

This module provides FastText Model Wrapper.
"""

import fasttext
from mlflow.pyfunc import PythonModel


class FastTextModelWrapper(PythonModel):
    """A wrapper class for training, testing, and predicting with a FastText model.

    This wrapper provides methods for training a model, making predictions, and evaluating the model's performance.
    """

    def __init__(self, params):
        """Initialize the FastTextModelWrapper with the provided configuration.

        Args:
            params (dict): A dictionary containing the parameters for training, testing, and model saving.

        """
        self.params = params
        self.model = None

    def load_model(self):
        """Load a pre-trained FastText model from the specified location in the configuration.

        The model will be loaded using the path specified by 'model_location' in the configuration.
        """
        if "model_location" not in self.params:
            raise ValueError("Model location is not specified in the configuration.")
        self.model = fasttext.load_model(self.params["model_location"])

    def clear_model(self):
        """Clear the current model from memory."""
        self.model = None

    def train(self):
        """Train a FastText model using the provided parameters in the configuration.

        The model is saved at the path specified by 'model_location' in the configuration.

        Returns:
            str: The path where the trained model is saved.

        """
        fasttext_params = {
            key: value
            for key, value in self.params.items()
            if key not in ["model_location", "test_input"]
        }

        self.model = fasttext.train_supervised(**fasttext_params)
        self.model.save_model(self.params["model_location"])
        return self.params["model_location"]

    def predict(self, text: str, threshold: float, k: int = 1) -> tuple:
        """Predict the labels for a given text input using the trained model.

        Args:
            text (str): The text for which predictions are to be made.
            threshold (float): The probability threshold to filter predictions.
            k (int, optional): The number of predictions to return. Defaults to -1 (return all predictions).

        Returns:
            tuple: A tuple containing two elements:
                - List of predicted labels
                - Corresponding list of prediction probabilities

        """
        if not self.model:
            raise ValueError("Model is not loaded. Please train or load a model first.")

        return self.model.predict(text, k=k, threshold=threshold)

    def evaluate(self, test_file: str = None, threshold: float = 0.65) -> dict:
        """Evaluate the model's performance on a test dataset.

        Args:
            test_file (str, optional): Path to the test file. If None, the 'test_file' key in the configuration is used.
            threshold (float, optional): The threshold for predictions to be considered relevant. Default is 0.65.

        Returns:
            dict: A dictionary containing the following evaluation metrics:
                - 'num_test': The number of test examples.
                - 'precision': The precision score.
                - 'recall': The recall score.
                - 'f1': The f1 score.

        """
        if not self.model:
            raise ValueError("Model is not loaded. Please train or load a model first.")

        test_file_path = test_file if test_file else self.params.get("test_file")

        if not test_file_path:
            raise ValueError(
                "Test file path is not provided in the configuration or as an argument."
            )

        results = self.model.test(test_file_path, k=-1, threshold=threshold)
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
