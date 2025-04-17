"""Experiment file.

This module provides MLflow experiments with FastText model for POS classification.
"""

import mlflow
import os
import logging
from itertools import product
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

from pos_classifier.model.fasttext_wrapper import FastTextModelWrapper
from pos_classifier.config.logging_config import setup_logging
from pos_classifier.data.data_loader import load_data
from pos_classifier.data.preprocessing import (
    preprocess_data,
    prepare_data_for_fasttext,
    split_data,
)
from pos_classifier.config.config import (
    FASTTEXT_TRAIN_FILE,
    FASTTEXT_TEST_FILE,
    EXPERIMENT_MODEL_PATH,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    TRAIN_DATA_PATH,
)

setup_logging()

logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
experiment_params = {"epoch": [15, 20, 25], "lr": [0.05, 0.1], "word_ngrams": [1, 2]}
param_combinations = list(product(*experiment_params.values()))
param_keys = list(experiment_params.keys())


def prepare_data_for_experiment():
    """Preprocess, split and prepare data for experiments."""
    logging.info("Preparing data for experiment")
    df = load_data(TRAIN_DATA_PATH)
    df = preprocess_data(df)
    train_df, test_df = split_data(df)
    prepare_data_for_fasttext(train_df, FASTTEXT_TRAIN_FILE)
    prepare_data_for_fasttext(test_df, FASTTEXT_TEST_FILE)


def run_experiments():
    """Run a series of FastText training experiments with different hyperparameter combinations, log results to MLflow, and register the best model.

    The function:
    - Creates or sets the MLflow experiment.
    - Iterates over predefined combinations of hyperparameters.
    - Trains a FastText model for each combination.
    - Logs training parameters, metrics, and the model to MLflow.
    - Registers the model with input/output schema and signature.

    """
    if not mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME):
        mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    os.makedirs(EXPERIMENT_MODEL_PATH, exist_ok=True)

    for combo in param_combinations:
        logging.info(f"Running experiment with parameters: {combo[0]}")
        params = dict(zip(param_keys, combo))
        params.update(
            {
                "input": str(FASTTEXT_TRAIN_FILE),
                "test_input": str(FASTTEXT_TEST_FILE),
                "verbose": 2,
                "model_location": f"{EXPERIMENT_MODEL_PATH}/fasttext_model_e{params['epoch']}_lr{params['lr']}_wn{params['word_ngrams']}.bin",
            }
        )

        with mlflow.start_run(run_name="FastText Experiment") as run:
            mlflow.set_tag("run_id", run.info.run_id)
            mlflow.log_params(params)

            model = FastTextModelWrapper(params)
            model_location = model.train()

            metrics = model.evaluate(params["test_input"])
            mlflow.log_metrics(metrics)

            input_schema = Schema([ColSpec("string", "product_description")])
            output_schema = Schema(
                [ColSpec("string", "predicted_label"), ColSpec("float", "probability")]
            )
            signature = ModelSignature(inputs=input_schema, outputs=output_schema)

            model.clear_model()
            mlflow.pyfunc.log_model(
                artifact_path="fasttext_model",
                artifacts={"fasttext_model_path": model_location},
                python_model=model,
                signature=signature,
                registered_model_name="fasttext_pyfunc_model",
            )


if __name__ == "__main__":
    prepare_data_for_experiment()
    run_experiments()
