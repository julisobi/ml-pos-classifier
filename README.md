# POS data classifier

This project demonstrates an end-to-end machine learning pipeline with separate modules for training and inference.

## Prerequisites

The current project is running using the following technologies:

    Python version >= 3.10
    Poetry
    Docker

## Project setup

1. Clone this repository:
```shell
git clone https://github.com/julisobi/ml-pos-classifier.git`
```
2. Add data files:
Place the following data files into the `data` folder:
- `Training_Data.csv`
- `Query_and_Validation_Data.csv`

## Docker

This project includes two main modules:
- Train: model retraining
- Api: FastAPI app for serving predictions

Build docker image:
```shell
docker build -t pos-classifier .
```

To retrain the model:
```shell
docker run -v artifacts:/app/artifacts -e MODE=train pos-classifier
```
`-v artifacts:/app/artifacts`: Mounts the artifacts volume to store the trained model.
`-e MODE=train`: Instructs the container to execute the training script.

To run FastAPI app:
```shell
docker run -v artifacts:/app/artifacts -e MODE=api -p 8000:8000 pos-classifier
```
`-v artifacts:/app/artifacts`: Mounts the artifacts volume, giving the container access to the trained model.
`-e MODE=api`: Instructs the container to start the FastAPI server.
`-p 8000:8000`: Exposes port 8000 for accessing the API locally.
Access the API docs at http://0.0.0.0:8000/docs.

## Local Development

Running application and training outside Docker:

```shell
poetry install
poetry config virtualenvs.in-project true
poetry shell
```

To train the model locally:
```shell
poetry run python src/pos_classifier/train.py
```

To start the monitoring dashboard (built with Streamlit):
```shell
poetry run streamlit run app/monitoring/monitoring.py
```

To run the FastAPI app locally:
```shell
poetry run uvicorn app.pos_api:app
```
Access the API docs at http://127.0.0.1:8000/docs.

##  Running FastText experiments with MLflow

The `experiments` module orchestrates a series of experiments using different hyperparameter combinations for the FastText model. Each experiment logs parameters, metrics, and models to MLflow.

```shell
poetry run python experiments/run_experiment.py
```
With the MLflow UI running, navigate to http://127.0.0.1:5001.

## Code Quality

This project uses `pre-commit` to ensure consistent code formatting and quality.

Install the Git hooks:
```shell
pre-commit install
```

Run on all files:
```shell
pre-commit run --all-files
```
