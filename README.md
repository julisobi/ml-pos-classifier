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
2. Add "Training_Data.csv" and "Query_and_Validation_Data.csv" to the [data](data) folder.

## Docker

This project includes two main modules:
- Training: For model retraining
- Inference: FastAPI app for serving predictions
  Docker and Docker Compose are used to build and run both services.

Build docker image:
```shell
docker build -t pos-classifier .
```

Retrain model:
```shell
docker run -v artifacts:/app/artifacts -e MODE=train pos-classifier
```

Run API:
```shell
docker run -v artifacts:/app/artifacts -e MODE=api -p 8000:8000 pos-classifier
```

## Local Development

Running application and training outside Docker:

```shell
poetry install
poetry config virtualenvs.in-project true
poetry shell
```

Run model training:

```shell
poetry run python src/pos_classifier/train.py
```

Run monitoring dashboard:

```shell
streamlit run app/monitoring/monitoring.py
```

Run FastAPI (remember to add /docs to see the endpoints)

```shell
poetry run uvicorn app.pos_api:app
```

