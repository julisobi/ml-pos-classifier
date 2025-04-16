FROM python:3.10-slim

ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONBUFFERED=1

ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.8.3

RUN python -m pip install "poetry==$POETRY_VERSION"

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libglib2.0-0 \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/* \

RUN poetry config virtualenvs.in-project true && poetry install --no-interaction --without dev

COPY ./data /app/data
COPY ./src /app/src
COPY ./app /app/app
COPY ./artifacts /app/artifacts
ENV PYTHONPATH=/app/src

CMD ["bash", "-c", "\
  if [ \"$MODE\" = \"train\" ]; then \
    poetry run python src/pos_classifier/train.py; \
  elif [ \"$MODE\" = \"api\" ]; then \
    poetry run uvicorn app.pos_api:app --host 0.0.0.0 --port 8000; \
  else \
    echo 'Unknown MODE: '$MODE; exit 1; \
  fi"]

EXPOSE 8000
