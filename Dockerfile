FROM python:3.10-slim

ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONBUFFERED=1

ENV POETRY_VERSION=1.8.3

RUN python -m pip install "poetry==$POETRY_VERSION"

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.in-project true \
  && poetry install --no-interaction --without dev --no-root

COPY ./src /app/src

COPY ./app /app/app

COPY entrypoint.py /app/entrypoint.py

ENTRYPOINT ["python", "entrypoint.py"]

EXPOSE 8000
