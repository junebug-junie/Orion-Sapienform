# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/app

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

COPY services/orion-dream/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Copy service code
COPY services/orion-dream/app ./app

COPY orion /app/orion

EXPOSE 8620

# Entrypoint runs FastAPI but can trigger dream manually
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8620"]
