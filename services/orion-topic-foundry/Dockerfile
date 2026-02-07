FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      curl \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY services/orion-topic-foundry/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -U pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

RUN apt-get purge -y --auto-remove build-essential || true

COPY services/orion-topic-foundry/app ./app
COPY orion ./orion

ENV PYTHONPATH=/app

CMD ["/bin/sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8615} --log-level info"]
