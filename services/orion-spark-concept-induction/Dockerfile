FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

COPY services/orion-spark-concept-induction/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && python -m spacy download en_core_web_sm

# Copy shared Orion package
COPY orion /app/orion

# Runtime policy + world-pulse section config (substrate metabolism + capability act)
COPY config/autonomy /app/config/autonomy
COPY config/world_pulse /app/config/world_pulse

# Copy service code
COPY services/orion-spark-concept-induction /app

ENV PYTHONPATH="/app:${PYTHONPATH}"

EXPOSE 8510

CMD ["python", "-m", "app.main"]
