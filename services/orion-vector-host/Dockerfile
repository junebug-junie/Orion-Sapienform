FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

COPY services/orion-vector-host/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY orion /app/orion
COPY services/orion-vector-host/app ./app

ENV PYTHONPATH="/app:${PYTHONPATH}"

EXPOSE 8320

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8320"]
