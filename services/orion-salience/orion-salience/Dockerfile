FROM python:3.10-slim

WORKDIR /app

# Install cron
RUN apt-get update && apt-get install -y cron && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

COPY service.py .
COPY offline_worker.py .
COPY crontab.txt /etc/cron.d/salience-cron

RUN chmod 0644 /etc/cron.d/salience-cron && \
    crontab /etc/cron.d/salience-cron

EXPOSE 8091

CMD service cron start && uvicorn service:app --host 0.0.0.0 --port 8091
