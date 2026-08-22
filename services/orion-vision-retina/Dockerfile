# Build context: repo root (see docker-compose.yml)
FROM python:3.10-slim

WORKDIR /app

# ffmpeg added 2026-08-22 for app/clip_capture.py (AffectGPT on-demand
# video+audio clips) -- everything else here is unchanged from before.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY services/orion-vision-retina/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY services/orion-vision-retina/app ./app
COPY orion ./orion

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
