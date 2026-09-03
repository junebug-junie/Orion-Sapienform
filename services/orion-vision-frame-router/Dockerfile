# Build context: repo root (see docker-compose.yml)
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY services/orion-vision-frame-router/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY services/orion-vision-frame-router/app ./app
COPY orion ./orion
COPY config/vision_frame_router.yaml /app/config/vision_frame_router.yaml

ENV PYTHONPATH=/app
# Unlike its siblings (orion-vision-host, orion-vision-retina), this
# Dockerfile never set this. Confirmed live 2026-09-03: with it unset,
# stdout is block-buffered against Docker's log pipe, so this service's own
# loguru sink (a plain `print`) silently withheld its INFO-level app logs --
# including the one line that would have confirmed a restart actually
# resubscribed and started dispatching -- while stderr-based warnings still
# appeared immediately. Diagnosing this redeploy required cross-checking a
# downstream service's logs instead of this one's own.
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010"]
