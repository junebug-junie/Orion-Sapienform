FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn redis loguru pydantic-settings

COPY services/orion-bus-tap/app ./app
COPY services/orion-bus-tap/static ./static

CMD ["sh", "-c", "uvicorn app.main:app --host ${UI_BIND_HOST:-0.0.0.0} --port ${UI_BIND_PORT:-8101}"]
