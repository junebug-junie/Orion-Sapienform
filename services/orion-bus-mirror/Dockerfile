FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir aiosqlite redis loguru pydantic-settings

COPY services/orion-bus-mirror/app ./app

CMD ["python", "-m", "app.main"]
