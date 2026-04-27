FROM python:3.11-slim

WORKDIR /app

COPY services/orion-world-pulse/requirements.txt /app/services/orion-world-pulse/requirements.txt

RUN pip install --no-cache-dir -r services/orion-world-pulse/requirements.txt

COPY orion /app/orion
COPY services/orion-world-pulse /app/services/orion-world-pulse
COPY config/world_pulse /app/config/world_pulse

ENV PYTHONPATH=/app:/app/services/orion-world-pulse

CMD ["python", "services/orion-world-pulse/app/main.py"]
