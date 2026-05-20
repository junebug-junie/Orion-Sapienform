FROM python:3.11-slim

WORKDIR /app

COPY services/orion-knowledge-forge/requirements.txt /app/services/orion-knowledge-forge/requirements.txt

RUN pip install --no-cache-dir -r services/orion-knowledge-forge/requirements.txt

COPY orion /app/orion
COPY services/orion-knowledge-forge /app/services/orion-knowledge-forge

ENV PYTHONPATH=/app:/app/services/orion-knowledge-forge

CMD ["python", "services/orion-knowledge-forge/app/main.py"]
