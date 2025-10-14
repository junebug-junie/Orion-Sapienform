FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Dependencies
COPY services/orion-gdb-client/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY services/orion-gdb-client/app ./app

# Shared Orion schemas
COPY orion /app/orion

# Env setup
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Service port — must match .env PORT
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
