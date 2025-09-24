FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends sqlite3 \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip tooling
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first
COPY services/orion-collapse-mirror/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY services/orion-collapse-mirror/app ./app

# Expose consistent port
EXPOSE 8087

# Run via uvicorn (not python -m)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8087", "--reload"]
