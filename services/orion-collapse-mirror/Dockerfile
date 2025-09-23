FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

CMD ["python", "-m", "app.main"]
FROM python:3.12-slim

WORKDIR /app

# Install build tools in case some deps compile wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip tooling
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first (cache-friendly)
COPY services/orion-collapse-mirror/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY services/orion-collapse-mirror/app ./app

# Expose default port (if it runs a FastAPI server)
EXPOSE 8085

# Run the service
CMD ["python", "-m", "app.main"]
