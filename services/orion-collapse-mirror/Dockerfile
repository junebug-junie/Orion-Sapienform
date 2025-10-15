FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip tooling
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for layer caching
COPY services/orion-collapse-mirror/requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared orion libs (schemas, etc.)
COPY orion /app/orion

# Copy app code
COPY services/orion-collapse-mirror/app ./app

# Expose consistent port
EXPOSE 8087

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8087"]
