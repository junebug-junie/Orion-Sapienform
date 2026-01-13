FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements file first to leverage Docker layer caching
COPY services/orion-vector-writer/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the shared Orion library
COPY orion /app/orion

# Copy the application code for this service
COPY services/orion-vector-writer/app ./app

ENV PYTHONPATH="/app:${PYTHONPATH}"

EXPOSE 8301

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8301"]
