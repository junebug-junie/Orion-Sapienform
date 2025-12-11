# services/orion-security-watcher/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Copy requirements early for layer caching
COPY services/orion-security-watcher/requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy app
COPY services/orion-security-watcher/app ./app
COPY orion ./orion

EXPOSE 7120

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7120"]
