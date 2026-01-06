# services/orion-recall/Dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    wget \
    iputils-ping \
    iproute2 \
    dnsutils \
    netcat-traditional \
    procps \
    lsof \
    less \
    vim-tiny \
    nano \
    git \
    build-essential \
    python3-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip

# requirements
COPY services/orion-recall/requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Shared Orion core
COPY orion /app/orion

# Service code
COPY services/orion-recall /app

# Use a neutral expose; runtime port comes from $PORT
EXPOSE 8260

# IMPORTANT: use env var PORT (compose passes it)
CMD ["/bin/sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8260} --log-level info"]
