# services/orion-llm-gateway/Dockerfile

FROM python:3.12-slim

WORKDIR /app

# Core utilities (debug-friendly)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    wget \
    jq \
    iputils-ping \
    iproute2 \
    dnsutils \
    netcat-traditional \
    redis-tools \
    procps \
    lsof \
    strace \
    tree \
    less \
    vim-tiny \
    nano \
    git \
    build-essential \
    python3-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Copy requirements early for layer caching
COPY services/orion-llm-gateway/requirements.txt ./requirements.txt

# Install service dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy shared Orion library (schemas/utilities)
COPY orion /app/orion

# Copy this service code
COPY services/orion-llm-gateway /app

EXPOSE 8210  # reserved for future HTTP/health if you want it

CMD ["python", "-m", "app.main"]
