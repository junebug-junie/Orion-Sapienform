# services/orion-spark-introspector/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Core utilities (minimal but helpful for debugging in-cluster)
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
    openssh-client \
    build-essential \
    python3-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Copy requirements for this service
# (build context = repo root)
COPY services/orion-spark-introspector/requirements.txt ./requirements.txt

# Install Python deps
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy shared Orion package (schemas, core models, OrionBus, etc.)
COPY orion /app/orion

# Copy this service's code
COPY services/orion-spark-introspector /app

# No HTTP server here; it's a worker, so no EXPOSE needed
CMD ["python", "-m", "app.main"]
