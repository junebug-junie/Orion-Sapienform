FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Upgrade pip early
# Core utilities (minimal)
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

COPY services/orion-bus-tap/requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy shared Orion package (schemas, core models, etc.)
COPY orion /app/orion

# Copy orchestrator service code
COPY services/orion-bus-tap /app
COPY services/orion-bus-tap/static ./static

CMD ["sh", "-c", "uvicorn app.main:app --host ${UI_BIND_HOST:-0.0.0.0} --port ${UI_BIND_PORT:-8101}"]
