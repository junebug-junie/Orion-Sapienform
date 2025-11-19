FROM python:3.12-slim

WORKDIR /app

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

# Copy requirements early for layer caching
# (explicit path since build context = repo root)
COPY services/orion-cortex-exec/requirements.txt ./requirements.txt

# Install service dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# copy schemas into the container
COPY orion /app/orion

# Copy rest of hub service code
COPY services/orion-cortex-exec /app

EXPOSE 8070

# This CMD is now passed as an argument to the ENTRYPOINT script
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8070"]
