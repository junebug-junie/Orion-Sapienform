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

EXPOSE 8260

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8260", "--log-level", "info"]
