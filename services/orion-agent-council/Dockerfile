FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    iputils-ping \
    dnsutils \
    netcat-traditional \
    procps \
    lsof \
    tree \
    less \
    vim-tiny \
    nano \
    git \
    build-essential \
    python3-dev \
    pkg-config \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip

COPY services/orion-agent-council/requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Shared Orion lib
COPY orion /app/orion

# Service code
COPY services/orion-agent-council /app

CMD ["python", "-m", "app.main"]
