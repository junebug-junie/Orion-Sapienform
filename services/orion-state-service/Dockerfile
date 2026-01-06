# services/orion-state-service/Dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    iputils-ping \
    iproute2 \
    dnsutils \
    netcat-traditional \
    procps \
    lsof \
    less \
    vim-tiny \
    nano \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip

# requirements
COPY services/orion-state-service/requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Shared Orion core
COPY orion /app/orion

# Service code
COPY services/orion-state-service /app

EXPOSE 8270

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8270", "--log-level", "info"]
