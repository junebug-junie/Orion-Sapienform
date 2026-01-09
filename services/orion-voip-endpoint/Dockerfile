FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps: Python, Asterisk, TFTP, curl, CA roots
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository universe && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip \
        asterisk \
        tftpd-hpa \
        curl \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# VoIP service requirements
COPY services/orion-voip-endpoint/requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Shared Orion package (so we can import orion.core.bus.async_service)
COPY orion /app/orion

# VoIP service code
COPY services/orion-voip-endpoint/app /app/app

EXPOSE 8085

CMD ["python3", "-m", "app.main"]
