# services/orion-power-guard/Dockerfile
FROM python:3.12-slim

# Python runtime tweaks
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Basic networking + debug utils
# - curl        : HTTP checks (UPS web UI, bus, etc.)
# - iproute2    : `ip`, `ss`
# - iputils-ping: `ping`
# - dnsutils    : `dig`, `nslookup`
# - net-tools   : `netstat`, `ifconfig` (handy when things are weird)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        iproute2 \
        iputils-ping \
        dnsutils \
        snmp \
        curl \
        openssh-client \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Install Python deps
COPY services/orion-power-guard/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code into the image
# (If your `orion` package lives outside this folder and you don't
# mount the repo in compose, you can adjust this to COPY from higher up.)
COPY services/orion-power-guard /app
COPY orion /app/orion

# Default entrypoint — this replaces the `command:` in compose
CMD ["python", "-m", "app.main"]
