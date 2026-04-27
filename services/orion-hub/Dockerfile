FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ARG INSTALL_DEV_DEPS=false

WORKDIR /app

# Install Python + tools + network utilities + Docker CLI/Compose plugin
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    python3 \
    python3-pip \
    build-essential \
    python3-dev \
    pkg-config \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    redis-tools \
    netcat-traditional \
    iputils-ping \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc \
    && chmod a+r /etc/apt/keyrings/docker.asc \
    && printf "Types: deb\nURIs: https://download.docker.com/linux/ubuntu\nSuites: %s\nComponents: stable\nArchitectures: %s\nSigned-By: /etc/apt/keyrings/docker.asc\n" \
         "$(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")" \
         "$(dpkg --print-architecture)" \
         > /etc/apt/sources.list.d/docker.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
         docker-ce-cli \
         docker-buildx-plugin \
         docker-compose-plugin \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Copy requirements early for layer caching
COPY services/orion-hub/requirements.txt ./requirements.txt
COPY requirements-dev.txt /tmp/requirements-dev.txt

# Install service dependencies
RUN pip3 install --no-cache-dir -r requirements.txt
RUN if [ "${INSTALL_DEV_DEPS}" = "true" ]; then pip3 install --no-cache-dir -r /tmp/requirements-dev.txt; fi

# copy schemas into the container
COPY orion /app/orion

# Copy rest of hub service code
COPY services/orion-hub /app

# Copy the new entrypoint script into a standard location
COPY services/orion-hub/entrypoint.sh /usr/local/bin/entrypoint.sh

# Make the script executable
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set the entrypoint script to be executed when the container starts
ENTRYPOINT ["entrypoint.sh"]

EXPOSE 8080

CMD ["uvicorn", "scripts.main:app", "--host", "0.0.0.0", "--port", "8080"]
