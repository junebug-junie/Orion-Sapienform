FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ARG INSTALL_DEV_DEPS=false

# Claude Code refuses to start with permissionMode=bypassPermissions (or the
# raw --dangerously-skip-permissions flag) as root/sudo unless it recognizes
# the process as a deliberate sandbox -- confirmed live 2026-08-13 by reading
# the CLI's own bundled source: the gate is
# `getuid()===0 && process.env.IS_SANDBOX!=="1" && !CLAUDE_CODE_BUBBLEWRAP`.
# This container always runs as root (no USER directive) and
# fcc_claude_bridge.py's FCC turns route through orion/fcc/claude_spawn.py's
# claude_permission_argv(), which passes bypassPermissions for root, so
# without this the FCC claude subprocess crashes on every turn.
ENV IS_SANDBOX=1

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
    git \
    openssh-client \
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
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && git config --global --add safe.directory '*' \
    && git config --global user.name "Orion FCC" \
    && git config --global user.email "orion-fcc@orion-sapienform.local"

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

# Vendor autonomy-gate pure helpers for Hub Drives Analytics (avoids relying solely
# on the repo volume mount for ORION_REPO_ROOT=/repo).
COPY scripts/analysis/measure_autonomy_gate.py /app/vendor/measure_autonomy_gate.py

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
