# This service controls docker itself (docker-outside-of-docker): it shells
# out to the `docker` CLI, which talks to the HOST's docker daemon over
# /var/run/docker.sock (bind-mounted, see docker-compose.yml). The image
# needs the docker CLI + compose plugin installed, but never runs its own
# nested docker daemon.
FROM python:3.11-slim

# Neither the docker CLI nor a compose plugin come from plain Debian's own
# repos here -- confirmed live 2026-09-02 on this exact base image
# (python:3.11-slim/trixie): apt's own `docker.io` package installs dockerd/
# docker-proxy/docker-init but NOT /usr/bin/docker itself (checked via
# `dpkg -L docker.io` after install -- no docker binary in the file list at
# all), and "docker-compose-plugin"/"docker-compose-v2" both 404 from
# apt-cache. Getting either via Docker's own apt repo would mean adding a
# GPG key + repo just for two binaries. Simpler and just as reliable: fetch
# both directly as static binaries, pinned to the exact versions already
# live on circe's host (`docker --version` -> 29.4.1, `docker compose
# version` -> v5.1.3, both confirmed live) so container and host behave the
# same. We never run dockerd in here at all -- this container only ever
# talks to the HOST's daemon over the bind-mounted socket.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-29.4.1.tgz -o /tmp/docker.tgz \
    && tar -xzf /tmp/docker.tgz -C /tmp docker/docker \
    && mv /tmp/docker/docker /usr/bin/docker \
    && chmod +x /usr/bin/docker \
    && rm -rf /tmp/docker /tmp/docker.tgz \
    && mkdir -p /usr/libexec/docker/cli-plugins \
    && curl -fsSL "https://github.com/docker/compose/releases/download/v5.1.3/docker-compose-linux-x86_64" \
       -o /usr/libexec/docker/cli-plugins/docker-compose \
    && chmod +x /usr/libexec/docker/cli-plugins/docker-compose

WORKDIR /app
COPY services/orion-gpu-lane-controller/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY services/orion-gpu-lane-controller/app ./app
COPY orion ./orion

EXPOSE 8090
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8090"]
