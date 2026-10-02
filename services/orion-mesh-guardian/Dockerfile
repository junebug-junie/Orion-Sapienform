# services/orion-mesh-guardian/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# docker-cli + docker-compose (v2 plugin) + docker-buildx (tier-2 builds) for remediation. Not
# docker.io: on Debian trixie that package no longer ships the CLI, which left
# remediation unable to run (`docker`: not found) -- found live 2026-10-02.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates docker-cli docker-compose docker-buildx \
    && rm -rf /var/lib/apt/lists/* \
    && docker compose version && docker buildx version

RUN pip3 install --no-cache-dir --upgrade pip

COPY services/orion-mesh-guardian/requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY services/orion-mesh-guardian/app ./app
COPY orion ./orion
COPY config ./config

EXPOSE 7161

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7161"]
