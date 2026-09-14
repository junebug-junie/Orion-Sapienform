# services/orion-curiosity-peer/Dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Thin bus worker: no Claude CLI, no git, no docker client, no cursor-sdk.
# Spawns host-mounted Cursor Agent CLI (compose: /opt/cursor-agent + desktop
# auth under /root/.config/cursor). Repo mount (read-only) at compose time.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/orion-curiosity-peer/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY services/orion-curiosity-peer/app /app/app
COPY orion /app/orion

CMD ["python", "-m", "app.main"]
