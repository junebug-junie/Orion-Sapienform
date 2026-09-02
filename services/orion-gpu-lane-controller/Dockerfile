# This service controls docker itself (docker-outside-of-docker): it shells
# out to the `docker` CLI, which talks to the HOST's docker daemon over
# /var/run/docker.sock (bind-mounted, see docker-compose.yml). The image
# needs the docker CLI + compose plugin installed, but never runs its own
# nested docker daemon.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    docker.io docker-compose-v2 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY services/orion-gpu-lane-controller/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY services/orion-gpu-lane-controller/app ./app
COPY orion ./orion

EXPOSE 8090
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8090"]
