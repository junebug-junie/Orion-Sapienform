FROM python:3.11-slim

WORKDIR /app

ARG PAGEINDEX_REPO_URL=https://github.com/VectifyAI/PageIndex.git
ARG PAGEINDEX_REF=main

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY services/orion-pageindex/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN git clone --depth 1 --branch ${PAGEINDEX_REF} ${PAGEINDEX_REPO_URL} /opt/PageIndex \
    && sed -i 's/^python-dotenv==1\.2\.2$/python-dotenv==1.0.1/' /opt/PageIndex/requirements.txt \
    && pip install --no-cache-dir -r /opt/PageIndex/requirements.txt

COPY services/orion-pageindex /app
# Real blocker found while wiring the bus-native SystemHealthV1 heartbeat (see
# docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md): this Dockerfile
# never copied the shared `orion` package in, so `orion.core.bus.bus_service_chassis` was not
# importable at runtime despite being importable from a repo-root checkout during dev/tests.
COPY orion /app/orion

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8360", "--log-level", "info"]
