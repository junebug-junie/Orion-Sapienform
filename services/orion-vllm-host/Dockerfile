# services/orion-vllm-host/Dockerfile

FROM vllm/vllm-openai:v0.6.0

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# (Optional) if you have extra deps:
COPY services/orion-vllm-host/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/orion-vllm-host/app ./app
COPY config/llm_profiles.yaml ./config/llm_profiles.yaml
# Needed for the bus-native SystemHealthV1 heartbeat (orion.core.bus.bus_service_chassis) --
# see docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
COPY orion ./orion
# The base image vllm/vllm-openai usually has an entrypoint that runs the server.
# Our main.py wraps it.

ENV PYTHONPATH="/app:${PYTHONPATH}"

EXPOSE 8000
ENTRYPOINT ["python3", "-m", "app.main"]
