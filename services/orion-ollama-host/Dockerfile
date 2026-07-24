# Dockerfile
# Orion Ollama Host - Wrapper for Ollama
# We use the official Ollama image as base if possible, but the requirement is to run a "host" service.
# However, `orion-brain` previously had TWO containers: `brain` (Python) and `brain-llm` (Ollama).
# The user wants "orion-ollama-host" to be "Gut it down to just a host... similar to its counterparts from above".
# `orion-llamacpp-host` wraps `llama-server`. `vllm-host` wraps `vllm`.
# `ollama` is a standalone Go binary.
# If we want a Python wrapper that manages profiles (like llamacpp/vllm), we can keep the Python part.
# But Ollama manages its own models via `ollama pull`.
#
# Let's assume the user wants a single service that runs Ollama, maybe with a thin Python wrapper if needed for "bus" or "profiles",
# BUT the user said "Gut it down to just a host... should be a dumb hub and not do business logic".
# And "ollama will have only the job of being a host for ollama".
#
# If I look at `orion-llamacpp-host`, it downloads models based on profile.
# If `ollama-host` should do the same, it needs a Python wrapper to run `ollama pull`.
#
# So I will create a Python wrapper (using the existing one but gutted) that runs `ollama serve` and manages pulls.

FROM ollama/ollama:latest

# Install Python + pip to run our wrapper
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps for wrapper
COPY services/orion-ollama-host/requirements.txt /app/requirements.txt
# --break-system-packages: this base image's Ubuntu 24.04 layer enforces PEP 668
# (externally-managed-environment) -- confirmed live build failure without this flag.
# Same fix already applied in services/orion-llamacpp-host/Dockerfile for the same reason.
RUN pip3 install --break-system-packages --no-cache-dir -r /app/requirements.txt

# Copy wrapper code
COPY services/orion-ollama-host/app /app/app
COPY config /app/config
# Needed for the bus-native SystemHealthV1 heartbeat (orion.core.bus.bus_service_chassis) --
# see docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
COPY orion /app/orion

ENV HOST=0.0.0.0
ENV PORT=11434

# We will use our python wrapper as entrypoint
ENTRYPOINT ["python3", "-m", "app.main"]
