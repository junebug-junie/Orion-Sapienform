# services/orion-llamacpp-neural-host/Dockerfile

FROM python:3.11-slim

# Install build dependencies and curl for healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential cmake git curl && \
    rm -rf /var/lib/apt/lists/*

# Crucial: Enable CUDA support for llama-cpp-python
ENV CMAKE_ARGS="-DGGML_CUDA=on"
ENV FORCE_CMAKE=1

WORKDIR /app

# Install dependencies
# Using 0.2.79 as requested
RUN pip install --no-cache-dir llama-cpp-python>=0.2.79 fastapi uvicorn sse-starlette pydantic-settings

# Copy app code
COPY services/orion-llamacpp-neural-host/app /app/app
# We will likely reuse the config structure or create a new one.
# For now, let's assume we copy config from root like other services if needed,
# but the task says "Copy app/ and config/".
# Assuming config is at repo root `config/` or `services/orion-llamacpp-host/config`?
# The task says "Copy app/ and config/".
# `orion-llamacpp-host` copies `config` from repo root (context: ../..).
COPY config /app/config

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005"]
