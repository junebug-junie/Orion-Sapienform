# services/orion-llamacpp-neural-host/Dockerfile

FROM python:3.11-slim

# Install build dependencies and curl for healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential cmake git curl && \
    rm -rf /var/lib/apt/lists/*

# Crucial: Enable CUDA support for llama-cpp-python
ENV CMAKE_ARGS="-DGGML_CUDA=on"
ENV FORCE_CMAKE=1
# Ensure python output is buffered for logs
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY services/orion-llamacpp-neural-host/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy shared Orion package
COPY orion /app/orion

# Copy config
COPY config /app/config

# Copy app code
COPY services/orion-llamacpp-neural-host/app /app/app

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005"]
