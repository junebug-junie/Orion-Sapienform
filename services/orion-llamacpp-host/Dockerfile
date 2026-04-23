# services/orion-llamacpp-host/Dockerfile

# Pinned target for Qwen3 GGUF support + CUDA image stability.
# Qwen docs indicate llama.cpp b5092 as minimum for Qwen3/Qwen3MoE support.
# We pin above that floor (b5401) rather than tracking HEAD.
ARG LLAMACPP_IMAGE_TAG=server-cuda-b8660
FROM ghcr.io/ggml-org/llama.cpp:${LLAMACPP_IMAGE_TAG}

# Install Python + pip (base image is minimal)
# Added curl for healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps for wrapper
COPY services/orion-llamacpp-host/requirements.txt /app/requirements.txt
RUN pip3 install --break-system-packages --no-cache-dir -r /app/requirements.txt

# Copy wrapper code
COPY services/orion-llamacpp-host/app /app/app
COPY config /app/config
COPY orion /app/orion

# Run wrapper (which downloads + launches llama-server)
ENTRYPOINT ["python3", "-m", "app.main"]
