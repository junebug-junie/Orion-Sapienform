# services/llamacpp-host/Dockerfile

FROM ghcr.io/ggerganov/llama.cpp:server-cuda-b4719

# Install Python + pip (base image is minimal)
# Added curl for healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps for wrapper
COPY services/llamacpp-host/requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Copy wrapper code
COPY services/llamacpp-host/app /app/app
COPY config /app/config

# Run wrapper (which downloads + launches llama-server)
ENTRYPOINT ["python3", "-m", "app.main"]
