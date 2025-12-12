# services/orion-llamaccp/Dockerfile

FROM ghcr.io/ggerganov/llama.cpp:server-cuda-b4719

# Install Python + pip (base image is minimal)
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps for wrapper
COPY services/orion-llamaccp/requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Copy wrapper code
COPY services/orion-llamaccp/app /app/app
COPY config /app/config

# Run wrapper (which downloads + launches llama-server)
ENTRYPOINT ["python3", "-m", "app.main"]
