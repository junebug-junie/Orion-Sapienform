# Dockerfile
# Orion Brain – GPU-enabled for LLM + GPU TTS
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

WORKDIR /app

# System dependencies for Coqui TTS + audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    ffmpeg \
    libsndfile1 \
    build-essential \
    git \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Install brain requirements first for caching
COPY services/orion-brain/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install GPU TTS (Coqui)
RUN pip3 install --no-cache-dir TTS

# Copy shared Orion library
COPY orion /app/orion

# Copy all brain service code
COPY services/orion-brain /app

ENV PORT=8088
EXPOSE 8088

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8088", "--log-level", "info"]
