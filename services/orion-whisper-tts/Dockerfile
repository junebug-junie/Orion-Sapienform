# services/orion-whisper-tts/Dockerfile

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

WORKDIR /app

# System dependencies for Coqui TTS + audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    ffmpeg \
    libsndfile1 \
    espeak-ng \
    espeak-ng-data \
    build-essential \
    git \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Install service dependencies
COPY services/orion-whisper-tts/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy shared Orion library
COPY orion /app/orion

# Copy this service's code
COPY services/orion-whisper-tts /app

EXPOSE 7800

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7800", "--log-level", "info"]
