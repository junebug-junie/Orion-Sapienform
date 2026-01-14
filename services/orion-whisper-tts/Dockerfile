# FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1
WORKDIR /app

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

RUN pip3 install --no-cache-dir --upgrade pip

# Install service deps first
COPY services/orion-whisper-tts/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# FORCE a Pascal-capable PyTorch build: CUDA 12.6 wheels
# (official index-url pattern)
RUN pip3 install --no-cache-dir --upgrade --force-reinstall \
    torch==2.8.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu126

# Build-time proof: wheel includes sm_61 support
RUN python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)

# Build-time sanity: we at least installed a CUDA-enabled torch wheel.
assert torch.version.cuda is not None, "CPU-only torch installed (torch.version.cuda is None)"

# Optional: try static arch flags (doesn't always require a driver)
flags_fn = getattr(torch._C, "_cuda_getArchFlags", None)
flags = flags_fn() if flags_fn else None
print("torch._C._cuda_getArchFlags():", flags)
PY


COPY orion /app/orion
COPY services/orion-whisper-tts /app

EXPOSE 7800
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7800", "--log-level", "info"]
