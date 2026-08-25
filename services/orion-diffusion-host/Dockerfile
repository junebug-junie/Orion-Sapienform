# CUDA base image matches orion-vision-host's convention (design doc §3: the
# right analog for a GPU-resident inference service, not llama.cpp).
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y \
    python3 python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip

# Install CUDA torch wheels explicitly (avoid CPU torch) -- same pattern and
# index as services/orion-vision-host/Dockerfile.
RUN pip3 install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

# App deps
COPY services/orion-diffusion-host/requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy service code + Orion libs
COPY services/orion-diffusion-host/app ./app
COPY orion ./orion

EXPOSE 6700
# --workers 1 pinned explicitly (uvicorn's own default without this flag is
# also 1, but review finding: a future "improve throughput" edit adding
# --workers N here would give each worker process its own model copy on
# the same physical GPU with independently-locked, uncoordinated access --
# a silent multi-process OOM/thrash regression on a service with exactly
# one dedicated GPU. Explicit, so that edit has to remove this on purpose.
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "6700", "--workers", "1"]
