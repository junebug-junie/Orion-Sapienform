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

# CUDA_DEVICE_ORDER=PCI_BUS_ID -- confirmed live on this exact host
# (Circe, 2026-08-25) as a real, not theoretical, bug: CUDA's default
# device enumeration is "fastest first," not PCI-bus-id order, so on a
# heterogeneous multi-GPU host CUDA_VISIBLE_DEVICES=2 can silently select a
# *different* physical card than nvidia-smi's index 2 -- torch enumerates
# and re-numbers only the devices CUDA_VISIBLE_DEVICES exposes, using its
# own (non-PCI-bus) order. Reproduced on this deploy: CUDA_VISIBLE_DEVICES=2
# (intended: the empty Tesla PG500-216, physical index 2) instead loaded
# the model onto physical index 3, a busy Tesla V100-32GB already serving a
# llama.cpp worker -- while /health still reported success and gave no
# indication anything was wrong. Same root cause and same fix as
# services/orion-world-model/Dockerfile (found there first, same day) and
# the equivalent env.setdefault() every spawned subprocess in
# orion-llamacpp-host/orion-vllm-host already applies. See README.md
# "GPU 2 has had two collisions" section for the verification command to
# run after every deploy -- /health reporting success is not sufficient
# evidence the model landed on the intended card. Placed here (last ENV
# before CMD), not in the top ENV block, deliberately -- it doesn't affect
# anything the build steps above do, and putting it early would bust the
# Docker layer cache for the expensive torch install on every future
# no-op-for-this-value rebuild.
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

EXPOSE 6700
# --workers 1 pinned explicitly (uvicorn's own default without this flag is
# also 1, but review finding: a future "improve throughput" edit adding
# --workers N here would give each worker process its own model copy on
# the same physical GPU with independently-locked, uncoordinated access --
# a silent multi-process OOM/thrash regression on a service with exactly
# one dedicated GPU. Explicit, so that edit has to remove this on purpose.
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "6700", "--workers", "1"]
