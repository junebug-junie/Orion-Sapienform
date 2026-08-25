FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Without this, CUDA_VISIBLE_DEVICES=<N> does NOT reliably select nvidia-smi's
# GPU N: the CUDA runtime's own device enumeration defaults to
# FASTEST_FIRST, which can (and, live-verified on a 7-GPU host 2026-08-25,
# does) differ from nvidia-smi's PCI-bus-order index. CUDA_VISIBLE_DEVICES=4
# on that host silently landed torch on a V100-16GB, not nvidia-smi's index-4
# P100 -- caught by comparing `torch.cuda.get_device_name(0)` against
# `nvidia-smi`'s listing for the same physical machine, not by trusting the
# env var took effect. `orion-vllm-host` and `orion-llamacpp-host` already
# set this defensively in their own Python launch code
# (`env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")`); orion-vision-host
# had no equivalent, so this was silently dormant on any single-GPU host
# (nvidia-smi order is trivially correct with only one card) and only
# surfaces on a multi-GPU deploy. Baked into the image as a correctness
# requirement, not a preference -- CUDA_VISIBLE_DEVICES is meaningless
# without it. This alone is NOT the only guard: docker-compose `environment:`
# entries take precedence over a Dockerfile ENV default for the same key, so
# services/orion-vision-host/docker-compose.yml also pins this literally
# (not ${CUDA_DEVICE_ORDER}-interpolated) so a future .env adding this key
# can't silently override it either. Review that file's comment on the same
# line before assuming this ENV alone is unconditionally load-bearing.
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

# System deps
RUN apt-get update && apt-get install -y \
    python3 python3-pip curl \
    libgl1 libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip

# Install CUDA torch wheels explicitly (avoid CPU torch)
RUN pip3 install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

# App deps
COPY services/orion-vision-host/requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy service code + Orion libs
COPY services/orion-vision-host/app ./app
COPY orion ./orion

# Copy ROOT config into image (no volume mount)
COPY config/vision_profiles.yaml /app/config/vision_profiles.yaml

EXPOSE 6600
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "6600"]

