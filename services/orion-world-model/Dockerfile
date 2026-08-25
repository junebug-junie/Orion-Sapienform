FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID

# System deps
RUN apt-get update && apt-get install -y \
    python3 python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip

# Install CUDA torch wheels explicitly (avoid CPU torch) -- same pattern as
# services/orion-vision-host/Dockerfile. This is the only real GPU dependency
# this service needs; there is no diffusers/transformers backbone here (the
# encoder is a small MLP fusion layer, not a new vision/text backbone).
RUN pip3 install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu121

# App deps
COPY services/orion-world-model/requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy service code + Orion libs
COPY services/orion-world-model/app ./app
COPY orion ./orion

EXPOSE 6701
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "6701"]
