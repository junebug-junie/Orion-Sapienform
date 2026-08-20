# CUDA base image matches orion-vision-host's convention (design doc §3: the
# right analog for a GPU-resident inference service, not llama.cpp). No CUDA
# torch wheels installed in this patch -- there is no model to run yet
# (README.md "Current skeleton status").
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y \
    python3 python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip

# App deps
COPY services/orion-diffusion-host/requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy service code + Orion libs
COPY services/orion-diffusion-host/app ./app
COPY orion ./orion

EXPOSE 6700
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "6700"]
