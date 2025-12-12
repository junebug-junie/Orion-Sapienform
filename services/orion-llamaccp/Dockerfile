# services/orion-llamaccp/Dockerfile
# Orion llama.cpp – GPU-enabled GGUF server with Orion profile wiring

# IMPORTANT: use the *devel* image so we get nvcc / CUDA toolkit
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

WORKDIR /app

# System deps for Python + building llama.cpp
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git \
    build-essential \
    cmake \
    curl \
    libcurl4-openssl-dev \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Make sure CMake knows where CUDA lives
ENV CUDAToolkit_ROOT=/usr/local/cuda

# Build llama.cpp with CUDA support
RUN git clone https://github.com/ggerganov/llama.cpp.git /app/llama.cpp \
 && cd /app/llama.cpp \
 && cmake -B build -DGGML_CUDA=ON -DCUDAToolkit_ROOT=${CUDAToolkit_ROOT} \
 && cmake --build build -j

# Copy service requirements and install
COPY services/orion-llamaccp/requirements.txt ./requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy shared Orion library + config + this service's app code
COPY orion /app/orion
COPY config/llm_profiles.yaml /app/config/llm_profiles.yaml
COPY services/orion-llamaccp/app /app/app

ENV PYTHONPATH="/app:${PYTHONPATH}"

EXPOSE 8000

CMD ["python3", "-m", "app.main"]
