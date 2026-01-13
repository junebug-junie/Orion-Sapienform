# services/orion-llamacpp-neural-host/Dockerfile

# [FIX] Use NVIDIA CUDA Devel image to provide 'nvcc' for compilation
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install Python 3.11 and build tools
# We need software-properties-common to add the deadsnakes PPA for Python 3.11
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    cmake \
    git \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 2. Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --set python3 /usr/bin/python3.11 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# 3. Upgrade pip (Critical for newer wheels)
RUN python3 -m pip install --upgrade pip

# 4. Set CMAKE args for GPU build
# This tells llama-cpp-python to compile with CUDA support
ENV CMAKE_ARGS="-DGGML_CUDA=on"
ENV FORCE_CMAKE=1

# Configure Linker for CUDA Stubs
# The build process needs 'libcuda.so.1' (Driver API), but the driver isn't loaded during 'docker build'.
# We link against the stubs provided by the NVIDIA image and create a symlink for the versioned .so.1 file.
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/cuda-stubs.conf \
    && ldconfig

WORKDIR /app

# 5. Install dependencies
# (We assume requirements.txt contains 'llama-cpp-python')
COPY services/orion-llamacpp-neural-host/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 6. Copy shared code
COPY orion/ /app/orion/

# 7. Copy Service Code
COPY services/orion-llamacpp-neural-host/app /app/app
COPY config /app/config

# 8. Set PYTHONPATH
ENV PYTHONPATH=/app

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005"]
