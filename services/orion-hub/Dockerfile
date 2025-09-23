# Use the official NVIDIA CUDA image as the single base for everything.
# This ensures all paths and dependencies are consistent.
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

WORKDIR /app

# Install Python + tools
RUN apt-get update && apt-get install -y \
    python3 python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Copy requirements early for layer caching
COPY requirements.txt .

# Install service dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy bus client (relative path inside monorepo)
# Assumes build context = Orion-Sapienform root
COPY services/orion-bus/client /tmp/orion-bus-client
RUN pip3 install /tmp/orion-bus-client

# Copy rest of hub service code
COPY services/orion-hub /app

EXPOSE 8080

CMD ["uvicorn", "scripts.main:app", "--host", "0.0.0.0", "--port", "8080"]
