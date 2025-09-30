# Use the same base as orion-hub for consistency
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

WORKDIR /app

# Install Python + tools
# System deps
RUN apt-get update && apt-get install -y \
    python3 python3-pip curl \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Copy requirements early for layer caching
COPY services/orion-vision/requirements.txt ./requirements.txt

# Install service dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy bus client (relative path inside monorepo)
COPY services/orion-bus/client /tmp/orion-bus-client
RUN pip3 install /tmp/orion-bus-client

# Copy rest of vision service code
# Copy app
COPY services/orion-vision/app ./app
COPY services/orion-vision/app/haar/haarcascade_frontalface_default.xml /app/haar/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
