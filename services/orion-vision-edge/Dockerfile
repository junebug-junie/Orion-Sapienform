# Use the same base as orion-hub for consistency
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

WORKDIR /app

# Install Python + tools
RUN apt-get update && apt-get install -y \
    python3 python3-pip curl \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Copy requirements early for layer caching
COPY services/orion-vision-edge/requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy rest of vision service code
# Copy app
COPY services/orion-vision-edge/app ./app
COPY services/orion-vision-edge/app/haar/haarcascade_frontalface_default.xml /app/haar/
COPY orion ./orion

EXPOSE 7100

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7100"]
