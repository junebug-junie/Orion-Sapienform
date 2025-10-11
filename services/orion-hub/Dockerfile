FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

WORKDIR /app

# Install Python + tools
RUN apt-get update && apt-get install -y \
    python3 python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Copy requirements early for layer caching
# (explicit path since build context = repo root)
COPY services/orion-hub/requirements.txt ./requirements.txt

# Install service dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# copy schemas into the container
COPY orion /app/orion

# add orion/ to PYTHONPATH
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Copy rest of hub service code
COPY services/orion-hub /app

# === MODIFICATIONS START ===

# Copy the new entrypoint script into a standard location
COPY services/orion-hub/entrypoint.sh /usr/local/bin/entrypoint.sh

# Make the script executable
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set the entrypoint script to be executed when the container starts
ENTRYPOINT ["entrypoint.sh"]

# === MODIFICATIONS END ===

EXPOSE 8080

# This CMD is now passed as an argument to the ENTRYPOINT script
CMD ["uvicorn", "scripts.main:app", "--host", "0.0.0.0", "--port", "8080"]
