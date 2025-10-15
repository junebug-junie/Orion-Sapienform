FROM python:3.12-slim

WORKDIR /app

# Install system dependencies that might be needed for Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements file first to leverage Docker layer caching
COPY services/orion-rag/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the shared Orion library into the container
COPY orion /app/orion

# Copy the RAG service's specific application code
COPY services/orion-rag/app ./app

# Expose the port the service will run on (value is passed via env)
EXPOSE 8300

# The command to run the FastAPI application
# It will be delayed by the 'command' directive in docker-compose.yml
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8300"]

