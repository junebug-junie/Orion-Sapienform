FROM python:3.12-slim

WORKDIR /app

# Install build tools only if needed (optional but future-proof)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip tooling
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy only project metadata first for better layer caching
COPY pyproject.toml README.md ./

# Install in editable mode (this layer rebuilds only if metadata changes)
RUN pip install --no-cache-dir -e .

# Copy the rest of the source
COPY . .

CMD ["python", "-c", "import orion_llm_client; print('orion-llm-client ready')"]
