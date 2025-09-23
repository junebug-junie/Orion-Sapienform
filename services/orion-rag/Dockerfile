# Orion RAG Service
FROM python:3.12-slim

WORKDIR /app

# Install build deps (needed by some ML libs, e.g. llama-cpp-python)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip tooling
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for better layer caching
COPY services/orion-rag/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy in LLM client (relative to monorepo root)
COPY services/orion-llm-client /deps/orion-llm-client
RUN pip install --no-cache-dir /deps/orion-llm-client

# Copy the RAG service code
COPY services/orion-rag /app

# Environment variables
ENV PORT=8001 \
    BRAIN_URL=http://brain-service:8088 \
    MODEL=mistral:instruct \
    QDRANT_URL=http://qdrant:6333 \
    RAG_EMBED_MODEL=BAAI/bge-small-en-v1.5 \
    FASTEMBED_CACHE_PATH=/cache/fastembed \
    REDIS_URL=redis://orion-redis:6379/0 \
    SERVICE_NAME=orion-rag \
    EVENTS_ENABLE=true \
    EVENTS_STREAM=orion:evt:rag \
    BUS_OUT_ENABLE=true \
    BUS_OUT_STREAM=orion:bus:out

# Volumes for persisted data + cache
VOLUME ["/app/data", "/cache/fastembed"]

EXPOSE 8001

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8001", "--log-level", "info"]
