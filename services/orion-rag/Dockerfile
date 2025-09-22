# Dockerfile.rag (in /mnt/services/rag-service)
FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

COPY rag-service /app

COPY orion-llm-client /deps/orion-llm-client
RUN pip install --no-cache-dir /deps/orion-llm-client

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Environment variables
ENV PORT=8001 \
    BRAIN_URL=http://orion-brain-service:8088 \
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

# Volumes
VOLUME ["/app/data", "/cache/fastembed"]

EXPOSE 8001
CMD ["python","-m","uvicorn","app.server:app","--host","0.0.0.0","--port","8001","--log-level","info"]
