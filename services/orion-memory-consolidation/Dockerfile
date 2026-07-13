FROM python:3.12-slim

WORKDIR /app

# chroma-hnswlib (transitive dep of chromadb) builds a native C extension -- needs a
# compiler toolchain, matching the pattern already used by orion-dream/orion-rag/
# orion-vector-writer's Dockerfiles for the same dependency.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

COPY services/orion-memory-consolidation/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY orion /app/orion
COPY services/orion-memory-consolidation/app /app/app

ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
