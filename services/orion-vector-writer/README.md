# 🧠 Orion Vector Writer Service

The **Orion Vector Writer** is a microservice responsible for ingesting structured events (Collapse Mirrors, chat logs, and RAG documents) and storing them as vector embeddings in **ChromaDB** for long-term semantic retrieval.

---

## 🚀 Overview

This service subscribes to the **Orion Redis Bus**, listens for specific event channels, converts incoming payloads into embeddings using a SentenceTransformer model, and stores them into a Chroma collection for later retrieval and analysis.

### 🔄 Message Flow
1. **Redis Bus → Vector Writer**  
   Listens on event channels like:
   - `orion:collapse:triage`
   - `orion:chat:history:log`
   - `orion:rag:document:add`

2. **Vector Writer → ChromaDB**  
   Incoming messages are validated, flattened, embedded, and upserted into the collection defined by `.env`.

3. **ChromaDB → Retrieval**  
   Other Orion services (e.g., RAG, reflection, or analysis nodes) query Chroma to recall relevant context or historical data.

---

## ⚙️ Configuration

All runtime settings are controlled via the `.env` file.

### Example `.env`
```bash
# --- Service Identity ---
SERVICE_NAME=vector-writer
SERVICE_VERSION=1.0.0
PORT=8301

# --- Orion Bus ---
ORION_BUS_ENABLED=true
ORION_BUS_URL=redis://${PROJECT}-bus-core:6379/0

# --- Subscription Channels ---
SUBSCRIBE_CHANNEL_COLLAPSE=orion:collapse:triage
SUBSCRIBE_CHANNEL_CHAT=orion:chat:history:log
SUBSCRIBE_CHANNEL_RAG_DOC=orion:rag:document:add

# --- Publish Channel ---
PUBLISH_CHANNEL_VECTOR_CONFIRM=orion:vector:confirm

# --- Vector Store ---
VECTOR_DB_HOST=${PROJECT}-vector-db
VECTOR_DB_PORT=8000
VECTOR_DB_COLLECTION=orion_main_store
VECTOR_DB_CREATE_IF_MISSING=true
VECTOR_DB_RETRY_ATTEMPTS=10
VECTOR_DB_RETRY_DELAY=5
EMBEDDING_MODEL=all-MiniLM-L6-v2

# --- Runtime ---
LOG_LEVEL=INFO
BATCH_SIZE=1
```

---

## 🧩 Docker Compose

```yaml
services:
  vector-writer:
    build:
      context: ../..
      dockerfile: services/orion-vector-writer/Dockerfile
    container_name: ${PROJECT}-vector-writer
    restart: unless-stopped
    networks:
      - app-net
    ports:
      - "${PORT}:${PORT}"
    environment:
      - SERVICE_NAME=${SERVICE_NAME}
      - SERVICE_VERSION=${SERVICE_VERSION}
      - PORT=${PORT}
      - LOG_LEVEL=${LOG_LEVEL}
      - BATCH_SIZE=${BATCH_SIZE}
      - ORION_BUS_ENABLED=${ORION_BUS_ENABLED}
      - ORION_BUS_URL=${ORION_BUS_URL}
      - SUBSCRIBE_CHANNEL_COLLAPSE=${SUBSCRIBE_CHANNEL_COLLAPSE}
      - SUBSCRIBE_CHANNEL_CHAT=${SUBSCRIBE_CHANNEL_CHAT}
      - SUBSCRIBE_CHANNEL_RAG_DOC=${SUBSCRIBE_CHANNEL_RAG_DOC}
      - PUBLISH_CHANNEL_VECTOR_CONFIRM=${PUBLISH_CHANNEL_VECTOR_CONFIRM}
      - VECTOR_DB_HOST=${VECTOR_DB_HOST}
      - VECTOR_DB_PORT=${VECTOR_DB_PORT}
      - VECTOR_DB_COLLECTION=${VECTOR_DB_COLLECTION}
      - EMBEDDING_MODEL=${EMBEDDING_MODEL}

networks:
  app-net:
    name: ${NET}
    external: true
```

---

## 🧬 Models

### CollapseTriageEvent
- Represents events from `orion:collapse:triage`.
- Fields: `summary`, `observer`, `trigger`, etc.

### ChatMessageEvent
- Represents chat messages from `orion:chat:history:log`.
- Fields: `user`, `content`, `timestamp`.

### RAGDocumentEvent
- Represents documents explicitly added for retrieval.
- Fields: `text`, `metadata`.

Each model includes `.to_document()` for standardizing the structure before insertion into Chroma.

---

## 🔁 Worker Threads

Two threads start on service boot:
- **Listener Worker** — Subscribes to Redis bus channels and queues validated messages.
- **Batch Upsert Worker** — Periodically processes queued messages and upserts them into Chroma.

Example log flow:
```
👂 Subscribing to channels: ['orion:collapse:triage', ...]
📥 Queued document 12345 from channel orion:collapse:triage
⚙️ Batch upsert worker started. Batch size: 1
🧠 Connected to Chroma → collection 'orion_main_store'
✅ Upserted 1 documents into Chroma collection 'orion_main_store'
```

---

## 🔍 Exploring the Collection

You can explore Chroma data from your host:
```bash
curl http://localhost:8500/api/v1/collections
```
Or from Python:
```python
import chromadb
client = chromadb.HttpClient(host="localhost", port=8500)
coll = client.get_collection("orion_main_store")
print(coll.count())
print(coll.peek())
```

From container:
```docker exec -it orion-janus-vector-writer python /app/app/collection_browse.py```

```docker exec -it orion-janus-vector-writer python /app/app/query_similiarity.py```
---

## 🧭 Debugging & Tips

- Check logs:
  ```bash
  docker logs -f orion-janus-vector-writer | grep -E "✅|⚙️|📥"
  ```
- Ensure both containers (`vector-writer` and `vector-db`) share the same Docker network (`app-net`).
- Use `/health` endpoint for quick checks:
  ```bash
  curl http://localhost:8301/health
  ```

---

## 📚 Future Enhancements
- Web-based Chroma Explorer UI
- Batch metrics & latency reporting
- Smarter retry / exponential backoff
- Optional persistence into PostgreSQL for hybrid search

---

**Author:** Orion Mesh (Janus Node)  
**Maintainer:** Juniper Feld  
**Version:** 1.0.0
