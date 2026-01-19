📦 Orion Vector DB

Responsibility: Provides a persistent, standalone vector database for the Orion mesh.
1. Overview

This service runs the official chromadb/chroma container image. Its sole purpose is to provide a central, persistent server for storing and querying vector embeddings. It is the "database server" component of the vector store architecture.

It is designed to be accessed by other services on the mesh, primarily the orion-vector-writer (for ingestion) and the orion-rag service (for querying).
2. Configuration

Configuration is managed via the .env file in this directory, which defines:

    PORT: The host port to expose the ChromaDB API on.

    CHROMA_DATA_PATH: The path on the host machine where the persistent database files will be stored.

3. How to Run

This service is managed like any other in the Orion mesh.

From the project root (/Orion-Sapienform):

# Start the service
docker compose --env-file .env --env-file services/orion-vector-db/.env -f services/orion-vector-db/docker-compose.yml up -d

# Check its status
docker compose -f services/orion-vector-db/docker-compose.yml ps

# View its logs
docker compose -f services/orion-vector-db/docker-compose.yml logs -f


🧠 Orion Vector Store — Architecture Blueprint

This document outlines the architecture for a dedicated, mesh-wide vector store. This component will serve as the central long-term memory and retrieval backbone for services like RAG, chat history, and semantic search.

This design moves away from embedded databases (like the one previously in orion-rag) to a centralized, service-oriented architecture, mirroring the pattern used by orion-sql-db.
1. Architectural Overview

The architecture consists of two primary services:

    orion-vector-db (The Database Server): A standalone service that runs the official ChromaDB vector database in a container. Its only job is to store and retrieve vectorized data.

    orion-vector-writer (The Ingestion Service): A "headless" worker service that acts as the single, intelligent gateway for writing data to the vector database. It listens for pre-embedded upsert events on the Orion Bus and persists them into the orion-vector-db. Embeddings are produced upstream (for example by the chat memory service or a dedicated embedding host) and passed through unchanged.

This separation of concerns is critical. It means that services like collapse-mirror or orion-hub don't need to know anything about vector databases. They publish normalized upsert events to the bus with embeddings already attached, and the vector-writer handles the storage.
2. The "Schema Enforcer" Pattern

The orion-vector-writer functions as a schema enforcement and storage middleware. It now consumes `memory.vector.upsert.v1` envelopes that already contain embeddings and optional latent references. Upstream services are responsible for obtaining embeddings and populating the standard payload.

Subscription Channels:

The writer listens to pre-embedded upsert events, for example:

    orion:memory:vector:upsert (for all vectorizable memory docs)

Schema Mapping:

Upsert payloads are validated against a shared `VectorDocumentUpsertV1` Pydantic schema, which carries the document text, metadata, embedding, embedding model info, and optional latent references.

This makes the system extensible. To add a new data source, emit a compliant upsert envelope with the embedding already attached and the writer will persist it.

3. Data Flow Example (Collapse Mirror)

    [ orion-hub ] submits a collapse mirror form.

    [ orion-collapse-mirror ] processes it and publishes the full event to orion:collapse:triage.

    [ orion-chat-memory / embedding host ] generates an embedding for the collapse summary and publishes a `memory.vector.upsert.v1` message.

    [ orion-vector-writer ] (listening on orion:memory:vector:upsert):

        Receives the message.

        Validates it against the VectorDocumentUpsertV1 schema.

        Upserts the document, vector, and metadata into the vector store without recomputing embeddings.

4. Next Steps

With this blueprint, we can now proceed with the implementation, which will involve:

    Creating the docker-compose.yml and .env files for a new orion-vector-db service group.

    Creating the full Python application for the orion-vector-writer service, including its Docker configuration and application code.

5. Testing

Find the IP:
    docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' orion-athena-vector-db

This appendix shows a few **copy/paste commands** to inspect what Orion is writing into ChromaDB, directly from your running Docker containers.

It covers:
- Confirming Chroma is reachable
- Listing collections + counts
- Peeking at metadata samples
- Filtering by who requested embeddings (`requester_service`)
- Verifying the semantic embedding pipeline (gateway → vector-host → vector-writer → Chroma)

---

## 0) Quick note about `docker exec` and “not a TTY”

If you see:

> `the input device is not a TTY`

…it means your shell context can’t allocate a terminal. Use **no `-t`**.

- ✅ Good for one-liners:

```bash
docker exec <container> python -c "print('hi')"
```

- ✅ Good for multi-line scripts:

```bash
docker exec -i <container> python - <<'PY'
print('hi')
PY
```

---

## 1) Confirm Chroma is reachable (HTTP heartbeat)

Run from any container on the same Docker network (e.g. `orion-athena-vector-writer`).

```bash
docker exec orion-athena-vector-writer sh -lc \
  'curl -sf http://orion-athena-vector-db:8000/api/v1/heartbeat && echo OK'
```

If this fails:
- Chroma isn’t running, or
- the hostname is wrong, or
- containers aren’t on the same Docker network.

---

## 2) List collections and their counts

```bash
docker exec -i orion-athena-vector-writer python - <<'PY'
import chromadb
from chromadb.config import Settings

client = chromadb.HttpClient(
    host="orion-athena-vector-db",
    port=8000,
    settings=Settings(anonymized_telemetry=False),
)

print("heartbeat:", client.heartbeat())
cols = client.list_collections()

# Chroma versions differ: list_collections may return objects or strings.
names = [(c.name if hasattr(c, "name") else c) for c in cols]
print("collections:", names)

for name in names:
    col = client.get_collection(name)
    print(f"- {name}: {col.count()}")
PY
```

---

## 3) Peek at metadata samples (what’s actually stored)

This is the fastest way to answer: **“what kinds of documents are in here?”**

```bash
docker exec -i orion-athena-vector-writer python - <<'PY'
import chromadb
from chromadb.config import Settings
import pprint

client = chromadb.HttpClient(
    host="orion-athena-vector-db",
    port=8000,
    settings=Settings(anonymized_telemetry=False),
)

col = client.get_collection("orion_main_store")
print("count:", col.count())

sample = col.peek(limit=3)
print("\nmetadatas:")
pprint.pprint(sample.get("metadatas"))
PY
```

Tip: most of your debugging will be done by reading **metadatas**, not the raw text.

---

## 4) Find who is requesting embeddings (requester breakdown)

This tells you whether the pipeline is actually being exercised by the component you think it is.

```bash
docker exec -i orion-athena-vector-writer python - <<'PY'
import chromadb
from chromadb.config import Settings
from collections import Counter

client = chromadb.HttpClient(
    host="orion-athena-vector-db",
    port=8000,
    settings=Settings(anonymized_telemetry=False),
)

col = client.get_collection("orion_main_store")
res = col.get(where={"role": "embedding_request"}, include=["metadatas"], limit=500)

ctr = Counter()
for md in res.get("metadatas", []):
    ctr[md.get("requester_service")] += 1

print("requester_service counts:")
for k, v in ctr.most_common():
    print(f"  {k}: {v}")
PY
```

Expected (based on your recent run):
- `llm-gateway: <n>`
- `orion-chat-memory: <n>`

---

## 5) Verify “gateway assistant-turn → vector-host semantic embed → Chroma”

The contract:
- The gateway publishes the assistant turn as `embedding.generate.v1` on `orion:embedding:generate`.
- Vector-host generates a **semantic** embedding.
- Vector-writer upserts it into Chroma.

This query shows the latest embedding requests attributed to the gateway:

```bash
docker exec -i orion-athena-vector-writer python - <<'PY'
import chromadb
from chromadb.config import Settings

client = chromadb.HttpClient(
    host="orion-athena-vector-db",
    port=8000,
    settings=Settings(anonymized_telemetry=False),
)

col = client.get_collection("orion_main_store")
res = col.get(where={"requester_service": "llm-gateway"}, include=["metadatas"], limit=5)
print("matches:", len(res.get("metadatas", [])))

for md in res.get("metadatas", []):
    print(
        md.get("timestamp"),
        md.get("request_doc_id"),
        md.get("original_channel"),
        md.get("embedding_model"),
        md.get("embedding_dim"),
    )
PY
```

If `matches: 0`, common causes are:
- You filtered by the wrong service name (e.g. `orion-llm-gateway` vs `llm-gateway`).
- The gateway instance you’re hitting has bus disabled.

---

## 6) Browse the collapse mirror collection

If you store mirror entries separately (e.g. `orion_collapse`):

```bash
docker exec -i orion-athena-vector-writer python - <<'PY'
import chromadb
from chromadb.config import Settings
import pprint

client = chromadb.HttpClient(
    host="orion-athena-vector-db",
    port=8000,
    settings=Settings(anonymized_telemetry=False),
)

col = client.get_collection("orion_collapse")
print("count:", col.count())

sample = col.peek(limit=3)
print("\nmetadatas:")
pprint.pprint(sample.get("metadatas"))
PY
```

---

## 7) Prefer Docker DNS names (avoid hardcoded container IPs)

Avoid hardcoding container IPs like `172.18.x.y` unless you’re actively debugging networking.

Prefer:
- `orion-athena-vector-db` (Chroma)
- `orion-athena-vector-writer` (writer)

IPs can change after restarts.

---

## 8) Common troubleshooting

### “Chroma not connected” but scripts show it is
Often the writer tried to connect before Chroma was ready and didn’t retry.

Quick fix:
```bash
docker restart orion-athena-vector-writer
```

### List collections works but `count()` fails
Your Chroma client version might return collection **names** instead of objects.
Use:
```python
names = [(c.name if hasattr(c, "name") else c) for c in client.list_collections()]
col = client.get_collection(name)
```

### You see embeddings from chat-memory but not from gateway
Check requester breakdown (section 4), and confirm your gateway’s SERVICE_NAME.

---

## 9) Naming consistency tip

If you want records to show `requester_service: orion-llm-gateway` instead of `llm-gateway`, set:

```env
SERVICE_NAME=orion-llm-gateway
```

Then restart the gateway container. New records will use the updated name.
