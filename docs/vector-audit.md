# Vector Audit — Orion (Athena)

## Executive Summary
- Recall’s vector backend can return zero results because it **only queries collections listed in `RECALL_VECTOR_COLLECTIONS`** and returns `[]` when that setting is empty (no fallback).【F:services/orion-recall/app/storage/vector_adapter.py†L64-L97】【F:services/orion-recall/app/settings.py†L135-L139】
- Vector ingestion is split between **vector-host (embedding + semantic upsert)** and **vector-writer (Chroma ingestion)**. Semantic embeddings are produced by `orion-vector-host` and written by `orion-vector-writer`.【F:services/orion-vector-host/app/main.py†L39-L197】【F:services/orion-vector-writer/app/main.py†L164-L279】
- A shared, stable schema exists for vector upserts (`VectorDocumentUpsertV1`), but it is **not registered in the schema registry yet** (fixed in the patch set).【F:orion/schemas/vector/schemas.py†L23-L67】【F:orion/schemas/registry.py†L19-L121】

---

## Inventory — Services, Modules, Channels, and Flows

### Embedding + Vector Producers
**orion-vector-host**
- Subscribes to:
  - `orion:chat:history:log` (`chat.history.message.v1`) for chat history embeddings.
  - `orion:embedding:generate` (`embedding.generate.v1`) for embedding requests.
- Emits semantic `VectorUpsertV1` on `orion:vector:semantic:upsert` with metadata including `session_id`, `role`, `original_channel`, `correlation_id`, and `requester_service`.【F:services/orion-vector-host/app/main.py†L39-L197】【F:services/orion-vector-host/app/main.py†L248-L306】

**orion-llm-gateway**
- Publishes `embedding.generate.v1` requests on `orion:embedding:generate`.
- Emits **latent** `VectorUpsertV1` on `orion:vector:latent:upsert` when an LLM backend returns `spark_vector`.【F:services/orion-llm-gateway/app/embed_publish.py†L23-L50】【F:services/orion-llm-gateway/app/main.py†L62-L109】

### Vector Writer (Ingestion)
**orion-vector-writer**
- Subscribes to semantic + latent upserts and `memory.vector.upsert.v1`.
- Writes to Chroma via `VectorUpsertV1` and `VectorDocumentUpsertV1` (pre-embedded docs).【F:services/orion-vector-writer/app/main.py†L164-L279】
- Chat history normalization (from `chat.history.message.v1`) is supported and writes to `orion_chat` by default.【F:services/orion-vector-writer/app/chat_history.py†L18-L58】

### Recall (Query)
**orion-recall**
- Uses `chromadb.HttpClient` and queries only collections in `RECALL_VECTOR_COLLECTIONS`.
- Filters results by time window based on metadata timestamps.
- Does **not** use metadata filters by default (session filters are added in the patch set).【F:services/orion-recall/app/storage/vector_adapter.py†L49-L153】

### Canonical Vector Ingestion API
- **Semantic upsert**: `orion:vector:semantic:upsert` (`VectorUpsertV1`), produced by vector-host.  
- **Latent upsert**: `orion:vector:latent:upsert` (`VectorUpsertV1`), produced by llm-gateway.  
- **Document upsert**: `orion:memory:vector:upsert` (`VectorDocumentUpsertV1`) — now registered in bus channels + schema registry in the patch set.【F:orion/bus/channels.yaml†L616-L644】【F:orion/schemas/vector/schemas.py†L23-L90】【F:orion/schemas/registry.py†L19-L121】

---

## Schemas & Registry Status

### Schemas Defined
- `VectorWriteRequest`
- `VectorDocumentUpsertV1`
- `VectorUpsertV1`
- `EmbeddingGenerateV1`, `EmbeddingResultV1`【F:orion/schemas/vector/schemas.py†L1-L90】

### Registry
Before patch:
- `VectorWriteRequest`, `VectorUpsertV1`, `EmbeddingGenerateV1`, `EmbeddingResultV1` registered.
- `VectorDocumentUpsertV1` **missing** (fixed in patch).【F:orion/schemas/registry.py†L19-L121】

---

## Recall Query Behavior (Current)
- **Collections**: Uses `RECALL_VECTOR_COLLECTIONS`; if empty → returns `[]`.【F:services/orion-recall/app/storage/vector_adapter.py†L64-L97】
- **Time window**: Drops results without parsable timestamps (`timestamp`, `created_at`, etc.).【F:services/orion-recall/app/storage/vector_adapter.py†L16-L47】
- **Filters**: No session or role filtering in current code; patch adds `session_id` and optional metadata filters.【F:services/orion-recall/app/storage/vector_adapter.py†L49-L153】

### Root Cause for “Vector = 0”
If `RECALL_VECTOR_COLLECTIONS` is unset or blank, the vector adapter returns early, producing zero results even if Chroma has data.【F:services/orion-recall/app/storage/vector_adapter.py†L64-L97】【F:services/orion-recall/app/settings.py†L135-L139】

---

## Chroma Inventory & Counts (Runtime Audit)

### Attempted Query (from repo environment)
Chroma access from this environment failed with a **403 Forbidden** tenant error, so counts could not be retrieved here.

**Recommended command (run inside vector-writer container)**:
```bash
docker exec -i orion-athena-vector-writer python - <<'PY'
import chromadb
from chromadb.config import Settings as ChromaSettings

client = chromadb.HttpClient(host="orion-athena-vector-db", port=8000, settings=ChromaSettings(anonymized_telemetry=False))
collections = client.list_collections()
print("collections", [c.name for c in collections])
for col in collections:
    c = client.get_or_create_collection(name=col.name)
    print(col.name, "count", c.count())

main = client.get_or_create_collection("orion_main_store")
def count(where):
    res = main.get(where=where, include=["metadatas"], limit=1000)
    return len(res.get("ids", []))

print("chat_history channel", count({"original_channel": "orion:chat:history:log"}))
print("session_id present", count({"session_id": {"$ne": ""}}))
print("role user", count({"role": "user"}))
print("role assistant", count({"role": "assistant"}))
print("role embedding_request", count({"role": "embedding_request"}))
PY
```

---

## Proposed Stable Metadata Schema (v1) for Chat Turns

**doc_type**: `"chat_turn"`  
**schema_version**: `"v1"`

### Required / Strongly Recommended Keys
- `doc_id` (stable ID)
- `correlation_id` (trace ID)
- `session_id`
- `source_service`
- `node` (or `source_node`)
- `role` (`user` / `assistant`)
- `channel`
- `created_at`
- `envelope_id`
- `embedding_model`
- `embedding_dim`

**Mapping** is additive and based on existing envelope + chat payload fields:
`BaseEnvelope.correlation_id`, `BaseEnvelope.source`, `BaseEnvelope.created_at`, and `ChatHistoryMessageV1` payload metadata (`session_id`, `role`).【F:services/orion-vector-host/app/main.py†L39-L197】【F:orion/schemas/chat_history.py†L37-L95】

---

## Backwards-Compatible Migration Plan
1) **Recall defaults**: if `RECALL_VECTOR_COLLECTIONS` is empty, fall back to `VECTOR_DB_COLLECTION`. (Patch included.)【F:services/orion-recall/app/settings.py†L126-L191】
2) **Metadata filters**: add optional filters in vector queries:
   - `session_id` (primary filter)
   - `source_node` (if node_id present)
   - optional `vector_meta_filters` (profile-driven).【F:services/orion-recall/app/storage/vector_adapter.py†L49-L153】【F:services/orion-recall/app/worker.py†L256-L475】
3) **Keep existing flows**: vector-host + vector-writer continue to handle embeddings and upserts without breaking hub or existing producers.
4) **Optional additive**: publish `doc_type=chat_turn` `VectorDocumentUpsertV1` to `orion:memory:vector:upsert` if prompt+response in one doc is required (no breaking change).

---

## Runbook

### Validate Recall Vector Hits (curl + jq)
```bash
curl -sS http://localhost:8260/recall \
  -H 'content-type: application/json' \
  -d '{"query_text":"cpu card","session_id":"<session_id>","diagnostic":true}' | jq '.debug.backend_counts'
```

### Smoke Test (Bus + Shared Schema)
```bash
python scripts/vector_smoke_test.py
```

### Inspect Chat Vectors (Chroma)
```bash
python - <<'PY'
import chromadb
from chromadb.config import Settings as ChromaSettings

client = chromadb.HttpClient(host="orion-athena-vector-db", port=8000, settings=ChromaSettings(anonymized_telemetry=False))
col = client.get_or_create_collection("orion_main_store")
res = col.get(where={"original_channel": "orion:chat:history:log"}, include=["metadatas"], limit=5)
print(res)
PY
```
