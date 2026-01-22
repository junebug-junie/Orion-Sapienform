# Vector Memory Cleanup — Audit, Plan, Verification

## Executive Summary
- **Session-scoped recall is failing** because most vectors in `orion_main_store` are embedding_request artifacts that do not carry `session_id`. Chat history vectors should live in `orion_chat` and include `session_id`, `role`, and `speaker` metadata. 【F:services/orion-vector-host/app/main.py†L248-L306】【F:services/orion-vector-writer/app/chat_history.py†L18-L61】
- **Chroma metadata constraints** require primitive values only; hub emits `tags` as a list, which breaks vector-writer upserts without sanitization. 【F:services/orion-vector-writer/app/main.py†L32-L74】
- **MiniLM downloads on query** were caused by `query_texts` calls. We now compute embeddings via vector-host and use `query_embeddings` for deterministic BGE‑small retrieval. 【F:services/orion-recall/app/storage/vector_adapter.py†L1-L167】【F:services/orion-vector-host/app/main.py†L322-L340】

---

## Phase 1 — Codepoints (Ground Truth)

### Vector Writer Ingestion
**VectorUpsertV1 → Chroma**
- `services/orion-vector-writer/app/main.py` handles `VectorUpsertV1` and writes to Chroma via `collection.upsert`. Metadata is now sanitized before write. 【F:services/orion-vector-writer/app/main.py†L120-L233】

**chat.history.message.v1 → orion_chat**
- `services/orion-vector-writer/app/chat_history.py` maps `ChatHistoryMessageV1` to `VectorWriteRequest` and writes to `orion_chat`. Metadata now includes `role`, `speaker`, `created_at`, `original_channel`, `correlation_id`, and `envelope_id`. 【F:services/orion-vector-writer/app/chat_history.py†L18-L61】

### Recall Vector Query
**Chroma query path**
- `services/orion-recall/app/storage/vector_adapter.py` builds the Chroma query with `query_embeddings` and a `where` filter using `session_id` if present. 【F:services/orion-recall/app/storage/vector_adapter.py†L70-L167】

### Embedding Model Usage
**Stored embeddings**
- `orion-vector-host` generates embeddings via BGE (HF or vLLM) and publishes `vector.upsert.v1`. 【F:services/orion-vector-host/app/embedder.py†L18-L84】【F:services/orion-vector-host/app/main.py†L74-L118】

**Query embeddings**
- Recall uses **vector-host’s embedding endpoint** and never calls `query_texts`. This avoids client‑side MiniLM downloads. 【F:services/orion-recall/app/storage/vector_adapter.py†L70-L167】【F:services/orion-vector-host/app/main.py†L322-L340】

---

## Patch Set Summary

### PATCH 1 — Metadata Sanitizer (vector-writer)
**Why**: Chroma requires primitive metadata; hub emits list `tags`.  
**Update File: `services/orion-vector-writer/app/main.py`**
- `sanitize_metadata()` converts lists to comma‑strings and dicts to JSON.
- Applied to **all** Chroma upsert paths (semantic, latent, memory docs, chat history). 【F:services/orion-vector-writer/app/main.py†L32-L233】

### PATCH 2 — Chat Metadata Enrichment (orion_chat)
**Why**: `orion_chat` needs `role`, `speaker`, `created_at`, `original_channel`, `correlation_id`, `envelope_id` for deterministic recall.  
**Update File: `services/orion-vector-writer/app/chat_history.py`**【F:services/orion-vector-writer/app/chat_history.py†L18-L61】

### PATCH 3 — Recall Defaults + Session Filter
**Why**: recall should target chat memory by default and filter by `session_id`.  
**Update File: `services/orion-recall/app/settings.py`**
- Defaults `RECALL_VECTOR_COLLECTIONS` to `orion_chat`.
- Adds `RECALL_VECTOR_EMBEDDING_URL` for BGE query embeddings. 【F:services/orion-recall/app/settings.py†L135-L201】

**Update File: `services/orion-recall/app/storage/vector_adapter.py`**
- Uses `query_embeddings` and `where={"session_id": ...}` when provided. 【F:services/orion-recall/app/storage/vector_adapter.py†L70-L167】

### PATCH 4 — Remove MiniLM Surprise
**Why**: `query_texts` triggers implicit MiniLM downloads.  
**Update File: `services/orion-recall/app/storage/vector_adapter.py`**
- Compute embeddings via vector-host `/embedding` and always call `query_embeddings`. 【F:services/orion-recall/app/storage/vector_adapter.py†L70-L167】

**Update File: `services/orion-vector-host/app/main.py`**
- Adds `/embedding` endpoint using the same embedder as vector-host’s pipeline. 【F:services/orion-vector-host/app/main.py†L322-L340】

---

## Verification Commands

### 1) Bus‑level ingestion (chat.history.message.v1)
```bash
python scripts/vector_smoke_test.py
```

### 2) Recall check (session‑scoped)
```bash
curl -sS http://localhost:8260/recall \
  -H 'content-type: application/json' \
  -d '{"query_text":"cpu card","session_id":"<session_id>","diagnostic":true}' | jq '.debug.backend_counts'
```

### 3) Chroma inspection (orion_chat)
```bash
docker exec -i orion-athena-vector-writer python - <<'PY'
import chromadb
from chromadb.config import Settings as ChromaSettings

client = chromadb.HttpClient(host="orion-athena-vector-db", port=8000, settings=ChromaSettings(anonymized_telemetry=False))
col = client.get_or_create_collection("orion_chat")
res = col.get(where={"session_id": {"$ne": ""}}, include=["metadatas"], limit=5)
print(res)
PY
```

### 4) No‑MiniLM check
Watch recall + vector-host logs during the query; there should be **no** all‑MiniLM downloads.
```bash
docker logs -f orion-athena-vector-host
```

---

## Notes on Channels & Schemas
- **No new bus channels added** in this patch set.
- **No new bus schemas added**; existing schemas are reused (`ChatHistoryMessageV1`, `EmbeddingGenerateV1`, `EmbeddingResultV1`).
