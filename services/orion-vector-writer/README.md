# Orion Vector Writer

The **Vector Writer** service subscribes to vector upsert envelopes and persists them into the Vector DB (ChromaDB/Qdrant) for semantic recall. By default it can request embeddings from the LLM gateway when none are supplied, while still accepting precomputed vectors from upstream services.

## Contracts

### Consumed Channels
Configured via `VECTOR_WRITER_SUBSCRIBE_CHANNELS` (list).

| Default Channel | Kind(s) | Description |
| :--- | :--- | :--- |
| `orion:memory:vector:upsert` | `memory.vector.upsert.v1` | Writes pre-embedded documents into the vector store. |

### Published Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion:vector:confirm` | `PUBLISH_CHANNEL_VECTOR_CONFIRM` | `vector.write.confirm` | Confirmation of write (optional). |

### Environment Variables
Provenance: `.env_example` → `docker-compose.yml` → `settings.py`

| Variable | Default (Settings) | Description |
| :--- | :--- | :--- |
| `VECTOR_WRITER_SUBSCRIBE_CHANNELS` | (See above) | List of input channels. |
| `PUBLISH_CHANNEL_VECTOR_CONFIRM` | `orion:vector:confirm` | Confirmation channel. |
| `ORION_HEALTH_CHANNEL` | `orion:system:health` | Health check channel. |
| `VECTOR_WRITER_EMBEDDINGS_ENABLED` | `true` | Request embeddings for messages that omit vectors. |
| `VECTOR_WRITER_REQUIRE_EMBEDDINGS` | `false` | Fail hard if embeddings cannot be resolved. |
| `VECTOR_WRITER_EMBEDDING_CHANNEL` | `orion:embedding:generate` | Bus channel for embedding requests. |
| `VECTOR_WRITER_EMBEDDING_REPLY_PREFIX` | `orion:embedding:result:` | Reply channel prefix for embedding RPC. |
| `VECTOR_WRITER_EMBEDDING_PROFILE` | `default` | Embedding profile name sent to the LLM gateway. |
| `VECTOR_WRITER_EMBEDDING_TIMEOUT_SEC` | `30` | Timeout for embedding RPC requests. |

## Running & Testing

### Run via Docker
```bash
docker-compose up -d orion-vector-writer
```

### Smoke Test
Monitor logs for successful ingestion.
```bash
docker-compose logs -f orion-vector-writer
```
