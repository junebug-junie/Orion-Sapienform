# Orion Vector Writer

The **Vector Writer** service subscribes to vector upsert envelopes and persists them into the Vector DB (ChromaDB/Qdrant) for semantic recall. It is a **pure sink**: embeddings must be precomputed upstream (for example by `orion-chat-memory` or an embedding host) and provided in the payload.

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
| `HEALTH_CHANNEL` | `system.health` | Health check channel. |

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
