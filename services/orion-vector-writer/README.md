# Orion Vector Writer

The **Vector Writer** service subscribes to vector upsert envelopes and persists them into the Vector DB (ChromaDB/Qdrant) for semantic recall. It only writes vectors provided by upstream services and never requests embeddings itself.

## Contracts

### Consumed Channels
Configured via `VECTOR_WRITER_SUBSCRIBE_CHANNELS` (list).

| Default Channel | Kind(s) | Description |
| :--- | :--- | :--- |
| `orion:vector:semantic:upsert` | `vector.upsert.v1` | Writes semantic embeddings into the primary vector store. |
| `orion:vector:latent:upsert` | `vector.upsert.v1` | Writes latent vectors into the latent collection. |
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
| `VECTOR_WRITER_REQUIRE_EMBEDDINGS` | `false` | Fail hard if embeddings cannot be resolved. |
| `VECTOR_DB_COLLECTION_LATENT` | `orion_latent_store` | Dedicated latent vector collection. |

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
