# Orion Chat Memory

The **orion-chat-memory** service turns chat transcripts and finalized chunk windows into vector upsert events. It does **not** compute embeddings locally—instead it requests them from a dedicated embedding host (bus RPC or HTTP) and publishes `memory.vector.upsert.v1` envelopes that already contain embeddings and latent references.

## What it does
- Listens to chat/chunk channels on the Orion bus.
- Requests embeddings upstream using the configured mode (`bus` or `http`).
- Publishes normalized `memory.vector.upsert.v1` payloads with precomputed embeddings, model metadata, and optional latent references.
- Skips chunk embedding until a window finalizes (or every N messages when configured).

## Configuration (env lineage)

| Variable | Default | Notes |
| --- | --- | --- |
| `CHAT_MEMORY_INPUT_CHANNELS` | `["orion:chat:history:log"]` | Bus patterns to consume. |
| `CHAT_MEMORY_COLLECTION` | `orion_chat` | Target collection name to include in upsert payloads. |
| `CHAT_MEMORY_UPSERT_CHANNEL` | `orion:memory:vector:upsert` | Bus channel to publish upsert envelopes. |
| `CHAT_MEMORY_CHUNK_INTERVAL` | `0` | When >0, embed every Nth chunk message; otherwise only on finalization flags. |
| `CHAT_MEMORY_EMBED_ENABLE` | `true` | Toggle embedding requests. |
| `CHAT_MEMORY_EMBED_MODE` | `bus` | `bus` or `http`. |
| `CHAT_MEMORY_EMBED_REQUEST_CHANNEL` | `orion:embedding:generate` | Bus request channel (bus mode). |
| `CHAT_MEMORY_EMBED_RESULT_CHANNEL` | `orion:embedding:result` | Bus reply channel prefix (bus mode). |
| `CHAT_MEMORY_EMBED_HOST_URL` | _(none)_ | Full HTTP endpoint for embedding requests (http mode). |
| `CHAT_MEMORY_INCLUDE_LATENTS` | `false` | When true, ask the embedder for latent references. |
| `CHAT_MEMORY_EMBED_TIMEOUT_MS` | `10000` | Embedding request timeout. |
| `CHAT_MEMORY_EMBED_PROFILE` | `default` | Profile name forwarded to the embedding host. |

Bus + health metadata is inherited from the shared chassis (`ORION_BUS_URL`, `SERVICE_NAME`, etc.).

## Run
```
docker compose -f services/orion-chat-memory/docker-compose.yml up --build
```

## Notes
- Latents are **off by default**. When enabled, only reference handles (`latent_ref`) and small summaries are propagated—no large latent payloads ride the bus.
- Upserts are skipped if an embedding is not returned by the embedding host to keep the downstream vector writer a pure sink.
