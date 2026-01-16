# Chat history → bus → vector memory

This service flow persists Orion Hub chat messages into vector memory using the Titanium envelope.

## Event contract
- **Kind:** `chat.history.message.v1`
- **Channel:** `CHAT_HISTORY_LOG_CHANNEL` (defaults to `orion:chat:history:log`)
- **Schema:** `orion.schemas.chat_history.ChatHistoryMessageEnvelope`
- **Payload fields:**
  - `message_id` (UUID string, defaults to a new UUID; `id` accepted as alias)
  - `session_id`/`conversation_id`
  - `role` (`user|assistant|system|tool`)
  - `speaker` (user id or system name; `user` alias supported)
  - `content`
  - `timestamp` (ISO-8601 string, auto-populated)
  - optional `model`, `provider`, `tags` (list[str])
- **Vector document mapping (`to_document`):**
  - `id` → `message_id`
  - `text` → `content`
  - `metadata` → role, speaker, session_id, timestamp, kind, correlation_id, source_service/node/version, source_channel, model/provider, tags

## Publishing (orion-hub)
- **WebSocket** (`services/orion-hub/scripts/websocket_handler.py`):
  - Publishes a `chat.history.message.v1` envelope for the inbound user transcript and another for the assistant reply (same `correlation_id`).
  - Uses the configurable `CHAT_HISTORY_LOG_CHANNEL`/`CHANNEL_CHAT_HISTORY_LOG`.
  - Continues emitting the legacy `chat.history` envelope for SQL Writer compatibility.
- **HTTP /api/chat** (`services/orion-hub/scripts/api_routes.py`):
  - Emits versioned chat history envelopes for the latest user prompt (if available) and assistant response.
  - Re-publishes the legacy chat history payload to the same channel for backward compatibility.
- **Env lineage:** `.env_example` → `docker-compose.yml` → `scripts/settings.py` (`chat_history_channel` property) → publisher helpers.

## Ingestion (orion-vector-host → orion-vector-writer)
- **Subscription:** `orion-vector-host` listens on `orion:chat:history:log` and filters roles via `VECTOR_HOST_EMBED_ROLES`.
- **Embedding:** `orion-vector-host` generates semantic embeddings and publishes `VectorUpsertV1` on `orion:vector:semantic:upsert`.
- **Vector write:** `orion-vector-writer` listens on `orion:vector:semantic:upsert` and writes the embedding to the semantic collection (default `VECTOR_DB_COLLECTION`).
- **Metadata logged:** `source_service`, `role`, `session_id`, `correlation_id`, and `original_channel` are included with each vector upsert.

## Configuration quick reference
- **Hub**
  - `.env_example`: `CHAT_HISTORY_LOG_CHANNEL` (new) + legacy `CHANNEL_CHAT_HISTORY_LOG`
  - `docker-compose.yml`: forwards `CHAT_HISTORY_LOG_CHANNEL` (falls back to legacy variable)
  - `scripts/settings.py`: `chat_history_channel` property reads new var or legacy fallback
- **Vector Host**
  - `.env_example`: `VECTOR_HOST_CHAT_HISTORY_CHANNEL`, `VECTOR_HOST_EMBED_BACKEND`, `VECTOR_HOST_EMBEDDING_MODEL`
  - `docker-compose.yml`: forwards chat history + embedding env vars
  - `app/settings.py`: validates embedding model + roles list
- **Vector Writer**
  - `.env_example`: `VECTOR_WRITER_SUBSCRIBE_CHANNELS`, `VECTOR_DB_COLLECTION`, `VECTOR_DB_COLLECTION_LATENT`
  - `docker-compose.yml`: forwards vector upsert channels + collections
