# Orion Vector Host

The **Vector Host** service generates semantic embeddings and publishes vector upserts for the vector writer. It listens to chat history events (passive embedding) and to explicit embedding requests on the bus (active embedding).

## Semantic embeddings

Vector-host computes semantic embeddings for **all assistant texts** regardless of which backend produced them (ollama/llamacpp/vllm/cola). The provider selection (currently vLLM) only controls the engine used to compute embeddings, not which assistant responses are embedded.

## Contracts

### Consumed Channels
| Channel | Schema | Description |
| :--- | :--- | :--- |
| `orion:chat:history:log` | `ChatHistoryMessageV1` | Embeds chat messages and publishes semantic upserts. |
| `orion:embedding:generate` | `EmbeddingGenerateV1` | Generates semantic embeddings and replies with `EmbeddingResultV1` while also emitting semantic upserts. |

### Published Channels
| Channel | Schema | Description |
| :--- | :--- | :--- |
| `orion:vector:semantic:upsert` | `VectorUpsertV1` | Semantic vector upserts for the vector writer. |
| `orion:embedding:result:*` | `EmbeddingResultV1` | Embedding RPC replies. |

## Environment Variables
Provenance: `.env_example` → `docker-compose.yml` → `settings.py`

| Variable | Default | Description |
| :--- | :--- | :--- |
| `VECTOR_HOST_EMBED_BACKEND` | `vllm` | Embedding backend (vLLM). |
| `VECTOR_HOST_EMBEDDING_MODEL` | (required) | Embedding model name sent to the backend. |
| `VECTOR_HOST_SEMANTIC_COLLECTION` | `orion_main_store` | Semantic collection for vector upserts. |
| `VECTOR_HOST_EMBED_ROLES` | `["user","assistant"]` | Chat roles to embed from history. |

## Smoke Tests

1) **Semantic path**  
Publish `ChatHistoryMessageV1` on `orion:chat:history:log`, confirm:
   - `orion-vector-host` emits `VectorUpsertV1` on `orion:vector:semantic:upsert`.
   - `orion-vector-writer` writes into the semantic collection.

2) **Request path**  
Publish `EmbeddingGenerateV1` on `orion:embedding:generate`, confirm:
   - `orion-vector-host` replies on `orion:embedding:result:*`.
   - `orion-vector-host` emits `VectorUpsertV1` on `orion:vector:semantic:upsert`.
   - `orion-vector-writer` writes into the semantic collection.

3) **Latent path**  
Trigger vLLM/llama-cola via `orion-llm-gateway`, confirm:
   - `orion-llm-gateway` publishes `VectorUpsertV1` on `orion:vector:latent:upsert`.
   - `orion-vector-writer` writes into the latent collection.
