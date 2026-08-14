# Orion Vector Host

The **Vector Host** service generates semantic embeddings and publishes vector upserts for the vector writer. It listens to chat history events (passive embedding) and to explicit embedding requests on the bus (active embedding).

## Semantic embeddings

Vector-host computes semantic embeddings for **all assistant texts** regardless of which backend produced them (ollama/llamacpp/vllm/cola). The provider selection (currently HF) only controls the engine used to compute embeddings, not which assistant responses are embedded.

## Contracts

### Consumed Channels
| Channel | Schema | Description |
| :--- | :--- | :--- |
| `orion:chat:history:log` | `ChatHistoryMessageV1` | Embeds chat messages and feeds OrionTissue directly (`app/tissue_feed.py`). Does **not** publish a vector upsert -- see "Dead pipeline killed" below. |
| `orion:chat:history:turn` | `ChatHistoryTurnV1` | Embeds chat turns and feeds OrionTissue directly. Does **not** publish a vector upsert -- same kill as above. |
| `orion:chat:gpt:log` | `ChatGptMessageV1` | Embeds ChatGPT imported messages into `orion_chat_gpt` when GPT ingest is enabled. |
| `orion:chat:gpt:turn` | `ChatGptLogTurnV1` | Embeds ChatGPT imported turns into `orion_chat_gpt_turns` when GPT ingest is enabled. |
| `orion:embedding:generate` | `EmbeddingGenerateV1` | Generates semantic embeddings and replies with `EmbeddingResultV1` while also emitting semantic upserts. |

### Dead pipeline killed (2026-08-14)

`_handle_chat_history` / `_handle_chat_turn` used to publish `VectorUpsertV1`
on `orion:vector:semantic:upsert` targeting the `orion_chat` /
`orion_chat_turns` Chroma collections. That write path had no reader:
orion-recall's Chroma retrieval path was already amputated in May 2026
(`RECALL_ENABLE_VECTOR` defaults `false`, see
`docs/superpowers/pr-reports/2026-05-29-orion-recall-vector-amputation-pr.md`).
The vector-upsert publish was removed outright (killed, not narrowed) so
`orion-vector-writer` never writes to those two collections again. The
embed + `feed_tissue()` call was kept in both handlers because it is a
separate, still-live OrionTissue stimulus source, not a fallback to the
killed write -- see `app/tissue_feed.py`'s module docstring. Existing
`orion_chat` / `orion_chat_turns` Chroma data is orphaned but was not
deleted by this change.

### Published Channels
| Channel | Schema | Description |
| :--- | :--- | :--- |
| `orion:vector:semantic:upsert` | `VectorUpsertV1` | Semantic vector upserts for the vector writer. |
| `orion:embedding:result:*` | `EmbeddingResultV1` | Embedding RPC replies. |

## Environment Variables
Provenance: `.env_example` → `docker-compose.yml` → `settings.py`

| Variable | Default | Description |
| :--- | :--- | :--- |
| `VECTOR_HOST_EMBED_BACKEND` | `hf` | Embedding backend (`hf` or `vllm`). |
| `VECTOR_HOST_EMBEDDING_MODEL` | (required) | HuggingFace model name (HF) or model name sent to vLLM. |
| `VECTOR_HOST_EMBEDDING_DEVICE` | `cpu` | Device for HF embeddings (`cpu` or `cuda`). |
| `VECTOR_HOST_SEMANTIC_COLLECTION` | `orion_main_store` | Semantic collection for vector upserts. |
| `VECTOR_HOST_GPT_ENABLED` | `false` | Enable ChatGPT channel ingestion in vector-host. |
| `VECTOR_HOST_GPT_MESSAGE_CHANNEL` | `orion:chat:gpt:log` | ChatGPT message channel. |
| `VECTOR_HOST_GPT_TURN_CHANNEL` | `orion:chat:gpt:turn` | ChatGPT turn channel. |
| `VECTOR_HOST_GPT_MESSAGE_COLLECTION` | `orion_chat_gpt` | Collection for ChatGPT imported message embeddings. |
| `VECTOR_HOST_GPT_TURN_COLLECTION` | `orion_chat_gpt_turns` | Collection for ChatGPT imported turn embeddings. |
| `VECTOR_HOST_EMBED_ROLES` | `["user","assistant"]` | Chat roles to embed from history. |

`VECTOR_HOST_CHAT_MESSAGE_COLLECTION` / `VECTOR_HOST_CHAT_TURN_COLLECTION` were
removed 2026-08-14 along with the dead vector-upsert write -- see above.

## Substrate Brain State page (`/spark/ui`)

Relocated 2026-07-28 from the retired `orion-spark-introspector` service (see
`docs/superpowers/specs/2026-07-28-spark-introspector-retirement-and-honest-substrate-
convergence.md`). Still contains OrionTissue's own real decay/diffusion tensor physics
(`app/tissue_feed.py`, `orion/spark/orion_tissue.py`) exposed via `GET /api/tissue/state` and
`WS /ws/tissue` -- fed by real per-turn embedding deltas, honestly labeled
(`embedding_similarity`/`novelty_zscore`/`polarity_diff`/`mean_abs_activation`, no invented
mood taxonomy).

**The page itself (`app/static/index.html` + `tissue_viz.js`) no longer displays those four
tensor stats.** They're real but theory-orphaned (no independent link to reasoning quality,
mood, or wellbeing) and, unlike the three signals below, weren't independently verified to
resist saturation before being wired into a visual driver. The page's synthwave visualizer
(kept as-is -- same grid/sun/hue mechanics) is now driven by three real signals it polls from
orion-hub's `GET /api/self-brain/frames/tail` every 5s (`BRAIN_FRAME_INTERVAL_SEC`), each
individually live-verified against real running data before being wired in:

| Signal | Source | Live-verified range |
|--------|--------|----------------------|
| Prediction Confidence | `honesty_metrics` region, `orion-substrate-runtime` | 0.79–0.99 (2min real window) |
| Coalition Stability | `frame.spotlight.coalition_stability` | 0.3–0.9 (7min real window) |
| Field Anomaly | `field_anomaly` region, mood-arc encoder recon_loss | confirmed real anomalous→calm transition |

Full candidate-rejection trace (why `self_state`, `node_kind`/`lane` `max()`/`min()`, and raw
`bus_synaptic` `gap_zscore` were checked and dropped) lives in
`services/orion-substrate-runtime/README.md`'s Brain Frames section and the spec doc above.

## Smoke Tests

1) **Chat tissue-feed path (no vector write)**  
Publish `ChatHistoryMessageV1` on `orion:chat:history:log`, confirm:
   - `orion-vector-host` computes the embedding and calls `feed_tissue()` (check `GET /api/tissue/state` for movement).
   - `orion-vector-host` does **not** emit `VectorUpsertV1` for this envelope (killed 2026-08-14).

2) **Request path**  
Publish `EmbeddingGenerateV1` on `orion:embedding:generate`, confirm:
   - `orion-vector-host` replies on `orion:embedding:result:*`.
   - `orion-vector-host` emits `VectorUpsertV1` on `orion:vector:semantic:upsert`.
   - `orion-vector-writer` writes into the semantic collection.

3) **Latent path**  
Trigger vLLM/llama-cola via `orion-llm-gateway`, confirm:
   - `orion-llm-gateway` publishes `VectorUpsertV1` on `orion:vector:latent:upsert`.
   - `orion-vector-writer` writes into the latent collection.
