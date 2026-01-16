# Orion LLM Gateway

The **LLM Gateway** provides a unified interface to various LLM backends (OpenAI, Anthropic, Local, etc.). It accepts standard `ChatRequestPayload` messages and returns normalized `ChatResultPayload` responses.

It now supports **latent vector emission** for vLLM/llama-cola responses (when the backend returns a spark vector), publishing those latents to the vector writer while leaving semantic embeddings to orion-vector-host.

## Contracts

### Consumed Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion-exec:request:LLMGatewayService` | `CHANNEL_LLM_INTAKE` | `llm.chat.request` | Chat requests. |
| `orion:spark:introspect:candidate` | `CHANNEL_SPARK_INTROSPECT_CANDIDATE` | `spark.introspect` | Spark introspection requests. |

### Published Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| (Caller-defined) | (via `reply_to`) | `llm.chat.result` | Chat completion result. |

### Environment Variables
Provenance: `.env_example` → `docker-compose.yml` → `settings.py`

| Variable | Default (Settings) | Description |
| :--- | :--- | :--- |
| `CHANNEL_LLM_INTAKE` | `orion-exec:request:LLMGatewayService` | Primary intake. |
| `CHANNEL_VECTOR_LATENT_UPSERT` | `orion:vector:latent:upsert` | Latent vector upsert channel. |
| `ORION_VECTOR_LATENT_COLLECTION` | `orion_latent_store` | Latent vector collection. |
| `ORION_LLM_VLLM_URL` | `None` | URL for vLLM host. |
| `ORION_LLM_LLAMACPP_URL` | `None` | URL for LlamaCpp Chat host. |

## Running & Testing

### Run via Docker
```bash
docker-compose up -d orion-llm-gateway
```

### Smoke Test
```bash
# Verify using curl or internal tool, or via harness trace
python scripts/bus_harness.py tap
```
