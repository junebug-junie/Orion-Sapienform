# Orion Recall Service

The **Recall Service** is the cognitive memory retrieval engine. It performs hybrid searches across Vector (semantic), SQL (structured), and RDF (graph) stores to provide relevant context to the cognitive runtime.

## Contracts

### Consumed Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion-exec:request:RecallService` | `CHANNEL_RECALL_REQUEST` | `recall.query.request` | Query requests. |

### Published Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| (Caller-defined) | (via `reply_to`) | `recall.query.result` | Retrieved memory fragments. |

### Environment Variables
Provenance: `.env_example` → `docker-compose.yml` → `settings.py`

| Variable | Default (Settings) | Description |
| :--- | :--- | :--- |
| `CHANNEL_RECALL_REQUEST` | `orion-exec:request:RecallService` | Intake channel. |
| `CHANNEL_RECALL_DEFAULT_REPLY_PREFIX` | `orion-exec:result:RecallService` | Default reply prefix (if not specified by caller). |

## Running & Testing

### Run via Docker
```bash
docker-compose up -d orion-recall
```

### Smoke Test
Recall is implicitly tested when running a "Brain" or "Agent" request that requires memory.
```bash
python scripts/bus_harness.py brain "what did I dream about yesterday?"
```
