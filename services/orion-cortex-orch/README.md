# Orion Cortex Orchestrator

The **Cortex Orchestrator** (Orch) is the entry point for the Cognitive Runtime. It accepts high-level client requests (via `orion-cortex:request`), manages the session state, and delegates execution planning to **Cortex Exec**.

## Contracts

### Consumed Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion-cortex:request` | `ORCH_REQUEST_CHANNEL` | `cortex.orch.request` | Client requests (Brain, Agent, Council modes). |

### Published Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion-cortex-exec:request` | `CORTEX_EXEC_REQUEST_CHANNEL` | `cortex.exec.request` | Delegation to Cortex Exec. |
| (Caller-defined) | (via `reply_to`) | `cortex.orch.result` | Final result sent back to client. |

### Environment Variables
Provenance: `.env_example` → `docker-compose.yml` → `settings.py`

| Variable | Default (Settings) | Description |
| :--- | :--- | :--- |
| `ORCH_REQUEST_CHANNEL` | `orion-cortex:request` | Input channel. |
| `CORTEX_EXEC_REQUEST_CHANNEL` | `orion-cortex-exec:request` | Output channel to Exec. |
| `REDIS_URL` | ... | Redis connection. |

## Running & Testing

### Run via Docker
```bash
docker-compose up -d orion-cortex-orch
```

### Smoke Test
Use the bus harness in "Brain" mode.
```bash
python scripts/bus_harness.py brain "hello world"
```
