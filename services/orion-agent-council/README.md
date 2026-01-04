# Orion Agent Council

The **Agent Council** service simulates a multi-agent deliberation process. It is currently a stub/prototype for higher-order reasoning.

## Contracts

### Consumed Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion:agent-council:intake` | `CHANNEL_COUNCIL_INTAKE` | `council.request` | Deliberation requests. |

### Published Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion-exec:request:LLMGatewayService` | `CHANNEL_LLM_INTAKE` | `llm.chat.request` | Reasoning calls. |
| (Caller-defined) | (via `reply_to`) | `council.result` | Final verdict. |

### Environment Variables
Provenance: `.env_example` → `docker-compose.yml` → `settings.py`

| Variable | Default (Settings) | Description |
| :--- | :--- | :--- |
| `CHANNEL_COUNCIL_INTAKE` | `orion:agent-council:intake` | Intake channel. |
| `CHANNEL_LLM_INTAKE` | `orion-exec:request:LLMGatewayService` | LLM channel. |

## Running & Testing

### Run via Docker
```bash
docker-compose up -d orion-agent-council
```
