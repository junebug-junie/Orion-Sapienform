# Orion Dream

The **Dream** service implements the "Night Cycle" of the cognitive architecture. It aggregates daily logs (chat, collapse, biometrics), performs synthesis (dreaming), and generates insights or "morning papers" for the system.

## Contracts

### Consumed Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion:dream:trigger` | `CHANNEL_DREAM_TRIGGER` | `dream.trigger` | Manual or scheduled trigger. |
| `orion:collapse:sql-write` | `CHANNEL_COLLAPSE_SQL_PUBLISH` | `collapse.mirror` | Listener. |
| `orion:tags:enriched` | `CHANNEL_COLLAPSE_TAGS_PUBLISH` | `tags.enriched` | Listener. |
| `orion:telemetry:biometrics` | `CHANNEL_TELEMETRY_PUBLISH` | `biometrics.telemetry` | Listener. |
| `orion:chat:history:log` | `CHANNEL_CHAT` | `chat.log` | Listener. |

### Published Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion:brain:intake` | `CHANNEL_BRAIN_INTAKE` | `dream.synthesis` | Final dream output. |

### Environment Variables
Provenance: `.env_example` → `docker-compose.yml` → `settings.py`

| Variable | Default (Settings) | Description |
| :--- | :--- | :--- |
| `CHANNEL_DREAM_TRIGGER` | `orion:dream:trigger` | Trigger channel. |
| `CHANNEL_BRAIN_INTAKE` | `orion:brain:intake` | Output channel. |

## Running & Testing

### Run via Docker
```bash
docker-compose up -d orion-dream
```
