# Orion Meta Tags

The **Meta Tags** service provides automated enrichment for collapse events and other content. It uses LLMs to extract entities, sentiment, and tags, adding structured metadata to the original payload.

## Contracts

### Consumed Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion:collapse:triage` | `CHANNEL_EVENTS_TRIAGE` | `collapse.mirror` | Raw entries needing enrichment. |

### Published Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion:tags:enriched` | `CHANNEL_EVENTS_TAGGED` | `tags.enriched` | Enriched metadata payload. |

### Environment Variables
Provenance: `.env_example` → `docker-compose.yml` → `settings.py`

| Variable | Default (Settings) | Description |
| :--- | :--- | :--- |
| `CHANNEL_EVENTS_TRIAGE` | `orion:collapse:triage` | Intake channel. |
| `CHANNEL_EVENTS_TAGGED` | `orion:tags:enriched` | Output channel. |

## Running & Testing

### Run via Docker
```bash
docker-compose up -d orion-meta-tags
```
