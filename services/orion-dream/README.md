# Orion Dream

## Modernization stance (Phase 0/1)

This service is a **donor / bridge / readout façade** while the canonical dream path moves to **cortex-orch → cortex-exec → RecallService (`dream.v1`) → LLM → `dream.result.v1` → SQL Writer → `dreams` table**.

| Concern | Canonical owner |
|--------|------------------|
| Trigger normalization | `orion-cortex-orch` (Hunter on `orion:dream:trigger` → `cortex.orch.request`, `verb=dream_cycle`) |
| Plan execution | `orion-cortex-exec` |
| Memory retrieval | `orion-recall` via profile `dream.v1` (no direct Vector/RDF/SQL in the verb plan) |
| Typed artifact | `DreamResultV1` / envelope kind `dream.result.v1` |
| Durable storage | `orion-sql-writer` → PostgreSQL `dreams` |
| Wake readout | This service: **SQL-first** (`GET /dreams/wakeup/today`), optional `DREAM_LOG_DIR` JSON fallback |

### Legacy code (retire after spine is proven)

Do not use for new flows; remove after E2E smoke green.

- `app/dream_cycle.py` — direct gather + gateway synthesis (superseded by `dream_cycle` verb)
- `app/aggregators_sql.py`, `app/aggregators_rdf.py`, `app/aggregators_vector.py`
- File-log-first assumptions in `wake_readout.py` (kept only as fallback)

### HTTP / bus behavior

- **No Hunter** in this process: `dream.trigger` is consumed by **cortex-orch** so triggers are not duplicated.
- `POST /dreams/run` publishes `dream.trigger` on `CHANNEL_DREAM_TRIGGER` for compatibility.

## Contracts (historical)

### Consumed Channels (legacy listeners may still run separately)

| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion:dream:trigger` | `CHANNEL_DREAM_TRIGGER` | `dream.trigger` | Published by clients; **handled by cortex-orch**. |
| `orion:collapse:sql-write` | `CHANNEL_COLLAPSE_SQL_PUBLISH` | `collapse.mirror` | `memory_listener` / legacy paths. |

### Environment Variables

| Variable | Default (Settings) | Description |
| :--- | :--- | :--- |
| `CHANNEL_DREAM_TRIGGER` | `orion:dream:trigger` | Trigger channel. |
| `POSTGRES_URI` | (see `settings.py`) | Used by SQL wake readout. |
| `DREAM_LOG_DIR` | `/app/logs/dreams` | Optional JSON fallback for readout. |

## Running & Testing

### Run via Docker

```bash
docker-compose up -d orion-dream
```
