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

## Dream cycle v2: sleep that changes something

A dream used to be a story written into `dreams` that nothing acted on. v2 makes
sleep the time Orion does work it can't do while awake, and makes every dream
scoreable. Default off (`ORION_DREAM_CYCLE_ENABLED`).

```
sleep pressure --(>= threshold AND idle AND >= min interval)--> replay
     |                                                             |
     |   weighted count of what the day left unprocessed          +--> REM compaction (staged, existing)
     |   since the last sleep: degraded/critical metacog,         |
     |   reverie compaction asks, resonance alerts, touched       +--> recombination --> dream_hypothesis
     |   active crystallizations. Reads exactly 0 after a sleep.       dream arm:   distant replay pairs
     |                                                                  control arm: random pairs, same prompt
     v
Hub curiosity kickoff shows each hypothesis once, arm hidden. Orion alone
decides whether to form a :Prior from one (formed_from "dream_hypothesis:<id>").
scripts/dream_hypothesis_scorecard.py compares adoption/support per arm.
```

| Piece | File |
|---|---|
| Candidates, weights, pressure, replay selection (deterministic) | `app/replay.py` |
| Pairing (both arms) + LLM link prompt + hollow guard | `app/recombine.py` |
| Orchestration + sleep loop | `app/cycle.py` |
| Reads (4 producer tables, chat idle) / writes (v2 tables only) | `app/cycle_store.py` |
| LLM gateway RPC (background lane) | `app/llm.py` |
| Contract | `orion/schemas/dream_cycle.py` |
| Offer / prompt section / scorecard | `orion/dream/hypotheses.py` |
| Migration | `services/orion-sql-db/manual_migration_dream_cycle_v2.sql` |

HTTP: `GET /dreams/cycle/pressure` (read-only), `POST /dreams/cycle/run?force=true`.

Writes nothing to canonical memory. The dream never writes a belief.

The legacy direct-gather path (`dream_cycle.py`, `aggregators_*`, `memory_listener.py`)
was deleted in the same patch.

### HTTP / bus behavior

- **No Hunter** in this process: `dream.trigger` is consumed by **cortex-orch** so triggers are not duplicated.
- `POST /dreams/run` publishes `dream.trigger` on `CHANNEL_DREAM_TRIGGER` for compatibility.

## Contracts (historical)

### Channels

| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion:dream:trigger` | `CHANNEL_DREAM_TRIGGER` | `dream.trigger` | Published by clients; **handled by cortex-orch**. |

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
