# Crystallization event observability — unify dead lifecycle channels + new concept-relation decisions

**Date:** 2026-07-13
**Status:** Proposal — ready for implementation
**Problem:** Two telemetry-shaped channels in this codebase are confirmed dead — published, never consumed. `orion:memory:crystallization:{proposed,validated,approved,rejected,quarantined,project,retrieved,reinforced,auto_activated}` (`orion/memory/crystallization/bus_emit.py::CHANNEL_DEFAULTS`, 9 channels) have zero subscribers anywhere in the repo (grep-confirmed). `orion:recall:telemetry` (`RecallDecisionV1`) has one reference, `orion/signals/registry.py`, which is a static organ-topology catalog entry, not a live consumer. Building a 10th channel (`concept_relation.decided.v1`) the same way repeats a demonstrated failure mode instead of fixing it.

## Goals

- One consumer, one destination, for all 9 existing crystallization lifecycle events **and** a new `concept_relation.decided.v1` event, in a single patch.
- Capture `unrelated`/below-floor concept-relation decisions, which currently leave **zero trace anywhere** (nothing gets written to `memory_crystallizations.provenance` unless the decision was `same`/`refines`/`contradicts`) — this is the primary reason a Postgres-only view is insufficient and a real event stream is required.
- Queryable without a dashboard: a table, not a UI. UI is a separate, later patch.

## Non-goals

- No Hub debug panel (separate patch, reads this table once it has real data).
- No OpenTelemetry/Grafana wiring — that stack exists (`services/orion-signal-gateway/`, network-reachable via `app-net`) but has zero adoption anywhere else in the repo; standing up the first non-gateway OTel integration is out of scope until this simpler mechanism proves there's traffic worth tracing.
- No changes to `orion:recall:telemetry` / `orion/signals/registry.py` — separate pre-existing gap, not blocking this one.
- No retention/archival policy beyond default Postgres — revisit if row count becomes real.

## Schema

New table, `orion/core/storage/sql/crystallization_event_log.sql` (append-only):

```sql
CREATE TABLE IF NOT EXISTS crystallization_event_log (
    event_id            BIGSERIAL PRIMARY KEY,
    event_kind          TEXT NOT NULL,          -- e.g. 'memory.crystallization.reinforced.v1', 'concept_relation.decided.v1'
    crystallization_id  TEXT,                    -- nullable: concept_relation 'unrelated' has no target
    payload             JSONB NOT NULL,          -- full envelope payload, unmodified
    received_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_cel_kind ON crystallization_event_log (event_kind);
CREATE INDEX IF NOT EXISTS idx_cel_crys ON crystallization_event_log (crystallization_id);
CREATE INDEX IF NOT EXISTS idx_cel_received ON crystallization_event_log (received_at DESC);
```

No new Pydantic model for the row — `payload` stores whatever the source event already validated (`MemoryCrystallizationV1` for lifecycle events; the new decision shape below for concept-relation). This is a log, not a second source of truth.

New event, `orion/memory/crystallization/concept_relation.py`:

```python
class ConceptRelationDecisionEventV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_crystallization_id: str
    relation: ConceptRelation                 # same|refines|contradicts|unrelated
    target_crystallization_id: str | None
    confidence: float
    candidate_count: int                       # len(similar_existing) actually sent to the LLM
    acted: bool                                 # True only for same (confidence >= floor)
```

Published unconditionally at the end of `resolve_concept_relation` — including the `similar_existing == []` short-circuit and the below-floor case — on a new channel `orion:memory:crystallization:concept_relation_decided`, kind `concept_relation.decided.v1`. Registered in `orion/bus/channels.yaml` next to the existing 9.

## Architecture

```text
producers (already exist, 2 confirmed):
  orion/memory/crystallization/intake_pipeline.py  → bus_emit.emit_crystallization_lifecycle (9 kinds)
  services/orion-hub/scripts/crystallization_routes.py → same, on governor approve/reject/etc
  orion/memory/crystallization/concept_relation.py → NEW: publish ConceptRelationDecisionEventV1

        │  10 channels total (9 existing + 1 new)
        ▼
NEW consumer: services/orion-hub/scripts/crystallization_event_cache.py
  mirrors the existing pattern in signals_inspect_cache.py / cognition_trace_cache.py —
  a small class holding an OrionBusAsync subscription, already-proven shape in this exact
  directory, started alongside Hub's other background subscribers at app startup.

        │  INSERT INTO crystallization_event_log
        ▼
Postgres (existing DSN, existing pool — no new infra)
```

**Why Hub, not a new service:** two of the three producers already run inside/adjacent to Hub's process (governor actions are literally Hub routes); Hub already hosts the exact subscriber-class pattern this needs (`signals_inspect_cache.py`, `cognition_trace_cache.py` — both subscribe-and-cache classes, started the same way); this is a thin seam onto proven infrastructure, not a new deployable.

**Third producer to confirm during implementation, not blocking the spec:** `services/orion-memory-crystallizer/app/worker.py` also imports `emit_crystallization_lifecycle` per grep — verify whether this is a live, distinct producer or dead/legacy code before wiring; either way the consumer subscribes to the channel, not the producer, so this doesn't change the design.

## Choke points

| File | Role |
|---|---|
| `orion/memory/crystallization/bus_emit.py` | Existing 9 channels — unchanged |
| `orion/memory/crystallization/concept_relation.py` | New: `ConceptRelationDecisionEventV1` + publish call in `resolve_concept_relation` |
| `orion/bus/channels.yaml` | Register the 10th channel |
| `services/orion-hub/scripts/crystallization_event_cache.py` (new) | The consumer — mirror `cognition_trace_cache.py`'s class shape exactly |
| `orion/core/storage/sql/crystallization_event_log.sql` (new) | Table DDL |
| `services/orion-hub/app/main.py` | Start the new subscriber alongside existing ones (wherever `signals_inspect_cache`/`cognition_trace_cache` get instantiated) |

## Acceptance checks

- [ ] Publish a `concept_relation.decided.v1` with `similar_existing=[]` (the cost-avoidance short-circuit) — confirm a row lands with `relation=unrelated, acted=false, candidate_count=0`. This is the case that currently leaves zero trace anywhere; proving it's captured is the acceptance bar that matters most.
- [ ] Publish each of the 9 existing lifecycle kinds (fixture envelopes, no live pipeline needed) — confirm 9 rows land with correct `event_kind`/`crystallization_id`.
- [ ] Consumer restart mid-stream doesn't duplicate or drop events under normal redelivery (match whatever guarantee `cognition_trace_cache.py` already relies on — don't invent a new one).
- [ ] `SELECT event_kind, count(*) FROM crystallization_event_log GROUP BY event_kind` — the whole point, works day one.

## Env/config changes

None required to ship — reuses Hub's existing `POSTGRES_URI` and `ORION_BUS_URL`. No new keys.

## Testing

Unit: `ConceptRelationDecisionEventV1` schema validation; publish call fires on all 4 relation outcomes including the empty-candidates short-circuit (currently untested — the review pass tested the decision object, not that it gets published).
Integration: fixture envelopes on all 10 channels → consumer → assert rows, mirroring whatever test pattern covers `cognition_trace_cache.py` today.

## Risks

| Severity | Risk | Mitigation |
|---|---|---|
| Low | 10th channel becomes an 11th dead thing if nobody queries the table | Explicitly out of scope to build a dashboard yet — but the acceptance check above (`GROUP BY event_kind`) is the manual query that replaces one-off `psql` sessions immediately, day one, no UI needed |
| Low | `orion-memory-crystallizer` producer unconfirmed | Doesn't block design — consumer is channel-scoped, not producer-scoped; confirm at implementation time |

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d --build
```
No memory-consolidation restart needed for the consumer itself; needed once the `concept_relation.py` publish call ships (env/code change to that service).
