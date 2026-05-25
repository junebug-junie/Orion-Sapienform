# Orion Bus Substrate Trace Adoption — Transport Legibility v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add bounded, periodic transport substrate traces for `services/orion-bus` (Redis core + exporter) via an optional `bus-observer` sidecar — without turning every Redis message into grammar spam.

**Architecture:** **Option B** — new `bus-observer` Python sidecar under `services/orion-bus` (not bus-tap, not bus-mirror). `orion-bus-tap` is a WebSocket UI that subscribes to `orion:*` for live display; `orion-bus-mirror` archives every envelope to SQLite — both violate the transport-only, non-packet-log stance. Observer polls Redis on an interval: `PING`, `XLEN` per configured stream, threshold-based backpressure, optional catalog check against `orion/bus/channels.yaml`. Pure `grammar_emit.py` + fail-open `grammar_publish.py` mirroring cortex-exec/biometrics. Default publish **off**.

**Tech Stack:** Python 3.12, `redis[hiredis]`, Pydantic v2 (`orion/schemas/grammar.py`), `orion.grammar.publish`, asyncio poll loop, Docker Compose, pytest 8.3.x.

**Design source:** User handoff “Orion Bus Substrate Trace Adoption — Transport Legibility v1” (2026-05-25).

**Doctrine (burned in):** Do not make the bus into a packet-to-grammar firehose. Make transport health, contract violations, and delivery anomalies substrate-legible.

**Locked review patches (2026-05-25):**

| Patch | Detail |
|-------|--------|
| `model_copy` tests | Use field names (`publish_orion_bus_grammar`, `bus_observer_streams`), not env aliases |
| Atom encoding | No `metadata` dict — bounded `summary` / `dimensions` / `payload_ref` only |
| Publish wrapper | Matches `publish_grammar_event(bus, event, source_name=..., channel=...)` |
| Uncataloged role | `bus_configured_stream_uncataloged` — configured stream ∉ catalog; not “rogue message” |
| Default streams | `orion:evt:gateway,orion:bus:out` only; `orion:grammar:event` operator-opt-in |
| Catalog producer | `orion-bus` only on `orion:grammar:event`; verify `schema_id: GrammarEventV1` |
| Redaction test | `test_summaries_never_include_redis_values_or_envelope_material` |

**Non-goals:** Layer 3 `bus_transport_reducer`, field digestion, per-message traces, new `GrammarEventKind` / `RelationType`, fatal grammar publish, changing `bus-core` / `bus-exporter` images.

---

## Worktree and branch hygiene (mandatory)

```bash
cd /mnt/scripts/Orion-Sapienform
git fetch origin main
git worktree add .worktrees/feat-orion-bus-substrate-trace-v1 \
  -b feat/orion-bus-substrate-trace-v1 origin/main
cd .worktrees/feat-orion-bus-substrate-trace-v1
git check-ignore -q .worktrees   # must succeed
```

**Rules:**

- All implementation commits happen **only** inside `.worktrees/feat-orion-bus-substrate-trace-v1`.
- Do **not** copy changed files back to the main workspace checkout except syncing `services/orion-bus/.env` from `.env_example` on the operator machine (`.env` is gitignored).
- PR and push from `feat/orion-bus-substrate-trace-v1` only.

---

## Preflight findings

| Question | Finding |
|----------|---------|
| `orion-bus` runtime | Redis `bus-core` + `redis_exporter` only — no Python app today |
| `orion-bus-tap` | FastAPI WebSocket tap + heartbeat — **not** periodic transport rollup |
| `orion-bus-mirror` | Subscribes `orion:*`, writes every message to SQLite — **anti-pattern** for this PR |
| Grammar schemas | `orion/schemas/grammar.py` — closed kinds; **no `metadata` on atoms** — use `summary` KV + `dimensions` + `payload_ref` |
| Shared publisher | `orion/grammar/publish.py` → `publish_grammar_event` |
| Reference emitters | `services/orion-cortex-exec/app/grammar_emit.py`, `services/orion-biometrics/app/grammar_emit.py` |
| Reference publish | `services/orion-cortex-exec/app/grammar_publish.py` |
| Registry | `GrammarEventV1` already in `orion/schemas/registry.py` — **no change** |
| Grammar channel | `orion:grammar:event` — add `orion-bus` to `producer_services` |
| Context-engineering docs | `docs/context-engineering/` **not present on `main`** — create service-local context files from handoff + templates inline below |
| Stream keys in catalog | `orion:evt:gateway`, `orion:bus:out` **not** listed in `channels.yaml` today — catalog violation role applies to configured observer streams missing from catalog |

### Architecture decision: Option B (locked)

| Option | Verdict |
|--------|---------|
| A — extend bus-tap | **Reject** — tap is operator UI + full-pattern subscribe; wrong ownership and seam |
| B — bus-observer under orion-bus | **Selected** — minimal transport observer co-located with broker compose |
| C — bus-mirror | **Reject** — packet archive contradicts non-goal |

### Semantic roles — v1 implement vs defer

| Role | v1 |
|------|-----|
| `bus_observer_tick_started` | Implement |
| `bus_health_observed` | Implement (`PING`) |
| `bus_stream_depth_observed` | Implement (`XLEN`) |
| `bus_backpressure_observed` | Implement (depth ≥ warning threshold) |
| `bus_configured_stream_uncataloged` | Implement (configured observer stream name ∉ `channels.yaml` — **not** live rogue producers) |
| `bus_observer_tick_completed` | Implement |
| `bus_observer_tick_failed` | Implement (tick exception path) |
| `bus_stream_lag_observed` | **Defer** — needs consumer-group `XINFO` semantics not wired in v1 |
| `bus_schema_validation_failed` | **Defer** — no bus schema validator service |
| `bus_delivery_anomaly_observed` | **Defer** — no subscriber tracking |
| `bus_metrics_scrape_observed` | **Defer** — exporter scrape not required for acceptance |
| `bus_metrics_scrape_failed` | **Defer** | |
| `bus_memory_pressure_observed` | **Defer** | |

### Trace and id conventions

```text
trace_id         = bus.transport:{node_id}:{sample_window_id}
sample_window_id = UTC tick start, e.g. 20260525T170000Z  (strftime %Y%m%dT%H%M%SZ)
atom_id          = {trace_id}:{semantic_role}
event_id         = gev_{sha1(trace_id|event_kind|body_key)[:16]}
source_service   = orion-bus
channel          = orion:grammar:event (when PUBLISH_ORION_BUS_GRAMMAR=true)
```

**`GrammarAtomV1` has no `metadata` dict.** Encode bounded transport facts in `summary` (plus `dimensions`, `source_event_id`, `payload_ref`) as `key=value` tokens — compatible with v1 schema. Never put Redis message values, envelope JSON, or raw payloads in `summary` or `text_value`.

### Observer stream defaults (v1)

Default `BUS_OBSERVER_STREAMS` is **only**:

```text
orion:evt:gateway,orion:bus:out
```

Do **not** default-observe `orion:grammar:event`. The observer publishes there when enabled; observing the same channel is operator-opt-in only (avoids self-referential monitoring loops). `orion:evt:gateway` / `orion:bus:out` may be legacy Redis stream keys, not current catalog channel names — the uncataloged role reflects **configured stream names**, not “rogue message on bus.”

### Bus catalog publish gate

`OrionBusAsync.publish()` enforces catalog when `ORION_BUS_ENFORCE_CATALOG=true` and validates payloads against the channel `schema_id`. Before publishing, confirm `orion:grammar:event` already has `schema_id: GrammarEventV1` (it does on `main`) and add **`orion-bus` only** as producer — no `orion-bus-tap` unless tap actually publishes.

---

## File structure

| Path | Responsibility |
|------|----------------|
| `services/orion-bus/app/__init__.py` | Package marker |
| `services/orion-bus/app/settings.py` | Observer + publish settings |
| `services/orion-bus/app/bus_observer.py` | Redis poll rollup, catalog load |
| `services/orion-bus/app/grammar_emit.py` | `BusTransportGrammarCollector`, `build_bus_transport_grammar_events` |
| `services/orion-bus/app/grammar_publish.py` | Fail-open publish wrapper |
| `services/orion-bus/app/main.py` | Async poll loop entry |
| `services/orion-bus/requirements.txt` | `redis`, `pydantic-settings`, `loguru`, `PyYAML`, `pytest` |
| `services/orion-bus/Dockerfile` | Observer image (repo-root context) |
| `services/orion-bus/docker-compose.yml` | Add optional `bus-observer` service |
| `services/orion-bus/.env_example` | New keys |
| `services/orion-bus/README.md` | Transport trace docs |
| `services/orion-bus/Makefile` | Optional `observer-logs` target |
| `services/orion-bus/AGENT_CONTEXT.md` | Agent orientation |
| `services/orion-bus/SERVICE_PORTS.yaml` | Port manifest (handoff classification) |
| `services/orion-bus/SUBSTRATE_TRACE_MAP.md` | Role → layer map |
| `services/orion-bus/LAYER_PIPELINE_PLAN.md` | Layers 1–11 plan + deferred reducer |
| `services/orion-bus/tests/test_orion_bus_grammar_emit.py` | Emitter tests |
| `services/orion-bus/tests/test_orion_bus_grammar_publish_fail_open.py` | Fail-open tests |
| `services/orion-bus/tests/test_orion_bus_observer_rollup.py` | Observer rollup unit tests |
| `orion/bus/channels.yaml` | `orion-bus` producer on `orion:grammar:event` |
| `scripts/smoke_orion_bus_substrate_trace.sh` | Tests + docker/log/SQL instructions |

**Do not modify:** `bus-core` command/healthcheck, `bus-exporter` image, `orion-bus-tap`, `orion-bus-mirror`, field digester, substrate runtime reducers.

---

# Phase 0 — Context files (no runtime code)

### Task 0: Service context engineering files

**Files:**
- Create: `services/orion-bus/AGENT_CONTEXT.md`
- Create: `services/orion-bus/SERVICE_PORTS.yaml`
- Create: `services/orion-bus/SUBSTRATE_TRACE_MAP.md`
- Create: `services/orion-bus/LAYER_PIPELINE_PLAN.md`

- [ ] **Step 1: Create `AGENT_CONTEXT.md`**

```markdown
# orion-bus — Agent Context

## Role
`orion-bus` is **transport_infrastructure** (mesh nervous-system conduit), not an organ.

## Native contracts
- Redis Streams / pub-sub (`redis_transport`)
- Prometheus metrics via `redis_exporter` (`redis_exporter_metrics`)
- Operator `redis-cli` inspection (`operator_stream_inspection`) — no substrate trace by default

## Substrate trace stance
Emit **bounded periodic transport rollups** only:
health, stream depth, backpressure, catalog violations on configured streams.
Never emit full message payloads or per-packet traces.

## Implementation
- `bus-core` + `bus-exporter` — unchanged Redis stack
- `bus-observer` (optional) — Python sidecar emitting `GrammarEventV1` on `orion:grammar:event` when enabled

## Publishing
Default `PUBLISH_ORION_BUS_GRAMMAR=false`. Fail-open on publish errors.

## Downstream (deferred)
`bus_transport_reducer` → `StateDeltaV1(target_kind=transport_bus)` → field pressure hints.
```

- [ ] **Step 2: Create `SERVICE_PORTS.yaml`** (from handoff; status `draft`)

```yaml
service: orion-bus
status: draft

ports:
  - name: redis_transport
    role: transport_infrastructure
    native_contract: redis_streams
    substrate_trace_required: true
    trace_roles:
      - bus_health_observed
      - bus_stream_depth_observed
      - bus_stream_lag_observed
      - bus_backpressure_observed
      - bus_configured_stream_uncataloged
      - bus_schema_validation_failed
      - bus_delivery_anomaly_observed
    downstream_layers: [1, 2, 3, 4, 5, 6, 10, 11]
    evidence_refs:
      - stream_key
      - sample_window_id
      - observed_at
      - node_id
    redaction:
      never_emit:
        - full_message_payload
        - credential_material
        - private_user_text
        - raw_model_prompt
        - raw_model_completion

  - name: redis_exporter_metrics
    role: transport_infrastructure
    native_contract: prometheus_metrics
    substrate_trace_required: true
    trace_roles:
      - bus_metrics_scrape_observed
      - bus_metrics_scrape_failed
      - bus_memory_pressure_observed
    downstream_layers: [1, 2, 3, 4, 5, 6, 10, 11]
    evidence_refs:
      - scrape_window_id
      - metric_name
      - node_id

  - name: operator_stream_inspection
    role: operator_read_surface
    native_contract: redis_cli
    substrate_trace_required: false
    notes: Manual read-only inspection; no trace required by default.
```

- [ ] **Step 3: Create `SUBSTRATE_TRACE_MAP.md`**

```markdown
# orion-bus substrate trace map

| semantic_role | v1 | atom_type | layer | summary hints |
|---------------|----|-----------|-------|---------------|
| bus_observer_tick_started | yes | signal | transport | sample_window_id, node_id |
| bus_health_observed | yes | observation | transport | redis_ping_ok |
| bus_stream_depth_observed | yes | observation | transport | stream_key, stream_length |
| bus_backpressure_observed | yes | uncertainty_marker | transport | stream_key, threshold, severity |
| bus_configured_stream_uncataloged | yes | uncertainty_marker | transport | stream_key; summary: "Configured observer stream is not declared in channel catalog" |
| bus_observer_tick_completed | yes | signal | transport | streams_observed count |
| bus_observer_tick_failed | yes | uncertainty_marker | transport | error_kind |
| bus_stream_lag_observed | deferred | — | — | needs consumer-group lag |
| bus_schema_validation_failed | deferred | — | — | no validator |
| bus_delivery_anomaly_observed | deferred | — | — | no subscriber map |
| bus_metrics_* | deferred | — | — | exporter scrape optional |
```

- [ ] **Step 4: Create `LAYER_PIPELINE_PLAN.md`**

```markdown
Layer 1: bus-observer emits substrate traces for transport-relevant conditions
Layer 2: traces persist to grammar_events (via orion-sql-writer consumer)
Layer 3: deferred bus_transport_reducer
Layer 4: deferred transport pressure into field lattice
Layer 5: attention may later select transport/backpressure anomalies
Layer 6: self-state may later reflect transport degradation
Layer 7: proposals may later suggest inspect/restart/defer actions
Layer 8: restart/control must require policy/operator review
Layer 9: any bus restart/snapshot/replay dispatch must be dry-run first
Layer 10: transport anomalies and operator actions become feedback
Layer 11: repeated lag/backpressure/schema violations become motifs

Reducer follow-up:
  bus transport traces → bus_transport_reducer → StateDeltaV1(target_kind=transport_bus)
  pressure_hints: bus_health, transport_lag, backpressure, schema_violation_pressure, delivery_confidence
```

- [ ] **Step 5: Commit**

```bash
git add services/orion-bus/AGENT_CONTEXT.md services/orion-bus/SERVICE_PORTS.yaml \
  services/orion-bus/SUBSTRATE_TRACE_MAP.md services/orion-bus/LAYER_PIPELINE_PLAN.md
git commit -m "docs(orion-bus): add context-engineering service files"
```

---

# Phase 1 — Settings, env, compose, Dockerfile

### Task 1: `settings.py` + `.env_example` + `.env` sync + compose

**Files:**
- Create: `services/orion-bus/app/settings.py`
- Modify: `services/orion-bus/.env_example`
- Modify: `services/orion-bus/.env` (local only, gitignored)
- Modify: `services/orion-bus/docker-compose.yml`
- Create: `services/orion-bus/requirements.txt`

- [ ] **Step 1: Write the failing test** — deferred to Task 4 (settings consumed by observer tests). For settings-only, skip dedicated test; verify via `test_orion_bus_observer_rollup` import.

- [ ] **Step 2: Create `requirements.txt`**

```text
redis[hiredis]==5.0.7
pydantic==2.11.5
pydantic-settings==2.5.2
PyYAML==6.0.2
loguru==0.7.2
pytest==8.3.4
pytest-asyncio==0.24.0
```

- [ ] **Step 3: Create `app/settings.py`**

```python
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    SERVICE_NAME: str = Field(default="orion-bus")
    SERVICE_VERSION: str = Field(default="0.1.0")

    REDIS_URL: str = Field(default="redis://bus-core:6379/0", alias="REDIS_URL")

    publish_orion_bus_grammar: bool = Field(False, alias="PUBLISH_ORION_BUS_GRAMMAR")
    grammar_event_channel: str = Field("orion:grammar:event", alias="GRAMMAR_EVENT_CHANNEL")

    bus_observer_enabled: bool = Field(True, alias="BUS_OBSERVER_ENABLED")
    bus_observer_poll_interval_sec: float = Field(10.0, alias="BUS_OBSERVER_POLL_INTERVAL_SEC")
    bus_observer_streams: str = Field(
        "orion:evt:gateway,orion:bus:out",
        alias="BUS_OBSERVER_STREAMS",
    )
    bus_stream_depth_warning: int = Field(25000, alias="BUS_STREAM_DEPTH_WARNING")
    bus_stream_depth_critical: int = Field(100000, alias="BUS_STREAM_DEPTH_CRITICAL")
    bus_observer_node_id: str = Field("athena", alias="BUS_OBSERVER_NODE_ID")

    channels_catalog_path: str = Field(
        "orion/bus/channels.yaml",
        alias="BUS_CHANNELS_CATALOG_PATH",
    )

    @property
    def observer_stream_list(self) -> list[str]:
        return [s.strip() for s in self.bus_observer_streams.split(",") if s.strip()]


settings = Settings()
```

- [ ] **Step 4: Update `.env_example`** (append)

```bash
# Substrate transport traces (bounded rollups; default off)
PUBLISH_ORION_BUS_GRAMMAR=false
GRAMMAR_EVENT_CHANNEL=orion:grammar:event
BUS_OBSERVER_ENABLED=true
BUS_OBSERVER_POLL_INTERVAL_SEC=10
BUS_OBSERVER_STREAMS=orion:evt:gateway,orion:bus:out
# Optional operator override (self-referential if publish enabled):
# BUS_OBSERVER_STREAMS=orion:evt:gateway,orion:bus:out,orion:grammar:event
BUS_STREAM_DEPTH_WARNING=25000
BUS_STREAM_DEPTH_CRITICAL=100000
BUS_OBSERVER_NODE_ID=athena
BUS_CHANNELS_CATALOG_PATH=orion/bus/channels.yaml
```

- [ ] **Step 5: Sync `.env`** — copy the same keys into `services/orion-bus/.env` in the worktree (not committed).

- [ ] **Step 6: Append `bus-observer` to `docker-compose.yml`**

```yaml
  bus-observer:
    build:
      context: ../..
      dockerfile: services/orion-bus/Dockerfile
    container_name: orion-${PROJECT}-bus-observer
    env_file:
      - .env
    environment:
      REDIS_URL: redis://bus-core:6379/0
      PUBLISH_ORION_BUS_GRAMMAR: ${PUBLISH_ORION_BUS_GRAMMAR:-false}
      GRAMMAR_EVENT_CHANNEL: ${GRAMMAR_EVENT_CHANNEL:-orion:grammar:event}
      BUS_OBSERVER_ENABLED: ${BUS_OBSERVER_ENABLED:-true}
      BUS_OBSERVER_POLL_INTERVAL_SEC: ${BUS_OBSERVER_POLL_INTERVAL_SEC:-10}
      BUS_OBSERVER_STREAMS: ${BUS_OBSERVER_STREAMS:-orion:evt:gateway,orion:bus:out}
      BUS_STREAM_DEPTH_WARNING: ${BUS_STREAM_DEPTH_WARNING:-25000}
      BUS_STREAM_DEPTH_CRITICAL: ${BUS_STREAM_DEPTH_CRITICAL:-100000}
      BUS_OBSERVER_NODE_ID: ${BUS_OBSERVER_NODE_ID:-athena}
    depends_on:
      bus-core:
        condition: service_healthy
    restart: unless-stopped
    networks:
      - app-net
```

- [ ] **Step 7: Commit**

```bash
git add services/orion-bus/app/settings.py services/orion-bus/requirements.txt \
  services/orion-bus/.env_example services/orion-bus/docker-compose.yml
git commit -m "feat(orion-bus): add bus-observer settings and compose service"
```

### Task 2: Dockerfile + `main.py` skeleton

**Files:**
- Create: `services/orion-bus/Dockerfile`
- Create: `services/orion-bus/app/__init__.py`
- Create: `services/orion-bus/app/main.py`

- [ ] **Step 1: Create `Dockerfile`** (mirror bus-tap repo-root build)

```dockerfile
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
RUN pip3 install --no-cache-dir --upgrade pip
COPY services/orion-bus/requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
COPY orion /app/orion
COPY services/orion-bus /app
CMD ["python", "-m", "app.main"]
```

- [ ] **Step 2: Create `app/main.py`**

```python
from __future__ import annotations

import asyncio

from loguru import logger

from app.bus_observer import run_bus_observer_loop
from app.settings import settings


def main() -> None:
    if not settings.bus_observer_enabled:
        logger.info("bus-observer disabled (BUS_OBSERVER_ENABLED=false); exiting")
        return
    asyncio.run(run_bus_observer_loop())


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add services/orion-bus/Dockerfile services/orion-bus/app/__init__.py services/orion-bus/app/main.py
git commit -m "feat(orion-bus): add bus-observer container entrypoint"
```

### Task 3: Bus channel catalog

**Files:**
- Modify: `orion/bus/channels.yaml` (`orion:grammar:event` producers)

- [ ] **Step 1: Verify existing catalog entry before editing**

Confirm `orion:grammar:event` already has:

```yaml
schema_id: GrammarEventV1
message_kind: grammar.event.v1
```

(On `main` this is already true.) Confirm `GrammarEventV1` resolves in `orion/schemas/registry.py` (no registry edit needed).

- [ ] **Step 2: Add `orion-bus` to `producer_services` only**

Under `name: "orion:grammar:event"`, append `- orion-bus`. **Do not** add `orion-bus-tap` — tap does not publish grammar in v1. Only the service that actually calls `publish_grammar_event` with `source_name=orion-bus` must be listed.

- [ ] **Step 3: Preflight publish path**

When `PUBLISH_ORION_BUS_GRAMMAR=true`, `OrionBusAsync.publish()` must pass catalog + schema validation. If local dev sets `ORION_BUS_ENFORCE_CATALOG=true`, a missing producer entry or wrong schema will fail publish (observer should still fail-open at the wrapper).

- [ ] **Step 4: Commit**

```bash
git add orion/bus/channels.yaml
git commit -m "chore(bus): register orion-bus as grammar event producer"
```

**Registry:** `GrammarEventV1` already registered — **no** `orion/schemas/registry.py` change.

---

# Phase 2 — Grammar emitter (TDD)

### Task 4: `grammar_emit.py` + tests

**Files:**
- Create: `services/orion-bus/app/grammar_emit.py`
- Create: `services/orion-bus/tests/test_orion_bus_grammar_emit.py`
- Create: `services/orion-bus/tests/__init__.py` (empty)

- [ ] **Step 1: Write the failing test**

Create `services/orion-bus/tests/test_orion_bus_grammar_emit.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone
from typing import get_args

import pytest

from app.grammar_emit import (
    BusTransportGrammarCollector,
    build_bus_transport_grammar_events,
    bus_transport_trace_id,
)
from orion.schemas.grammar import AtomType, GrammarEventKind, RelationType

FIXED_OBS = datetime(2026, 5, 25, 17, 0, 0, tzinfo=timezone.utc)
WINDOW = "20260525T170000Z"
NODE = "athena"


def test_trace_id_format() -> None:
    assert bus_transport_trace_id(NODE, WINDOW) == f"bus.transport:{NODE}:{WINDOW}"


def test_builds_transport_rollup_trace() -> None:
    collector = BusTransportGrammarCollector(
        node_id=NODE,
        sample_window_id=WINDOW,
        observed_at=FIXED_OBS,
        code_version="0.1.0",
    )
    collector.record_tick_started()
    collector.record_health_observed(redis_ping_ok=True)
    collector.record_stream_depth(stream_key="orion:evt:gateway", stream_length=123)
    collector.record_backpressure(
        stream_key="orion:evt:gateway",
        stream_length=50000,
        threshold=25000,
        severity="warning",
    )
    collector.record_uncataloged_stream(stream_key="orion:evt:gateway")
    collector.record_tick_completed(streams_observed=3)

    events = build_bus_transport_grammar_events(collector)
    assert events
    kinds = {e.event_kind for e in events}
    assert kinds <= set(get_args(GrammarEventKind))
    assert "trace_started" in kinds
    assert "trace_ended" in kinds

    roles = {e.atom.semantic_role for e in events if e.atom}
    assert roles >= {
        "bus_observer_tick_started",
        "bus_health_observed",
        "bus_stream_depth_observed",
        "bus_backpressure_observed",
        "bus_configured_stream_uncataloged",
        "bus_observer_tick_completed",
    }

    uncataloged = next(
        e.atom for e in events if e.atom and e.atom.semantic_role == "bus_configured_stream_uncataloged"
    )
    assert "not declared in channel catalog" in uncataloged.summary.lower()

    for event in events:
        if event.atom:
            assert event.atom.atom_type in get_args(AtomType)
        if event.edge:
            assert event.edge.relation_type in get_args(RelationType)

    health = next(e.atom for e in events if e.atom and e.atom.semantic_role == "bus_health_observed")
    assert "redis_ping_ok=true" in health.summary
    assert "node_id=athena" in health.summary


def test_no_payload_blobs_in_summaries() -> None:
    collector = BusTransportGrammarCollector(
        node_id=NODE,
        sample_window_id=WINDOW,
        observed_at=FIXED_OBS,
    )
    collector.record_tick_started()
    collector.record_stream_depth(stream_key="orion:bus:out", stream_length=1)
    collector.record_tick_completed(streams_observed=1)
    events = build_bus_transport_grammar_events(collector)
    for event in events:
        if event.atom:
            assert "envelope" not in event.atom.summary.lower()
            assert "payload" not in event.atom.summary.lower()


def test_summaries_never_include_redis_values_or_envelope_material() -> None:
    """Redaction guard: summaries are bounded KV hints, not packet dumps."""
    collector = BusTransportGrammarCollector(
        node_id=NODE,
        sample_window_id=WINDOW,
        observed_at=FIXED_OBS,
    )
    collector.record_tick_started()
    collector.record_stream_depth(stream_key="orion:bus:out", stream_length=42)
    collector.record_tick_completed(streams_observed=1)
    events = build_bus_transport_grammar_events(collector)
    forbidden_fragments = (
        "{",
        "}",
        '"kind"',
        "grammar.event",
        "BaseEnvelope",
        "XREAD",
        "XRANGE",
        "redis://",
        "password",
    )
    for event in events:
        if not event.atom:
            continue
        summary = event.atom.summary
        assert event.atom.text_value is None
        for frag in forbidden_fragments:
            assert frag not in summary, f"forbidden fragment {frag!r} in {summary!r}"
        # stream_length is allowed as a bounded count, not raw stream entry bodies
        assert "stream_length=42" in summary or event.atom.semantic_role != "bus_stream_depth_observed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/scripts/Orion-Sapienform/.worktrees/feat-orion-bus-substrate-trace-v1 && PYTHONPATH=services/orion-bus:. pytest services/orion-bus/tests/test_orion_bus_grammar_emit.py -v`

Expected: FAIL — `ModuleNotFoundError: app.grammar_emit`

- [ ] **Step 3: Implement `grammar_emit.py`**

```python
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from orion.schemas.grammar import (
    GrammarAtomV1,
    GrammarEdgeV1,
    GrammarEventV1,
    GrammarProvenanceV1,
)


def bus_transport_trace_id(node_id: str, sample_window_id: str) -> str:
    return f"bus.transport:{node_id}:{sample_window_id}"


def _hash_id(*parts: object, prefix: str) -> str:
    raw = "|".join(str(p) for p in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


@dataclass
class BusTransportGrammarCollector:
    node_id: str
    sample_window_id: str
    observed_at: datetime
    code_version: str | None = None
    _atoms: dict[str, GrammarAtomV1] = field(default_factory=dict)
    _edge_specs: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def trace_id(self) -> str:
        return bus_transport_trace_id(self.node_id, self.sample_window_id)

    def _provenance(self, payload_ref: str) -> GrammarProvenanceV1:
        return GrammarProvenanceV1(
            source_service="orion-bus",
            source_component="bus_transport_grammar_emit",
            source_event_id=f"{self.node_id}:{self.sample_window_id}",
            source_trace_id=self.trace_id,
            source_payload_ref=payload_ref,
            code_version=self.code_version,
        )

    def _atom_id(self, role: str) -> str:
        return f"{self.trace_id}:{role}"

    def _put_atom(self, role: str, atom: GrammarAtomV1) -> None:
        self._atoms[role] = atom

    def record_tick_started(self) -> None:
        self._put_atom(
            "bus_observer_tick_started",
            GrammarAtomV1(
                atom_id=self._atom_id("bus_observer_tick_started"),
                trace_id=self.trace_id,
                atom_type="signal",
                semantic_role="bus_observer_tick_started",
                layer="transport",
                dimensions=["bus", "transport", "observer"],
                summary=(
                    f"Bus observer tick started node_id={self.node_id} "
                    f"sample_window_id={self.sample_window_id}"
                ),
                confidence=1.0,
                salience=0.5,
                source_event_id=self.sample_window_id,
                payload_ref=f"bus.transport.tick:{self.sample_window_id}",
            ),
        )

    def record_tick_completed(self, *, streams_observed: int) -> None:
        self._put_atom(
            "bus_observer_tick_completed",
            GrammarAtomV1(
                atom_id=self._atom_id("bus_observer_tick_completed"),
                trace_id=self.trace_id,
                atom_type="signal",
                semantic_role="bus_observer_tick_completed",
                layer="transport",
                dimensions=["bus", "transport", "observer"],
                summary=(
                    f"Bus observer tick completed streams_observed={streams_observed} "
                    f"sample_window_id={self.sample_window_id}"
                ),
                confidence=1.0,
                salience=0.5,
                source_event_id=self.sample_window_id,
                payload_ref=f"bus.transport.tick_done:{self.sample_window_id}",
            ),
        )
        if "bus_observer_tick_started" in self._atoms:
            self._edge_specs.append(
                (
                    self._atoms["bus_observer_tick_started"].atom_id,
                    self._atoms["bus_observer_tick_completed"].atom_id,
                    "temporal_successor",
                )
            )

    def record_tick_failed(self, *, error_kind: str) -> None:
        self._put_atom(
            "bus_observer_tick_failed",
            GrammarAtomV1(
                atom_id=self._atom_id("bus_observer_tick_failed"),
                trace_id=self.trace_id,
                atom_type="uncertainty_marker",
                semantic_role="bus_observer_tick_failed",
                layer="transport",
                dimensions=["bus", "transport", "failure"],
                summary=f"Bus observer tick failed error_kind={error_kind}",
                confidence=0.9,
                salience=0.9,
                source_event_id=self.sample_window_id,
                payload_ref=f"bus.transport.tick_failed:{self.sample_window_id}",
            ),
        )

    def record_health_observed(self, *, redis_ping_ok: bool) -> None:
        self._put_atom(
            "bus_health_observed",
            GrammarAtomV1(
                atom_id=self._atom_id("bus_health_observed"),
                trace_id=self.trace_id,
                atom_type="observation",
                semantic_role="bus_health_observed",
                layer="transport",
                dimensions=["bus", "health", "redis"],
                summary=(
                    f"Redis bus core health probe node_id={self.node_id} "
                    f"redis_ping_ok={str(redis_ping_ok).lower()} "
                    f"sample_window_id={self.sample_window_id}"
                ),
                confidence=1.0,
                salience=0.8,
                source_event_id=self.sample_window_id,
                payload_ref=f"bus.transport.health:{self.sample_window_id}",
            ),
        )
        if "bus_observer_tick_started" in self._atoms:
            self._edge_specs.append(
                (
                    self._atoms["bus_observer_tick_started"].atom_id,
                    self._atoms["bus_health_observed"].atom_id,
                    "contains",
                )
            )

    def record_stream_depth(self, *, stream_key: str, stream_length: int) -> None:
        role = f"bus_stream_depth_observed:{stream_key}"
        self._put_atom(
            role,
            GrammarAtomV1(
                atom_id=self._atom_id(role),
                trace_id=self.trace_id,
                atom_type="observation",
                semantic_role="bus_stream_depth_observed",
                layer="transport",
                dimensions=["bus", "stream", "depth"],
                summary=(
                    f"Observed Redis stream depth stream_key={stream_key} "
                    f"stream_length={stream_length} sample_window_id={self.sample_window_id}"
                ),
                confidence=1.0,
                salience=0.7,
                source_event_id=self.sample_window_id,
                payload_ref=f"bus.transport.depth:{stream_key}:{self.sample_window_id}",
            ),
        )
        if "bus_health_observed" in self._atoms:
            self._edge_specs.append(
                (
                    self._atoms["bus_health_observed"].atom_id,
                    self._atoms[role].atom_id,
                    "contains",
                )
            )

    def record_backpressure(
        self,
        *,
        stream_key: str,
        stream_length: int,
        threshold: int,
        severity: str,
    ) -> None:
        role = f"bus_backpressure_observed:{stream_key}"
        self._put_atom(
            role,
            GrammarAtomV1(
                atom_id=self._atom_id(role),
                trace_id=self.trace_id,
                atom_type="uncertainty_marker",
                semantic_role="bus_backpressure_observed",
                layer="transport",
                dimensions=["bus", "backpressure", "stream"],
                summary=(
                    f"Bus stream depth exceeded threshold stream_key={stream_key} "
                    f"stream_length={stream_length} threshold={threshold} severity={severity}"
                ),
                confidence=0.95,
                salience=0.85,
                source_event_id=self.sample_window_id,
                payload_ref=f"bus.transport.backpressure:{stream_key}:{self.sample_window_id}",
            ),
        )
        depth_role = f"bus_stream_depth_observed:{stream_key}"
        if depth_role in self._atoms:
            self._edge_specs.append(
                (
                    self._atoms[depth_role].atom_id,
                    self._atoms[role].atom_id,
                    "derived_from",
                )
            )

    def record_uncataloged_stream(self, *, stream_key: str) -> None:
        role = f"bus_configured_stream_uncataloged:{stream_key}"
        self._put_atom(
            role,
            GrammarAtomV1(
                atom_id=self._atom_id(role),
                trace_id=self.trace_id,
                atom_type="uncertainty_marker",
                semantic_role="bus_configured_stream_uncataloged",
                layer="transport",
                dimensions=["bus", "catalog", "contract"],
                summary=(
                    f"Configured observer stream is not declared in channel catalog "
                    f"stream_key={stream_key} source=bus_observer "
                    f"sample_window_id={self.sample_window_id}"
                ),
                confidence=0.9,
                salience=0.8,
                source_event_id=self.sample_window_id,
                payload_ref=f"bus.transport.uncataloged_stream:{stream_key}",
            ),
        )


def _event(
    *,
    event_kind: str,
    trace_id: str,
    emitted_at: datetime,
    observed_at: datetime,
    provenance: GrammarProvenanceV1,
    atom: GrammarAtomV1 | None = None,
    edge: GrammarEdgeV1 | None = None,
    parent_event_id: str | None = None,
    root_event_id: str | None = None,
    layer: str | None = None,
    dimensions: list[str] | None = None,
) -> GrammarEventV1:
    body_key = atom.atom_id if atom else edge.edge_id if edge else uuid4().hex
    return GrammarEventV1(
        event_id=_hash_id(trace_id, event_kind, body_key, prefix="gev"),
        event_kind=event_kind,  # type: ignore[arg-type]
        trace_id=trace_id,
        parent_event_id=parent_event_id,
        root_event_id=root_event_id,
        emitted_at=emitted_at,
        observed_at=observed_at,
        layer=layer,
        dimensions=dimensions or [],
        atom=atom,
        edge=edge,
        provenance=provenance,
    )


def build_bus_transport_grammar_events(
    collector: BusTransportGrammarCollector,
) -> list[GrammarEventV1]:
    observed_at = collector.observed_at
    emitted_at = datetime.now(timezone.utc)
    trace_id = collector.trace_id
    provenance = collector._provenance(f"bus.transport.trace:{collector.sample_window_id}")

    root = _event(
        event_kind="trace_started",
        trace_id=trace_id,
        emitted_at=emitted_at,
        observed_at=observed_at,
        provenance=provenance,
        layer="transport",
        dimensions=["bus", "transport"],
    )
    root_id = root.event_id
    events: list[GrammarEventV1] = [root]

    for atom in collector._atoms.values():
        events.append(
            _event(
                event_kind="atom_emitted",
                trace_id=trace_id,
                emitted_at=emitted_at,
                observed_at=observed_at,
                provenance=provenance,
                atom=atom,
                parent_event_id=root_id,
                root_event_id=root_id,
                layer=atom.layer,
                dimensions=atom.dimensions,
            )
        )

    for from_atom_id, to_atom_id, relation_type in collector._edge_specs:
        edge = GrammarEdgeV1(
            edge_id=f"{trace_id}:edge:{from_atom_id}:{relation_type}:{to_atom_id}",
            trace_id=trace_id,
            from_atom_id=from_atom_id,
            to_atom_id=to_atom_id,
            relation_type=relation_type,  # type: ignore[arg-type]
            confidence=0.9,
            salience=0.6,
            evidence_event_ids=[collector.sample_window_id],
        )
        events.append(
            _event(
                event_kind="edge_emitted",
                trace_id=trace_id,
                emitted_at=emitted_at,
                observed_at=observed_at,
                provenance=provenance,
                edge=edge,
                parent_event_id=root_id,
                root_event_id=root_id,
                layer="transport",
                dimensions=["bus", "transport"],
            )
        )

    events.append(
        _event(
            event_kind="trace_ended",
            trace_id=trace_id,
            emitted_at=datetime.now(timezone.utc),
            observed_at=observed_at,
            provenance=provenance,
            parent_event_id=root_id,
            root_event_id=root_id,
            layer="transport",
            dimensions=["bus", "transport"],
        )
    )
    return events
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=services/orion-bus:. pytest services/orion-bus/tests/test_orion_bus_grammar_emit.py -q`

Expected: PASS (includes `test_summaries_never_include_redis_values_or_envelope_material`)

- [ ] **Step 5: Commit**

```bash
git add services/orion-bus/app/grammar_emit.py services/orion-bus/tests/
git commit -m "feat(orion-bus): add transport grammar emitter"
```

---

# Phase 3 — Publish fail-open + observer rollup

### Task 5: `grammar_publish.py` + fail-open test

**Files:**
- Create: `services/orion-bus/app/grammar_publish.py`
- Create: `services/orion-bus/tests/test_orion_bus_grammar_publish_fail_open.py`

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from app.grammar_emit import BusTransportGrammarCollector, build_bus_transport_grammar_events
from app.grammar_publish import publish_bus_transport_grammar_trace


@pytest.mark.asyncio
async def test_publish_failure_is_non_fatal() -> None:
    bus = AsyncMock()
    bus.publish = AsyncMock(side_effect=RuntimeError("bus down"))
    collector = BusTransportGrammarCollector(
        node_id="athena",
        sample_window_id="20260525T170000Z",
        observed_at=datetime.now(timezone.utc),
        code_version="0.1.0",
    )
    collector.record_tick_started()
    collector.record_health_observed(redis_ping_ok=True)
    collector.record_tick_completed(streams_observed=0)
    events = build_bus_transport_grammar_events(collector)
    await publish_bus_transport_grammar_trace(
        bus,
        events,
        channel="orion:grammar:event",
        source_name="orion-bus",
        enabled=True,
    )
```

- [ ] **Step 2: Run test — expect FAIL** (`ModuleNotFoundError`)

Run: `PYTHONPATH=services/orion-bus:. pytest services/orion-bus/tests/test_orion_bus_grammar_publish_fail_open.py -v`

- [ ] **Step 3: Implement `grammar_publish.py`**

```python
from __future__ import annotations

import logging
from typing import Any

from orion.grammar.publish import publish_grammar_event
from orion.schemas.grammar import GrammarEventV1

logger = logging.getLogger("orion.bus.grammar_publish")


async def publish_bus_transport_grammar_trace(
    bus: Any,
    events: list[GrammarEventV1],
    *,
    channel: str,
    source_name: str = "orion-bus",
    enabled: bool = True,
) -> None:
    if not enabled or not events:
        return
    for event in events:
        try:
            await publish_grammar_event(
                bus,
                event,
                source_name=source_name,
                correlation_id=None,
                channel=channel,
            )
        except Exception:
            logger.warning(
                "bus_grammar_publish_failed trace_id=%s event_kind=%s",
                event.trace_id,
                event.event_kind,
                exc_info=True,
            )
```

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add services/orion-bus/app/grammar_publish.py \
  services/orion-bus/tests/test_orion_bus_grammar_publish_fail_open.py
git commit -m "feat(orion-bus): fail-open grammar publish"
```

### Task 6: `bus_observer.py` + rollup tests

**Files:**
- Create: `services/orion-bus/app/bus_observer.py`
- Create: `services/orion-bus/tests/test_orion_bus_observer_rollup.py`

- [ ] **Step 1: Write the failing test** (mock Redis, no docker)

```python
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.bus_observer import ObserverRollup, build_rollup_from_redis_snapshot
from app.settings import Settings


@pytest.mark.asyncio
async def test_rollup_records_depth_and_backpressure() -> None:
    # Use Python field names (not env aliases) — model_copy(update=...) does not re-validate aliases.
    settings = Settings(
        bus_observer_node_id="athena",
        bus_stream_depth_warning=100,
        bus_stream_depth_critical=1000,
        bus_observer_streams="orion:evt:gateway",
    )
    snapshot = {
        "ping_ok": True,
        "stream_lengths": {"orion:evt:gateway": 150},
        "catalog_names": {"orion:grammar:event"},
    }
    rollup = build_rollup_from_redis_snapshot(
        settings=settings,
        snapshot=snapshot,
        observed_at=datetime(2026, 5, 25, 17, 0, 0, tzinfo=timezone.utc),
        sample_window_id="20260525T170000Z",
    )
    assert rollup.ping_ok is True
    assert rollup.stream_lengths["orion:evt:gateway"] == 150
    collector = rollup.to_collector(code_version="0.1.0")
    roles = {a.semantic_role for a in collector._atoms.values()}
    assert "bus_stream_depth_observed" in roles
    assert "bus_backpressure_observed" in roles
    assert "bus_configured_stream_uncataloged" in roles


@pytest.mark.asyncio
async def test_run_tick_publishes_when_enabled() -> None:
    with patch("app.bus_observer._fetch_redis_snapshot", new_callable=AsyncMock) as snap:
        snap.return_value = {
            "ping_ok": True,
            "stream_lengths": {"orion:evt:gateway": 1},
            "catalog_names": {"orion:evt:gateway"},
        }
        bus = AsyncMock()
        from app.bus_observer import run_observer_tick
        from app.settings import settings as default_settings

        s = default_settings.model_copy(
            update={
                "publish_orion_bus_grammar": True,
                "bus_observer_streams": "orion:evt:gateway",
            }
        )
        await run_observer_tick(bus=bus, settings=s)
        assert bus.publish.await_count >= 1
```

- [ ] **Step 2: Run test — expect FAIL**

- [ ] **Step 3: Implement `bus_observer.py`** (core logic)

```python
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from redis import asyncio as aioredis

from orion.core.bus.async_service import OrionBusAsync

from app.grammar_emit import BusTransportGrammarCollector, build_bus_transport_grammar_events
from app.grammar_publish import publish_bus_transport_grammar_trace
from app.settings import Settings, settings


def _sample_window_id(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_channel_catalog_names(catalog_path: str) -> set[str]:
    path = Path(catalog_path)
    if not path.is_file():
        repo_root = Path(__file__).resolve().parents[3]
        path = repo_root / catalog_path
    if not path.is_file():
        return set()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names: set[str] = set()
    for ch in data.get("channels") or []:
        if isinstance(ch, dict) and ch.get("name"):
            names.add(str(ch["name"]))
    return names


@dataclass
class ObserverRollup:
    node_id: str
    sample_window_id: str
    observed_at: datetime
    ping_ok: bool
    stream_lengths: dict[str, int] = field(default_factory=dict)
    uncataloged_streams: list[str] = field(default_factory=list)
    backpressure: list[tuple[str, int, int, str]] = field(default_factory=list)

    def to_collector(self, *, code_version: str | None) -> BusTransportGrammarCollector:
        c = BusTransportGrammarCollector(
            node_id=self.node_id,
            sample_window_id=self.sample_window_id,
            observed_at=self.observed_at,
            code_version=code_version,
        )
        c.record_tick_started()
        c.record_health_observed(redis_ping_ok=self.ping_ok)
        for stream_key, length in sorted(self.stream_lengths.items()):
            c.record_stream_depth(stream_key=stream_key, stream_length=length)
        for stream_key, length, threshold, severity in self.backpressure:
            c.record_backpressure(
                stream_key=stream_key,
                stream_length=length,
                threshold=threshold,
                severity=severity,
            )
        for stream_key in self.uncataloged_streams:
            c.record_uncataloged_stream(stream_key=stream_key)
        c.record_tick_completed(streams_observed=len(self.stream_lengths))
        return c


def build_rollup_from_redis_snapshot(
    *,
    settings: Settings,
    snapshot: dict[str, Any],
    observed_at: datetime,
    sample_window_id: str,
) -> ObserverRollup:
    ping_ok = bool(snapshot.get("ping_ok"))
    stream_lengths: dict[str, int] = dict(snapshot.get("stream_lengths") or {})
    catalog_names: set[str] = set(snapshot.get("catalog_names") or [])
    uncataloged = [
        sk
        for sk in settings.observer_stream_list
        if sk not in catalog_names
    ]
    backpressure: list[tuple[str, int, int, str]] = []
    for stream_key, length in stream_lengths.items():
        if length >= settings.bus_stream_depth_critical:
            backpressure.append(
                (stream_key, length, settings.bus_stream_depth_critical, "critical")
            )
        elif length >= settings.bus_stream_depth_warning:
            backpressure.append(
                (stream_key, length, settings.bus_stream_depth_warning, "warning")
            )
    return ObserverRollup(
        node_id=settings.bus_observer_node_id,
        sample_window_id=sample_window_id,
        observed_at=observed_at,
        ping_ok=ping_ok,
        stream_lengths=stream_lengths,
        uncataloged_streams=uncataloged,
        backpressure=backpressure,
    )


async def _fetch_redis_snapshot(settings: Settings) -> dict[str, Any]:
    client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    try:
        ping_ok = (await client.ping()) is True
        stream_lengths: dict[str, int] = {}
        for stream_key in settings.observer_stream_list:
            try:
                stream_lengths[stream_key] = int(await client.xlen(stream_key))
            except Exception:
                stream_lengths[stream_key] = 0
        catalog_names = load_channel_catalog_names(settings.channels_catalog_path)
        return {
            "ping_ok": ping_ok,
            "stream_lengths": stream_lengths,
            "catalog_names": catalog_names,
        }
    finally:
        await client.aclose()


async def run_observer_tick(*, bus: Any, settings: Settings) -> None:
    observed_at = datetime.now(timezone.utc)
    window = _sample_window_id(observed_at)
    try:
        snapshot = await _fetch_redis_snapshot(settings)
        rollup = build_rollup_from_redis_snapshot(
            settings=settings,
            snapshot=snapshot,
            observed_at=observed_at,
            sample_window_id=window,
        )
        collector = rollup.to_collector(code_version=settings.SERVICE_VERSION)
        events = build_bus_transport_grammar_events(collector)
        await publish_bus_transport_grammar_trace(
            bus,
            events,
            channel=settings.grammar_event_channel,
            source_name=settings.SERVICE_NAME,
            enabled=settings.publish_orion_bus_grammar,
        )
        logger.debug(
            "bus observer tick ok window={} streams={}",
            window,
            len(rollup.stream_lengths),
        )
    except Exception as exc:
        logger.warning("bus observer tick failed: {}", exc, exc_info=True)
        fail_collector = BusTransportGrammarCollector(
            node_id=settings.bus_observer_node_id,
            sample_window_id=window,
            observed_at=observed_at,
            code_version=settings.SERVICE_VERSION,
        )
        fail_collector.record_tick_started()
        fail_collector.record_tick_failed(error_kind=type(exc).__name__)
        events = build_bus_transport_grammar_events(fail_collector)
        await publish_bus_transport_grammar_trace(
            bus,
            events,
            channel=settings.grammar_event_channel,
            source_name=settings.SERVICE_NAME,
            enabled=settings.publish_orion_bus_grammar,
        )


async def run_bus_observer_loop() -> None:
    bus = OrionBusAsync(settings.REDIS_URL)
    await bus.connect()
    logger.info(
        "bus-observer started node={} interval={}s publish={}",
        settings.bus_observer_node_id,
        settings.bus_observer_poll_interval_sec,
        settings.publish_orion_bus_grammar,
    )
    try:
        while True:
            await run_observer_tick(bus=bus, settings=settings)
            await asyncio.sleep(settings.bus_observer_poll_interval_sec)
    finally:
        await bus.close()
```

- [ ] **Step 4: Run rollup + publish tests**

Run:

```bash
PYTHONPATH=services/orion-bus:. pytest \
  services/orion-bus/tests/test_orion_bus_grammar_emit.py \
  services/orion-bus/tests/test_orion_bus_grammar_publish_fail_open.py \
  services/orion-bus/tests/test_orion_bus_observer_rollup.py \
  -q
```

Expected: PASS

- [ ] **Step 5: Compileall**

```bash
PYTHONPATH=. python -m compileall services/orion-bus -q
```

- [ ] **Step 6: Commit**

```bash
git add services/orion-bus/app/bus_observer.py \
  services/orion-bus/tests/test_orion_bus_observer_rollup.py
git commit -m "feat(orion-bus): add periodic transport observer rollup"
```

---

# Phase 4 — README, Makefile, smoke script

### Task 7: README + Makefile

**Files:**
- Modify: `services/orion-bus/README.md`
- Modify: `services/orion-bus/Makefile`

- [ ] **Step 1: Add README section "Substrate transport traces"**

Document: `bus-observer`, env flags, default publish off, bounded rollups only, deferred reducer, smoke script path.

- [ ] **Step 2: Add Makefile targets**

```makefile
OBSERVER ?= orion-${PROJECT}-bus-observer

observer-logs:
	@docker logs --tail=200 $(OBSERVER) 2>&1

observer-ps:
	@docker ps --format 'table {{.Names}}\t{{.Image}}' | awk 'NR==1 || /bus/'
```

- [ ] **Step 3: Commit**

```bash
git add services/orion-bus/README.md services/orion-bus/Makefile
git commit -m "docs(orion-bus): document transport substrate traces"
```

### Task 8: Smoke script

**Files:**
- Create: `scripts/smoke_orion_bus_substrate_trace.sh`

- [ ] **Step 1: Create smoke script**

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"

echo "=== 1. Unit tests ==="
PYTHONPATH=services/orion-bus:. "$PY" -m pytest \
  services/orion-bus/tests/test_orion_bus_grammar_emit.py \
  services/orion-bus/tests/test_orion_bus_grammar_publish_fail_open.py \
  services/orion-bus/tests/test_orion_bus_observer_rollup.py \
  -q

echo "=== 2. Compileall ==="
PYTHONPATH=. "$PY" -m compileall services/orion-bus -q

echo "=== 3. Docker bus containers ==="
docker ps --format 'table {{.Names}}\t{{.Image}}' | grep -Ei 'bus' || true

OBSERVER="$(docker ps --format '{{.Names}}' | grep -E 'bus-observer' | head -1 || true)"
if [ -n "${OBSERVER}" ]; then
  echo "=== 4. Observer logs (tail 200) ==="
  docker logs --tail=200 "${OBSERVER}" 2>&1 | grep -Ei 'bus|trace|grammar|failed|error' || true
else
  echo "WARN: no bus-observer container found; start with: cd services/orion-bus && make up"
fi

echo "=== 5. SQL proof (run against grammar_events DB) ==="
cat <<'SQL'
select
    created_at
  , source_service
  , trace_id
  , event_json::jsonb #>> '{atom,semantic_role}' as semantic_role
  , event_json::jsonb #>> '{atom,summary}' as summary
from grammar_events
where source_service in ('orion-bus', 'orion-bus-tap')
  and trace_id like 'bus.transport:%'
order by created_at desc
limit 30;
SQL
```

- [ ] **Step 2: chmod +x and commit**

```bash
chmod +x scripts/smoke_orion_bus_substrate_trace.sh
git add scripts/smoke_orion_bus_substrate_trace.sh
git commit -m "chore(orion-bus): add substrate trace smoke script"
```

---

# Phase 5 — Verification (mandatory before PR)

### Task 9: Full verification pass

- [ ] **Step 1: Run unit tests**

```bash
cd .worktrees/feat-orion-bus-substrate-trace-v1
PYTHONPATH=services/orion-bus:. pytest \
  services/orion-bus/tests/test_orion_bus_grammar_emit.py \
  services/orion-bus/tests/test_orion_bus_grammar_publish_fail_open.py \
  services/orion-bus/tests/test_orion_bus_observer_rollup.py \
  -q
```

Expected: all passed

- [ ] **Step 2: Optional live stack** (operator)

```bash
cd services/orion-bus
# Ensure .env has PUBLISH_ORION_BUS_GRAMMAR=true for proof only
make up
../../scripts/smoke_orion_bus_substrate_trace.sh
```

- [ ] **Step 3: Confirm `bus-core` / `bus-exporter` unchanged** — diff only adds `bus-observer`; Redis image/command identical.

---

# Phase 6 — Code review subagent + fixes

**REQUIRED SUB-SKILL:** `requesting-code-review` — dispatch code-reviewer subagent on the full branch diff vs `origin/main`.

- [ ] **Step 1: Dispatch subagent** with prompt: review `feat/orion-bus-substrate-trace-v1` for (1) no per-message traces, (2) redaction in summaries, (3) fail-open publish, (4) closed grammar enums, (5) no bleed to main checkout, (6) channel catalog accuracy.

- [ ] **Step 2: Fix all blocking issues** — new commits on feature branch only.

- [ ] **Step 3: Re-run Task 9 tests after fixes**

---

# Phase 7 — PR report, push, create PR

### Task 10: PR markdown report

**Files:**
- Create: `docs/superpowers/pr-reports/2026-05-25-orion-bus-substrate-trace-v1-pr.md`

- [ ] **Step 1: Write PR report** using template:

```markdown
# PR: Orion Bus Substrate Trace — Transport Legibility v1

**Branch:** `feat/orion-bus-substrate-trace-v1`
**Base:** `main`

## service role
transport_infrastructure

## native contract
Redis core + Redis exporter (+ optional bus-observer sidecar)

## substrate trace stance
bounded transport rollups and anomalies only — not per-packet grammar

## implemented roles
- bus_observer_tick_started
- bus_health_observed
- bus_stream_depth_observed
- bus_backpressure_observed
- bus_configured_stream_uncataloged
- bus_observer_tick_completed
- bus_observer_tick_failed

## deferred roles
- bus_stream_lag_observed
- bus_schema_validation_failed
- bus_delivery_anomaly_observed
- bus_metrics_scrape_observed / failed / bus_memory_pressure_observed

## tests run
<paste pytest output>

## live proof
<SQL/log output or not_verified_live>

## downstream follow-up
bus_transport_reducer → transport pressure → field digestion
```

- [ ] **Step 2: Commit report**

```bash
git add docs/superpowers/pr-reports/2026-05-25-orion-bus-substrate-trace-v1-pr.md
git commit -m "docs: add orion-bus substrate trace PR report"
```

### Task 11: Push and open PR

- [ ] **Step 1: Push branch**

```bash
git push -u origin feat/orion-bus-substrate-trace-v1
```

- [ ] **Step 2: Create PR via gh**

```bash
gh pr create --title "feat(orion-bus): transport substrate trace v1" --body "$(cat <<'EOF'
## Summary
- Add optional `bus-observer` sidecar under `services/orion-bus` for bounded periodic transport substrate traces
- Emit `GrammarEventV1` on `orion:grammar:event` when `PUBLISH_ORION_BUS_GRAMMAR=true` (default off)
- Context-engineering files: AGENT_CONTEXT, SERVICE_PORTS, SUBSTRATE_TRACE_MAP, LAYER_PIPELINE_PLAN
- No per-message / packet-log traces; Redis core + exporter unchanged

## Test plan
- [ ] `PYTHONPATH=services/orion-bus:. pytest services/orion-bus/tests/ -q`
- [ ] `scripts/smoke_orion_bus_substrate_trace.sh`
- [ ] Optional: enable publish, SQL proof on `grammar_events` for `trace_id like 'bus.transport:%'`

EOF
)"
```

---

## Self-review (plan author checklist)

| Check | Result |
|-------|--------|
| Spec coverage: context files, observer, emit, publish, catalog, tests, smoke, PR | Covered in Phases 0–7 |
| Placeholder scan | No TBD steps; code blocks provided |
| Type consistency | `BusTransportGrammarCollector`, `ObserverRollup`, settings field names in tests |
| `model_copy` in tests | Uses `publish_orion_bus_grammar` / `bus_observer_streams` field names, not env aliases |
| Uncataloged role semantics | `bus_configured_stream_uncataloged` — configured streams only, not live rogue producers |
| Default observer streams | `orion:evt:gateway,orion:bus:out` only; `orion:grammar:event` operator-opt-in |
| Catalog producer | `orion-bus` only; verify `schema_id: GrammarEventV1` before edit |
| Redaction test | `test_summaries_never_include_redis_values_or_envelope_material` |
| Option A/B decision documented | Option B locked with rationale |
| `metadata` on atoms | Corrected — use `summary` KV only |
| Registry change | Correctly omitted |
| `.env` sync called out | Yes, Task 1 Step 5 |
| No bleed rule | Worktree section |
| Code review + PR push | Phase 6–7 |

---

## Execution handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-25-orion-bus-substrate-trace-v1.md`. Two execution options:**

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task (Phases 0–7), review between tasks.
2. **Inline Execution** — run phases in this session with `executing-plans` checkpoints.

**Which approach?**
