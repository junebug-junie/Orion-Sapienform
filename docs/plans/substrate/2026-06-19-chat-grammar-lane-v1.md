# Chat Grammar Lane V1 Implementation Plan

> **For agentic workers:** Use subagent-driven-development. Implement task by task, commit after each, get review before proceeding.

**Goal:** Close the chat lane gap. Chat turns currently produce in-memory `SubstrateMoleculeV1` molecules for the repair pressure appraisal but never emit `GrammarEventV1` onto `orion:grammar:event`. This means the most semantically rich signal — what the user actually says — is invisible to the substrate reduction pipeline and never reaches the field digester.

After this plan, a chat turn flows:
```text
orion-hub chat turn (websocket_handler.py)
  → GrammarEventV1 on orion:grammar:event (hub.chat:athena:<turn_id>)
  → orion-substrate-runtime chat_grammar_consumer cursor
  → orion/substrate/chat_loop/ reducer
  → StateDeltaV1(target_kind=chat_turn)
  → substrate_reduction_receipts
  → orion-field-digester
  → node conversation_load / repair_pressure channels
```

**Existing work reused (do not redo):**
- `orion/substrate/appraisal/` — repair pressure appraisal pipeline already built; hub already calls it
- `orion/mind/substrate_emit.py` — molecule helpers; do not modify
- Reducer pattern exactly mirrors `orion/substrate/execution_loop/`

## Architecture constraints

- **No new GrammarEventKind** — use `trace_started`, `atom_emitted`, `edge_emitted`, `trace_ended`
- **No new AtomType** — use `observation`, `entity`, `signal`, `stance_influence`
- **No new RelationType** — use `derived_from`, `references`, `contains`
- **No raw user text in grammar events** — use `payload_ref` and bounded summaries only
- **Publisher is fail-open** — chat must work whether grammar publishing is on or off
- **Env-flag gated** — `PUBLISH_HUB_CHAT_GRAMMAR=false` default
- **Reducer env-flag gated** — `ENABLE_CHAT_GRAMMAR_REDUCER=false` default

## Trace shape

```text
trace_id: hub.chat:<node_id>:<turn_id>

trace_started
  atom: session_context   (entity)         — session_id, no user content
  atom: user_utterance    (observation)    — payload_ref only, word_count, redacted
  atom: repair_signal     (signal)         — appraisal dimensions if available; else omit
  edge: user_utterance → derived_from → session_context
  edge: repair_signal  → derived_from → user_utterance   (if repair_signal present)
trace_ended
```

## Pressure hints

Reducer extracts from the `repair_signal` atom (or defaults to 0.0 if absent):
```json
{
  "conversation_load": 0.0–1.0,
  "repair_pressure":   0.0–1.0,
  "topic_coherence":   0.0–1.0
}
```

## File structure

| File | Action | Responsibility |
|------|--------|----------------|
| `services/orion-hub/scripts/grammar_emit.py` | Create | Pure builder: chat turn → list[GrammarEventV1] |
| `services/orion-hub/scripts/grammar_publish.py` | Create | Fail-open Redis publisher |
| `services/orion-hub/tests/test_hub_grammar_emit.py` | Create | Emit unit tests |
| `services/orion-hub/tests/test_hub_grammar_publish_fail_open.py` | Create | Publisher fail-open tests |
| `services/orion-hub/scripts/websocket_handler.py` | Modify | Wire grammar_publish after substrate_effect_pipeline call |
| `services/orion-hub/app/settings.py` | Modify | Add `PUBLISH_HUB_CHAT_GRAMMAR`, `GRAMMAR_EVENT_CHANNEL` |
| `orion/schemas/chat_projection.py` | Create | `ChatSessionProjectionV1`, `ChatTurnStateV1` |
| `orion/substrate/chat_loop/__init__.py` | Create | |
| `orion/substrate/chat_loop/constants.py` | Create | Cursor name, projection ID, reducer ID, source service |
| `orion/substrate/chat_loop/grammar_extract.py` | Create | Extract semantic state from chat grammar events |
| `orion/substrate/chat_loop/reducer.py` | Create | `reduce_chat_trace_events` → StateDeltaV1 + receipt |
| `orion/substrate/chat_loop/pipeline.py` | Create | Group by trace_id, run reducer, emit receipts |
| `services/orion-sql-db/manual_migration_chat_substrate_loop.sql` | Create | Chat projection + cursor DDL |
| `services/orion-substrate-runtime/app/store.py` | Modify | Chat grammar fetch, cursor, projection I/O |
| `services/orion-substrate-runtime/app/worker.py` | Modify | `_chat_poll_loop`, `_chat_tick`, ReducerSpec entry |
| `services/orion-substrate-runtime/app/settings.py` | Modify | `ENABLE_CHAT_GRAMMAR_REDUCER`, `CHAT_GRAMMAR_BATCH_LIMIT` |
| `services/orion-substrate-runtime/.env_example` | Modify | New flags |
| `services/orion-field-digester/app/ingest/state_deltas.py` | Modify | `chat_turn` delta → perturbations |
| `config/field/biometrics_lattice.yaml` | Modify | Add `conversation_load`, `repair_pressure` node channels |
| `tests/test_chat_grammar_emit.py` | Create | Emit schema validity, redaction, stable IDs |
| `tests/test_chat_substrate_reducer.py` | Create | Reducer, extract, pressure hints |
| `tests/test_chat_substrate_pipeline.py` | Create | Pipeline grouping, idempotence |
| `tests/test_field_chat_perturbations.py` | Create | Field delta → perturbations |

**Do NOT modify:** `orion/schemas/grammar.py`, `orion/mind/substrate_emit.py`, `orion/substrate/appraisal/*`, `services/orion-hub/scripts/substrate_effect_pipeline.py`

---

## Task 1: Hub grammar emitter

**Files:** `services/orion-hub/scripts/grammar_emit.py`, `tests/test_hub_grammar_emit.py`

Build the pure event builder. No Redis, no SQL, no side effects.

### grammar_emit.py

```python
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from orion.schemas.grammar import (
    GrammarAtomV1,
    GrammarEdgeV1,
    GrammarEventV1,
    GrammarProvenanceV1,
)

SOURCE_SERVICE = "orion-hub"
SOURCE_COMPONENT = "hub_chat_grammar_emit"
TRACE_PREFIX = "hub.chat"


def _hash_id(*parts: object, prefix: str) -> str:
    raw = "|".join(str(p) for p in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode()).hexdigest()[:16]}"


def _event(
    *,
    event_kind: str,
    trace_id: str,
    provenance: GrammarProvenanceV1,
    emitted_at: datetime,
    observed_at: datetime,
    atom: GrammarAtomV1 | None = None,
    edge: GrammarEdgeV1 | None = None,
    parent_event_id: str | None = None,
    root_event_id: str | None = None,
    layer: str | None = None,
    dimensions: list[str] | None = None,
) -> GrammarEventV1:
    from uuid import uuid4
    body_key = (atom.atom_id if atom else edge.edge_id if edge else uuid4().hex)
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


def build_chat_turn_grammar_events(
    *,
    turn_id: str,
    session_id: str,
    node_id: str = "athena",
    word_count: int = 0,
    repair_pressure_level: float = 0.0,
    repair_pressure_confidence: float = 0.0,
    repair_pressure_dims: dict[str, float] | None = None,
    has_repair_signal: bool = False,
    observed_at: datetime | None = None,
    code_version: str | None = None,
) -> list[GrammarEventV1]:
    now = datetime.now(timezone.utc)
    observed = observed_at or now
    ts = observed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    trace_id = f"{TRACE_PREFIX}:{node_id}:{turn_id}"

    provenance = GrammarProvenanceV1(
        source_service=SOURCE_SERVICE,
        source_component=SOURCE_COMPONENT,
        source_event_id=turn_id,
        source_trace_id=trace_id,
        source_payload_ref=f"hub.chat:{session_id}:{turn_id}",
        code_version=code_version,
    )

    root_event = _event(
        event_kind="trace_started",
        trace_id=trace_id,
        provenance=provenance,
        emitted_at=now,
        observed_at=observed,
        layer="chat",
        dimensions=["conversation", "turn", "session"],
    )
    root_id = root_event.event_id

    def atom_id(role: str) -> str:
        return f"{trace_id}:{role}"

    session_atom = GrammarAtomV1(
        atom_id=atom_id("session_context"),
        trace_id=trace_id,
        atom_type="entity",
        semantic_role="session_context",
        layer="context",
        dimensions=["session", "conversation"],
        summary=f"Chat session {session_id} on {node_id}",
        text_value=session_id,
        confidence=1.0,
        salience=0.6,
        source_event_id=turn_id,
        payload_ref=f"hub.session:{session_id}",
    )

    utterance_atom = GrammarAtomV1(
        atom_id=atom_id("user_utterance"),
        trace_id=trace_id,
        atom_type="observation",
        semantic_role="user_utterance",
        layer="raw_input",
        dimensions=["conversation", "turn", "user"],
        summary=f"User message in session {session_id} ({word_count} words)",
        confidence=1.0,
        salience=min(1.0, 0.4 + word_count / 200.0),
        source_event_id=turn_id,
        payload_ref=f"hub.chat:{session_id}:{turn_id}",
    )

    atoms: dict[str, GrammarAtomV1] = {
        "session_context": session_atom,
        "user_utterance": utterance_atom,
    }

    if has_repair_signal:
        dims = repair_pressure_dims or {}
        repair_atom = GrammarAtomV1(
            atom_id=atom_id("repair_signal"),
            trace_id=trace_id,
            atom_type="signal",
            semantic_role="repair_signal",
            layer="organ_signal",
            dimensions=["repair", "conversation", "pressure"],
            summary=f"Repair pressure level={repair_pressure_level:.2f} confidence={repair_pressure_confidence:.2f}",
            confidence=repair_pressure_confidence,
            salience=repair_pressure_level,
            source_event_id=turn_id,
            payload_ref=f"hub.repair_pressure:{turn_id}",
        )
        atoms["repair_signal"] = repair_atom

    edge_specs: list[tuple[str, str, str]] = [
        ("user_utterance", "session_context", "derived_from"),
    ]
    if "repair_signal" in atoms:
        edge_specs.append(("repair_signal", "user_utterance", "derived_from"))

    events: list[GrammarEventV1] = [root_event]

    for atom in atoms.values():
        events.append(_event(
            event_kind="atom_emitted",
            trace_id=trace_id,
            provenance=provenance,
            emitted_at=now,
            observed_at=observed,
            atom=atom,
            parent_event_id=root_id,
            root_event_id=root_id,
            layer=atom.layer,
            dimensions=atom.dimensions,
        ))

    for from_key, to_key, rel in edge_specs:
        edge = GrammarEdgeV1(
            edge_id=f"{trace_id}:edge:{from_key}:{rel}:{to_key}",
            trace_id=trace_id,
            from_atom_id=atoms[from_key].atom_id,
            to_atom_id=atoms[to_key].atom_id,
            relation_type=rel,  # type: ignore[arg-type]
            confidence=0.95,
            salience=0.7,
            layer_from=atoms[from_key].layer,
            layer_to=atoms[to_key].layer,
            evidence_event_ids=[turn_id],
        )
        events.append(_event(
            event_kind="edge_emitted",
            trace_id=trace_id,
            provenance=provenance,
            emitted_at=now,
            observed_at=observed,
            edge=edge,
            parent_event_id=root_id,
            root_event_id=root_id,
            layer=edge.layer_to,
            dimensions=["conversation", "turn"],
        ))

    events.append(_event(
        event_kind="trace_ended",
        trace_id=trace_id,
        provenance=provenance,
        emitted_at=datetime.now(timezone.utc),
        observed_at=observed,
        parent_event_id=root_id,
        root_event_id=root_id,
        layer="chat",
        dimensions=["conversation", "turn", "session"],
    ))

    return events
```

### Tests (tests/test_hub_grammar_emit.py)

- Valid GrammarEventV1 instances for all events
- trace_started / atom_emitted / edge_emitted / trace_ended sequence
- user_utterance has no raw text (text_value is None)
- stable event IDs (same inputs → same IDs)
- repair_signal present only when has_repair_signal=True
- all AtomType values are valid closed literals
- all RelationType values are valid closed literals

- [ ] Write tests, verify they fail
- [ ] Create grammar_emit.py
- [ ] Run tests, verify they pass
- [ ] Commit: `feat(hub): add chat turn grammar event emitter`

---

## Task 2: Hub grammar publisher + hub wiring

**Files:** `services/orion-hub/scripts/grammar_publish.py`, `services/orion-hub/tests/test_hub_grammar_publish_fail_open.py`, `services/orion-hub/app/settings.py`, `services/orion-hub/scripts/websocket_handler.py`

### grammar_publish.py

```python
from __future__ import annotations
import json
import logging
from typing import Any
from orion.schemas.grammar import GrammarEventV1

logger = logging.getLogger("orion-hub.grammar_publish")

def publish_chat_grammar_events(
    events: list[GrammarEventV1],
    *,
    redis_client: Any,
    channel: str,
    enabled: bool,
) -> None:
    """Fail-open. Chat must work whether publishing is on or off."""
    if not enabled or not events:
        return
    try:
        for event in events:
            redis_client.publish(channel, json.dumps(event.model_dump(mode="json")))
    except Exception:
        logger.warning("hub_chat_grammar_publish_failed", exc_info=True)
```

### settings.py additions

Add to `services/orion-hub/app/settings.py`:
```python
publish_hub_chat_grammar: bool = Field(False, alias="PUBLISH_HUB_CHAT_GRAMMAR")
grammar_event_channel: str = Field("orion:grammar:event", alias="GRAMMAR_EVENT_CHANNEL")
```

### websocket_handler.py wiring

After the existing `run_substrate_effect_pipeline` call (line ~833), add:

```python
# Emit chat grammar trace (fail-open, behind env flag)
try:
    from scripts.grammar_emit import build_chat_turn_grammar_events
    from scripts.grammar_publish import publish_chat_grammar_events
    _settings = get_settings()
    if _settings.publish_hub_chat_grammar:
        _grammar_events = build_chat_turn_grammar_events(
            turn_id=trace_id,
            session_id=str(session_id or "anonymous"),
            word_count=len((transcript or "").split()),
            repair_pressure_level=float((substrate_summary or {}).get("level", 0.0)),
            repair_pressure_confidence=float((substrate_summary or {}).get("confidence", 0.0)),
            has_repair_signal=substrate_summary is not None,
        )
        publish_chat_grammar_events(
            _grammar_events,
            redis_client=_get_redis(),  # use existing redis accessor
            channel=_settings.grammar_event_channel,
            enabled=True,
        )
except Exception:
    logger.warning("hub_chat_grammar_wire_failed corr=%s", trace_id, exc_info=True)
```

**Read `websocket_handler.py` to find the right Redis accessor (`_get_redis` or equivalent) and the correct import path for `get_settings`.** Adapt imports to match the hub's actual patterns.

### Publisher tests

- publish with enabled=True calls redis_client.publish
- publish with enabled=False is a no-op
- publish with redis_client raising does not raise (fail-open)
- publish with malformed events does not raise

- [ ] Write tests, verify they fail
- [ ] Create grammar_publish.py, update settings, wire websocket_handler
- [ ] Run tests, verify they pass
- [ ] Commit: `feat(hub): add chat grammar publisher and wire into chat handler`

---

## Task 3: Chat projection schema + reducer + pipeline

**Files:** `orion/schemas/chat_projection.py`, `orion/substrate/chat_loop/`

### Chat projection schema

```python
# orion/schemas/chat_projection.py
from __future__ import annotations
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

class ChatTurnStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trace_id: str
    turn_id: str
    session_id: str
    node_id: str
    observed_at: datetime
    word_count: int = 0
    repair_pressure_level: float = 0.0
    repair_pressure_confidence: float = 0.0
    has_repair_signal: bool = False
    evidence_event_ids: list[str] = Field(default_factory=list)
    last_updated_at: datetime

class ChatSessionProjectionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["chat_session_projection.v1"] = "chat_session_projection.v1"
    projection_id: str
    generated_at: datetime
    turns: dict[str, ChatTurnStateV1] = Field(default_factory=dict)
    total_turn_count: int = 0
    sessions: list[str] = Field(default_factory=list)
```

### constants.py

```python
CHAT_GRAMMAR_CURSOR_NAME = "chat_grammar_consumer"
CHAT_SESSION_PROJECTION_ID = "active_chat_session"
CHAT_REDUCER_ID = "chat_turn_reducer"
CHAT_SOURCE_SERVICE = "orion-hub"
CHAT_TRACE_PREFIX = "hub.chat:"
```

### grammar_extract.py

Extract `ChatTurnStateV1` from a list of grammar events for one trace:
- Parse `hub.chat:<node_id>:<turn_id>` from trace_id
- From `user_utterance` atom: word_count from summary, payload_ref
- From `repair_signal` atom (if present): confidence, salience → repair_pressure_level/confidence
- Noop if source_service != "orion-hub"

### reducer.py

`reduce_chat_trace_events(events, projection, now) → (projection, receipt)`:
- Extract ChatTurnStateV1 via grammar_extract
- Update projection.turns[turn_id]
- Produce StateDeltaV1(target_kind="chat_turn", target_id=session_id, operation="update")
- Pressure hints: `conversation_load`, `repair_pressure`, `topic_coherence`
- Stable delta_id: `chat_turn:<session_id>:update:<trace_id>`
- Evidence event IDs capped at 50

Pressure hint formulas:
```
conversation_load  = min(1.0, word_count / 150.0)  (150 words ≈ full)
repair_pressure    = repair_pressure_level          (direct from appraisal)
topic_coherence    = max(0.0, 1.0 - repair_pressure_level)
```

### pipeline.py

`process_chat_grammar_events(events, load_projection, save_projection, save_receipt, now)`:
- Group by trace_id
- Filter to hub.chat: prefix only
- Run reducer per trace
- Save receipt, update projection

### Tests

- `tests/test_chat_substrate_reducer.py`: extract, pressure hints, stable IDs, noop on wrong source
- `tests/test_chat_substrate_pipeline.py`: grouping, idempotence, wrong-prefix filtering

- [ ] Write tests, verify they fail
- [ ] Implement schema + chat_loop package
- [ ] Run tests, verify they pass
- [ ] Commit: `feat(substrate): add chat turn reducer and pipeline`

---

## Task 4: SQL migration

**File:** `services/orion-sql-db/manual_migration_chat_substrate_loop.sql`

```sql
-- Chat substrate loop (apply before enabling chat grammar reducer)
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_chat_substrate_loop.sql

create table if not exists substrate_chat_session_projection (
    projection_id text primary key,
    generated_at timestamptz not null,
    projection_json jsonb not null,
    created_at timestamptz not null default now()
);

create index if not exists idx_substrate_chat_proj_generated_at
    on substrate_chat_session_projection (generated_at desc);

-- Chat grammar cursor (substrate-runtime tracks position in grammar_events)
insert into substrate_grammar_cursors (cursor_name, created_at, event_id)
values ('chat_grammar_consumer', now(), null)
on conflict (cursor_name) do nothing;
```

- [ ] Create migration file
- [ ] Commit: `feat(substrate): add chat projection and cursor DDL`

---

## Task 5: Substrate-runtime wiring

**Files:** `services/orion-substrate-runtime/app/store.py`, `app/worker.py`, `app/settings.py`, `.env_example`

### settings.py

Add:
```python
enable_chat_grammar_reducer: bool = Field(False, alias="ENABLE_CHAT_GRAMMAR_REDUCER")
chat_grammar_batch_limit: int = Field(100, alias="CHAT_GRAMMAR_BATCH_LIMIT")
```

### store.py

Add alongside existing execution/transport methods:
- `fetch_chat_grammar_events(limit)` — fetch from grammar_events WHERE trace_id LIKE 'hub.chat:%'
- `advance_chat_cursor(event_id, created_at)` — update substrate_grammar_cursors WHERE cursor_name='chat_grammar_consumer'
- `load_chat_projection(projection_id)` → `ChatSessionProjectionV1 | None`
- `save_chat_projection(projection)` — upsert substrate_chat_session_projection
- Add `CHAT_GRAMMAR_CURSOR_NAME` to `GRAMMAR_CURSOR_REGISTRY`

### worker.py

Add `ReducerSpec` entry and `_chat_poll_loop` / `_chat_tick` following the exact pattern of `_execution_poll_loop` / `_execution_tick`.

### .env_example

Add:
```
# Chat grammar reducer (set true after PUBLISH_HUB_CHAT_GRAMMAR=true on hub)
ENABLE_CHAT_GRAMMAR_REDUCER=false
CHAT_GRAMMAR_BATCH_LIMIT=100
```

- [ ] Implement store + worker + settings
- [ ] Commit: `feat(substrate-runtime): wire chat grammar reducer loop`

---

## Task 6: Field digestion + lattice channels

**Files:** `services/orion-field-digester/app/ingest/state_deltas.py`, `config/field/biometrics_lattice.yaml`, `tests/test_field_chat_perturbations.py`

### state_deltas.py addition

Add a `chat_turn` block:
```python
if delta.target_kind == "chat_turn":
    hints = dict(after.get("pressure_hints") or {})
    node_key = _node_key(str(after.get("node_id") or delta.target_id))
    for channel, key in (
        ("conversation_load", "conversation_load"),
        ("repair_pressure", "repair_pressure"),
    ):
        if key in hints:
            out.append(Perturbation(
                node_id=node_key,
                channel=channel,
                intensity=float(hints[key]),
                label=delta.delta_id,
                mode="replace",
            ))
```

### biometrics_lattice.yaml

Add to `node_channels`:
```yaml
- conversation_load
- repair_pressure
```

### Tests

- `chat_turn` delta with hints → correct perturbations
- missing hints → no perturbations  
- `noop` operation → no perturbations

- [ ] Write tests, verify they fail
- [ ] Implement state_deltas addition + lattice config
- [ ] Run tests, verify pass
- [ ] Commit: `feat(field-digester): map chat_turn deltas to conversation_load and repair_pressure`

---

## Acceptance criteria

- [ ] `build_chat_turn_grammar_events` returns valid GrammarEventV1 instances; no raw user text in any atom
- [ ] Publisher is fail-open (exception in redis.publish does not propagate to chat handler)
- [ ] `PUBLISH_HUB_CHAT_GRAMMAR=false` (default) means no Redis calls at all
- [ ] Reducer extracts word_count and repair pressure from grammar events without touching raw user content
- [ ] StateDeltaV1(target_kind=chat_turn) has stable delta_id
- [ ] Field digester maps chat_turn → conversation_load and repair_pressure perturbations
- [ ] All reducers in substrate-runtime remain independent (chat failure does not affect biometrics/execution/transport)
- [ ] `ENABLE_CHAT_GRAMMAR_REDUCER=false` default — safe to deploy before hub is publishing

## Non-goals

- Do not embed raw user text anywhere in grammar events
- Do not add new GrammarEventKind, AtomType, or RelationType literals
- Do not modify `orion/substrate/appraisal/` or `substrate_effect_pipeline.py`
- Do not wire a live bus subscription in this plan (same stance as signal bridge)
- Do not add topic modeling, NLP, or LLM calls to the reducer
