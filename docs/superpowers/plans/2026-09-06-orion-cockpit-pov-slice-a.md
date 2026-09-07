# Orion Cockpit POV (Slice A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Soft HUD Cockpit on Hub chat turns with a durable Turn Sighting timeline, live hop streaming, rewind scrubbing, and first real hop producers (stance decision, FCC motor steps, finalize/closure) — gaps explicit for stages deferred to Slice B/C.

**Architecture:** New `CockpitHopV1` bus event is the append-only sighting record. Hub assigns monotonic `seq` during a unified turn, publishes hops to `orion:cockpit:hop` (sql-writer persists) and mirrors `cockpit_hop` WS frames to the chat session. Soft HUD modal loads `GET /api/chat/turn/{correlation_id}/cockpit` then follows WS. Turn Trace panel stays unchanged.

**Tech Stack:** Pydantic schemas + `orion/schemas/registry.py`, Redis bus (`orion/bus/channels.yaml`), Postgres via orion-sql-writer, FastAPI Hub routes, Hub WebSocket JSON frames, vanilla Hub static JS/CSS (Soft HUD modal).

**Spec:** `docs/superpowers/specs/2026-09-06-orion-cockpit-pov-design.md`

**Out of this plan (separate plans later):** Slice B (stance_inputs, motor_boot full prefix, association guts), Slice C (remaining offboarding side-effects). This plan must still leave the Soft HUD thick for what it *does* record, and show **gap** beads for deferred stages.

## Global Constraints

- Unified Orion turn path only (`execute_unified_turn` / harness governor / FCC motor) — not classic PlanRunner thickness
- Operator Hub debug surface; full `raw` payloads (same honesty class as unredacted harness turn-trace)
- Never fabricate hops; missing stages → `status="gap"`
- Keep Turn Trace panel mounted and working
- Soft HUD aesthetic (frosted glass / soft vignette) — not Hard HUD / Cinematic
- Env parity: if `.env_example` changes, run `python scripts/sync_local_env_from_example.py` from repo root
- Work only in a git worktree (not shared main checkout); commit per task
- Register new schemas in **both** `_REGISTRY` and `SCHEMA_REGISTRY` in `orion/schemas/registry.py`
- Do not destroy or shrink `graphify-out/`; after code changes run `scripts/safe_graphify_update.sh` once before PR, not mid-task

## File structure (locked)

| File | Responsibility |
|------|----------------|
| `orion/schemas/cockpit_sighting.py` | `CockpitHopV1`, stage literals, channel constant |
| `orion/schemas/registry.py` | Dual registration |
| `orion/bus/channels.yaml` | `orion:cockpit:hop` |
| `orion/cockpit/__init__.py` | Package export |
| `orion/cockpit/builders.py` | Pure builders: thought / step / run / outcome / closure / gap → `CockpitHopV1` |
| `orion/cockpit/sequencer.py` | Monotonic `seq` helper (in-process per correlation_id) |
| `orion/cockpit/publish.py` | `publish_cockpit_hop(bus, hop)` envelope publish |
| `orion/cockpit/tests/test_builders.py` | Builder unit tests |
| `orion/cockpit/tests/test_sequencer.py` | Seq monotonicity |
| `services/orion-sql-writer/app/models/cockpit_turn_sighting.py` | SQLAlchemy hop row model |
| `services/orion-sql-writer/app/cockpit_turn_sighting_persist.py` | `append_cockpit_hop(sess, payload)` |
| `services/orion-sql-writer/app/worker.py` | MODEL_MAP + special branch |
| `services/orion-sql-writer/app/settings.py` | `DEFAULT_ROUTE_MAP` kind → model |
| `services/orion-sql-writer/app/models/__init__.py` | Export model |
| `services/orion-sql-writer/.env_example` | Subscribe + route map entry |
| `services/orion-sql-writer/tests/test_cockpit_turn_sighting_sql_shape.py` | Persist/route tests |
| `services/orion-hub/scripts/chat_cockpit_routes.py` | `GET .../cockpit` (+ optional hop-by-seq) |
| `services/orion-hub/scripts/api_routes.py` | `include_router` |
| `services/orion-hub/tests/test_chat_cockpit_routes.py` | API tests |
| `services/orion-hub/scripts/harness_step_relay.py` | Also queue `cockpit_hop` frames from motor steps |
| `orion/hub/cockpit_emit.py` | Pure helpers: pre-motor gaps+stance frames; step→hop; run artifact→hops |
| `orion/hub/turn_orchestrator.py` | Call emit helpers; send WS; publish bus; drain `claude_step` → `cockpit_hop` |
| `services/orion-hub/tests/test_turn_orchestrator_cockpit_hops.py` | WS/bus hop emission tests |
| `services/orion-hub/static/js/cockpit-hud.js` | Soft HUD modal API |
| `services/orion-hub/static/css/cockpit-hud.css` | Soft HUD styles |
| `services/orion-hub/templates/index.html` | Modal root + CSS/JS link |
| `services/orion-hub/static/js/app.js` | Cockpit button near Turn Trace; WS handler |
| `services/orion-hub/tests/test_cockpit_hud_ui.py` | Node behavioral UI tests |
| `services/orion-hub/README.md` | Short Cockpit operator note |

---

### Task 1: CockpitHopV1 schema + registry + bus channel

**Files:**
- Create: `orion/schemas/cockpit_sighting.py`
- Modify: `orion/schemas/registry.py` (add to `_REGISTRY` and `SCHEMA_REGISTRY` beside other harness/thought entries)
- Modify: `orion/bus/channels.yaml` (add `orion:cockpit:hop` after harness run step block ~L2985)
- Test: `orion/schemas/tests/test_cockpit_sighting_schema.py` (create if `orion/schemas/tests/` exists; otherwise `tests/test_cockpit_sighting_schema.py` at repo root matching local convention — prefer colocated under `orion/schemas/tests/` only if that dir already has peers; else `orion/cockpit/tests/test_schema_roundtrip.py`)

**Interfaces:**
- Produces: `COCKPIT_HOP_CHANNEL = "orion:cockpit:hop"`, `CockpitHopV1`, `CockpitStageV1` literal union, message kind `cockpit.hop.v1`

- [ ] **Step 1: Write the failing test**

Create `orion/cockpit/tests/test_schema_roundtrip.py`:

```python
from orion.schemas.cockpit_sighting import CockpitHopV1, COCKPIT_HOP_CHANNEL
from orion.schemas.registry import resolve


def test_cockpit_hop_roundtrip_and_registry():
    hop = CockpitHopV1(
        correlation_id="corr-1",
        seq=1,
        stage="stance_decision",
        visor_line="stance · proceed",
        status="ok",
        summary={"disposition": "proceed"},
        raw={"disposition": "proceed"},
        producer="orion-hub",
    )
    dumped = hop.model_dump(mode="json")
    again = CockpitHopV1.model_validate(dumped)
    assert again.seq == 1
    assert again.stage == "stance_decision"
    assert COCKPIT_HOP_CHANNEL == "orion:cockpit:hop"
    assert resolve("CockpitHopV1") is CockpitHopV1


def test_gap_status_allowed():
    hop = CockpitHopV1(
        correlation_id="corr-1",
        seq=2,
        stage="motor_boot",
        visor_line="gap · motor_boot not recorded (Slice B)",
        status="gap",
        summary={"deferred_to": "slice_b"},
        raw={},
        producer="orion-hub",
    )
    assert hop.status == "gap"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest orion/cockpit/tests/test_schema_roundtrip.py -v`  
Expected: FAIL (module/import not found)

- [ ] **Step 3: Write minimal implementation**

`orion/schemas/cockpit_sighting.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

COCKPIT_HOP_CHANNEL = "orion:cockpit:hop"

CockpitStageV1 = Literal[
    "ingress",
    "association",
    "stance_inputs",
    "stance_decision",
    "motor_boot",
    "motor_hop",
    "draft_appraisal",
    "finalize",
    "closure",
]

CockpitHopStatusV1 = Literal["started", "ok", "failed", "skipped", "gap"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CockpitHopV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["cockpit.hop.v1"] = "cockpit.hop.v1"
    correlation_id: str = Field(min_length=1)
    seq: int = Field(ge=0)
    ts: datetime = Field(default_factory=_utc_now)
    stage: CockpitStageV1
    visor_line: str = Field(min_length=1, max_length=512)
    status: CockpitHopStatusV1
    summary: dict[str, Any] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)
    producer: str = Field(min_length=1, max_length=128)
```

Register in `registry.py`:
- Import `CockpitHopV1`
- `_REGISTRY["CockpitHopV1"] = CockpitHopV1`
- `SCHEMA_REGISTRY["CockpitHopV1"] = SchemaRegistration(model=CockpitHopV1, kind="cockpit.hop.v1")`

`channels.yaml` entry:

```yaml
  - name: "orion:cockpit:hop"
    kind: "event"
    schema_id: "CockpitHopV1"
    message_kind: "cockpit.hop.v1"
    producer_services: ["orion-hub"]
    consumer_services: ["orion-sql-writer", "orion-hub"]
    stability: "experimental"
    since: "2026-09-06"
```

Add empty `orion/cockpit/__init__.py` and `orion/cockpit/tests/__init__.py` if needed for collection.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest orion/cockpit/tests/test_schema_roundtrip.py -v`  
Also: `python scripts/check_schema_registry.py` and `python scripts/check_bus_channels.py`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/schemas/cockpit_sighting.py orion/schemas/registry.py orion/bus/channels.yaml orion/cockpit/
git commit -m "$(cat <<'EOF'
feat(cockpit): add CockpitHopV1 schema and bus channel

Turn Sighting hop contract for Soft HUD live+rewind spine.
EOF
)"
```

---

### Task 2: Hop builders + in-process sequencer

**Files:**
- Create: `orion/cockpit/builders.py`
- Create: `orion/cockpit/sequencer.py`
- Create: `orion/cockpit/publish.py`
- Test: `orion/cockpit/tests/test_builders.py`, `orion/cockpit/tests/test_sequencer.py`

**Interfaces:**
- Consumes: `CockpitHopV1` from Task 1
- Produces:
  - `next_seq(correlation_id: str) -> int` (starts at 0, increments)
  - `reset_seq(correlation_id: str) -> None` (tests / turn start)
  - `hop_from_thought(*, correlation_id, seq, thought: dict) -> CockpitHopV1`
  - `hop_from_motor_step(*, correlation_id, seq, step_index: int, step: dict) -> CockpitHopV1`
  - `hop_from_run_artifact(*, correlation_id, seq, run: dict) -> list[CockpitHopV1]` (draft_appraisal + finalize)
  - `hop_from_outcome(*, correlation_id, seq, outcome: dict) -> CockpitHopV1`
  - `hop_from_closure(*, correlation_id, seq, closure: dict) -> CockpitHopV1`
  - `gap_hop(*, correlation_id, seq, stage, deferred_to: str = "slice_b") -> CockpitHopV1`
  - `async def publish_cockpit_hop(bus, hop: CockpitHopV1, *, channel: str = COCKPIT_HOP_CHANNEL) -> None`

- [ ] **Step 1: Write failing tests**

`orion/cockpit/tests/test_sequencer.py`:

```python
from orion.cockpit.sequencer import next_seq, reset_seq


def test_next_seq_monotonic_per_correlation():
    reset_seq("corr-a")
    reset_seq("corr-b")
    assert next_seq("corr-a") == 0
    assert next_seq("corr-a") == 1
    assert next_seq("corr-b") == 0
    assert next_seq("corr-a") == 2
```

`orion/cockpit/tests/test_builders.py`:

```python
from orion.cockpit.builders import (
    gap_hop,
    hop_from_motor_step,
    hop_from_thought,
)


def test_hop_from_thought_proceed():
    hop = hop_from_thought(
        correlation_id="c1",
        seq=0,
        thought={
            "disposition": "proceed",
            "disposition_reasons": ["ok"],
            "imperative": "stay with it",
            "tone": "steady",
        },
    )
    assert hop.stage == "stance_decision"
    assert hop.status == "ok"
    assert "proceed" in hop.visor_line
    assert hop.raw["disposition"] == "proceed"
    assert hop.producer == "orion-hub"


def test_hop_from_motor_step():
    hop = hop_from_motor_step(
        correlation_id="c1",
        seq=3,
        step_index=2,
        step={"type": "tool_use", "name": "Read", "input": {"path": "x"}},
    )
    assert hop.stage == "motor_hop"
    assert hop.summary["step_index"] == 2
    assert hop.raw["step"]["name"] == "Read"


def test_gap_hop_motor_boot():
    hop = gap_hop(correlation_id="c1", seq=1, stage="motor_boot")
    assert hop.status == "gap"
    assert hop.stage == "motor_boot"
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `pytest orion/cockpit/tests/test_sequencer.py orion/cockpit/tests/test_builders.py -v`

- [ ] **Step 3: Implement**

`orion/cockpit/sequencer.py` — module-level `dict[str, int]` counter; `next_seq` / `reset_seq`.

`orion/cockpit/builders.py` — map thought disposition into visor_line; put full thought dict in `raw`; motor step stores `{step_index, step}` in raw; `hop_from_run_artifact` returns two hops when both draft/appraisal and finalize fields exist (stage `draft_appraisal` then `finalize`), one hop if only one side present; gap as above; `producer="orion-hub"` for Hub-built hops.

`orion/cockpit/publish.py` — mirror `orion/harness/step_stream.py` envelope pattern with `kind="cockpit.hop.v1"` and `payload=hop.model_dump(mode="json")`.

- [ ] **Step 4: Run tests — expect PASS**

Run: `pytest orion/cockpit/tests/test_sequencer.py orion/cockpit/tests/test_builders.py -v`

- [ ] **Step 5: Commit**

```bash
git add orion/cockpit/
git commit -m "$(cat <<'EOF'
feat(cockpit): hop builders and per-turn seq helper

Pure adapters from thought/step/finalize facts into CockpitHopV1.
EOF
)"
```

---

### Task 3: sql-writer durable append store

**Files:**
- Create: `services/orion-sql-writer/app/models/cockpit_turn_sighting.py`
- Create: `services/orion-sql-writer/app/cockpit_turn_sighting_persist.py`
- Modify: `services/orion-sql-writer/app/models/__init__.py`
- Modify: `services/orion-sql-writer/app/worker.py` (MODEL_MAP + persist branch like harness_turn_trace)
- Modify: `services/orion-sql-writer/app/settings.py` (`DEFAULT_ROUTE_MAP["cockpit.hop.v1"] = "CockpitTurnSightingSQL"`)
- Modify: `services/orion-sql-writer/.env_example` (subscribe `orion:cockpit:hop`, route map entry)
- Test: `services/orion-sql-writer/tests/test_cockpit_turn_sighting_sql_shape.py`

**Interfaces:**
- Consumes: bus payload `cockpit.hop.v1` / `CockpitHopV1` dump
- Produces: `append_cockpit_hop(sess, payload: dict) -> bool`; table `cockpit_turn_sighting` PK `(correlation_id, seq)`

- [ ] **Step 1: Write failing tests**

Mirror `test_harness_turn_trace_sql_shape.py` / `test_thought_decision_sql_shape.py`:

```python
def test_default_route_map_points_cockpit_hop_at_sighting_sql():
    from app.settings import DEFAULT_ROUTE_MAP
    assert DEFAULT_ROUTE_MAP.get("cockpit.hop.v1") == "CockpitTurnSightingSQL"


def test_append_cockpit_hop_inserts_row(sqlite_sess):  # use the suite's sess fixture pattern
    from app.cockpit_turn_sighting_persist import append_cockpit_hop
    ok = append_cockpit_hop(
        sqlite_sess,
        {
            "schema_version": "cockpit.hop.v1",
            "correlation_id": "corr-1",
            "seq": 0,
            "stage": "stance_decision",
            "visor_line": "stance · proceed",
            "status": "ok",
            "summary": {},
            "raw": {"disposition": "proceed"},
            "producer": "orion-hub",
        },
    )
    assert ok is True
```

Also assert `.env_example` lists the channel (string search) and MODEL_MAP registers the class.

- [ ] **Step 2: Run — expect FAIL**

Run: `pytest services/orion-sql-writer/tests/test_cockpit_turn_sighting_sql_shape.py -v`

- [ ] **Step 3: Implement model + persist + wiring**

Table columns: `correlation_id`, `seq`, `ts`, `stage`, `visor_line`, `status`, `summary` JSONB, `raw` JSONB, `producer`, `created_at`. Composite primary key `(correlation_id, seq)`. `append_cockpit_hop` inserts; on conflict do nothing / return False (idempotent republish).

Wire worker like thought_decision / harness_turn_trace special cases.

Sync local env:

```bash
python scripts/sync_local_env_from_example.py
```

- [ ] **Step 4: Run — expect PASS**

Run: `pytest services/orion-sql-writer/tests/test_cockpit_turn_sighting_sql_shape.py -v`

- [ ] **Step 5: Commit**

```bash
git add services/orion-sql-writer/
git commit -m "$(cat <<'EOF'
feat(sql-writer): persist cockpit turn sighting hops

Append-only (correlation_id, seq) store for Soft HUD rewind.
EOF
)"
```

---

### Task 4: Hub GET `/api/chat/turn/{correlation_id}/cockpit`

**Files:**
- Create: `services/orion-hub/scripts/chat_cockpit_routes.py`
- Modify: `services/orion-hub/scripts/api_routes.py` (import + `include_router`)
- Test: `services/orion-hub/tests/test_chat_cockpit_routes.py`

**Interfaces:**
- Produces: `async def get_cockpit_timeline(correlation_id: str) -> dict` with shape:

```python
{
  "correlation_id": str,
  "hops": [ /* CockpitHopV1 dumps ordered by seq */ ],
  "complete": bool,  # True if any hop present OR explicit terminal marker later; Slice A: True when hops non-empty
  "gaps": [str],     # stages from canonical list with no hop rows
}
```

- Route: `GET /api/chat/turn/{correlation_id}/cockpit` — never 404 on empty (mirror turn-trace honesty)

- [ ] **Step 1: Write failing tests**

Monkeypatch the Postgres loader (copy pattern from `test_chat_turn_trace_routes.py`):

```python
import pytest
from scripts import chat_cockpit_routes as mod


@pytest.mark.asyncio
async def test_empty_timeline_reports_all_canonical_gaps(monkeypatch):
    monkeypatch.setattr(mod, "_load_hops", lambda cid: [])
    body = await mod.get_cockpit_timeline("corr-1")
    assert body["correlation_id"] == "corr-1"
    assert body["hops"] == []
    assert body["complete"] is False
    assert "stance_decision" in body["gaps"]
    assert "motor_hop" in body["gaps"]


@pytest.mark.asyncio
async def test_timeline_orders_by_seq_and_shrinks_gaps(monkeypatch):
    monkeypatch.setattr(
        mod,
        "_load_hops",
        lambda cid: [
            {"correlation_id": cid, "seq": 1, "stage": "motor_hop", "status": "ok"},
            {"correlation_id": cid, "seq": 0, "stage": "stance_decision", "status": "ok"},
        ],
    )
    body = await mod.get_cockpit_timeline("corr-1")
    assert [h["seq"] for h in body["hops"]] == [0, 1]
    assert "stance_decision" not in body["gaps"]
    assert "motor_hop" not in body["gaps"]
    assert "motor_boot" in body["gaps"]
```

- [ ] **Step 2: Run — expect FAIL**

Run: `pytest services/orion-hub/tests/test_chat_cockpit_routes.py -v`

- [ ] **Step 3: Implement routes**

`_load_hops(correlation_id)` SELECT from `cockpit_turn_sighting ORDER BY seq`. Compute `gaps` as canonical stages with zero rows (note: many `motor_hop` rows still clear the `motor_hop` gap). Include router in `api_routes.py` next to turn-trace router.

- [ ] **Step 4: Run — expect PASS**

Run: `pytest services/orion-hub/tests/test_chat_cockpit_routes.py -v`

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/scripts/chat_cockpit_routes.py services/orion-hub/scripts/api_routes.py services/orion-hub/tests/test_chat_cockpit_routes.py
git commit -m "$(cat <<'EOF'
feat(hub): GET cockpit timeline for turn sighting rewind

Ordered hops plus explicit stage gaps for Soft HUD scrubber.
EOF
)"
```

---

### Task 5: Live emit — WS `cockpit_hop` + bus publish from unified turn

**Files:**
- Modify: `services/orion-hub/scripts/harness_step_relay.py` (`_dispatch_step` also queues cockpit frame)
- Modify: `orion/hub/turn_orchestrator.py` (reset seq at turn start; emit stance hop; emit Slice-A gap hops for deferred stages; on run success emit draft/finalize hops; publish each hop)
- Test: `services/orion-hub/tests/test_hub_harness_step_relay.py` (extend) + `services/orion-hub/tests/test_turn_orchestrator_cockpit_hops.py`

**Interfaces:**
- WS frame shape (in addition to existing `claude_step`):

```python
{
  "kind": "cockpit_hop",
  "correlation_id": str,
  "hop": { /* CockpitHopV1 dump */ },
}
```

- Terminal (optional Slice A): `{"kind": "cockpit_timeline_complete", "correlation_id": str}` after final hop batch
- Bus: each hop also `await publish_cockpit_hop(bus, hop)` when bus available; fail-open (log, never raise into chat)

**Emission order for Slice A (per turn):**
1. `reset_seq(correlation_id)`
2. `gap_hop` for `ingress`, `association`, `stance_inputs`, `motor_boot` (deferred Slice B) — so the rail is honest
3. `hop_from_thought` when stance resolves to proceed (and also on defer/refuse before return)
4. Each motor step → `hop_from_motor_step` via relay path (preferred: build hop in relay or in orchestrator drain when `claude_step` arrives — pick **orchestrator drain** so seq stays single-threaded: when draining `claude_step`, also build/publish/send `cockpit_hop`)
5. On successful `HarnessRunV1`: `hop_from_run_artifact` hops; if outcome/closure available on run path, emit those too; else gap or skip until artifact lands
6. `cockpit_timeline_complete`

**Important:** Prefer assigning `seq` only inside `turn_orchestrator` drain/emit helpers so relay does not race the counter. Relay keeps emitting `claude_step`; orchestrator converts to cockpit hops when draining.

- [ ] **Step 1: Write failing tests**

```python
def test_orchestrator_emits_gap_then_stance_cockpit_hops():
    # Drive a minimal fake execute path or unit-test a new helper
    # emit_slice_a_pre_motor_hops(correlation_id, thought_dict) -> list[dict]
    from orion.hub.cockpit_emit import emit_slice_a_pre_motor_hops
    frames = emit_slice_a_pre_motor_hops(
        "corr-1",
        {"disposition": "proceed", "disposition_reasons": ["x"], "imperative": "go", "tone": "calm"},
    )
    kinds = [f.get("kind") for f in frames]
    assert kinds.count("cockpit_hop") >= 5  # 4 gaps + stance
    stages = [f["hop"]["stage"] for f in frames if f["kind"] == "cockpit_hop"]
    assert stages[:4] == ["ingress", "association", "stance_inputs", "motor_boot"]
    assert stages[4] == "stance_decision"
```

Put helper in `orion/hub/cockpit_emit.py` to keep `turn_orchestrator.py` thinner (allowed new file).

Also test: given a drained `claude_step` item, helper returns a `cockpit_hop` with `stage=motor_hop` and incremented seq.

- [ ] **Step 2: Run — expect FAIL**

Run: `pytest services/orion-hub/tests/test_turn_orchestrator_cockpit_hops.py -v`

- [ ] **Step 3: Implement `orion/hub/cockpit_emit.py` + wire into `run_unified_turn`**

- Create helpers that return WS frame dicts and optionally publish to bus
- In `run_unified_turn`, after stance decision known, send pre-motor frames
- When draining relay queue items with `kind=="claude_step"`, also emit cockpit hop frames
- After success frames, emit finalize/closure hops from run artifact fields + `cockpit_timeline_complete`
- Fail-open on publish errors

- [ ] **Step 4: Run — expect PASS**

Run:
```bash
pytest services/orion-hub/tests/test_turn_orchestrator_cockpit_hops.py services/orion-hub/tests/test_hub_harness_step_relay.py -v
```

- [ ] **Step 5: Commit**

```bash
git add orion/hub/cockpit_emit.py orion/hub/turn_orchestrator.py services/orion-hub/tests/test_turn_orchestrator_cockpit_hops.py
git commit -m "$(cat <<'EOF'
feat(hub): stream cockpit_hop frames on unified turns

Live Soft HUD spine: gaps, stance, motor steps, finalize hops.
EOF
)"
```

---

### Task 6: Soft HUD modal UI (visor, rail, scrubber, inspector)

**Files:**
- Create: `services/orion-hub/static/js/cockpit-hud.js`
- Create: `services/orion-hub/static/css/cockpit-hud.css`
- Modify: `services/orion-hub/templates/index.html` (link CSS/JS; modal root `div#cockpitHudRoot`)
- Test: `services/orion-hub/tests/test_cockpit_hud_ui.py` (node eval pattern from `test_turn_trace_panel_ui.py`)

**Interfaces:**
- Global `window.OrionCockpitHud`:
  - `open({ correlationId, apiBaseUrl })`
  - `close()`
  - `ingestHop(hop)` — append/reorder by seq; update rail/visor if follow-mode
  - `markComplete()`
  - `buildShell()` / `render(state)` for tests

**UI behavior (Slice A):**
- Full-screen Soft HUD (frosted glass center, soft vignette) — reuse modal open/close patterns from chat stance debug (`app.js` ~L4350) / substrate-effect fixed inset
- Hop rail from `state.hops`
- Scrubber index into hops; play steps on an interval
- Inspector shows `JSON.stringify(hop.raw, null, 2)` for selected hop + summary fields
- On open: `fetch(/api/chat/turn/{id}/cockpit)` then follow live `ingestHop`
- Gap hops styled muted on the rail

- [ ] **Step 1: Write failing UI tests**

```python
def test_build_shell_includes_soft_hud_hooks():
    html = _node_call("buildShell", {"correlationId": "c1"})
    assert "cockpit-hud" in html
    assert "cockpit-inspector" in html
    assert "cockpit-scrubber" in html


def test_render_orders_hops_and_marks_gap():
    html = _node_call(
        "renderFixture",
        {
            "hops": [
                {"seq": 0, "stage": "ingress", "status": "gap", "visor_line": "gap"},
                {"seq": 1, "stage": "stance_decision", "status": "ok", "visor_line": "proceed", "raw": {"x": 1}},
            ],
            "selectedSeq": 1,
        },
    )
    assert "stance_decision" in html
    assert "cockpit-hop-gap" in html or "status=\"gap\"" in html or "data-status=\"gap\"" in html
```

- [ ] **Step 2: Run — expect FAIL**

Run: `pytest services/orion-hub/tests/test_cockpit_hud_ui.py -v`

- [ ] **Step 3: Implement JS/CSS/HTML shell**

Keep JS focused: state `{hopsBySeq, order, selectedSeq, followLive, playing}`; no framework. Soft HUD colors: cool blue-gray glass, avoid purple neon / hard green terminal.

- [ ] **Step 4: Run — expect PASS**

Run: `pytest services/orion-hub/tests/test_cockpit_hud_ui.py -v`

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/static/js/cockpit-hud.js services/orion-hub/static/css/cockpit-hud.css services/orion-hub/templates/index.html services/orion-hub/tests/test_cockpit_hud_ui.py
git commit -m "$(cat <<'EOF'
feat(hub): Soft HUD cockpit modal shell

Visor, hop rail, scrubber, and inspector for turn sighting.
EOF
)"
```

---

### Task 7: Wire Cockpit button + WS ingest; preserve Turn Trace

**Files:**
- Modify: `services/orion-hub/static/js/app.js`
  - Near `appendTurnTracePanel` (~L6935): also append Cockpit button that calls `OrionCockpitHud.open({ correlationId, apiBaseUrl })`
  - In WS handler near `claude_step` (~L11370): on `d.kind === 'cockpit_hop'` call `OrionCockpitHud.ingestHop(d.hop)` if modal open for that correlation_id; on `cockpit_timeline_complete` call `markComplete()`
- Modify: `services/orion-hub/tests/test_turn_trace_panel_ui.py` or add assertions in `test_cockpit_hud_ui.py` that `app.js` still contains `mountTurnTracePanel` / `appendTurnTracePanel` and also `OrionCockpitHud`
- Modify: `services/orion-hub/README.md` — short “Cockpit (Soft HUD)” operator blurb under chat/debug

**Interfaces:**
- Consumes: Task 6 API + Task 5 WS frames + Task 4 GET

- [ ] **Step 1: Write failing wiring test**

```python
def test_app_js_wires_cockpit_beside_turn_trace():
    text = (REPO_ROOT / "services/orion-hub/static/js/app.js").read_text()
    assert "appendTurnTracePanel" in text
    assert "OrionCockpitHud" in text
    assert "cockpit_hop" in text
```

- [ ] **Step 2: Run — expect FAIL** (OrionCockpitHud / cockpit_hop missing)

- [ ] **Step 3: Wire app.js + README**

Implement button HTML near Turn Trace host. Ensure script tag order in `index.html`: `cockpit-hud.js` before `app.js`.

- [ ] **Step 4: Run — expect PASS**

```bash
pytest services/orion-hub/tests/test_cockpit_hud_ui.py services/orion-hub/tests/test_turn_trace_panel_ui.py -v
```

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/static/js/app.js services/orion-hub/templates/index.html services/orion-hub/README.md services/orion-hub/tests/
git commit -m "$(cat <<'EOF'
feat(hub): wire Cockpit button and live hop ingest

Turn Trace stays; Soft HUD opens beside it and follows WS.
EOF
)"
```

---

### Task 8: Agent-check gates + Slice B/C hooks doc note

**Files:**
- Modify: `docs/superpowers/specs/2026-09-06-orion-cockpit-pov-design.md` — status line to “Slice A planned; B/C follow-on plans”
- Create (short): `docs/superpowers/plans/2026-09-06-orion-cockpit-pov-slice-b-stub.md` listing only the deferred stages/files (stance_inputs capture site, motor_boot prefix assembly site, association reads) — stub is OK as a checklist pointer, not a full writing-plans doc
- Run gates

- [ ] **Step 1: Run focused gates**

```bash
python scripts/check_schema_registry.py
python scripts/check_bus_channels.py
pytest orion/cockpit/tests -q
pytest services/orion-sql-writer/tests/test_cockpit_turn_sighting_sql_shape.py -q
pytest services/orion-hub/tests/test_chat_cockpit_routes.py services/orion-hub/tests/test_turn_orchestrator_cockpit_hops.py services/orion-hub/tests/test_cockpit_hud_ui.py services/orion-hub/tests/test_turn_trace_panel_ui.py -q
```

Expected: all PASS

- [ ] **Step 2: safe graphify update once**

```bash
scripts/safe_graphify_update.sh
```

- [ ] **Step 3: Commit docs/hooks if any graphify artifacts are intentionally updated (only if the wrapper produced a legitimate growth; do not force-add a shrunk graph)**

```bash
git add docs/superpowers/specs/2026-09-06-orion-cockpit-pov-design.md docs/superpowers/plans/2026-09-06-orion-cockpit-pov-slice-b-stub.md
git commit -m "$(cat <<'EOF'
docs(cockpit): mark Slice A plan complete; stub B follow-ons

EOF
)"
```

- [ ] **Step 4: Manual smoke (operator)**

After deploy (Hub + sql-writer + harness-governor as needed):

1. Send an Orion-mode Hub chat turn  
2. Click **Cockpit** mid-turn — hops should append  
3. After complete, scrub to a `motor_hop` and confirm inspector raw shows tool payload  
4. Confirm gap beads exist for `motor_boot` / `stance_inputs`  
5. Confirm Turn Trace still opens  

Restart commands (print for Juniper, do not sudo):

```bash
# from the service worktree, via safe wrapper:
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
# harness-governor only if emit path required a rebuild
```

---

## Spec coverage self-check

| Spec requirement | Task |
|------------------|------|
| Soft HUD modal pretty | 6 |
| Thick dual surface (visor + inspector raw) | 6–7 |
| Live + rewind | 4–5, 6–7 |
| Keep Turn Trace | 7 |
| Timeline hops with seq/stage/visor/raw/status | 1–2 |
| Canonical stages + gaps | 2, 4–5 |
| Persist timeline | 3 |
| Hub API | 4 |
| WS live | 5, 7 |
| First producers: stance, motor, finalize | 5 |
| Slice B/C deferred but scheduled | 8 stub |
| Failure honesty / no empty-shell | 4–5 |

## Placeholder scan

None intentional. Implementers must not leave `TBD` in code; gap hops are the honest deferral mechanism.

## Type consistency

- Schema id: `CockpitHopV1`
- Kind: `cockpit.hop.v1`
- Channel: `orion:cockpit:hop`
- WS: `kind: "cockpit_hop"` with `hop` body; complete: `cockpit_timeline_complete`
- SQL model: `CockpitTurnSightingSQL`
- Table: `cockpit_turn_sighting`
- JS: `window.OrionCockpitHud`

---

## Follow-on plans (not this PR)

- **Slice B:** capture real `stance_inputs` bundle + `motor_boot` exact prefix at assembly site + association reads (replace gap hops)
- **Slice C:** remaining offboarding side-effects beyond outcome/closure already emitted in A
