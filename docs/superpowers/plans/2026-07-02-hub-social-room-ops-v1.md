# Hub Social Room Ops v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Hub `hub_direct` social room memory-backed, inspectable via the Social Inspection panel, and operator-legible (Mode vs Compute), by fixing social-memory schema drift and two Hub wiring gaps.

**Architecture:** Apply a manual Postgres migration so `orion-social-memory` `/summary` and `/inspection` stop 500ing; store websocket turn `routing_debug` in the existing in-memory inspection cache (parity with HTTP `/api/chat`); augment hub-direct inspection snapshots with sections built from `route_debug` fields already on the turn (no bridge policy fakes). Document Mode/Compute/social-room override semantics in Hub README and tooltips.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy, vanilla JS, pytest, Postgres (`conjourney` DSN), existing `orion/inspection/social.py` + `social_room_inspection_cache`.

**Spec:** `docs/superpowers/specs/2026-07-02-hub-social-room-ops-v1-design.md`

---

## File map

| File | Responsibility |
|------|----------------|
| `services/orion-sql-db/manual_migration_social_memory_calibration_v1.sql` | Add missing social-memory columns (`IF NOT EXISTS`) |
| `services/orion-social-memory/README.md` | Operator migration + smoke curls |
| `orion/inspection/social.py` | `build_hub_direct_inspection_sections()` |
| `services/orion-hub/scripts/social_room.py` | Merge hub sections in `build_social_inspection_debug()` |
| `services/orion-hub/scripts/websocket_handler.py` | Call `social_room_inspection_cache.store()` after WS social turns |
| `services/orion-hub/tests/test_hub_direct_inspection_cache.py` | WS cache regression |
| `tests/test_social_inspection.py` | Hub-direct section unit tests |
| `services/orion-hub/tests/test_social_room.py` | Integration via `build_social_inspection_debug` |
| `services/orion-hub/README.md` | Mode/Compute/social-room table |
| `services/orion-hub/templates/index.html` | Mode/Compute tooltips |
| `services/orion-social-room-bridge/.env_example` | Bridge-only env comment |

---

### Task 1: Social-memory schema migration

**Files:**
- Create: `services/orion-sql-db/manual_migration_social_memory_calibration_v1.sql`
- Create: `services/orion-social-memory/README.md`

- [ ] **Step 1: Create migration SQL**

Create `services/orion-sql-db/manual_migration_social_memory_calibration_v1.sql`:

```sql
-- Hub social-room ops v1: align live Postgres with orion-social-memory models.
-- Run against the conjourney DSN used by orion-social-memory (see service README).

BEGIN;

-- social_participant_continuity (hygiene + artifact dialogue columns)
ALTER TABLE social_participant_continuity ADD COLUMN IF NOT EXISTS shared_artifact_proposal JSONB;
ALTER TABLE social_participant_continuity ADD COLUMN IF NOT EXISTS shared_artifact_revision JSONB;
ALTER TABLE social_participant_continuity ADD COLUMN IF NOT EXISTS shared_artifact_confirmation JSONB;
ALTER TABLE social_participant_continuity ADD COLUMN IF NOT EXISTS calibration_signals JSONB;
ALTER TABLE social_participant_continuity ADD COLUMN IF NOT EXISTS peer_calibration JSONB;
ALTER TABLE social_participant_continuity ADD COLUMN IF NOT EXISTS trust_boundary JSONB;
ALTER TABLE social_participant_continuity ADD COLUMN IF NOT EXISTS memory_freshness JSONB;
ALTER TABLE social_participant_continuity ADD COLUMN IF NOT EXISTS decay_signals JSONB;
ALTER TABLE social_participant_continuity ADD COLUMN IF NOT EXISTS regrounding_decisions JSONB;

-- social_room_continuity (thread/deliberation/gif/hygiene columns)
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS shared_artifact_proposal JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS shared_artifact_revision JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS shared_artifact_confirmation JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS active_threads JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS current_thread_key TEXT;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS current_thread_summary TEXT NOT NULL DEFAULT '';
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS handoff_signal JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS active_claims JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS recent_claim_revisions JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS claim_attributions JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS claim_consensus_states JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS claim_divergence_signals JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS bridge_summary JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS clarifying_question JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS deliberation_decision JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS turn_handoff JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS closure_signal JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS floor_decision JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS gif_usage_state JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS active_commitments JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS calibration_signals JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS peer_calibrations JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS trust_boundaries JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS memory_freshness JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS decay_signals JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS regrounding_decisions JSONB;

COMMIT;
```

- [ ] **Step 2: Create service README with operator steps**

Create `services/orion-social-memory/README.md`:

```markdown
# orion-social-memory

Relational continuity synthesizer for social-room turns (`orion:chat:social:stored`).

## Database migration (required on existing installs)

`Base.metadata.create_all()` does **not** add columns to existing tables. After pulling hub-social-room-ops-v1, run:

```bash
psql "$DATABASE_URL" -f services/orion-sql-db/manual_migration_social_memory_calibration_v1.sql
```

Use the same DSN as `DATABASE_URL` in this service's `.env` (default database: `conjourney`).

## Smoke checks

```bash
curl -fsS 'http://localhost:8765/health'
curl -fsS 'http://localhost:8765/summary?platform=hub&room_id=hub-direct&participant_id=juniper'
curl -fsS 'http://localhost:8765/inspection?platform=hub&room_id=hub-direct&participant_id=juniper'
```

Restart `orion-social-memory` after migration if it was crash-looping on schema errors.
```

- [ ] **Step 3: Apply migration on operator Postgres**

Run (adjust host/credentials to match Athena):

```bash
psql "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney" \
  -f services/orion-sql-db/manual_migration_social_memory_calibration_v1.sql
```

Expected: `ALTER TABLE` lines for each column (or `NOTICE: column … already exists, skipping`).

- [ ] **Step 4: Verify social-memory endpoints**

```bash
curl -fsS 'http://localhost:8765/summary?platform=hub&room_id=hub-direct&participant_id=juniper' | head -c 200
curl -fsS 'http://localhost:8765/inspection?platform=hub&room_id=hub-direct&participant_id=juniper' | head -c 200
```

Expected: HTTP 200 JSON (empty participant/room objects OK on fresh room).

- [ ] **Step 5: Commit**

```bash
git add services/orion-sql-db/manual_migration_social_memory_calibration_v1.sql services/orion-social-memory/README.md
git commit -m "fix(social-memory): add calibration/hygiene column migration for hub-direct room"
```

---

### Task 2: Hub-direct inspection section builder

**Files:**
- Modify: `orion/inspection/social.py`
- Test: `tests/test_social_inspection.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_social_inspection.py`:

```python
from orion.inspection.social import build_hub_direct_inspection_sections


def test_hub_direct_inspection_sections_include_routing_recall_and_gif_na() -> None:
    route_debug = {
        "verb": "chat_social_room",
        "mode": "brain",
        "llm_route": "chat",
        "social_room_mode": "hub_direct",
        "chat_profile": "social_room",
        "recall_profile": "social.room.v1",
        "recall_enabled": True,
        "social_redaction_posture": "strict",
    }
    sections = build_hub_direct_inspection_sections(route_debug=route_debug, metadata={})
    kinds = {section.section_kind for section in sections}
    assert "routing" in kinds
    assert "context_window" in kinds
    assert "gif" in kinds
    gif = next(item for item in sections if item.section_kind == "gif")
    assert any("bridge-only" in trace.summary.lower() for trace in gif.decision_traces)


def test_hub_direct_inspection_sections_include_skill_when_present() -> None:
    route_debug = {
        "verb": "chat_social_room",
        "social_room_mode": "hub_direct",
        "social_skill_selection": {"selected_skill": "social_self_ground", "used": True},
        "social_skill_result": {"summary": "Oríon is here as a peer."},
    }
    sections = build_hub_direct_inspection_sections(route_debug=route_debug, metadata={})
    assert any(section.section_kind == "artifact_dialogue" for section in sections)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. pytest tests/test_social_inspection.py::test_hub_direct_inspection_sections_include_routing_recall_and_gif_na tests/test_social_inspection.py::test_hub_direct_inspection_sections_include_skill_when_present -q
```

Expected: FAIL with `ImportError` or `cannot import name 'build_hub_direct_inspection_sections'`

- [ ] **Step 3: Implement `build_hub_direct_inspection_sections`**

Add to `orion/inspection/social.py` (after `_make_section`, before `build_social_inspection_snapshot`):

```python
def build_hub_direct_inspection_sections(
    *,
    route_debug: Dict[str, Any],
    metadata: Dict[str, Any] | None = None,
) -> list[SocialInspectionSectionV1]:
    """Materialize hub-local inspection sections when bridge turn-policy surfaces are absent."""
    debug = dict(route_debug or {})
    meta = dict(metadata or {})
    sections: list[SocialInspectionSectionV1] = []

    verb = _compact_text(debug.get("verb") or meta.get("verb"), limit=80)
    mode = _compact_text(debug.get("mode") or debug.get("selected_ui_route"), limit=40)
    llm_route = _compact_text((debug.get("options") or {}).get("llm_route") or debug.get("llm_route"), limit=40)
    social_mode = _compact_text(debug.get("social_room_mode"), limit=40)
    routing_selected = [item for item in (f"verb={verb}" if verb else "", f"mode={mode}" if mode else "", f"compute={llm_route}" if llm_route else "", f"social_room_mode={social_mode}" if social_mode else "") if item]
    routing_section = _make_section(
        section_kind="routing",
        included=[],
        selected=routing_selected,
        softened=[],
        excluded=[],
        why="Hub-direct routing uses chat_social_room; compute lane comes from the Hub Compute dropdown.",
        traces=[
            _trace(
                trace_kind="hub_direct_route",
                decision_state="active",
                summary=f"{verb or 'chat_social_room'} on compute lane {llm_route or 'quick'}",
                why="Social room toggle forces the social verb; llm_route is orthogonal.",
            )
        ],
        metadata={"source": "hub_direct"},
    )
    if routing_section is not None:
        sections.append(routing_section)

    recall_profile = _compact_text(debug.get("recall_profile") or (debug.get("options") or {}).get("recall_profile"), limit=80)
    recall_enabled = debug.get("recall_enabled")
    recall_selected = [
        item
        for item in (
            f"profile={recall_profile}" if recall_profile else "",
            f"enabled={recall_enabled}" if recall_enabled is not None else "",
            "lane=social",
        )
        if item
    ]
    context_section = _make_section(
        section_kind="context_window",
        included=[],
        selected=recall_selected,
        softened=[],
        excluded=[],
        why="Hub-direct recall uses social.room.v1 when enabled.",
        traces=[
            _trace(
                trace_kind="hub_direct_recall",
                decision_state="active",
                summary=recall_profile or "social.room.v1",
                why="Recall profile is pinned for social_room turns.",
            )
        ],
    )
    if context_section is not None:
        sections.append(context_section)

    posture = _compact_text(debug.get("social_redaction_posture") or meta.get("social_redaction_posture"), limit=40)
    epistemic_section = _make_section(
        section_kind="epistemic",
        included=[],
        selected=[f"redaction_posture={posture}"] if posture else [],
        softened=[],
        excluded=[],
        why="Redaction posture gates keyword blocklist and artifact scope defaults.",
        traces=[],
    )
    if epistemic_section is not None:
        sections.append(epistemic_section)

    skill_selection = debug.get("social_skill_selection") if isinstance(debug.get("social_skill_selection"), dict) else {}
    skill_result = debug.get("social_skill_result") if isinstance(debug.get("social_skill_result"), dict) else {}
    if skill_selection.get("selected_skill") or skill_result.get("summary"):
        skill_section = _make_section(
            section_kind="artifact_dialogue",
            included=[_safe_text(skill_result.get("summary"), limit=180)] if skill_result.get("summary") else [],
            selected=[_compact_text(skill_selection.get("selected_skill"), limit=80)] if skill_selection.get("selected_skill") else [],
            softened=[],
            excluded=[],
            why="Deterministic social skill surfacing ran before the LLM reply.",
            traces=[],
        )
        if skill_section is not None:
            sections.append(skill_section)

    safety_section = _make_section(
        section_kind="safety",
        included=[],
        selected=[f"posture={posture}"] if posture else ["bounded recall"],
        softened=[],
        excluded=[],
        why="Hub-direct keeps PII regex hygiene regardless of posture.",
        traces=[],
    )
    if safety_section is not None:
        sections.append(safety_section)

    gif_section = _make_section(
        section_kind="gif",
        included=[],
        selected=["hub_direct — GIF policy N/A (bridge-only)"],
        softened=[],
        excluded=[],
        why="GIF observe/policy/post lives in orion-social-room-bridge, not Hub websocket chat.",
        traces=[
            _trace(
                trace_kind="hub_direct_gif",
                decision_state="inactive",
                summary="GIF policy N/A on hub_direct",
                why="Bridge-only transport seam; Hub UI is text-only.",
            )
        ],
    )
    if gif_section is not None:
        sections.append(gif_section)

    return sections
```

Export is via existing module; no `__init__.py` change needed if tests import from `orion.inspection.social`.

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=. pytest tests/test_social_inspection.py::test_hub_direct_inspection_sections_include_routing_recall_and_gif_na tests/test_social_inspection.py::test_hub_direct_inspection_sections_include_skill_when_present -q
```

Expected: PASS (2 tests)

- [ ] **Step 5: Wire into `build_social_inspection_debug`**

In `services/orion-hub/scripts/social_room.py`, add import:

```python
from orion.inspection.social import build_hub_direct_inspection_sections
```

At end of `build_social_inspection_debug`, before `return SocialInspectionSnapshotV1.model_validate(inspection).model_dump(mode="json")`:

```python
    hub_direct = str(
        route_debug.get("social_room_mode") or payload.get("social_room_mode") or ""
    ).strip().lower() == "hub_direct"
    if hub_direct:
        hub_sections = build_hub_direct_inspection_sections(route_debug=route_debug, metadata=metadata)
        if hub_sections:
            inspection.sections.extend(hub_sections)
            inspection.summary = inspection.summary or (
                f"Hub-direct turn · {len(inspection.sections)} sections · verb {route_debug.get('verb') or 'chat_social_room'}"
            )
```

- [ ] **Step 6: Add integration test in `services/orion-hub/tests/test_social_room.py`**

```python
def test_build_social_inspection_debug_hub_direct_has_sections() -> None:
    payload = {
        "chat_profile": "social_room",
        "social_room_mode": "hub_direct",
        "external_room": {"platform": "hub", "room_id": "hub-direct"},
        "external_participant": {"participant_id": "juniper"},
    }
    route_debug = {
        "verb": "chat_social_room",
        "mode": "brain",
        "llm_route": "chat",
        "social_room_mode": "hub_direct",
        "recall_profile": "social.room.v1",
        "recall_enabled": True,
        "social_redaction_posture": "strict",
    }
    snapshot = hub_social_room.build_social_inspection_debug(
        payload=payload,
        route_debug=route_debug,
        metadata={"social_redaction_posture": "strict"},
    )
    assert len(snapshot["sections"]) >= 3
    assert snapshot["platform"] == "hub"
    assert snapshot["room_id"] == "hub-direct"
```

- [ ] **Step 7: Run hub social_room tests**

```bash
PYTHONPATH=. pytest services/orion-hub/tests/test_social_room.py::test_build_social_inspection_debug_hub_direct_has_sections tests/test_social_inspection.py -q
```

Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add orion/inspection/social.py services/orion-hub/scripts/social_room.py tests/test_social_inspection.py services/orion-hub/tests/test_social_room.py
git commit -m "feat(hub-social-room): add hub_direct inspection sections for live panel"
```

---

### Task 3: Websocket inspection cache parity

**Files:**
- Modify: `services/orion-hub/scripts/websocket_handler.py`
- Create: `services/orion-hub/tests/test_hub_direct_inspection_cache.py`

- [ ] **Step 1: Write failing test**

Create `services/orion-hub/tests/test_hub_direct_inspection_cache.py`:

```python
from __future__ import annotations

import importlib


def test_store_social_room_inspection_from_route_debug() -> None:
    cache = importlib.import_module("scripts.social_room_inspection_cache")
    cache._latest.clear()
    route_debug = {
        "verb": "chat_social_room",
        "social_room_mode": "hub_direct",
        "social_inspection": {"platform": "hub", "room_id": "hub-direct", "sections": []},
    }
    cache.store("hub-direct", route_debug)
    entry = cache.get("hub-direct")
    assert entry is not None
    assert entry["routing_debug"]["verb"] == "chat_social_room"
    assert entry["room_id"] == "hub-direct"
    assert entry["stored_at"]
```

- [ ] **Step 2: Run test to verify it passes (cache module already exists)**

```bash
PYTHONPATH=services/orion-hub pytest services/orion-hub/tests/test_hub_direct_inspection_cache.py -q
```

Expected: PASS (validates cache API used by next step)

- [ ] **Step 3: Add store call in websocket handler**

In `services/orion-hub/scripts/websocket_handler.py`, add import near other script imports:

```python
from scripts import social_room_inspection_cache
```

After `route_debug` is finalized and before `await websocket.send_json(...)`, add:

```python
            if is_social_room_payload(data):
                _social_room_id = str((data.get("external_room") or {}).get("room_id") or "").strip()
                if _social_room_id and route_debug:
                    social_room_inspection_cache.store(_social_room_id, route_debug)
```

Place immediately before the `ws_payload = {` block (~line 1249) so `route_debug` includes `social_inspection`.

- [ ] **Step 4: Extend test to document websocket contract (optional assertion helper)**

Add to `test_hub_direct_inspection_cache.py`:

```python
def test_get_latest_returns_most_recent_room() -> None:
    cache = importlib.import_module("scripts.social_room_inspection_cache")
    cache._latest.clear()
    cache.store("hub-direct", {"verb": "chat_social_room", "corr": "a"})
    cache.store("other-room", {"verb": "chat_social_room", "corr": "b"})
    latest = cache.get_latest()
    assert latest is not None
    assert latest["room_id"] in {"hub-direct", "other-room"}
```

- [ ] **Step 5: Run tests**

```bash
PYTHONPATH=services/orion-hub pytest services/orion-hub/tests/test_hub_direct_inspection_cache.py -q
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add services/orion-hub/scripts/websocket_handler.py services/orion-hub/tests/test_hub_direct_inspection_cache.py
git commit -m "fix(hub): store social-room routing_debug from websocket turns for inspection poll"
```

---

### Task 4: Mode vs Compute operator clarity

**Files:**
- Modify: `services/orion-hub/README.md`
- Modify: `services/orion-hub/templates/index.html`
- Modify: `services/orion-social-room-bridge/.env_example`

- [ ] **Step 1: Add Mode/Compute table to Hub README**

In `services/orion-hub/README.md`, after the existing **Compute lane override** paragraph (~line 427), append:

```markdown
**Social room toggle vs Mode vs Compute:**

| Control | What it sets | Social room ON override |
|---------|----------------|-------------------------|
| **Mode** | Behavior lane (`brain`, `agent`, `council`, …) | Forces `mode=brain` |
| **Compute** | GPU/model lane via `llm_route` (`chat`, `quick`, …) | **Still applies** — pick which model serves `chat_social_room` |
| **Social room** checkbox | `chat_profile=social_room`, `social_room_mode=hub_direct` | Forces verb **`chat_social_room`** (not `chat_general` / `chat_quick`) |

Bridge env keys `SOCIAL_BRIDGE_HUB_MODE` / `SOCIAL_BRIDGE_HUB_VERB` affect **CallSyne bridge → Hub** calls only; the Hub UI toggle ignores them.
```

- [ ] **Step 2: Add tooltips to Mode and Compute selects**

In `services/orion-hub/templates/index.html`, on `#hubModeSelect` add:

```html
title="Behavior lane: agent/council/brain routing. Social room ON forces brain mode and chat_social_room verb."
```

On `#hubComputeSelect` add:

```html
title="GPU/model lane (llm_route): chat, quick, agent, metacog. Still applies when Social room is ON."
```

- [ ] **Step 3: Comment bridge env example**

In `services/orion-social-room-bridge/.env_example`, above `SOCIAL_BRIDGE_HUB_VERB`:

```env
# Bridge-only: verb sent when bridge calls Hub /api/chat. Empty = omit verbs (Hub resolves chat_social_room).
# Hub UI social-room toggle ignores this; do not set chat_quick if you want peer-mode social room from CallSyne.
SOCIAL_BRIDGE_HUB_VERB=
```

(Leave local `.env` unchanged unless operator opts in.)

- [ ] **Step 4: Commit**

```bash
git add services/orion-hub/README.md services/orion-hub/templates/index.html services/orion-social-room-bridge/.env_example
git commit -m "docs(hub): clarify Mode vs Compute vs social room override"
```

---

### Task 5: End-to-end verification

- [ ] **Step 1: Run focused test suite**

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. pytest tests/test_social_inspection.py services/orion-hub/tests/test_hub_direct_inspection_cache.py services/orion-hub/tests/test_social_room.py services/orion-hub/tests/test_hub_direct_social_room_isolation.py -q
```

Expected: all PASS

- [ ] **Step 2: Restart services (operator)**

```bash
docker compose -f services/orion-social-memory/docker-compose.yml restart
docker compose -f services/orion-hub/docker-compose.yml restart
```

- [ ] **Step 3: Manual Hub smoke**

1. Enable **Social room** toggle; send one message.
2. Confirm routing debug shows `verb: chat_social_room`, `llm_route: chat` (or selected compute).
3. Open Social Inspection — live sections ≥3; no red **Internal Server Error**.
4. `curl -fsS http://localhost:8080/api/social-room/inspection/latest | jq '.routing_debug.verb,.stored_at'`

Expected: `"chat_social_room"` and ISO timestamp.

- [ ] **Step 4: Final commit if any fixups needed**

Only if verification required small fixes; otherwise skip.

---

## Plan self-review (spec coverage)

| Spec requirement | Task |
|------------------|------|
| A — `/summary` 200 | Task 1 migration + smoke |
| B — `/inspection` 200 | Task 1 migration + smoke |
| C — WS inspection cache | Task 3 |
| D — ≥3 hub-direct sections | Task 2 |
| E — Mode/Compute docs | Task 4 |
| Non-goals (no GIF/Cursor/bridge) | Explicit in spec; no tasks |

No TBD placeholders. Type names consistent: `build_hub_direct_inspection_sections`, `social_room_inspection_cache.store`.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-02-hub-social-room-ops-v1.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration  
2. **Inline Execution** — run tasks in this session with executing-plans checkpoints  

Which approach do you want?
