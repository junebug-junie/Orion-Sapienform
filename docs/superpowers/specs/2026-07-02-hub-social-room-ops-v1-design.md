# Hub Social Room Ops v1 — design

**Date:** 2026-07-02  
**Status:** Approved (scope A)  
**Scope:** `orion-social-memory` schema, `orion-hub` websocket inspection + panel, Mode/Compute operator clarity  
**Related:** [2026-06-30 hub social-room toggle plan](../plans/2026-06-30-hub-social-room-toggle.md); live trace `46a5d6cf-8e8d-46d5-89cb-e509186b1f7c`

---

## 1. Purpose

The Hub **Social room** toggle ships the `chat_social_room` chat rail correctly, but the **operational layer** around it is broken or incomplete on Athena:

| Gap | Live evidence |
|-----|----------------|
| Social-memory schema drift | `GET /summary` and `/inspection` → 500; `UndefinedColumn: calibration_signals` |
| Websocket inspection cache | `/api/social-room/inspection/latest` always null for UI turns |
| Hub-direct inspection panel | `sections=0`; UI shows “No live social inspection captured” |
| Mode vs Compute semantics | Operators confuse Mode, Compute, and bridge `.env` knobs |

**Goal:** Make hub_direct social room **memory-backed, inspectable, and operator-legible** without expanding scope to CallSyne bridge, GIFs, multi-participant, or multi-reply autonomy.

**Success criteria (all required):**

| ID | Criterion |
|----|-----------|
| A | `GET /summary?platform=hub&room_id=hub-direct&participant_id=juniper` → 200 after migration |
| B | `GET /api/social-memory/inspection?…` (Hub proxy) → 200 |
| C | Hub websocket turn stores `social_room_inspection_cache`; `/api/social-room/inspection/latest` non-null |
| D | Hub-direct live inspection shows ≥3 sections (routing, recall/context, safety or gif N/A) without bridge policy |
| E | Hub README + Mode/Compute tooltips document social-room overrides |

---

## 2. Problem anatomy (verified live)

Trace `46a5d6cf-8e8d-46d5-89cb-e509186b1f7c`:

```text
hub_direct_social_memory_fetch_failed → /summary 500
Routing: mode=brain verb=chat_social_room llm_route=chat  ✅
hub_ingress_result: memory_used=True recall_count=5 final_len=246  ✅
social_inspection_snapshot_built sections=0 traces=0
/api/social-memory/inspection → 500 Internal Server Error
/api/social-room/inspection/latest → routing_debug null (WS path never stores)
```

**Choke points:**

1. `orion-social-memory` — `Base.metadata.create_all()` does not alter existing tables; code expects columns absent from live Postgres.
2. `orion-hub/scripts/api_routes.py:2485` — only HTTP `/api/chat` calls `social_room_inspection_cache.store()`.
3. `orion-hub/scripts/social_room.py:build_social_inspection_debug` — passes empty bridge/memory surfaces on hub_direct → zero sections.
4. Operator docs — Mode/Compute/social-room interaction undocumented in one place.

---

## 3. Chosen approach

**Thin seams, repo-native (Approach 1).**

| Approach | Verdict |
|----------|---------|
| A — Thin seams (migration + WS cache + hub sections + docs) | **Selected** |
| B — Shared post-turn helper refactor | Deferred; optional if WS patch duplicates >10 lines |
| C — UI-only band-aids | Rejected; leaves schema blocker |

---

## 4. Architecture

```text
Hub UI (social room ON, hub_direct)
  │
  ├─ prefetch GET orion-social-memory/summary  ← requires migration
  ├─ build_chat_request → chat_social_room + llm_route from Compute dropdown
  ├─ cortex → reply
  ├─ build_social_inspection_debug (+ hub_direct sections)
  ├─ social_room_inspection_cache.store(room_id, route_debug)  ← NEW on WS path
  └─ publish_social_room_turn → bus → social-memory synthesizer

Social Inspection panel
  ├─ Live: routing_debug.social_inspection (WS + poll cache)
  └─ Memory: GET /api/social-memory/inspection → social-memory /inspection
```

**Isolation preserved:** `platform=hub`, `room_id=hub-direct`. No bridge, CallSyne, or GIF transport.

---

## 5. Component design

### 5.1 Social-memory schema migration (blocker)

**Add:** `services/orion-sql-db/manual_migration_social_memory_calibration_v1.sql`

Use `ALTER TABLE … ADD COLUMN IF NOT EXISTS` for JSON/text columns present in `services/orion-social-memory/app/models.py` but missing on live DB, at minimum:

**`social_participant_continuity`:**  
`shared_artifact_proposal`, `shared_artifact_revision`, `shared_artifact_confirmation`,  
`calibration_signals`, `peer_calibration`, `trust_boundary`, `memory_freshness`, `decay_signals`, `regrounding_decisions`

**`social_room_continuity`:**  
`shared_artifact_*`, `active_threads`, `current_thread_key`, `current_thread_summary`, `handoff_signal`,  
`active_claims`, `recent_claim_revisions`, `claim_attributions`, `claim_consensus_states`, `claim_divergence_signals`,  
`bridge_summary`, `clarifying_question`, `deliberation_decision`, `turn_handoff`, `closure_signal`, `floor_decision`,  
`gif_usage_state`, `active_commitments`, `calibration_signals`, `peer_calibrations`, `trust_boundaries`,  
`memory_freshness`, `decay_signals`, `regrounding_decisions`

**Operator step:** Run against `conjourney` DSN (same as `orion-social-memory` `DATABASE_URL`). Document in new `services/orion-social-memory/README.md`.

**Non-goals:** Alembic; auto-migrate on service boot.

### 5.2 Websocket inspection cache parity

**Modify:** `services/orion-hub/scripts/websocket_handler.py`

After successful social-room turn, when `route_debug` exists and `external_room.room_id` is set:

```python
from scripts import social_room_inspection_cache

_social_room_id = str((data.get("external_room") or {}).get("room_id") or "").strip()
if _social_room_id and route_debug:
    social_room_inspection_cache.store(_social_room_id, route_debug)
```

Mirror `api_routes.py:2483-2485`. Import path consistent with existing hub script imports.

### 5.3 Hub-direct live inspection sections

**Modify:** `orion/inspection/social.py` — add `build_hub_direct_inspection_sections(route_debug, metadata) -> list[SocialInspectionSectionV1]`

Materialize sections from hub-local `route_debug` (no bridge fakes):

| `section_kind` | Content |
|----------------|---------|
| `routing` | `verb`, `mode`, `llm_route`, `social_room_mode`, `chat_profile` |
| `context_window` | recall profile, enabled/required, lane `social` |
| `epistemic` | `social_redaction_posture` |
| `artifact_dialogue` | skill selection/result when present |
| `safety` | redaction posture + bounded recall note |
| `gif` | trace: `hub_direct — GIF policy N/A (bridge-only)` |

**Modify:** `orion-hub/scripts/social_room.py` — in `build_social_inspection_debug`, when `payload.get("social_room_mode") == "hub_direct"` or `route_debug.get("social_room_mode") == "hub_direct"`, merge hub sections into snapshot before return.

**Optional UI tweak:** `social-inspection.js` — map gif N/A to badge label `N/A (hub)` when trace contains `bridge-only`.

### 5.4 Mode vs Compute semantics

**Modify:**

- `services/orion-hub/README.md` — table: Mode vs Compute vs Social room override (see brainstorming design).
- `services/orion-hub/templates/index.html` — `title` on `#hubModeSelect` and `#hubComputeSelect`.
- `services/orion-social-room-bridge/.env_example` — comment on `SOCIAL_BRIDGE_HUB_VERB` (bridge-only; hub toggle ignores).

**Non-goals:** Rename dropdowns; change default compute lane; enable bridge.

---

## 6. Non-goals (scope A)

- Cursor as room participant  
- Hub GIF delivery  
- Bridge consecutive-turn / autonomy on hub_direct  
- Enabling or reconfiguring CallSyne bridge  
- New bus channels or schema registry entries  
- HTTP `/api/chat` path changes beyond existing cache behavior  

---

## 7. Files touched

| File | Change |
|------|--------|
| `services/orion-sql-db/manual_migration_social_memory_calibration_v1.sql` | Create |
| `services/orion-social-memory/README.md` | Create — migration + health checks |
| `orion/inspection/social.py` | Hub-direct section builder |
| `services/orion-hub/scripts/social_room.py` | Merge hub sections into snapshot |
| `services/orion-hub/scripts/websocket_handler.py` | Inspection cache store |
| `services/orion-hub/tests/test_hub_direct_inspection_cache.py` | Create |
| `services/orion-hub/tests/test_social_room.py` | Hub-direct inspection sections |
| `tests/test_social_inspection.py` | Unit tests for hub sections |
| `services/orion-hub/README.md` | Mode/Compute table |
| `services/orion-hub/templates/index.html` | Tooltips |
| `services/orion-social-room-bridge/.env_example` | Bridge env comment |

---

## 8. Acceptance checks

```bash
# After migration on conjourney:
curl -fsS 'http://localhost:8765/summary?platform=hub&room_id=hub-direct&participant_id=juniper'
curl -fsS 'http://localhost:8080/api/social-memory/inspection?platform=hub&room_id=hub-direct&participant_id=juniper'

# After hub patch + social room toggle ON, send one message:
curl -fsS 'http://localhost:8080/api/social-room/inspection/latest' | jq '.routing_debug.verb,.stored_at'

# Unit tests:
PYTHONPATH=. pytest tests/test_social_inspection.py services/orion-hub/tests/test_hub_direct_inspection_cache.py services/orion-hub/tests/test_social_room.py -q
```

**Manual:** Social Inspection panel shows non-empty Route badge (`chat_social_room` / `chat`), live sections ≥3, memory pane loads without 500.

---

## 9. Risks

| Risk | Mitigation |
|------|------------|
| Migration incomplete (other missing columns) | Migration lists all model columns; smoke curls after apply |
| In-memory cache lost on hub restart | Acceptable for debug; WS response still carries routing_debug |
| Hub sections diverge from bridge inspection | Explicit `hub_direct` branch; bridge surfaces unchanged |

---

## 10. Recommended implementation order

1. Migration + social-memory README  
2. Websocket inspection cache  
3. Hub-direct inspection sections + tests  
4. Docs/tooltips  

Implementation plan: `docs/superpowers/plans/2026-07-02-hub-social-room-ops-v1.md`
