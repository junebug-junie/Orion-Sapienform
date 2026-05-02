# Memory Cards — Phase 4 Hub completion (delta spec)

**Date:** 2026-05-02  
**Status:** Draft for implementation planning  
**Parent spec:** [Orion Memory Cards v1](2026-05-01-orion-memory-cards-v1-design.md) (approved)  
**Companion:** [Memory Cards v1 offboarding](../guides/2026-05-01-memory-cards-v1-offboarding.md)

This document scopes **only** Hub operator UX and HTTP parity for Phase 4. It does not repeat parent ground rules (§2), lane semantics (§3), data model (§4), or Phase 5–6 work. Where this doc conflicts with the parent, **the parent wins** unless this doc explicitly supersedes a *path* or *integration* detail listed below.

---

## 1. Purpose

Finish Phase 4 so operators can curate memory cards from the Hub without ad-hoc HTTP clients: full route surface per parent §10, Memory tab UI, and **`lane` passed from Hub into recall/cortex payloads** wherever Hub already distinguishes social vs chat context.

---

## 2. Normative constraints (inherited)

- Parent **§2** ground rules: no default profile churn except `self.factual.v1.yaml`; auto-extractor off by default; Stage 2 `NotImplementedError` if forced; cards are data not identity mutation.
- Parent **§3** lane invariant: `RecallQueryV1.lane` set from request context; filter rule unchanged (`'all' in visibility_scope OR lane in visibility_scope OR lane is None`).
- Parent **§12** safety items that touch Hub: mutations logged in history; visibility enforced at recall/inject (Hub does not bypass DAL validation where applicable).

---

## 3. Router placement (supersedes parent §13 path table for Hub API only)

**Decision:** Hub memory HTTP handlers may live in a **dedicated module** (e.g. `services/orion-hub/scripts/memory_routes.py`) included from `main.py`, **or** be merged into `api_routes.py`. Either is acceptable if:

1. All routes in §4 of this doc are reachable behind the same `ensure_session` pattern as the rest of Hub API.
2. Parent spec **§13** path row for “Hub API routes” is updated in a follow-up edit to the parent doc **or** this delta remains the canonical pointer for “where routes live” until the parent is amended.

---

## 4. HTTP API parity (must implement)

All routes require `RECALL_PG_DSN` pool available; behavior when DSN is unset matches existing Hub pattern (fail closed or skip router—match current `main.py` contract).

| Method | Path | Behavior |
|--------|------|----------|
| POST | `/api/memory/cards` | Create card; DAL writes + history |
| GET | `/api/memory/cards` | List with filters: `status`, `types`, `anchor_class`, `project`, `priority`, `limit`, `offset` (clamp limit/offset per existing undo-telemetry branch semantics) |
| GET | `/api/memory/cards/{id_or_slug}` | Single card + edges + recent history |
| PATCH | `/api/memory/cards/{id_or_slug}` | Partial update; writes history |
| POST | `/api/memory/cards/{id}/status` | Status transition with optional reason; writes history |
| POST | `/api/memory/edges` | Add edge (respect DAL cycle guard) |
| DELETE | `/api/memory/edges/{id}` | Remove edge |
| GET | `/api/memory/history` | List history with `card_id` and/or `edge_id` filter (query params—exact names documented in OpenAPI or route docstring) |
| POST | `/api/memory/history/{id}/reverse` | Reverse history entry; map unsupported ops to client errors per implemented contract (404/400 already aligned on follow-up branch) |
| GET | `/api/memory/cards/{id}/neighborhood` | Flat neighbor list grouped by `edge_type`; default `hops=1` |
| POST | `/api/memory/sessions/{id}/distill` | Remains **501** until Phase 5 distiller is wired (parent §10) |

Request/response bodies use **`orion.core.contracts.memory_cards`** types or JSON shapes derived from them; do not invent parallel schemas.

---

## 5. Memory tab UI (`templates/index.html`)

- New nav control `memoryTabButton` with `data-hash-target="memory"` consistent with existing tab patterns.
- Section `#memory` (or equivalent `data-panel="memory"`) with three toggles: **Review Queue**, **All Cards**, **Activity Log**.
- Load `static/js/memory.js` with the same cache-bust approach as other Hub scripts.

---

## 6. Client behavior (`static/js/memory.js`)

Minimum viable behavior aligned with parent §10:

| Area | Requirement |
|------|-------------|
| API client | Single module wrapping §4 endpoints against Hub’s API base |
| Review Queue | List `status=pending_review`; actions: approve, reject, open edit flow |
| All Cards | Filterable list; row → detail (fields, edges in/out by type, bounded history, edit) |
| Activity Log | Reverse-chronological history; **Reverse this** only where product rules allow (e.g. create rows with `actor=auto_extractor` when that path exists—Phase 5); until auto-extractor ships, UI may hide or no-op with clear copy |
| Highlight-to-remember | Text selection in chat transcript → modal; Stage 1 pre-fill may call **client-side** heuristics or a thin Hub endpoint if one already exists—**no** new Stage 2 LLM path |
| Neighborhood | Cytoscape via CDN; data from neighborhood GET |

**Out of scope for this delta:** Playwright UI automation (parent §14); full bulk-select polish can be incremental if approve/reject single-card paths ship first.

---

## 7. `lane` wiring (Hub → cortex/recall)

- Where Hub builds payloads for **social** vs **standard chat** turns, set `lane` to **`social`** or **`chat`** respectively to match parent §3 naming.
- Any code path that already sets social-specific fields for cortex must pass the same `lane` into recall query construction so `cards_adapter` and fusion see consistent visibility.
- Document in implementation plan every Hub entrypoint touched (file + function) to avoid drift.

---

## 8. Verification

- Extend or add `services/orion-hub/tests/test_memory_api.py` (parent §13) for new routes: happy path + auth/session + representative 404/400 for reverse.
- Re-run existing memory contract / reverse-history tests when touching shared DAL contracts.
- Manual: open Memory tab with `RECALL_PG_DSN` set, create card, list, patch, edge add/remove, history list, neighborhood JSON, reverse where supported.

---

## 9. Explicitly out of scope

- Phase 5 auto-extractor, distiller CLI, Hub 501 → distiller wiring.
- Phase 6 runbook, CI matrix expansion, full E2E shell (tracked separately).
- Recall `cards_adapter` SQL optimization and Graphtri merge decision (tracked in hardening backlog—not blocking Hub route parity).

---

## 10. Implementation ordering (suggested)

1. HTTP routes missing today (PATCH, status, edges, history GET, neighborhood) on top of existing create/list/get/reverse/distill-501.
2. `lane` on Hub→cortex/recall payloads.
3. `index.html` shell + `memory.js` wired to routes.
4. Hub tests + manual smoke.

---

## 11. Self-review checklist (pre-merge of this doc)

- [x] No placeholder TBD for required routes.
- [x] Distill 501 and Phase 5 scope clearly deferred.
- [x] Parent §2/§3/§12 referenced for non-Hub rules.
