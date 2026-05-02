# Memory Cards — Phase 4 Hub completion (delta spec)

**Date:** 2026-05-02  
**Status:** Draft for implementation planning  
**Parent spec:** [Orion Memory Cards v1](2026-05-01-orion-memory-cards-v1-design.md) (approved)  
**Companion:** [Memory Cards v1 offboarding](../guides/2026-05-01-memory-cards-v1-offboarding.md)

This document scopes **only** Hub operator UX and HTTP parity for Phase 4. It does not repeat parent ground rules (§2), lane semantics (§3), or data model (§4). **Phase 5, Phase 6, recall/cortex hardening, and v1.5-adjacent work are out of scope** for this delta; see **§9** for the full inventory and pointers to the parent spec. Where this doc conflicts with the parent, **the parent wins** unless this doc explicitly supersedes a *path* or *integration* detail listed below.

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

**UI polish deferred with Phase 4:** Playwright UI automation (parent §14); full bulk-select polish can ship incrementally after single-card approve/reject paths. Other deferrals: **§9**.

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

## 9. Out of scope — extended phases and adjacent work

Implementers must not expand Phase 4 Hub tasks into the areas below without a separate approved spec or plan. The **normative** descriptions remain in the parent design ([§11 Phase 5](2026-05-01-orion-memory-cards-v1-design.md), [§12 Phase 6](2026-05-01-orion-memory-cards-v1-design.md), [§14](2026-05-01-orion-memory-cards-v1-design.md)); this section is a **boundary checklist** so Hub completion does not accidentally absorb extraction, ops, or recall-scale work.

### 9.1 Phase 5 — Auto-extractor and operator distiller (parent §11)

| Item | Out of scope for Phase 4 Hub delta | Parent / paths |
|------|-------------------------------------|----------------|
| Stage 1 extractor productization | Replacing the stub in `orion/core/storage/memory_extraction.py` with full §5 regex tables; `extract_candidates` / `fingerprint` behavior as specified | Parent §5 patterns, §11 |
| Auto-extractor pipeline | Implementing per-candidate logic in `memory_extractor.py` (fingerprint, dedupe vs existing cards, contradiction vs `always_inject`, session counting, auto-promote threshold, DAL inserts with `actor=auto_extractor`) | Parent §11 |
| Default and safety flags | Any change that turns `ORION_AUTO_EXTRACTOR_ENABLED` on by default; enabling Stage 2 before v1.5 (`ORION_AUTO_EXTRACTOR_STAGE2_ENABLED` must keep raising `NotImplementedError`) | Parent §2, §11 |
| Distiller CLI | Completing `services/orion-recall/scripts/distill_memory_cards.py` (JSONL detect, dry-run YAML, `--apply` via DAL with `operator_distiller` provenance) | Parent §11 |
| Hub distill endpoint | Replacing `POST /api/memory/sessions/{id}/distill` **501** with in-process or `subprocess` invocation of the distiller | Parent §10–§11 |
| Tests listed for Phase 5 | `tests/test_memory_extraction.py`, distiller CLI tests, extractor integration beyond “disabled / Stage 2 raises” already covered | Parent §13 |

**Phase 4 allowance:** Hub may surface Activity Log rows and placeholders for “Reverse this” where product rules apply later; no requirement to implement extractor-backed flows in this delta.

### 9.2 Phase 6 — Safety verification, runbook, CI, E2E (parent §12)

| Item | Out of scope for Phase 4 Hub delta | Notes |
|------|-------------------------------------|-------|
| PR safety checklist execution | Running and attaching `rg` / `git diff` evidence from parent §12 | Phase 6 |
| Operator runbook | Authoring `docs/superpowers/runbooks/2026-05-01-memory-cards-v1-runbook.md` | Parent §12 |
| Real E2E smoke | Replacing placeholder `scripts/smoke_memory_cards_e2e.sh` with the full flow (schema → DAL → Hub tab → recall bundle → reverse auto-promotion) | Parent §12 |
| CI matrix | Ensuring images install updated `requirements.txt` and running the expanded pytest set across services | [Offboarding](../guides/2026-05-01-memory-cards-v1-offboarding.md) — Phase 6 + ops |
| Spec-listed tests not yet present | DAL roundtrip, cards backend scoring, visibility lane integration, `memory_inject` snapshot, extraction tables, `chat.general.v1` empty-store parity (where not already done) | Parent §12–§13; offboarding “Spec-listed tests not yet added” |

Phase 4 Hub work **may** add or extend `services/orion-hub/tests/test_memory_api.py` as listed in §8 of this doc; it must not be blocked on the full Phase 6 checklist.

### 9.3 Recall and cortex hardening (between v1 phases / parallel backlog)

These items improve correctness or scale **outside** the Hub HTTP/UI slice. Track in implementation planning as **follow-on** work, not part of this delta’s exit criteria.

| Item | Description | Typical owners |
|------|-------------|----------------|
| Visibility integration test | End-to-end assertion that intimate-scoped cards never appear for `lane="social"` across `cards_adapter`, `memory_inject`, and fusion | recall + cortex-orch tests |
| Graphtri / browse-only paths | Product decision: merge cards with the same lane rules as multi-signal recall, or document “cards not merged” for graphtri-only paths | recall + spec note |
| `cards_adapter` scale | Moving filters/limits into SQL vs full-table fetch + Python scoring for large datasets | recall |
| Readiness / pool health | Operator-visible probe when `RECALL_ENABLE_CARDS` is on but the PG pool is unhealthy | recall ops |
| `memory_inject` async migration | Optional future if `conversation_front` becomes async; snapshot test for empty `known_facts_block` byte-identical prompt | cortex-orch |

### 9.4 v1.5 and parent “explicitly not in v1” (parent §14)

| Item | Status |
|------|--------|
| Stage 2 LLM extraction | **v1.5** — not in Memory Cards v1; must not be implemented under this Phase 4 delta |
| Playwright Hub UI tests | Explicitly not in v1 per parent §14; manual smoke only for Hub |
| New external runtime families | Forbidden per parent §2; Phase 4 Hub must not add npm build steps, new databases, etc. |

### 9.5 Profile and default churn (parent §2)

Out of scope: modifying `RECALL_DEFAULT_PROFILE`, editing recall YAML profiles other than the single allowed exception (`self.factual.v1.yaml`), or adding profiles beyond what the parent already specified (e.g. `biographical.v1.yaml`). Hub Phase 4 does not change recall profile files.

---

## 10. Implementation ordering (suggested)

1. HTTP routes missing today (PATCH, status, edges, history GET, neighborhood) on top of existing create/list/get/reverse/distill-501.
2. `lane` on Hub→cortex/recall payloads.
3. `index.html` shell + `memory.js` wired to routes.
4. Hub tests + manual smoke.

---

## 11. Self-review checklist (pre-merge of this doc)

- [x] No placeholder TBD for required routes.
- [x] Distill 501 and Phase 5 scope clearly deferred (§9.1).
- [x] Phase 6 and recall hardening boundaries documented (§9.2–§9.3).
- [x] v1.5 / §14 exclusions documented (§9.4); profile churn excluded (§9.5).
- [x] Parent §2/§3/§12 referenced for non-Hub rules (§2).
