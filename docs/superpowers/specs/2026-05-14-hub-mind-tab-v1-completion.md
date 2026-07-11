# Hub Mind tab — v1 completion (analytics UI, client modularization, API polish)

**Date:** 2026-05-14  
**Status:** Draft — ready for operator review → implementation plan  
**Parent:** [Hub Mind tab, modal, and analytics (2026-05-03)](./2026-05-03-hub-mind-tab-and-modal-design.md)  
**Related:** [Orion’s Mind service (2026-05-02)](./2026-05-02-orions-mind-service-design.md)

---

## 1. Purpose

The **2026-05-03** spec’s **M1–M4** trajectory is largely implemented on `main`: session-gated `GET /api/mind/runs/recent`, filters, aggregates in the JSON contract, `#mind` tab + hash routing, chat-anchored modal with list + drill-down, `localStorage` prefs (`orion.hub.mind.prefs.v1`) and “default Mind on send.”

This document captures **remaining v1 gaps** between that spec and the current codebase so a single implementation pass (or small PR series) can close them **without** expanding scope into **v2** (server-stored operator defaults) or **open inquiries** (livestream).

---

## 2. Non-goals (unchanged from parent)

- **v2 control plane** — Hub DB / admin UI for Mind defaults; cross-browser operator prefs (see parent §9).
- **Orion livestream** — streaming in-flight Mind (parent §10).
- **Browser → `orion-mind` HTTP** — still forbidden.
- **Operator-global `mind_runs` reads** without explicit authz product + audit story — **out of scope** unless paired with a separate security spec. Current **session-scoped** policy remains the default.

---

## 3. Gap A — Surface `aggregates` in the Mind tab UI

### 3.1 Problem

`GET /api/mind/runs/recent` already returns `aggregates` including:

- `total_runs`, `ok_count`, `failed_count`
- `top_error_codes` (up to 3 rows: `error_code`, `run_count`)
- `top_router_profile_ids` (up to 3 rows: `router_profile_id`, `run_count`)
- `time_buckets` (hour buckets: `bucket_utc`, `run_count`, capped server-side)

The Hub tab **only renders** the three headline counts. **Charts and top-N lists** from parent §4.1–§4.3 are not shown, so operators cannot visually validate “window shape” or dominant failure modes without exporting data elsewhere.

### 3.2 Requirements

1. **Top lists (read-only)**  
   - Render **top `error_code`** and **top `router_profile_id`** from `aggregates` (empty state when arrays are empty or all null codes).
   - Compact layout (e.g. two small columns under the existing summary tiles, or a second row of cards). Use existing Tailwind / Hub typography; no new design system.

2. **Time-bucket chart**  
   - Use **`aggregates.time_buckets`** only (no extra API).  
   - X-axis: bucket time (respect operator locale formatting **or** fixed ISO UTC short — pick one and document in plan).  
   - Y-axis: `run_count`.  
   - Handle **empty** buckets and **single-bucket** windows without layout collapse.  
   - Cap rendered points to what the API already limits (≤ 72 rows); if payload is larger, slice client-side defensively.

3. **Consistency**  
   - After refresh, **totals in tiles** must still match the **sum implied by the same filter window** as the table query (same params as today). If chart totals and tile totals disagree because of different rounding, document the rule (prefer: tiles = server `aggregates` counts; chart = same payload).

4. **Performance**  
   - No continuous polling beyond existing refresh controls. Chart redraw only on successful `/recent` fetch.

### 3.3 Acceptance

- With seeded or real `mind_runs`, changing **hours** or **filters** updates **tiles + top lists + chart + table** together from one `/recent` response.
- Manual check from parent §7: *“tab chart counts align with table for same window”* — define quantitatively: sum of `time_buckets[].run_count` should equal `total_runs` **when every run falls in a bucket**; if server excludes some rows from buckets, document that edge case in the plan.

---

## 4. Gap B — Client modularization (`mind_hub.js`)

### 4.1 Problem

Parent spec **Approach 1** (§3.2) calls for **`static/js/mind_hub.js`** owning fetch helpers, table/list rendering, drill-down helpers, chart data prep, and prefs I/O. Current code implements this **inline in `app.js`**, increasing bundle coupling and making reuse between tab and modal harder to reason about.

### 4.2 Requirements

1. **New module** `services/orion-hub/static/js/mind_hub.js` (exact name; cache-bust via existing `HUB_UI_ASSET_VERSION` / template pattern).  
2. **Exports** (ES module **or** IIFE global consistent with Hub’s existing JS loading — match whatever `app.js` uses today; no new bundler unless repo already adds one).  
3. **Move** (not duplicate): Mind-specific helpers and state that are **purely client-side** — e.g. `refreshMindRuns`, `renderMindRows`, modal list/detail renderers, prefs read/write, chart render entrypoint, `formatMindTs` if only used by Mind.  
4. **`app.js` remains** the integration layer: tab hash wiring, DOM element lookup, passing `API_BASE_URL` / `orionSessionId`, attaching listeners, `openMindRunsModal` trigger from chat bubbles.  
5. **Regression**: All existing tests that grep `app.js` for Mind strings must be updated to assert on **`mind_hub.js`** where appropriate, **or** tests assert on **rendered HTML contract** (`index.html` ids) only — prefer updating `test_mind_hub_tab.py` / related tests so CI guards the split.

### 4.3 Acceptance

- `app.js` Mind block shrinks to orchestration; **no behavior regression** on: `#mind` load, filters, modal open/close/focus, default Mind on send.
- Lighthouse / manual: one extra script request is acceptable; document load order in plan.

---

## 5. Gap C — Modal drill-down UX polish (parent §5)

### 5.1 Problem

Drill-down exists with collapsible `<details>` and structured snippets for brief/decision. Parent spec asked for **copy affordances** on raw JSON and **loop-oriented** trajectory emphasis where data allows.

### 5.2 Requirements

1. **Copy buttons** (minimum): “Copy run JSON” for the **raw run payload** shown in modal detail; optional second copy for **pretty `result_jsonb` only**.  
2. **Trajectory emphasis**: If `result_jsonb` exposes a stable list/shape for loops/phases (schema-dependent), render a **short summary list** above the raw JSON (non-blocking if shape missing — fall back to JSON-only).  
3. **Esc** closes modal **if not already** wired; **focus trap** already required — add automated or manual checklist item to confirm no regression.

### 5.3 Acceptance

- Operator can copy JSON **without** DevTools.
- Empty-safe: missing `result_jsonb` does not throw; message is inline.

---

## 6. Gap D — `/recent` pagination (`next_cursor`)

### 6.1 Problem

Contract defines `next_cursor` (nullable). Implementation always returns **`null`**; large sessions cannot page past the first `limit` rows without narrowing filters/time window.

### 6.2 Requirements (v1 minimal)

Choose **one**:

- **D1 — Honest v1:** Document in Hub README + OpenAPI-style comment that **`next_cursor` is reserved** and always `null` until implemented; **or**  
- **D2 — Opaque cursor:** Implement keyset pagination `(created_at_utc, mind_run_id)` **scoped by `session_id` + same filter set**, return `next_cursor` only when `len(items)==limit`.

### 6.3 Acceptance (if D2)

- Test: insert > `limit` rows in one window; first page returns `next_cursor`; second request returns next slice; no cross-session leakage.

---

## 7. Gap E — Error / empty states (parent §6)

### 7.1 Problem

Parent matrix specifies explicit behavior for **pool unavailable (503)** and other failures. Implementation should be verified and extended if any path only logs to console.

### 7.2 Requirements

1. **`503` / `postgres_pool_unavailable`**: Mind tab and modal show **inline, human-readable** copy (“Mind database unavailable for this Hub instance”) — not a raw stack trace.  
2. **Network errors**: Distinguish **offline** vs **4xx/5xx** where possible; single retry policy **out of scope** unless trivial.  
3. **Empty `/recent`**: Preserve current empty table state; chart/top lists show **zero / empty** states, not broken axes.

### 7.3 Acceptance

- Playwright or unit test **optional**; minimum is **documented manual checklist** in the implementation plan + one pytest for **503 JSON shape** from `mind_routes` if not already covered.

---

## 8. Testing matrix (minimum)

| Area | Requirement |
|------|----------------|
| **API** | If D2: new tests for cursor stability under filters + `session_id` scoping. |
| **UI** | Update or add `test_mind_hub_tab.py` (or equivalent) for **new DOM ids** for chart container + top-list regions **or** snapshot key substrings in `index.html`. |
| **Client split** | Import smoke: page loads with `mind_hub.js` present; no duplicate symbol definitions. |
| **Manual** | Parent §7 manual bullets + “copy JSON” + “chart totals vs table” check. |

---

## 9. Phasing suggestion (for `docs/superpowers/plans/…`)

| Phase | Deliverable |
|-------|-------------|
| **P1** | Gap A — aggregates UI (top lists + chart), tests/manual checklist. |
| **P2** | Gap B — extract `mind_hub.js`, update tests. |
| **P3** | Gap C — copy + trajectory summary polish. |
| **P4** | Gap D — cursor **or** explicit doc-only closure. |
| **P5** | Gap E — 503/empty hardening sweep. |

Phases **P1–P3** are the highest user-visible value; **P4–P5** can ship in the same release if small.

---

## 10. Self-review

- [x] Does not contradict parent spec or Mind service spec (read path stays Hub → PG; no client → Mind HTTP).
- [x] v2 / livestream remain explicitly out of scope.
- [x] Each gap has measurable acceptance criteria.
- [x] Chart implementation detail (canvas vs SVG vs simple HTML bars) intentionally left to **implementation plan** per parent §11 style.

## 11. Approval

Operator: **\_\_\_\_\_\_\_\_**  Date: **\_\_\_\_\_\_\_\_**  
After approval: spawn **`docs/superpowers/plans/2026-05-14-hub-mind-v1-completion.md`** (or dated plan) via **writing-plans** / normal planning workflow.
