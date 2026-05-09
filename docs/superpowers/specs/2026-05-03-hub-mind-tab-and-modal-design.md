# Design: Hub Mind tab, chat modal, and analytics

**Date:** 2026-05-03  
**Status:** Approved for implementation planning (operator)  
**Related:** [Orion’s Mind service (2026-05-02)](./2026-05-02-orions-mind-service-design.md)

**Charter:** Add a **global Hub “Mind” tab** for read-only analytics over `mind_runs`, a **chat-anchored modal** (Inspect / Memory graph pattern) for correlation-scoped run lists and drill-down, and **v1 operator preferences stored only in this browser** (`localStorage`). **Do not** call `services/orion-mind` HTTP from the browser; **run** Mind only via existing **Hub → Orch** paths. **Do not** implement server-wide Mind defaults in v1 (**v2**).

---

## 1. Goals

| Goal | Detail |
|------|--------|
| **Global visibility** | Operators see recent Mind runs, filter/summarize them, and chart activity over time without pasting a `correlation_id` first. |
| **In-context introspection** | From chat, open a modal listing **all** `mind_runs` rows for the message’s `correlation_id` (newest first), then drill into **one** run’s interior (loops / trajectory / decision / brief from `result_jsonb`). |
| **Shared UI logic** | One client module owns list + drill-down + chart helpers; tab and modal are thin shells (**Approach 1**). |
| **Local prefs (v1)** | Optional toggles (e.g. default Mind enabled on send) persist **only** in `localStorage` on this browser. |
| **Spec alignment** | Matches Mind service spec: Hub reads artifacts via Hub API + Postgres; canonical Mind **execution** remains behind Orch. |

## 2. Non-goals (v1)

- **Orion livestream** — real-time, Cursor-style narration of in-flight cognition (streaming tokens / phases). **Open inquiry** (see §10).
- **Server-backed** operator or admin defaults for Mind (Hub DB, env UI, multi-session). **Deferred to v2** (see §9).
- **Hub → orion-mind** direct HTTP from the client.
- Replacing logs, Spark, or OTel; this is a **Mind-runs–centric** surface only.

## 3. Architecture

### 3.1 Data flow

- **Read:** Browser → Hub `GET` routes → same **`mind_runs`** Postgres pool and **`ensure_session`** pattern as [`mind_routes.py`](../../../services/orion-hub/scripts/mind_routes.py).
- **Run Mind:** Unchanged — client continues to send chat (or probe) requests such that Orch invokes Mind when `context.metadata["mind_enabled"]` is exactly `true` (per Mind service spec). v1 prefs may set that flag on send from **this browser** only.

### 3.2 Client layout (Approach 1)

| Piece | Responsibility |
|-------|------------------|
| **`static/js/mind_hub.js`** (name may vary) | Fetch helpers, run list table rendering, single-run drill-down (structured sections for trajectory / decision / brief / diagnostics), time-bucket chart data prep, `localStorage` read/write for prefs. |
| **`app.js`** | Hash tab `#mind` / `setActiveTab`, mount tab panel, wire chat-bubble affordance → open modal with `correlation_id`, apply “default Mind on send” when composing requests (if pref on). |
| **`templates/index.html`** | New tab button + `#mind` panel markup; script tag with cache-bust query aligned with `HUB_UI_ASSET_VERSION` / existing patterns. |

Reuse Hub styling (Tailwind classes) consistent with Memory / Inspect modals.

### 3.3 Hub API additions (read-only)

Existing:

- `GET /api/mind/runs/{mind_run_id}`
- `GET /api/mind/runs?correlation_id=&limit=`

**New (v1):** `GET /api/mind/runs/recent` (exact path may mirror naming above), **session-gated**, query parameters at minimum:

- `since_utc` / `until_utc` **or** `hours` (implementer picks one style; document in plan),
- optional filters: `ok` (bool), `trigger`, `error_code`, `router_profile_id`,
- `limit` (bounded, e.g. max 500) and optional **cursor** for pagination if needed.

SQL: filter `mind_runs` by `created_at_utc` and optional columns; `ORDER BY created_at_utc DESC`. Use existing index on `(created_at_utc DESC)` from Mind spec; add migration only if profiling requires it.

**Security:** Same rules as other Mind routes — no anonymous cross-session leakage; session must be established first.

## 4. Global Mind tab (UX)

1. **Controls:** Time range selector, filters (ok / trigger / error_code / router_profile_id), refresh.
2. **Summary tiles:** For the **current filtered window** — total runs, ok vs failed counts, top `error_code` (small list), top `router_profile_id` (small list).
3. **Time-bucket chart:** Count of runs per bucket (e.g. hour or day) over the selected window; cap number of buckets for performance.
4. **Table:** One row per run (key columns: `created_at_utc`, `ok`, `trigger`, `error_code`, `router_profile_id`, `correlation_id`, `mind_run_id` truncated + copy). Row click opens **same drill-down viewer** as modal (inline expand, drawer, or secondary column — implementer chooses; document in plan).
5. **Operator help (static):** Short copy + link to Mind service spec on how Mind is invoked (Orch + `mind_enabled`); no server config in v1.

### 4.1 `localStorage` prefs (v1)

Persist **only** in the browser (namespaced keys, e.g. `orion.hub.mind.*`):

- **Default Mind on send** — boolean; when true, Hub client attaches `mind_enabled: true` on outbound chat/orch requests per existing client contract.
- **Chart / table prefs** — default time window hours, table page size, last-used filters (optional).

**v2:** Server-stored prefs — §9.

## 5. Chat-anchored modal (UX)

- **Affordance:** Control on **assistant** bubbles (minimum); whether **user** bubbles show the chip is **implementation default: assistant only** unless product extends with clear empty-state (“no runs yet”) for user-only correlations.
- **Open:** Reads `correlation_id` from the same metadata path Inspect / Memory graph use for that bubble.
- **Body:**
  1. **List:** `GET /api/mind/runs?correlation_id=&limit=` — vertical list, newest first; show `ok`, `trigger`, `created_at_utc`, short error if any.
  2. **Drill-down:** Selecting a row loads `GET /api/mind/runs/{mind_run_id}` if not already in row payload; render **loop-oriented** trajectory (`result_jsonb`), plus decision, brief, timing/diagnostics if present. Prefer **collapsible sections** over a single monolithic `<pre>` where practical; keep raw JSON available (copy button).
- **Accessibility:** Focus trap while open; Esc closes; return focus to trigger.

## 6. Error handling and empty states

| Condition | Behavior |
|-----------|----------|
| `503` / pool unavailable | Inline error; copy explains Mind DB not wired for this Hub deployment. |
| Recent API returns `[]` | Empty state in tab. |
| Modal / correlation has no runs | “No Mind runs for this correlation yet.” |
| Run row references missing id | Disable drill-down; log once in console in dev. |

## 7. Testing

| Layer | Minimum |
|-------|---------|
| **Hub API** | Tests for `GET .../recent`: seeded rows, ordering, filter by `ok` / `trigger`, bound on `limit`, session required. |
| **Client** | If repo has patterns for DOM/modal smoke, add load check for new script + tab panel ids; otherwise document manual checklist. |
| **Manual** | Mind-enabled turn → row in `mind_runs` → modal lists run → drill matches DB; tab chart counts align with table for same window. |

## 8. Phasing (implementation plan input)

| Phase | Content |
|-------|---------|
| **M1** | `GET /api/mind/runs/recent` + tests; wire pool/session as existing mind routes. |
| **M2** | `mind_hub.js` + shared list/drill renderer; Mind tab + hash routing. |
| **M3** | Chat modal + bubble affordance; reuse shared renderer. |
| **M4** | `localStorage` prefs + hook “default Mind on send” into existing send path. |
| **M5** | Tiles + time-bucket chart; polish empty/error states. |

Order may be adjusted in the implementation plan if parallelizing M2/M3 helps.

## 9. v2 (deferred) — server-side control plane

**Not in v1.** Track for a later spec:

- Hub- or service-stored **defaults** for Mind invocation (e.g. deployment-wide “Mind on” policy).
- **Admin UI** or config API with authz for operators vs end-users.
- Cross-session **sync** of operator preferences (same Hub, different browsers).

## 10. Open inquiries (deferred) — Orion livestream

**Goal (research):** Show what Orion is **currently** thinking in near real time (analogy: Cursor-style reasoning stream).

**Why deferred:** Today’s Mind path is **synchronous HTTP** from Orch to `orion-mind` with persistence **after** completion; Hub reads **completed** `mind_runs`. A true livestream implies **streaming or incremental events** (WebSocket, SSE, or bus fan-out) from Orch/Mind/Exec with correlation scoping, redaction, and back-pressure — **not** defined in the Mind v1 contract.

**Research directions (non-binding):** event schema for `mind.phase.*` or reuse of existing Hub WS reasoning fields; whether narration is **Mind-only** or **whole-stack**; operator-only gating.

---

## 11. Self-review checklist

- [x] No contradiction with Mind service spec (Hub reads PG; run via Orch; no client → Mind HTTP).
- [x] v1 / v2 / open inquiry boundaries explicit.
- [x] New API surface named; details left to implementation plan where appropriate (exact query param names, chart library).
- [x] Accessibility and empty states addressed at high level.

## 12. Approval

Brainstorming sign-off: **Approach 1** (shared `mind_hub.js` + thin `app.js` / template wiring); sections **§1–§7** and **§10–§11** approved in thread 2026-05-03. Proceed to **`docs/superpowers/plans/…`** via **writing-plans** after operator review of this committed file.
