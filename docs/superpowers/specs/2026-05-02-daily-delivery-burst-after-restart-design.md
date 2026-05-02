# Daily delivery burst after mesh restart — design

**Date:** 2026-05-02  
**Status:** Approved for implementation planning (operator confirmed all three pillars: document, durable cursors, Hub UX)

---

## 1. Purpose

Operators who **restart the mesh often** and **refresh the Hub** reported a single moment where **daily journal**, **daily pulse**, **world pulse**, **journal pass** (when scheduled), and **emails** seemed to “queue” and then **arrive together**, with **timestamps or copy reflecting earlier wall-clock**. Chat with Orion **correlated** with noticing the burst but is **not** required for backend delivery.

This spec defines **three coordinated pillars**: (1) documented behavior and observability, (2) **durable** scheduler cursors in `orion-actions`, (3) Hub notification UX so hydration after refresh is less overwhelming.

---

## 2. Root cause model (normative)

These statements are the shared mental model for implementation and docs.

1. **`orion-actions` daily cadence uses in-memory cursors** (`last_daily_run`, `last_journal_run` in the scheduler lifespan). After **process restart**, those maps are empty. For each daily job, `should_run_daily` compares local wall time to the configured local hour/minute; if `last_ran_date` is missing and local time is **already past** the threshold, the job is **eligible again for the same calendar day**. That can produce **duplicate** pulse / world pulse / metacog / **journal** work and **duplicate** notify/email relative to a “already ran this morning” expectation. **Note:** `ACTIONS_DAILY_RUN_ON_STARTUP` applies to **daily pulse, world pulse, and daily metacog** only; **daily journal** uses `journal_should_run` alone (no startup override in current code).

2. **The scheduler loop runs multiple jobs in one iteration** (~45s cadence). Restarts plus eligibility can make **several** downstream calls (cortex, world-pulse trigger, notify) appear in a **tight burst**.

3. **Hub in-app notifications** are buffered in the Hub process’s **`NotificationCache`** (in-memory deque, bus-fed). **`GET /api/notifications`** returns that cache. A **full page load or refresh** calls `loadNotifications()` and can **paint many rows at once**; payload fields may include **`created_at`** from upstream when the job ran, which reads as “earlier” than the refresh.

4. **Email** is driven by **`orion-notify`** when producers call `/notify` (and related paths). Bursts of email follow **bursts of eligible work and notify calls**, not from “opening a Hub conversation” as a gate.

5. **Workflow schedules** (`journal_pass`, etc.) already use **durable** JSON persistence (`ACTIONS_WORKFLOW_SCHEDULE_STORE_PATH`). This spec’s cursors apply to **built-in daily triggers**, not to replacing workflow schedule storage.

---

## 3. Pillar 1 — Document + observe

### 3.1 Documentation

- **`services/orion-actions/README.md`**: Add a short subsection **“Daily scheduler and restarts”** stating that without durable cursors (Pillar 2), **same-day re-run after restart past the local cutoff** is expected; point to env flags (`ACTIONS_DAILY_RUN_ON_STARTUP`, timezone, daily hours).
- **`services/orion-hub/README.md`**: Clarify that **`/api/notifications`** is **in-process memory** of recent bus-fed events, not a full historical inbox; refresh loads the cache snapshot.

### 3.2 Observability (minimal)

- On each scheduler iteration (or only when a daily job **fires**), emit **structured logs** including: `service`, `scheduler_tick`, `local_date` (per job), `job_key` (e.g. `daily_pulse`, `world_pulse`, `daily_journal`), `restart_dedupe_source` (`memory` vs `durable` once Pillar 2 exists), and **`correlation_id`** where already available.
- Optional metric counters (if the stack already exposes Prometheus elsewhere, align with that pattern; **do not** introduce a new metrics system solely for this spec).

---

## 4. Pillar 2 — Durable scheduler cursors

### 4.1 Goal

Persist “**last completed local calendar date**” (or equivalent) per **built-in daily job** so that **process restart alone** does **not** re-eligible the same calendar day’s run **after** it has already completed successfully.

### 4.2 Job keys (initial set)

Align with current scheduler responsibilities in `services/orion-actions/app/main.py`:

- `daily_pulse` (`ACTION_DAILY_PULSE_V1`)
- `world_pulse` (scheduler branch keyed `"world_pulse"` in `last_daily_run`)
- `daily_metacog` (`ACTION_DAILY_METACOG_V1`)
- `daily_journal` (uses `last_journal_run` today — fold into the same store as a keyed cursor for consistency)

### 4.3 Storage shape and location

- **Preferred v1:** A small **JSON file** on disk (same operational pattern as `WorkflowScheduleStore`), controlled by a new setting **`ACTIONS_SCHEDULER_CURSOR_STORE_PATH`** (default under the same directory family as `ACTIONS_WORKFLOW_SCHEDULE_STORE_PATH`, e.g. alongside `/tmp/orion-actions/` in dev).
- **Contents:** Mapping `job_key -> last_completed_local_date` (ISO `YYYY-MM-DD`) in the **configured `ACTIONS_DAILY_TIMEZONE`** (same timezone already used for `should_run_daily` and `build_daily_window`).
- **Atomic write:** reuse the **write temp + replace** pattern from `workflow_schedule_store.py`.

**Rationale:** `orion-actions` does not own a Postgres DSN today; introducing SQL solely for two strings per job is heavier than a file consistent with existing schedule persistence.

### 4.4 Semantics

- **On successful completion** of a daily job (after the same success criteria used today to advance `last_daily_run` / `last_journal_run`), **persist** the cursor for that `job_key` to **that job’s `local_date` string** returned from `should_run_daily` / journal equivalent.
- **On startup:** **hydrate** in-memory state from file before the first scheduler tick.
- **Race:** single writer process assumption matches workflow store; if multiple Actions replicas ever exist, document **single replica** requirement or add file locking in a follow-up (out of scope unless multi-replica is already supported).

### 4.5 Interaction with existing flags

- **`ACTIONS_DAILY_RUN_ON_STARTUP`:** Today this flag gates **only** `daily_pulse`, `world_pulse`, and `daily_metacog` (not daily journal). After cursors exist, “run on startup” for those jobs must mean **“run once if there is **no** successful completion recorded for the **intended** target date”**, not “ignore cursor.” Exact semantics: if startup forced run is still desired when **cursor absent** (first deploy), allow first tick to run **once** and then persist; if **cursor present** for today, **do not** force duplicate for that job unless a separate **explicit override** env exists (default **no** new override in v1).
- **`ACTIONS_DAILY_RUN_ONCE_DATE` / forced date:** Manual override must still work: completing a forced run updates the cursor to the **forced local date** associated with that window.

### 4.6 Dedupe and notify

- Existing **notify dedupe keys** and **journal dedupe** remain authoritative for **downstream** idempotency. Cursors prevent **starting** duplicate work; they do not replace notify dedupe.

---

## 5. Pillar 3 — Hub UX

### 5.1 Goals

Reduce the feeling of a **spam burst** when the tray hydrates from `/api/notifications`, and make **timing** legible without implying false causality from chat.

### 5.2 Requirements

1. **Display `created_at` vs `received_at`:** `NotificationCache` already sets `received_at` when the Hub ingests the bus message. The UI should show **both** (compact: “Created … · Received …”) where both exist.
2. **Initial load batching:** On first `loadNotifications()` after page load, if **N ≥ configurable threshold** (default **5**), show **one** summary toast or banner: “**N notifications** loaded (see tray)” instead of firing **N** individual toasts (if toasts are driven per notification today).
3. **Tray ordering:** Keep **newest-first** by `received_at` (fallback `created_at`) unless operator toggles sort (optional; default fixed sort is acceptable for v1).
4. **Copy:** Tooltip or footnote: “Notifications reflect **server** recent events; refresh reloads the Hub cache, not full history.”

### 5.3 Non-goals (Hub)

- Replacing the in-memory cache with **full SQL-backed** notification history for the tray (would be a larger product; optional future spec).
- Changing **email** policy or notify rules (remains `orion-notify`).

---

## 6. Testing and verification

| Area | Scenario |
|------|----------|
| Actions | Restart Actions **after** daily cutoff with cursors file showing **today** already completed → **no** second dispatch for that job until next local day. |
| Actions | Fresh data dir (no file) → first eligible tick runs and **writes** file. |
| Actions | `ACTIONS_DAILY_RUN_ON_STARTUP=true` with cursor already **today** → **no** duplicate for that job. |
| Hub | Load page with **10** synthetic notifications in cache → **at most one** “batch loaded” surface + tray lists all with created/received labels. |

---

## 7. Implementation order

1. **Pillar 1** docs + logs (low risk, clarifies operator expectations).  
2. **Pillar 2** durable cursors (fixes duplicate work root cause).  
3. **Pillar 3** Hub UX (improves perception; safe after behavior is clearer).

---

## 8. Out of scope

- Changing **world-pulse** or **cortex** semantics beyond what Actions already triggers.  
- Multi-tenant cursor namespaces (single operator / recipient group assumed unless settings already encode multi-tenant).  
- Backdating or rewriting already-sent emails.
