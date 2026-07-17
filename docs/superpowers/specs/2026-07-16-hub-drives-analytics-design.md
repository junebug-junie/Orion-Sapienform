# Hub Drives Analytics — operator orientation surface

**Date:** 2026-07-16  
**Status:** Design approved — ready for implementation plan  
**Mode:** Implementation-bound design (operator surface + slim SQL persistence; no cognition-loop change)  
**Related:**
- `orion/autonomy/README.md` + `orion/autonomy/drives_and_autonomy_retrospective.md`
- `orion/spark/concept_induction/drives.py` (`DriveEngine`, `DRIVE_KEYS`)
- `orion/core/schemas/drives.py` (`DriveAuditV1`, `tick_attribution`)
- `services/orion-sql-writer/app/models/drive_audit.py` (slim `drive_audits` sink)
- `scripts/analysis/measure_autonomy_gate.py` (GO / NO-GO / SATURATED / UNMEASURABLE)
- Hub isolation precedents: Causal Geometry tab, Pressure Analytics tab

---

## Arsonist summary

The drives program already runs: six homeostatic pressures, per-tick audits, Postgres history, and offline gate scripts. Operators still cannot *see* it without CLI archaeology and per-turn chat side panels. This patch adds a session-preserving Hub **Drives** tab that orients Juniper on program health — gauges, real contributor history, KPIs, dual time series, divergence, goals, and cross-links — with goal-aligned coloring and honest degradation after stack rebuilds. The one non-UI load-bearing change: stop discarding `tick_attribution` / `tension_kinds` at the sql-writer sink so contributor history is real, not cosplay.

---

## Core question

How do we give Juniper a Hub tab that makes the drives economy legible at a glance — current pressures, who contributes, whether goals match wanting, and whether the tick rail is alive — using durable Postgres evidence, without killing the Hub session on tab toggle, and without lying when history is thin after a rebuild?

---

## Decisions locked (brainstorm)

| Topic | Choice |
|---|---|
| Scope | All brainstorm cards + dual tick time series |
| Contributors | Persist slim `tick_attribution` + `tension_kinds` on `drive_audits` in the same pass |
| Coloring | Goal-aligned regimes; togglable `combined` / `align` / `funnel` (default `combined`) |
| Time series | Dual: tick-rate buckets + six pressure sparklines, shared window |
| Window | Default 24h; picker 1h / 6h / 24h / 7d with honest coverage degradation |
| Subjects | Allowlist ∪ discovered subjects, each badged with audit coverage |
| Mutations | Strictly read-only |
| Isolation | Causal-geometry pattern (standalone page + dedicated JS + iframe tab) |
| Docs | Update `orion/autonomy/README.md`, `services/orion-hub/README.md`, and `services/orion-spark-concept-induction/README.md` (DriveEngine producer) |
| Orientation | Per-card / per-chart tooltips: definition, why designed this way, how operators should read it |

---

## Ground truth

| Claim | Evidence |
|---|---|
| Six drives are live | `DRIVE_KEYS` in `orion/spark/concept_induction/drives.py`; leaky integrator, activate ≥0.62 / deactivate <0.42 |
| Audits already carry attribution on the wire | `DriveAuditV1.tick_attribution`, `tension_kinds` in `orion/core/schemas/drives.py`; built in `audit.py` |
| Postgres currently drops attribution | `DriveAuditSQL` intentionally omits `tick_attribution` / `tension_kinds` / evidence (sql-writer model docstring) |
| Hub can read Postgres | `app.state.memory_pg_pool` via `RECALL_PG_DSN` (same pool as causal-geometry) |
| Session-safe Hub tabs exist | Causal Geometry + Pressure Analytics: iframe + dedicated JS + `app.js` hash wiring |
| Gate metrics already exist | `scripts/analysis/measure_autonomy_gate.py` pure functions + saturation verdicts |
| Autonomy docs are the drives home | `orion/autonomy/README.md` → retrospective; concept-induction README has divergence-audit section only |
| Pressure Analytics ≠ drives | `#pressure` is substrate mutation pressure, not `DriveEngine` |

---

## Architecture

### Hub shell isolation

```text
index.html  --tab button + iframe section-->  /drives-analytics
app.js      --hash #drives?window=&subject=&color=-->  setActiveTab only
drives-analytics.html + drives-analytics.js  --fetch-->  /api/drives-analytics/*
```

- Standalone route `/drives-analytics` renders `templates/drives-analytics.html`.
- Bundle: `static/js/drives-analytics.js` — self-contained; **must not** reference `window.OrionHub`.
- In-shell: `<section id="drives" data-panel="drives">` with lazy iframe `src="/drives-analytics"`.
- Tab: `#drivesTabButton`, label **Drives**, hash `#drives`.
- Refresh button reloads **only** `drivesAnalyticsPanelFrame.contentWindow`, never the Hub shell.
- Operator state in hash query: `window`, `subject`, `color` — survives refresh; does not depend on server session.

### API surface (all read-only, all degrade)

Reuse `memory_pg_pool`. Never 500 on missing pool / undefined table / empty window — return `{degraded, error, source, ...}` like causal-geometry.

| Endpoint | Purpose |
|---|---|
| `GET /api/drives-analytics/subjects` | Allowlist ∪ distinct `drive_audits.subject`, each with `{row_count, oldest_ts, newest_ts}` |
| `GET /api/drives-analytics/snapshot?subject=` | Latest audit: pressures, activations, active_drives, dominant, tick_attribution, tension_kinds, observed_at, staleness |
| `GET /api/drives-analytics/window?subject=&hours=` | Dominant histogram, mean pressure/drive, active_count distribution, co-activation / saturation KPIs, `coverage_hours`, `row_count`, `retention_note` |
| `GET /api/drives-analytics/series?subject=&hours=` | Downsampled per-drive pressure series + tick-rate buckets |
| `GET /api/drives-analytics/goal-alignment?subject=` | Per-drive goal match + funnel/gate posture for coloring modes |
| `GET /api/drives-analytics/divergence?subject=` | `drive_state.v1` (concept store) vs latest audit; store-path warning |

KPI / saturation math: **import** pure functions from `scripts/analysis/measure_autonomy_gate.py` (do not fork thresholds).

Default subject: `orion`. Default hours: `24`. Allowed hours: `1, 6, 24, 168` (7d).

### Schema change (same changeset as UI)

Wire payload already has the fields. Only the SQL sink changes.

1. `DriveAuditSQL`: add `tick_attribution` (`_JSONB`) and `tension_kinds` (`_JSONB`), nullable.
2. sql-writer worker: stop filtering those two columns out of write rows.
3. Migration: `services/orion-sql-db/manual_migration_drive_audits_v2_tick_attribution.sql`.
4. Boot DDL parity in sql-writer `app/main.py` (same pattern as existing drive_audits indexes).
5. **No** bus/schema/registry change — `DriveAuditV1` already defines both fields.
6. **No backfill.** Pre-migration rows stay `NULL`. UI must say: “attribution not recorded before \<first_non_null_ts\>.”

Bounds: attribution is 6 fixed keys; tension_kinds is a short list per tick — acceptable append cost given existing thousands-of-rows/day audit volume and 90-day retention.

### Polling

Gentle interval (≈30s), pause on `document.visibilityState === 'hidden'`, immediate refresh on resume — copy causal-geometry.

---

## UI cards

All seven ship in v1. Every card has a help control (ⓘ) whose tooltip / popover covers three sentences minimum:

1. **What it is** (definition),
2. **Why it is designed this way** (intent),
3. **How to read it as an operator** (meaning / failure modes).

Tooltip copy lives in `drives-analytics.js` as a single `TOOLTIP_COPY` map keyed by card id — easy to keep in sync with README wording.

### 1. Six-drive gauge card

- One thermometer-style gauge per `DRIVE_KEYS`.
- Fill = pressure `[0,1]`.
- Color = alignment regime (see Coloring), not naive low=green/high=red.
- Active badge when drive is in `active_drives` / pressure past activate threshold.
- Dominant drive highlighted.
- Staleness chip if `observed_at` older than 5 minutes.

### 2. Contributors card

- Per-drive stacked bars from `tick_attribution` × `tension_kinds`.
- Toggle: **Live (latest tick)** vs **Window (aggregated over selected hours)**.
- Window aggregation: sum/average attribution only over rows where `tick_attribution IS NOT NULL`; surface count of attributed vs null rows.
- Explicit label when window is partly pre-migration.

### 3. Program-health KPI strip

- Tick rate, active-drive count, co-activation %, top-dominant share, gate verdict (`GO` / `NO-GO` / `SATURATED` / `UNMEASURABLE`), last-audit age.
- Chips colored by funnel/gate posture when color mode is `funnel` or `combined`.

### 4. Time series card (dual)

- Top: audit **tick-rate** over time (counts per bucket).
- Below: **six pressure sparklines**, shared window and x-axis.
- After rebuild with thin history: render available points + coverage banner (`coverage_hours`, `row_count`, `retention_note`).

### 5. Divergence / integrity card

- Side-by-side `drive_state.v1` (LocalProfileStore via configured `CONCEPT_STORE_PATH`) vs latest `drive_audits` pressures.
- Max absolute delta per drive.
- Loud banner if store path is the known host-local fallback default, or if `autonomy_state_v2` compare is frozen/historical only.
- Never claim two live signals when the second is dead.

### 6. Goal pipeline card (read-only)

- Active goals, proposal rate, dominant `drive_origin`, archive backlog counts.
- Serves goal-alignment coloring and answers “is pressure becoming wanting → goals?”
- Links to existing Hub autonomy surfaces; **no** promote / dismiss / complete controls on this tab.

### 7. Cross-link strip

- Spark Cognitive EKG (`/spark/ui`), Pressure Analytics (`#pressure`), Causal Geometry, autonomy readiness, raw snapshot JSON in `<details>`.
- Clarifies Pressure ≠ Drives.

### Controls row

- Subject selector (allowlist ∪ discovered, coverage badges).
- Window picker (1h / 6h / 24h / 7d) with degradation banner.
- Color mode toggle: `combined` | `align` | `funnel`.
- Refresh + last-loaded timestamp.
- Auto-refresh checkbox (optional; default on with visibility pause).

---

## Coloring model

Hash param `color=` with default `combined`.

| Mode | Gauges | KPI strip / goals card |
|---|---|---|
| `align` | Per-drive goal match | Same align rules applied to strip chips where applicable |
| `funnel` | Neutral / pressure-only fill with gate outline | Proposal→promote→complete health + gate verdict |
| `combined` (default) | Align rules | Funnel / gate rules |

**Align rules (per drive):**

- **Green:** elevated pressure *and* matching active/recent goal (`drive_origin` or dominant-drive match).
- **Yellow:** elevated pressure *without* matching goal (wanting without trajectory).
- **Red:** saturation / monoculture / starvation / drive↔goal mismatch / stale audit rail.

Regime always overrides raw magnitude: all-six-pinned is **red** even when pressures are high.

---

## Documentation (required in the same changeset)

### `orion/autonomy/README.md`

Add a section **"Hub Drives Analytics"** that explains:

- Why the surface exists (orientation on the live `DriveEngine` economy; not a mutation console).
- What “good” means here (churn + goal match, not all-high pressures).
- Link to this spec, the retrospective, and Hub README.
- Point operators at `#drives` and the offline gate script for deeper measurement.

### `services/orion-hub/README.md`

Add a subsection under operator tabs (near Pressure / Causal Geometry / Self):

- What the Drives tab is and is not (≠ Pressure Analytics).
- Session-preserving iframe pattern.
- Endpoints list (read-only).
- Hash params (`window`, `subject`, `color`).
- Dependency on `RECALL_PG_DSN` / `memory_pg_pool` and post-migration attribution columns.
- Restart notes after sql-writer + Hub deploy.

### `services/orion-spark-concept-induction/README.md`

Extend the existing drive / divergence section:

- Confirm this service is the **producer** of `DriveAuditV1` (including `tick_attribution`).
- Note that Hub analytics consumes Postgres history written by sql-writer — concept-induction itself does not serve the Hub UI.
- Link Hub Drives tab + autonomy README for operator meaning.

### In-tab tooltips

Tooltip text must stay consistent with the README explanations. Implementation tests assert each card id has a non-empty `TOOLTIP_COPY` entry covering definition / design / operator reading (string-structure check, not full prose lock).

---

## Files likely to touch

**Hub**

- `services/orion-hub/templates/drives-analytics.html` (new)
- `services/orion-hub/static/js/drives-analytics.js` (new)
- `services/orion-hub/templates/index.html` (tab + iframe section)
- `services/orion-hub/static/js/app.js` (tab / hash / refresh wiring)
- `services/orion-hub/scripts/api_routes.py` (or thin `drives_analytics_routes.py` included from main)
- `services/orion-hub/tests/test_drives_analytics_page.py` (new)
- `services/orion-hub/tests/test_drives_analytics_api.py` (new)
- `services/orion-hub/README.md`

**Persistence**

- `services/orion-sql-writer/app/models/drive_audit.py`
- `services/orion-sql-writer/app/worker.py` (column filter / derivations)
- `services/orion-sql-writer/app/main.py` (boot DDL if needed)
- `services/orion-sql-writer/tests/` (persist attribution regression)
- `services/orion-sql-db/manual_migration_drive_audits_v2_tick_attribution.sql` (new)

**Docs / shared**

- `orion/autonomy/README.md`
- `services/orion-spark-concept-induction/README.md`
- `docs/superpowers/specs/2026-07-16-hub-drives-analytics-design.md` (this file)

**Reuse without forking**

- `scripts/analysis/measure_autonomy_gate.py` (import pure helpers)
- `scripts/drive_state_divergence_audit.py` (factor or call read helpers for divergence card)

---

## Non-goals

- Mutating goals, tensions, or drive math from this tab.
- Merging with Pressure Analytics (`#pressure`).
- Backfilling historical `tick_attribution` for pre-migration rows.
- Replacing `measure_autonomy_gate.py` CLI (Hub surfaces the same ideas; CLI remains the offline instrument).
- Claiming `autonomy_state_v2` is a live second signal.
- Keyword / emotional detectors or chat-stance changes.
- New bus channels or schema registry entries.

---

## Acceptance checks

1. Hub tab `#drives` toggles without reloading Hub chat/session; iframe refresh only reloads the analytics page.
2. `GET /api/drives-analytics/snapshot` returns latest pressures for subject `orion` when pool is healthy; degrades honestly when not.
3. After sql-writer + migration deploy, new `drive_audits` rows store non-null `tick_attribution` for ticks that had tensions; UI window contributors use those rows.
4. Pre-migration window shows attribution coverage note, not fake zeros.
5. Dual time series render tick-rate + six sparklines; thin post-rebuild history shows coverage banner.
6. Color mode toggle switches align / funnel / combined behavior without reload.
7. Subject selector shows allowlist + discovered subjects with coverage badges.
8. Every card exposes a tooltip with definition / design / operator-reading.
9. Hub README + autonomy README + concept-induction README updated in the same PR.
10. Focused tests pass: page isolation, API degradation, sql-writer attribution persistence, tooltip map completeness.
11. No mutation routes mounted under `/api/drives-analytics/`.

---

## Risks / concerns

| Severity | Concern | Mitigation |
|---|---|---|
| Med | Attribution columns increase row size / write cost | Bounded 6 floats + short kinds list; retention already 90d; monitor row growth once |
| Med | Goal-alignment endpoint depends on goal store availability | Degrade coloring to yellow/neutral with explicit “goals unavailable” note |
| Med | CONCEPT_STORE_PATH wrong after rebuild | Divergence card banners fallback path (known 2026-07-13 incident class) |
| Low | Operator confuses Pressure tab with Drives | Naming, cross-link strip, README “is not” language |
| Low | 7d window on cold DB | Coverage degradation + downsample; never pretend full window |

---

## Recommended implementation slices

1. **Persistence first:** migration + sql-writer keep attribution + regression test (without this, contributors lie).
2. **API + snapshot/window/series** with degradation fixtures.
3. **Standalone page + gauges + KPI + series + tooltips.**
4. **Contributors + goal-alignment coloring + subject/window/color controls.**
5. **Divergence + goals card + cross-links.**
6. **Hub shell tab wiring + README trio + page/API tests.**

---

## Restart required (when implemented)

```bash
# After migration applied on the memory/sql DB used by RECALL_PG_DSN:
# restart sql-writer so new columns are written, then Hub so routes/UI load.
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

Exact compose flags follow each service’s README; print operator-facing restart commands in the PR report (do not sudo-restart from agents).

---

## Next step

After Juniper approves this written spec, produce an implementation plan via writing-plans, then implement in a dedicated worktree on a `feat/hub-drives-analytics` branch.
