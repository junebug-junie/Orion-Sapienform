# Hub Drives Analytics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a session-preserving Hub **Drives** tab that orients operators on the live six-drive economy — gauges, real contributor history, KPIs, dual time series, divergence, goals, tooltips — backed by persisting `tick_attribution` / `tension_kinds` on `drive_audits`.

**Architecture:** Causal-geometry isolation pattern (standalone `/drives-analytics` page + dedicated JS + iframe Hub tab). Postgres `drive_audits` is the history SoR (via Hub `memory_pg_pool`). sql-writer stops dropping slim attribution columns. KPI/saturation math imports pure helpers from `scripts/analysis/measure_autonomy_gate.py`. Strictly read-only.

**Tech Stack:** FastAPI (orion-hub), asyncpg pool, SQLAlchemy sql-writer models, vanilla JS + Tailwind CDN (Hub convention), Pytest gate tests.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-16-hub-drives-analytics-design.md` — follow it; do not invent mutation routes.
- `DRIVE_KEYS` = `("coherence", "continuity", "capability", "relational", "predictive", "autonomy")` from `orion.spark.concept_induction.drives`.
- Allowed windows: `1, 6, 24, 168` hours; default `24`. Default subject: `orion`. Default color mode: `combined`.
- Subject allowlist: `orion`, `relationship`, `juniper` (from `docs/architecture/autonomy_subjects.md`), unioned with distinct DB subjects.
- Hash state: `#drives?window=&subject=&color=` (also support bare `#drives`).
- Degrade never 500: missing pool / undefined table / empty window → `{degraded: true, error, source, ...}`.
- No bus/schema/registry changes. No backfill of old attribution rows.
- Work in an isolated worktree on `feat/hub-drives-analytics` (branch from current `docs/hub-drives-analytics` or main after merging the spec). Never commit from the shared checkout.
- `drives-analytics.js` must not reference `window.OrionHub`.
- Env: reuse `RECALL_PG_DSN` / `CONCEPT_STORE_PATH`; if any `.env_example` key is added, run `python scripts/sync_local_env_from_example.py`.

## File map

| File | Responsibility |
|---|---|
| `services/orion-sql-writer/app/models/drive_audit.py` | Add `tick_attribution`, `tension_kinds` columns |
| `services/orion-sql-writer/app/main.py` | Boot DDL `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` |
| `services/orion-sql-db/manual_migration_drive_audits_v4_tick_attribution.sql` | Manual/operator migration |
| `services/orion-sql-writer/tests/test_drive_audit_sql_shape.py` | Flip shape contract to require attribution columns |
| `services/orion-hub/scripts/drives_analytics.py` | Pure payload builders + SQL query helpers (keep `api_routes.py` thin) |
| `services/orion-hub/scripts/api_routes.py` | Mount page + GET routes only |
| `services/orion-hub/templates/drives-analytics.html` | Standalone page shell + card DOM ids |
| `services/orion-hub/static/js/drives-analytics.js` | Fetch, render, tooltips, polling, hash controls |
| `services/orion-hub/templates/index.html` | Tab button + iframe panel |
| `services/orion-hub/static/js/app.js` | Tab switch / hash / iframe refresh |
| `services/orion-hub/tests/test_drives_analytics_page.py` | Isolation + tooltip + tab wiring tests |
| `services/orion-hub/tests/test_drives_analytics_api.py` | API degrade + payload shape tests |
| `orion/autonomy/README.md` | Hub Drives Analytics section |
| `services/orion-hub/README.md` | Operator tab docs |
| `services/orion-spark-concept-induction/README.md` | Producer vs Hub consumer note |

---

### Task 1: Persist slim attribution on `drive_audits`

**Files:**
- Modify: `services/orion-sql-writer/app/models/drive_audit.py`
- Modify: `services/orion-sql-writer/app/main.py` (boot DDL near existing `drive_audits` block ~645–678)
- Create: `services/orion-sql-db/manual_migration_drive_audits_v4_tick_attribution.sql`
- Modify: `services/orion-sql-writer/tests/test_drive_audit_sql_shape.py`
- Test: `services/orion-sql-writer/tests/test_drive_audit_sql_shape.py`

**Interfaces:**
- Consumes: existing `_write_row` mapper-column filter (passes any column that exists on the model)
- Produces: `DriveAuditSQL.tick_attribution` (`_JSONB`, nullable), `DriveAuditSQL.tension_kinds` (`_JSONB`, nullable); wire fields from `DriveAuditV1` persist without worker special-cases

- [ ] **Step 1: Write the failing tests (flip shape contract)**

In `test_drive_audit_sql_shape.py`:

1. Add to `EXPECTED_COLUMNS`: `"tick_attribution"`, `"tension_kinds"`.
2. Replace `test_archive_fields_are_not_columns` so it still forbids `evidence_items` and `source_event_refs`, but **requires** `tick_attribution` and `tension_kinds` as columns.
3. Add:

```python
def test_tick_attribution_and_tension_kinds_pass_through() -> None:
    attribution = {
        "coherence": 0.1,
        "continuity": 0.0,
        "capability": 0.0,
        "relational": 0.2,
        "predictive": 0.5,
        "autonomy": 0.0,
    }
    kinds = ["substrate.world_coverage_gap", "tension.contradiction.v1"]
    row = _row_dict(
        _make_audit(tick_attribution=attribution, tension_kinds=kinds)
    )
    assert row["tick_attribution"] == attribution
    assert row["tension_kinds"] == kinds
    obj = DriveAuditSQL(**row)
    assert obj.tick_attribution["predictive"] == 0.5
    assert obj.tension_kinds[0] == "substrate.world_coverage_gap"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /mnt/scripts/Orion-Sapienform-hub-drives-analytics  # or feat worktree
pytest services/orion-sql-writer/tests/test_drive_audit_sql_shape.py -q
```

Expected: FAIL — `EXPECTED_COLUMNS` mismatch and/or `tick_attribution` not in mapper.

- [ ] **Step 3: Minimal model + DDL + migration**

Update `DriveAuditSQL` docstring: attribution/kinds are now stored (bounded); evidence/refs still dropped.

Add columns (same `_JSONB` pattern as `drive_pressures`):

```python
tick_attribution = Column(_JSONB, nullable=True)
tension_kinds = Column(_JSONB, nullable=True)
```

In `app/main.py` after the `summary` ALTER, add:

```python
conn.exec_driver_sql(
    "ALTER TABLE drive_audits ADD COLUMN IF NOT EXISTS tick_attribution JSONB;"
)
conn.exec_driver_sql(
    "ALTER TABLE drive_audits ADD COLUMN IF NOT EXISTS tension_kinds JSONB;"
)
```

Create `services/orion-sql-db/manual_migration_drive_audits_v4_tick_attribution.sql`:

```sql
-- Drive audit v4: persist slim tick_attribution + tension_kinds for Hub Drives Analytics.
-- Wire schema DriveAuditV1 already carries these; sql-writer previously dropped them.
-- No backfill. Pre-migration rows remain NULL.
ALTER TABLE drive_audits ADD COLUMN IF NOT EXISTS tick_attribution JSONB;
ALTER TABLE drive_audits ADD COLUMN IF NOT EXISTS tension_kinds JSONB;
```

No change needed in `_apply_drive_audit_derivations` — mapper filter passes new columns automatically.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest services/orion-sql-writer/tests/test_drive_audit_sql_shape.py -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/orion-sql-writer/app/models/drive_audit.py \
  services/orion-sql-writer/app/main.py \
  services/orion-sql-db/manual_migration_drive_audits_v4_tick_attribution.sql \
  services/orion-sql-writer/tests/test_drive_audit_sql_shape.py
git commit -m "$(cat <<'EOF'
feat(sql-writer): persist drive audit tick_attribution and tension_kinds

Hub Drives Analytics needs real contributor history; stop dropping the
slim attribution fields that DriveAuditV1 already carries on the wire.
EOF
)"
```

---

### Task 2: Hub pure helpers module (`drives_analytics.py`)

**Files:**
- Create: `services/orion-hub/scripts/drives_analytics.py`
- Create: `services/orion-hub/tests/test_drives_analytics_helpers.py`

**Interfaces:**
- Consumes: `DRIVE_KEYS`; `drive_stats_from_histogram`, `parse_dominant_rows`, `apply_dominant_counts`, `retention_caveat` from `scripts.analysis.measure_autonomy_gate` (import carefully — see note below)
- Produces:
  - `ALLOWED_HOURS: set[int] = {1, 6, 24, 168}`
  - `SUBJECT_ALLOWLIST: tuple[str, ...] = ("orion", "relationship", "juniper")`
  - `normalize_hours(hours: int | None) -> int`
  - `normalize_subject(subject: str | None) -> str`
  - `coverage_meta(*, requested_hours: int, oldest: datetime | None, newest: datetime | None, row_count: int) -> dict`
  - `build_window_kpis(*, active_count_hist: dict[int, int], dominant_counts: dict[str, int], mean_pressures: dict[str, float], coverage: dict) -> dict` (includes `gate_verdict_drive_only` using saturation clauses without requiring self-state pressure — see Step 3)
  - `aggregate_tick_attribution(rows: list[dict]) -> dict` with `{per_drive: {...}, attributed_row_count, null_attribution_row_count, first_attributed_at}`
  - `align_color_for_drive(*, pressure: float, has_matching_goal: bool, saturated: bool, starved: bool, stale: bool) -> Literal["green","yellow","red","neutral"]`
  - `downsample_series(points: list[dict], max_points: int = 240) -> list[dict]`
  - `bucket_tick_rates(timestamps: list[datetime], *, window_start: datetime, bucket_sec: int) -> list[dict]` returning `{t, count}`

**Import note:** Prefer adding repo root to `sys.path` the same way other hub scripts/tests do. If importing `scripts.analysis.measure_autonomy_gate` is fragile from the service, copy only the pure functions used into `drives_analytics.py` **with a comment pointing at the source + constants** — but prefer import. Unit-test that saturation thresholds match (`SATURATION_DOMINANT_SHARE`, etc.).

For hub window KPIs, resource_pressure may be unavailable; expose:

```python
def drive_economy_verdict_from_drive_stats(drive_stats) -> str:
    """Drive-rail verdict for the Hub strip.

    Uses measure_autonomy_gate saturation/coactivation thresholds.
    Returns UNMEASURABLE | SATURATED | NO-GO | GO_DRIVE_ONLY.
    GO_DRIVE_ONLY means coactivation bar met and not saturated; full GO
    still requires resource_pressure (offline gate only). Never claim full GO.
    """
```

- [ ] **Step 1: Write failing helper tests**

```python
# services/orion-hub/tests/test_drives_analytics_helpers.py
from datetime import datetime, timedelta, timezone

from scripts import drives_analytics as da


def test_normalize_hours_clamps_to_allowlist() -> None:
    assert da.normalize_hours(24) == 24
    assert da.normalize_hours(99) == 24  # default
    assert da.normalize_hours(168) == 168


def test_coverage_meta_reports_short_history() -> None:
    now = datetime(2026, 7, 16, 12, tzinfo=timezone.utc)
    oldest = now - timedelta(hours=3)
    meta = da.coverage_meta(
        requested_hours=24, oldest=oldest, newest=now, row_count=10
    )
    assert meta["row_count"] == 10
    assert meta["coverage_hours"] == 3.0
    assert "retention_note" in meta and meta["retention_note"]


def test_align_color_regimes() -> None:
    assert da.align_color_for_drive(
        pressure=0.8, has_matching_goal=True, saturated=False, starved=False, stale=False
    ) == "green"
    assert da.align_color_for_drive(
        pressure=0.8, has_matching_goal=False, saturated=False, starved=False, stale=False
    ) == "yellow"
    assert da.align_color_for_drive(
        pressure=0.9, has_matching_goal=True, saturated=True, starved=False, stale=False
    ) == "red"


def test_aggregate_tick_attribution_skips_nulls() -> None:
    rows = [
        {"tick_attribution": {"predictive": 0.5}, "observed_at": "2026-07-16T10:00:00+00:00"},
        {"tick_attribution": None, "observed_at": "2026-07-16T09:00:00+00:00"},
    ]
    out = da.aggregate_tick_attribution(rows)
    assert out["attributed_row_count"] == 1
    assert out["null_attribution_row_count"] == 1
    assert out["per_drive"]["predictive"] == 0.5
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest services/orion-hub/tests/test_drives_analytics_helpers.py -q
```

Expected: FAIL (module missing)

- [ ] **Step 3: Implement `drives_analytics.py`**

Implement the helpers listed in Interfaces. Keep functions pure (no asyncpg). Document `GO_DRIVE_ONLY` in docstring so UI never labels it as full gate GO.

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest services/orion-hub/tests/test_drives_analytics_helpers.py -q
```

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/scripts/drives_analytics.py \
  services/orion-hub/tests/test_drives_analytics_helpers.py
git commit -m "$(cat <<'EOF'
feat(hub): add drives analytics pure helpers

Coverage, attribution aggregation, and goal-align color regimes for the
Drives operator surface — no I/O yet.
EOF
)"
```

---

### Task 3: Read-only Hub API — subjects, snapshot, window, series

**Files:**
- Modify: `services/orion-hub/scripts/drives_analytics.py` (add async query helpers)
- Modify: `services/orion-hub/scripts/api_routes.py` (mount routes; keep handlers thin)
- Create: `services/orion-hub/tests/test_drives_analytics_api.py`

**Interfaces:**
- Consumes: `app.state.memory_pg_pool`; helpers from Task 2
- Produces HTTP:
  - `GET /api/drives-analytics/subjects`
  - `GET /api/drives-analytics/snapshot?subject=`
  - `GET /api/drives-analytics/window?subject=&hours=`
  - `GET /api/drives-analytics/series?subject=&hours=`

Payload contracts (always include `degraded: bool`, `source: dict`):

**snapshot success:**
```json
{
  "degraded": false,
  "source": {"table": "drive_audits", "subject": "orion"},
  "subject": "orion",
  "observed_at": "...",
  "drive_pressures": {},
  "active_drives": [],
  "dominant_drive": "predictive",
  "summary": "...",
  "tick_attribution": {},
  "tension_kinds": [],
  "stale": false
}
```

**window success:** includes `kpis`, `dominant_counts`, `mean_pressures`, `attribution` (aggregated), `coverage` from `coverage_meta`.

**series success:** `{tick_rate: [{t, count}], pressures: {drive: [{t, v}]}, coverage: {...}}`.

SQL window predicate (match gate):

```sql
WHERE subject = $1
  AND COALESCE(observed_at, created_at) >= $2
  AND COALESCE(observed_at, created_at) < $3
```

Cap series rows at 5000 server-side; downsample client helpers to ≤240 points.

Mirror causal-geometry degrade: catch `UndefinedTableError`, missing pool → degraded payload.

- [ ] **Step 1: Write failing API tests** (pattern from `test_causal_geometry_api.py`)

```python
import pytest
from scripts import api_routes


@pytest.mark.asyncio
async def test_snapshot_degrades_when_pool_missing(monkeypatch, hub_main) -> None:
    # Use the same hub_main / FakeConn fixtures as causal-geometry tests.
    monkeypatch.setattr(hub_main.app.state, "memory_pg_pool", None, raising=False)
    # If hub_main fixture sets state differently, patch where _snapshot reads pool.
    payload = await api_routes.api_drives_analytics_snapshot(subject="orion")
    assert payload["degraded"] is True
    assert "error" in payload["source"] or "error" in payload


@pytest.mark.asyncio
async def test_snapshot_returns_pressures_from_fake_row(monkeypatch, hub_main) -> None:
    # FakeConn returns one drive_audits row with drive_pressures JSON.
    ...
    payload = await api_routes.api_drives_analytics_snapshot(subject="orion")
    assert payload["degraded"] is False
    assert "predictive" in payload["drive_pressures"]


def test_no_mutation_routes_under_drives_analytics_prefix() -> None:
    mutating = {"POST", "PUT", "PATCH", "DELETE"}
    for route in api_routes.router.routes:
        path = getattr(route, "path", "")
        methods = getattr(route, "methods", set()) or set()
        if path.startswith("/api/drives-analytics"):
            assert methods.isdisjoint(mutating)
```

Copy FakeConn patterns from `services/orion-hub/tests/test_causal_geometry_api.py` rather than inventing a new mock style.

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest services/orion-hub/tests/test_drives_analytics_api.py -q
```

- [ ] **Step 3: Implement query helpers + routes**

Add async functions in `drives_analytics.py` that take `pool` and return payloads.

In `api_routes.py`:

```python
@router.get("/api/drives-analytics/subjects")
async def api_drives_analytics_subjects() -> Dict[str, Any]:
    ...

@router.get("/api/drives-analytics/snapshot")
async def api_drives_analytics_snapshot(subject: str = Query(default="orion")) -> Dict[str, Any]:
    ...
```

Also add the HTML page stub route early (or in Task 4) — if you add it here, return a minimal placeholder template until Task 4.

- [ ] **Step 4: Run — expect PASS**

```bash
pytest services/orion-hub/tests/test_drives_analytics_api.py services/orion-hub/tests/test_drives_analytics_helpers.py -q
```

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/scripts/drives_analytics.py \
  services/orion-hub/scripts/api_routes.py \
  services/orion-hub/tests/test_drives_analytics_api.py
git commit -m "$(cat <<'EOF'
feat(hub): add drives-analytics read APIs for snapshot window series

Degrading Postgres-backed endpoints for the operator Drives surface.
EOF
)"
```

---

### Task 4: Goal-alignment + divergence endpoints

**Files:**
- Modify: `services/orion-hub/scripts/drives_analytics.py`
- Modify: `services/orion-hub/scripts/api_routes.py`
- Modify: `services/orion-hub/tests/test_drives_analytics_api.py`

**Interfaces:**
- Consumes: `scripts.drive_state_divergence_audit.load_drive_state_v1` + `DEFAULT_CONCEPT_STORE_PATH` / env `CONCEPT_STORE_PATH`; `orion.autonomy.repository.AutonomyRepository` (or `_fetch_active_goals` via a thin wrapper) for goals — **fail open**
- Produces:
  - `GET /api/drives-analytics/goal-alignment?subject=`
  - `GET /api/drives-analytics/divergence?subject=`

**goal-alignment payload:**
```json
{
  "degraded": false,
  "goals_available": true,
  "active_goals": [{"artifact_id": "...", "drive_origin": "predictive", "proposal_status": "active", "goal_statement": "..."}],
  "per_drive": {
    "predictive": {"pressure": 0.7, "has_matching_goal": true, "color_align": "green"}
  },
  "funnel": {
    "proposed": 0, "active": 0, "planned": 0, "executing": 0, "completed": 0, "archived": 0
  },
  "notes": []
}
```

If Fuseki/graph unavailable: `goals_available: false`, `degraded: true`, colors fall back to yellow/neutral with note `"goals unavailable"`.

**divergence payload:**
```json
{
  "degraded": false,
  "store_path": "...",
  "store_path_is_fallback_default": false,
  "drive_state_pressures": {},
  "audit_pressures": {},
  "deltas": {},
  "max_abs_delta": 0.0,
  "autonomy_state_v2_note": "frozen/historical — not a live second signal"
}
```

Set `store_path_is_fallback_default` true when path equals `DEFAULT_CONCEPT_STORE_PATH` and env was unset (same loud warning class as the divergence audit script).

- [ ] **Step 1: Write failing tests** for goals-unavailable degrade and divergence fallback banner flag.

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Implement endpoints** (fail-open around graph + store I/O).

- [ ] **Step 4: Run — PASS**

```bash
pytest services/orion-hub/tests/test_drives_analytics_api.py -q
```

- [ ] **Step 5: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(hub): add drives goal-alignment and divergence APIs

Read-only goal match + concept-store vs audit integrity for coloring.
EOF
)"
```

---

### Task 5: Standalone page shell, tooltips map, polling

**Files:**
- Create: `services/orion-hub/templates/drives-analytics.html`
- Create: `services/orion-hub/static/js/drives-analytics.js`
- Modify: `services/orion-hub/scripts/api_routes.py` (`GET /drives-analytics` → template with `HUB_UI_ASSET_VERSION`)
- Create: `services/orion-hub/tests/test_drives_analytics_page.py`

**Interfaces:**
- Consumes: Task 3–4 APIs
- Produces: page with ids:
  - Controls: `drivesSubjectSelect`, `drivesWindowSelect`, `drivesColorMode`, `drivesRefreshButton`, `drivesAutoRefresh`, `drivesLastLoaded`, `drivesCoverageBanner`
  - Cards: `drivesGaugesCard`, `drivesContributorsCard`, `drivesKpiStrip`, `drivesSeriesCard`, `drivesDivergenceCard`, `drivesGoalsCard`, `drivesCrossLinks`
  - `TOOLTIP_COPY` keys: `gauges`, `contributors`, `kpi`, `series`, `divergence`, `goals`, `crosslinks` — each `{definition, design, reading}`

- [ ] **Step 1: Write page tests** (mirror `test_causal_geometry_page.py` + tooltip structure)

```python
def test_drives_analytics_route_template_and_bundle_are_standalone() -> None:
    ...
    assert "/static/js/drives-analytics.js?v={{HUB_UI_ASSET_VERSION}}" in template
    assert "window.OrionHub" not in script
    assert "'/api/drives-analytics/snapshot'" in script


def test_tooltip_copy_covers_all_cards() -> None:
    script = (HUB_ROOT / "static/js/drives-analytics.js").read_text()
    for key in ("gauges", "contributors", "kpi", "series", "divergence", "goals", "crosslinks"):
        assert f"{key}:" in script or f'"{key}"' in script
    assert "definition:" in script
    assert "design:" in script
    assert "reading:" in script


def test_polling_pauses_when_hidden() -> None:
    script = (HUB_ROOT / "static/js/drives-analytics.js").read_text()
    assert "visibilitychange" in script
    assert "setInterval(" in script
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Implement HTML + JS skeleton**

HTML: dark Hub-style page (copy causal_geometry.html chrome), cards as empty containers with ⓘ buttons calling `showTooltip(cardId)`.

JS skeleton:
- Parse hash query for `window`, `subject`, `color` (also accept parent hash if embedded — prefer own `location.hash` / `URLSearchParams` on the iframe URL; parent can set iframe `src="/drives-analytics#...?` or pass via query string `?window=24&subject=orion&color=combined` — **use query string on iframe src** for reliability: `/drives-analytics?window=24&subject=orion&color=combined`).
- Spec says hash `#drives?window=` on the Hub shell; iframe should read `URLSearchParams(window.location.search)` and Hub `app.js` should update iframe `src` query when controls change **or** let the iframe own controls and only sync shell hash for the tab key. **Decision for implementers:** iframe owns controls via its own URL search params; on change, `history.replaceState` inside iframe; parent shell only needs `#drives`. Document this in README.
- `refreshAll()` fetches all endpoints in parallel (`Promise.allSettled`), renders or degraded notices.
- Polling 30s + visibility pause (copy causal-geometry.js).

Minimal render for this task: text/JSON placeholders in each card is OK **only if** tests still require real tables later — prefer empty DOM hooks ready for Task 6.

- [ ] **Step 4: Run page tests — PASS**

```bash
pytest services/orion-hub/tests/test_drives_analytics_page.py -q
```

- [ ] **Step 5: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(hub): add standalone drives-analytics page shell and tooltips

Session-safe iframe target with polling and operator tooltip copy map.
EOF
)"
```

---

### Task 6: Render gauges, KPI strip, dual time series

**Files:**
- Modify: `services/orion-hub/static/js/drives-analytics.js`
- Modify: `services/orion-hub/templates/drives-analytics.html` (gauge/series containers)
- Modify: `services/orion-hub/tests/test_drives_analytics_page.py`

**Interfaces:**
- Consumes: snapshot + window + series + goal-alignment payloads
- Produces: DOM thermometers (CSS/SVG), KPI chips, canvas or SVG sparklines + tick-rate bars; colors from `color` mode (`combined`/`align`/`funnel`)

- [ ] **Step 1: Add failing render-contract tests**

```python
def test_drives_js_builds_gauges_not_raw_json_dump() -> None:
    script = (HUB_ROOT / "static/js/drives-analytics.js").read_text()
    assert "function renderGauges(" in script
    assert "function renderKpiStrip(" in script
    assert "function renderSeries(" in script
    assert "target.textContent = JSON.stringify(payload, null, 2);" not in script
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Implement renderers**

- Gauges: six columns; fill height = pressure%; class `drive-color-green|yellow|red|neutral`; active badge; dominant outline; stale chip from snapshot.
- KPI: chips for tick rate, active count, coactivation, top dominant share, `gate_verdict` (label `GO_DRIVE_ONLY` as “drive economy OK (partial)” — never “GO”), last age.
- Series: top bar chart tick_rate; below six sparklines sharing x-domain; show `coverage` banner when `coverage_hours < requested_hours`.

Color application:
- `align` / `combined`: gauge colors from `goal-alignment.per_drive.*.color_align` (recompute client-side with same rules if needed).
- `funnel`: gauges neutral fill; KPI chips colored by funnel/gate.

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(hub): render drives gauges KPI strip and dual time series

Goal-aligned thermometer gauges plus tick-rate and pressure sparklines.
EOF
)"
```

---

### Task 7: Contributors, controls, goals, divergence, cross-links

**Files:**
- Modify: `services/orion-hub/static/js/drives-analytics.js`
- Modify: `services/orion-hub/templates/drives-analytics.html`
- Modify: `services/orion-hub/tests/test_drives_analytics_page.py`

**Interfaces:**
- Consumes: subjects, window.attribution, snapshot.tick_attribution, goal-alignment, divergence
- Produces: full remaining cards + control wiring

- [ ] **Step 1: Tests**

```python
def test_drives_js_has_contributor_live_and_window_toggle() -> None:
    script = ...
    assert "Live" in script and "Window" in script
    assert "null_attribution_row_count" in script or "attribution not recorded" in script


def test_drives_js_has_color_mode_and_window_picker() -> None:
    assert "combined" in script and "align" in script and "funnel" in script
```

- [ ] **Step 2: FAIL → Step 3: Implement**

- Contributors: Live uses snapshot attribution; Window uses aggregated; banner when `null_attribution_row_count > 0`.
- Subject select: populate from `/subjects` with coverage badges (`N rows · oldest`).
- Divergence card: table of deltas + fallback banner.
- Goals card: list active goals read-only; funnel counts; **no** action buttons.
- Cross-links: `/spark/ui`, `/#pressure`, `/causal-geometry`, autonomy readiness note, `<details>` raw JSON.

- [ ] **Step 4: PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(hub): complete drives analytics cards and operator controls

Contributors, subject/window/color controls, divergence, goals, links.
EOF
)"
```

---

### Task 8: Hub shell tab wiring (session-preserving)

**Files:**
- Modify: `services/orion-hub/templates/index.html` (nav near Pressure/Causal Geometry; panel section near other iframes)
- Modify: `services/orion-hub/static/js/app.js` (`getElementById`, `setActiveTab`, hash, click, refresh)
- Modify: `services/orion-hub/tests/test_drives_analytics_page.py`

**Interfaces:**
- Consumes: `/drives-analytics` page
- Produces: `#drives` tab that toggles without full navigation; refresh reloads iframe only

Template snippets to add (mirror causal-geometry):

Nav:
```html
<a
  id="drivesAnalyticsTabButton"
  href="#drives"
  data-hash-target="#drives"
  class="px-3 py-1.5 text-xs font-semibold rounded-full bg-gray-800 text-gray-200 border border-gray-700 hover:bg-gray-700"
  role="button"
>Drives</a>
```

Panel:
```html
<section id="drives" data-panel="drives" class="hidden w-full bg-gray-900 rounded-2xl shadow-lg p-5 flex flex-col gap-4 min-h-[56rem]">
  ...
  <iframe id="drivesAnalyticsPanelFrame" src="/drives-analytics?window=24&subject=orion&color=combined" ...></iframe>
</section>
```

`app.js`: follow exact causal-geometry patterns for `isDrives`, `styleTabButton`, click → `setActiveTab("drives")` + `history.replaceState(null, "", "#drives")`, hash handler `h === "#drives"`, refresh → `drivesAnalyticsPanelFrame.contentWindow?.location.reload()`.

Assert shell does **not** load `drives-analytics.js` in `index.html`.

- [ ] **Step 1: Write failing shell tests** (copy from `test_causal_geometry_page.py` shell tests)

- [ ] **Step 2: FAIL → Step 3: Wire → Step 4: PASS**

```bash
pytest services/orion-hub/tests/test_drives_analytics_page.py -q
```

- [ ] **Step 5: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(hub): wire session-preserving Drives tab into Hub shell

Iframe isolation matching causal-geometry so chat session survives toggles.
EOF
)"
```

---

### Task 9: README trio + acceptance gate

**Files:**
- Modify: `orion/autonomy/README.md`
- Modify: `services/orion-hub/README.md`
- Modify: `services/orion-spark-concept-induction/README.md`
- Modify: tests if README path assertions are desired (optional — prefer content exists checks in page test only)

**Docs content requirements (from spec):**

1. **autonomy README** — section `## Hub Drives Analytics`: why (orientation, not mutation); what “good” means (churn + goal match); links to spec, retrospective, Hub README, `#drives`, `measure_autonomy_gate.py`.
2. **Hub README** — operator tab subsection: is/is-not Pressure; iframe pattern; endpoint list; query/hash params; `RECALL_PG_DSN` + v4 attribution columns; restart commands for sql-writer then Hub.
3. **concept-induction README** — producer of `DriveAuditV1` including `tick_attribution`; Hub reads Postgres via sql-writer; link Hub tab + autonomy README.

- [ ] **Step 1: Add a small docs presence test** (optional but recommended)

```python
def test_readmes_mention_hub_drives_analytics() -> None:
    auto = (REPO_ROOT / "orion/autonomy/README.md").read_text()
    hub = (HUB_ROOT / "README.md").read_text()
    spark = (REPO_ROOT / "services/orion-spark-concept-induction/README.md").read_text()
    assert "Hub Drives Analytics" in auto
    assert "Drives" in hub and "/api/drives-analytics" in hub
    assert "tick_attribution" in spark and "Drives" in spark
```

- [ ] **Step 2: FAIL → Step 3: Write README sections → Step 4: PASS**

- [ ] **Step 5: Full focused gate**

```bash
pytest services/orion-sql-writer/tests/test_drive_audit_sql_shape.py \
  services/orion-hub/tests/test_drives_analytics_helpers.py \
  services/orion-hub/tests/test_drives_analytics_api.py \
  services/orion-hub/tests/test_drives_analytics_page.py -q
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git commit -m "$(cat <<'EOF'
docs: explain Hub Drives Analytics for operators

Autonomy, Hub, and concept-induction READMEs cover why the surface
exists, how to read it, and the producer/consumer split.
EOF
)"
```

- [ ] **Step 7: PR report** using AGENTS.md §18 template; list restart:

```bash
# Apply migration if boot DDL hasn't run yet, then:
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

Run code-review skill in a subagent before declaring DONE.

---

## Spec coverage self-check

| Spec requirement | Task |
|---|---|
| Persist `tick_attribution` / `tension_kinds` | Task 1 |
| Migration + boot DDL | Task 1 |
| Subjects / snapshot / window / series APIs | Task 3 |
| Goal-alignment + divergence APIs | Task 4 |
| Standalone page + dedicated JS + tooltips | Task 5 |
| Gauges + KPI + dual series + color modes | Task 6 |
| Contributors + controls + goals + cross-links | Task 7 |
| Session-preserving Hub tab | Task 8 |
| README trio | Task 9 |
| Read-only / degrade / no backfill | Tasks 1–4, 9 |
| Reuse measure_autonomy_gate pure math | Task 2–3 |

## Placeholder scan

No TBD/TODO left in tasks. Implementers must still fill FakeConn details by copying `test_causal_geometry_api.py` fixtures — that file is the template, not a placeholder.

## Type / name consistency

- Route prefix: `/api/drives-analytics/*` and page `/drives-analytics`
- Tab key / panel id: `drives`
- Frame id: `drivesAnalyticsPanelFrame`
- Helper module: `scripts.drives_analytics` (Hub)
- Migration: `manual_migration_drive_audits_v4_tick_attribution.sql` (v3 already used)
