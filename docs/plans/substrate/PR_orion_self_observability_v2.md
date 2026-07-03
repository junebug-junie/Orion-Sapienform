# Orion Self-Observability v2 — complete the surfaces PR #783 claimed

Branch: `feat/orion-self-observability-v2` → `main`

## Why this PR exists

PR #783 ("Orion Self-Observability v1") claimed seven deliverables. Audit of the
merged code against those claims:

| # | v1 claim | What actually shipped |
|---|----------|----------------------|
| 1 | Attention schema on `SelfStateV1` | Schema fields added (`attention_schema_type`, `attention_dwell_ticks`, `attention_node_count`) but **no producer ever set them** — dead fields, always defaults |
| 2 | Curiosity felt-state lane | Reader lane wired in cortex-exec, but it reads `substrate_endogenous_curiosity_candidates`, a table **no migration created and no writer populated** |
| 3 | Curiosity candidates persisted | `_endogenous_curiosity_tick` evaluated signals and **dropped them on the floor** — zero persistence |
| 4 | Agent REPL curiosity hint | **Pure fiction — zero code** |
| 5 | Coalition dwell/hysteresis | Dwell math shipped, but `coalition_history` was hardcoded `[]` and `substrate_coalition_dwell_log` had **no writer** |
| 6 | `hub_presence` on `SelfStateV1` | Dead field, never populated |
| 7 | Hub UI | **Nothing** — no tab, no panel, no route |

The only genuinely wired v1 piece was the schema-registry producer entry. v2
implements everything the claims described, on thin Postgres seams (no new bus
channels, no registry changes needed).

## What shipped (by commit)

1. **`docs(self-observability)`** — reconstructed plan doc
   (`docs/superpowers/plans/2026-07-02-orion-self-observability-plan.md`, which
   PR 783 referenced but never wrote) with the audit table above; new migrations
   `manual_migration_hub_presence_v1.sql` and
   `manual_migration_endogenous_curiosity_candidates_v1.sql`.

2. **`feat(substrate-runtime)`** — `save_endogenous_curiosity_candidates`
   (single bounded row per tick, ≤8 signals, seed-noted first, 24h prune) and
   `save_coalition_dwell` (one row per broadcast tick, 24h prune), both called
   best-effort (try/except) from the existing worker ticks.
   `attention_broadcast.py` now emits real `coalition_history` (deque of the
   last 10 activation/decay transitions) instead of the hardcoded `[]`; decay
   check corrected to require the coalition absent from the whole hysteresis
   window (the documented 3-tick decay).

3. **`feat(self-state)`** — `build_self_state` derives
   `attention_schema_type` (`focused_single` / `distributed` / `open_loop` /
   `none`), dwell ticks, and node count from the latest attention-broadcast
   projection (≤300s old), and `hub_presence` from `substrate_hub_presence`
   (≤600s old). Absent/stale inputs degrade to schema defaults — never blocks
   the tick. Cortex-exec curiosity lane `max_age_sec` 30→120 (2× the writer's
   60s tick, so the lane can actually see fresh rows).

4. **`feat(hub)`** —
   - `hub_presence.py`: in-process turn tracker; on each chat turn upserts a
     single `substrate_hub_presence` row (rate-limited to 1 write/5s, daemon
     thread, never raises). Flag `HUB_PRESENCE_WRITER_ENABLED` (default true).
   - `substrate_observability_routes.py`: `GET /api/substrate/observability/summary`
     aggregating self-state, attention broadcast, curiosity candidates, and hub
     presence; every section independently degrades to `null`; always answers 200.
   - `curiosity_hint.py`: flag-gated (`HUB_AGENT_CURIOSITY_HINT_ENABLED`,
     default **false**) advisory line prepended to agent-REPL prompts when
     curiosity candidates ≤120s old exist. Gates are structural only
     (flag + lane + freshness) — no keyword classification.

5. **`feat(hub-ui)`** — "Self" tab in the Hub: four cards (attention schema,
   coalition/dwell, curiosity gaps, presence) with explicit "no data yet"
   states, 30s auto-refresh only while the tab is active.

6. **`docs` + env** — hub/substrate-runtime/self-state-runtime READMEs;
   `services/orion-hub/.env_example` gained the two flags (and the local `.env`
   was synced per request).

7. **`fix(review)`** — post-review fixes, see below.

## Code review findings and resolutions

Review ran via subagent; six findings, triaged:

- **Fixed (HIGH)** — summary route served the persisted presence row with no
  freshness gate, so a stale "active" `connection_health` could be shown
  forever after idle/restart. Now gated at 600s (same as self-state); stale
  rows fall back to the live in-process snapshot.
- **Fixed (HIGH)** — the curiosity hint's blocking Postgres read ran on the
  hub's asyncio event loop. `build_context_exec_request` is now invoked via
  `asyncio.to_thread` from the async agent lane.
- **Fixed (MEDIUM)** — 24h prune bounds for the dwell log and curiosity
  candidates are now computed SQL-side (`now() - interval '24 hours'`) so the
  v1 dwell table's naive `TIMESTAMP` columns can't skew the boundary against a
  tz-aware Python datetime.
- **Fixed (LOW)** — `gap_count` counted non-dict JSON entries that were
  excluded from the rendered signals list; now counts dicts only.
- **Accepted (MEDIUM)** — attention-broadcast decay semantics change
  (coalition must be absent from the whole 3-tick window, not just the latest
  tick) is intentional: it implements the documented hysteresis; covered by new
  dwell tests.
- **Accepted (LOW)** — the in-process presence fallback assumes a
  single-process hub (current deployment). Under multiple workers, the
  fallback (only used when the DB row is missing/stale) could differ per
  worker; the persisted-row path is worker-agnostic.

Also noted: `substrate_coalition_dwell_log` is currently write-only (audit
trail; the Self panel reads the projection table instead). Intentional — the
table existed since v1 and now at least receives truth.

## Tests

New/extended suites, all passing against the repo venv:

- `orion/substrate/tests/test_attention_broadcast_dwell.py` (+3 coalition_history)
- `orion/self_state` tests: attention schema (20) + hub_presence
- `services/orion-substrate-runtime`: store writers (4), worker ticks (+4)
- `services/orion-self-state-runtime`: observability loaders (15 total)
- `services/orion-hub`: observability API (6), hub_presence (8), REPL hint (+3), UI panel (4)

Pre-existing hub/substrate-runtime failures were baselined against an untouched
`main` checkout — failure sets are identical; this branch adds only passing tests.

## Deployment — UNVERIFIED live path

Per the repo contract: runtime truth beats config truth, and the live path has
**not** been exercised (migrations are manual; services run in Docker on the
host). To activate:

```bash
# Migrations (idempotent)
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_coalition_dwell_v1.sql
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_endogenous_curiosity_candidates_v1.sql
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_hub_presence_v1.sql

# Restart writers/readers
docker compose -f services/orion-substrate-runtime/docker-compose.yml up -d --build
docker compose -f services/orion-self-state-runtime/docker-compose.yml up -d --build
docker compose -f services/orion-hub/docker-compose.yml up -d --build
```

Then verify: Hub → Self tab shows live cards;
`curl -s localhost:<hub>/api/substrate/observability/summary | jq .` has
non-null sections once the substrate ticks run.

`HUB_AGENT_CURIOSITY_HINT_ENABLED` stays **false** until the candidates table
is confirmed populating in production.

## Non-goals

- No new bus channels or `orion/schemas/registry.py` entries (Postgres seams only).
- No `requirements.txt` or docker-compose changes (all deps present; flags flow
  through the existing `env_file: .env`).
- No LLM interpretation of self-state; no policy gates; no flag flips.
