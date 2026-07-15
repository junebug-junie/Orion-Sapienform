# PR report: kill the drive-audit graph path everywhere (phases 2+3) + stop a live Fuseki dual-write leak

Branch: `feat/drive-audit-graph-path-kill` (stacked on `feat/drive-audit-postgres-gate`, PR #1064)
Series: endogenous action follow-through — after #1064 made the measurement gate Postgres-only, this finishes the job: the last graph *reader* is repointed and the graph *writer* is deleted, making Postgres `drive_audits` the only possible sink for `memory.drives.audit.v1`.

## Summary

- `scripts/drive_history_reflection_synthesis.py` (last reader of the frozen Fuseki
  drives graph, starved since 2026-06-19) now reads Postgres `drive_audits` over the
  same DSN it already used for writes. No fallback; empty fetch → honest
  `insufficient_history`. The old SPARQL detail-join scope reduction dies:
  `mean_pressure_by_drive` / `active_drive_frequency` become genuine full-window
  statistics instead of an ≤8-event enrichment sample.
- `drive_audits` gains a `summary TEXT NULL` column (the one field the synthesis
  needed that the slim table dropped) — model, boot DDL with
  `ADD COLUMN IF NOT EXISTS`, migration v2. Generic worker passthrough verified;
  no worker change.
- orion-rdf-writer's drive-audit materialization is **deleted**, not skipped:
  handler, kind maps, dispatch branch, the channel subscription itself, compose
  passthrough, env keys — plus a do-not-re-add regression test. `channels.yaml`
  consumer annotation updated in the same changeset.
- **Live leak found and stopped by this patch**: the 2026-07-14 deploy recreated the
  rdf-writer container against the live `.env`, whose `RDF_SKIP_KINDS` never
  contained `memory.drives.audit.v1` (only `.env_example` had it, and the sync never
  bridged it). Fuseki dual-writes silently resumed the same day the motor nerve went
  live: **5,981 audits ≈ 500K triples since 07-13** into the 37.9M-triple drives
  graph, on the instance with a prior 99.99%-memory incident. Verified via container
  env inspection + write-committed log lines + SPARQL count. The rdf-writer restart
  below stops it permanently.

## Outcome moved

Before: drive audits had two sinks (one sanctioned, one silently resurrected), the
reflection synthesis read a frozen snapshot, and the "kill" was one env edit away
from un-killing. After: one sink, one reader path, both Postgres; the graph path is
structurally unrepresentable (no handler, no subscription, no skip-list entry to
flip); and the synthesis computes per-drive pressure/activation statistics it never
actually had.

## Current architecture (before this patch)

`orion:memory:drives:audit` → orion-rdf-writer (subscribed, actively writing ~20KB
per DriveEngine tick to Fuseki since 07-14) and → orion-sql-writer (from #1064).
`drive_history_reflection_synthesis.py` read Fuseki via SPARQL with a disclosed
scope reduction (pressures/active-drives enriched only for ≤8 cited events).

## Architecture touched

- `orion-sql-writer` (column addition, additive)
- `orion-rdf-writer` (drive-audit path deleted; other kinds untouched)
- `scripts/drive_history_reflection_synthesis.py` (source repoint)
- `orion/bus/channels.yaml` (consumer annotation — contract change in-changeset)

## Files changed

- `services/orion-sql-writer/app/models/drive_audit.py`: `summary` column + docstring
- `services/orion-sql-writer/app/main.py`: boot DDL column + `ADD COLUMN IF NOT EXISTS`
- `services/orion-sql-db/manual_migration_drive_audits_v2.sql`: standalone migration
- `services/orion-sql-writer/tests/test_drive_audit_sql_shape.py`: column set,
  passthrough, None handling
- `scripts/drive_history_reflection_synthesis.py`: Postgres fetch (subject +
  windowed, capped at existing 500), pure row→event conversion with defensive JSONB
  coercion, local `is_undefined_table_error` copy (named source), all
  SPARQL/enrichment machinery deleted (−551 gross lines)
- `tests/test_drive_history_reflection_synthesis.py`: 48 tests — row conversion edge
  cases, degrade paths, JSONB→aggregation flow, fetch-empty → `insufficient_history`
  with no LLM call and no write
- `services/orion-rdf-writer/app/autonomy.py`, `app/rdf_builder.py`,
  `app/settings.py`, `docker-compose.yml`, `.env_example`: drive-audit path removal
- `services/orion-rdf-writer/tests/test_autonomy_materialization.py`: materialization
  test replaced with not-materialized + not-subscribed regression tests
- `orion/bus/channels.yaml`: `orion-rdf-writer` removed from
  `orion:memory:drives:audit` consumers, with do-not-re-add comment

## Schema / bus / API changes

- Added: `drive_audits.summary TEXT NULL`
- Removed: orion-rdf-writer as a consumer of `orion:memory:drives:audit`
  (annotation + actual subscription); Fuseki drive-audit materialization
- Behavior changed: `drive_history_reflection_synthesis.py` CLI lost
  `--query-url` / `--sparql-timeout-sec` / `--detail-timeout-sec` (graph-read args
  with nothing left to configure)
- Compatibility: existing `drive_audits` rows get `summary NULL` (honest absence);
  the frozen Fuseki graph remains readable by ad-hoc queries, just unwritten and
  unread by any code path

## Env/config changes

- Removed keys: `CHANNEL_MEMORY_DRIVES_AUDIT` (rdf-writer `.env_example`,
  docker-compose, settings) and `memory.drives.audit.v1` dropped from
  `RDF_SKIP_KINDS` in `.env_example`
- Live `.env` synced by hand (branch-only change; sync script can't see unmerged
  `.env_example` edits): `services/orion-rdf-writer/.env` — `CHANNEL_MEMORY_DRIVES_AUDIT`
  removed; live `RDF_SKIP_KINDS` never contained the kind (that's the leak's root
  cause, documented above). Verified gitignored.
- Live DB: `manual_migration_drive_audits_v1.sql` and `_v2.sql` both applied to live
  Postgres this session (table + window index + summary column verified present).

## Tests run

```text
pytest tests/test_drive_history_reflection_synthesis.py \
       services/orion-sql-writer/tests/test_drive_audit_sql_shape.py \
       scripts/analysis/tests/test_measure_autonomy_gate.py -q
  → 94 passed

pytest services/orion-rdf-writer/tests -q
  → 40 passed

scripts/check_single_consumer_channels.py (live NUMSUB) → OK, 31 channels
channels.yaml parses; check_service_env_compose_parity FUSEKI_* failure is
pre-existing (untouched keys), identical on the base branch.
```

## Evals run

```text
No eval harness for these surfaces; the gap is tracked as issue #1066 (filed on
PR #1064) and unchanged by this patch.
```

## Docker/build/smoke checks

```text
Live verification this session (read-only + sanctioned DDL): container env
inspection proving the leak, SPARQL counts quantifying it, both migrations
applied and verified against live Postgres. Service rebuilds deferred to the
restart step below.
```

## Review findings fixed

Review verdict: approve with fixes, no CRITICAL/HIGH. The CRITICAL-class check —
does anything in production invoke the synthesis script with the deleted CLI
flags — came back clean (the script is manual/on-demand by design; nothing in
cron/Makefile/compose/orion-actions references it).

- Finding: (MEDIUM) root `.env_example` still declared `CHANNEL_MEMORY_DRIVES_AUDIT`
  after its only consumers were deleted — dead config inviting re-wiring.
  - Fix: removed from root `.env_example` and hand-synced out of the live root
    `.env` (gitignored, verified).
- Finding: (MEDIUM) `services/orion-memory-consolidation/README.md` — the script's
  actual operator runbook — still documented `--query-url` (deleted; argparse would
  exit 2) and described the Fuseki source in three places.
  - Fix: source description, pipeline step, and flag table all corrected; the one
    remaining Fuseki mention is the intentional historical note.
- Finding: (MEDIUM) the combined single-process pytest invocation across
  sql-writer + rdf-writer suites fails on a pre-existing `app`-package collision
  at collection (reproduced identically at the base commit).
  - Fix: none needed in code; the Tests section below states the suites as
    separate invocations, which is this repo's established workaround.
- Finding: (LOW) a JSONB column returned as text (non-default driver behavior)
  would silently empty the full-window statistics while reporting success.
  - Fix: `_maybe_parse_json_str` — text values are parsed, unparseable text warns
    instead of masquerading as measured-empty; regression test added.
- Accepted without change: (LOW) DB-outage and genuinely-empty history both surface
  as `insufficient_history` in `--json` output (deliberate; the stderr log line
  distinguishes them, and MIN_EVENTS prevents fabricated output). (LOW)
  `source_kind` stays `"rdf_memory_graph"` for same-day dedup continuity — renaming
  needs a dedup-transition plan, tracked as a named follow-up, not churned here.

## Restart required

```bash
# 1) rdf-writer FIRST — stops the live Fuseki dual-write leak (after merge)
docker compose --env-file .env --env-file services/orion-rdf-writer/.env \
  -f services/orion-rdf-writer/docker-compose.yml up -d --build

# 2) sql-writer — starts drive_audits rows flowing (incl. summary)
docker compose --env-file .env --env-file services/orion-sql-writer/.env \
  -f services/orion-sql-writer/docker-compose.yml up -d --build

# Verify the leak stopped (should show no new writes after restart):
docker logs orion-athena-rdf-writer --since 10m 2>&1 | grep drives.audit || echo "leak stopped"
# Verify rows flowing with summary:
psql -h localhost -p 55432 -U postgres -d conjourney \
  -c "SELECT count(*), count(summary) FROM drive_audits;"
```

## Risks / concerns

- Severity: LOW. Concern: reflection synthesis has no history until drive_audits
  accumulates (existing rows pre-summary have `summary NULL`). Mitigation: reducer's
  honest `insufficient_history` refusal; fills within days.
- Severity: LOW. Concern: the frozen Fuseki drives graph (37.9M triples) still
  occupies the instance. Mitigation: out of scope here; a drop/archive decision is a
  separate operator call now that nothing reads or writes it.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1069 (base: main — #1064
merged mid-build and its branch auto-deleted; this branch stacks directly on its
final commit, no rebase)
