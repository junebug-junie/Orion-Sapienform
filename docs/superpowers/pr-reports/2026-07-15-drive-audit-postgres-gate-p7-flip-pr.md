# PR report: drive-audit Postgres persistence + gate repoint (Option B) + P7 origination flip

Branch: `feat/drive-audit-postgres-gate`
Series: endogenous action motor-nerve follow-through — makes gate (b) drive co-activation measurable again from a live source, and operator-flips `ORION_ENDOGENOUS_ORIGINATION_ENABLED` (P7) per explicit Juniper approval.

## Summary

- `orion-sql-writer` now persists `memory.drives.audit.v1` (DriveAuditV1, published by
  orion-spark-concept-induction on every DriveEngine tick) into a new slim Postgres
  table `drive_audits` — the first live sink for this stream since orion-rdf-writer
  stopped materializing it to Fuseki on 2026-06-19 (`e9b233e9`, `RDF_SKIP_KINDS`).
- `scripts/analysis/measure_autonomy_gate.py` reads drive co-activation Postgres-first
  and only falls back to the frozen Fuseki DriveAudit graph for historical windows; the
  report names which source produced the number. Two dead sources still yield
  `UNMEASURABLE`, never a fabricated zero — the P6 guard is untouched.
- Stale NO-GO STATUS docstring on `orion/autonomy/endogenous_origination.py` refreshed
  to the current measured reality (gate (a) GO since 2026-07-13; gate (b) UNMEASURABLE
  from the dead source, being re-instrumented by this patch).
- **P7 flip (live-env operator action, not a commit):**
  `ORION_ENDOGENOUS_ORIGINATION_ENABLED=true` in
  `services/orion-spark-concept-induction/.env`. `.env_example` default stays `false`.
  Takes effect on that service's restart.

## Outcome moved

Gate (b) — drive co-activation, one of the two origination enable criteria — has been
structurally unmeasurable since 2026-06-19: the instrument read a Fuseki graph that
stopped receiving writes, and until P6 that read silently as "measured 0.0004, NO-GO"
instead of "measured nothing." After this patch, every DriveEngine tick lands a row in
`drive_audits` and the gate reads it. Combined with the P7 flip, Orion can now mint
endogenous tensions during exogenous silence AND we can finally watch co-activation
respond to that activity instead of arguing about a frozen snapshot.

## Current architecture (before this patch)

`ConceptWorker` publishes DriveAuditV1 on `orion:memory:drives:audit` (confirmed live —
a real message observed within 60s of subscribing). Sole subscriber was orion-rdf-writer,
which skips the kind. No Postgres persistence anywhere. `measure_autonomy_gate.py`'s
only drive source was the frozen Fuseki graph. `endogenous_origination.py` gated
always-off behind `ORION_ENDOGENOUS_ORIGINATION_ENABLED=false`, with a stale 2026-07-08
NO-GO docstring.

## Architecture touched

- `orion-sql-writer` (new consumer on an existing annotated channel — `channels.yaml`
  already listed it as a consumer of `orion:memory:drives:audit`; no contract change)
- `scripts/analysis/measure_autonomy_gate.py` (measurement instrument)
- Live env only: `orion-spark-concept-induction` (P7 flag)

## Files changed

- `services/orion-sql-writer/app/models/drive_audit.py`: new `DriveAuditSQL` — slim
  measurement table, `artifact_id` PK, JSON-with-JSONB-variant columns (SQLite tests /
  Postgres prod), index ownership documented
- `services/orion-sql-writer/app/worker.py`: `MODEL_MAP` entry +
  `_apply_drive_audit_derivations` (derives `active_count = len(active_drives)`,
  malformed → 0 never raises; maps `observed_at` ← artifact `ts`)
- `services/orion-sql-writer/app/settings.py`: `DEFAULT_ROUTE_MAP` entry + channel in
  the subscribe default list
- `services/orion-sql-writer/.env_example`: channel added to
  `SQL_WRITER_SUBSCRIBE_CHANNELS` (the env var is set live and the Pydantic alias
  REPLACES the default — settings-only would be dead wiring, a known incident class) +
  route entry in `SQL_WRITER_ROUTE_MAP_JSON`
- `services/orion-sql-writer/app/main.py`: boot DDL — table + expression index on
  `COALESCE(observed_at, created_at) DESC` (orchestrator fix: a bare `created_at`
  index would never serve the gate's window predicate since `ts` always populates
  `observed_at`)
- `services/orion-sql-db/manual_migration_drive_audits_v1.sql`: standalone migration,
  same DDL
- `services/orion-sql-writer/app/models/__init__.py`: export
- `services/orion-sql-writer/tests/test_drive_audit_sql_shape.py`: 12 tests — route/model
  map, channel in settings default AND `.env_example` (guards the alias-replaces-default
  incident class), column shape, derivation edge cases, `observed_at`←`ts`, SQLite merge
  idempotency
- `scripts/analysis/measure_autonomy_gate.py`: `fetch_drive_stats_postgres` (reuses the
  existing read-only autocommit connection; table-missing / empty-window / query-failure
  degrade to distinct notes, never raise), pure `resolve_drive_stats` source selection
  (Fuseki IO injected as a lazy callable — not queried at all when Postgres has rows),
  `render_report` names the drive source
- `scripts/analysis/tests/test_measure_autonomy_gate.py`: 10 new tests (36 total green)
- `orion/autonomy/endogenous_origination.py`: docstring-only STATUS refresh
- `services/orion-sql-writer/README.md`: channel table row + persistence paragraph

## Schema / bus / API changes

- Added: `drive_audits` Postgres table (new sink; no bus/schema contract change —
  channel, kind, and consumer annotation all pre-existed in `channels.yaml`)
- Removed: none
- Renamed: none
- Behavior changed: `measure_autonomy_gate.py` drive source is now Postgres-first
- Compatibility: Fuseki fallback keeps historical windows readable; gate report format
  gains a source line

## Env/config changes

- Added keys: `DRIVE_AUDITS_RETENTION_DAYS=90` (services/orion-sql-writer — startup
  prune window for the new table, 0 disables; added in the second review pass).
  Existing `SQL_WRITER_SUBSCRIBE_CHANNELS` / `SQL_WRITER_ROUTE_MAP_JSON` values
  extended.
- `.env_example` updated: `services/orion-sql-writer/.env_example`
- Local `.env` synced: hand-edited `services/orion-sql-writer/.env` (channel + route
  entry) — the sync script diffs against the main checkout's own `.env_example` and
  cannot see an unmerged branch's change; hand-editing is the sanctioned path for this
  case. Verified gitignored via `git check-ignore`.
- **Operator flip (approved)**: `services/orion-spark-concept-induction/.env`
  `ORION_ENDOGENOUS_ORIGINATION_ENABLED=false → true`. `.env_example` default
  unchanged (`false`).
- Skipped keys requiring operator action: none

## Tests run

```text
pytest services/orion-sql-writer/tests/test_drive_audit_sql_shape.py \
       services/orion-sql-writer/tests/test_action_outcome_sql_shape.py -q
  → 19 passed

pytest scripts/analysis/tests/test_measure_autonomy_gate.py -q
  → 36 passed (26 pre-existing + 10 new)

pytest services/orion-sql-writer/tests -q  (full dir, minus pre-broken dream module)
  → worktree: 18 failed / 119 passed / 3 errors
  → main:     18 failed / 107 passed / 3 errors  (identical pre-existing failures;
    delta is exactly the 12 new passing tests — nothing regressed)
```

## Evals run

```text
No eval harness exists for orion-sql-writer or the gate script; the gate script IS the
measurement instrument. Post-deploy live verification plan is in "Restart required".
```

## Docker/build/smoke checks

```text
Not run from this environment (code + tests only this cycle). The live path is
UNVERIFIED until the sql-writer restart below; first verification query included.
```

## Review findings fixed

- Finding: (orchestrator, pre-review) gate windows on COALESCE(observed_at, created_at)
  but the index was on bare created_at — never serves the query since ts always
  populates observed_at.
  - Fix: expression index `idx_drive_audits_window` in boot DDL + migration.
  - Evidence: `services/orion-sql-writer/app/main.py`, migration file.
- Finding: (review, MEDIUM — the one material caveat) Postgres-first suppresses in-window
  Fuseki history with no retention caveat: a `--window-days 120` run (the original NO-GO
  window) with a few days of drive_audits rows would present that tail as the full
  window — the "dead sensor read as behavioral finding" class, one notch softer.
  - Fix: `fetch_earliest_drive_audit_ts` (MIN over the same COALESCE, undefined-table
    degrades silently) feeding the existing `retention_caveat` machinery whenever the
    Postgres source wins.
  - Evidence: `test_fetch_earliest_drive_audit_ts_degrades`,
    `test_drive_audits_retention_caveat_fires_for_predating_window`; 50 tests green.
- Finding: (review, LOW) negative-count clamp in `parse_postgres_histogram_rows` could
  overwrite the real 0 bucket with a bogus foreign row `(-1, -4)` — and the test
  codified it.
  - Fix: negatives skipped like other malformed rows; test updated to assert the 0
    bucket survives.
  - Evidence: `test_parse_postgres_histogram_rows`.
- Finding: (review, LOW) `verdict_economy` docstring still said the drive source was
  Fuseki.
  - Fix: docstring corrected to "Postgres drive_audits, Fuseki fallback".
- Finding: (review, informational, accepted) the refreshed `endogenous_origination.py`
  docstring makes measurement/flag claims not provable from this diff — accepted: it is
  a STATUS docstring describing operator actions taken this cycle, with the measurement
  evidence on record in the 2026-07-07 design doc and this report.
- Review verdict: approve. Clean checks: full dead-wiring path trace (channel →
  settings → route map → MODEL_MAP → `_write_row` derivation ordering), genuine
  idempotent redelivery, zero writer/reader contract drift, all degrade paths
  non-fatal, UNMEASURABLE guard intact and test-locked.

### Second pass (/code-review high, 8 angles × verify): 10 findings, fixes applied

- Finding: (PLAUSIBLE, most severe) a single in-window Postgres row suppresses all
  in-window Fuseki history — verdict (b) computed on the drive_audits tail with only
  a caveat disclosing it; on long windows the headline can differ from the
  full-window value.
  - Fix: when the Postgres source wins with partial coverage (retention floor inside
    the window), the gate now also queries Fuseki and reports exactly how many
    in-window rows are excluded from the verdict. Full source-merging was
    deliberately NOT done: neither source covers the 2026-06-19→07-15 sensor gap, so
    any long window is partial regardless — disclosure with numbers is the honest
    altitude, and histogram-merging would silently double-count if RDF
    materialization is ever re-enabled.
- Finding: (PLAUSIBLE) retention-caveat control flow keyed off the display label
  (`drive_source.startswith("postgres")`) — a relabel would silently disable it.
  - Fix: `resolve_drive_stats` now returns a structured `source_kind`
    ("postgres"/"fuseki"/"none") that `run()` branches on; labels are display-only.
- Finding: (CONFIRMED) `drive_audits` had no retention while siblings
  (`orion_metacognitive_trace`, `grammar_events`) prune at startup.
  - Fix: `DRIVE_AUDITS_RETENTION_DAYS` (default 90, 0 disables) + startup DELETE on
    `COALESCE(observed_at, created_at)`, mirroring the metacog pattern.
    `.env_example` + live `.env` + README updated.
- Finding: (CONFIRMED) `DriveAuditSQL` paid `merge()`'s always-missing PK SELECT per
  event.
  - Fix: added to `INSERT_ONLY_MODELS` (fast path: add + duplicate-key skip); model
    docstring and redelivery test updated to insert-once semantics.
- Finding: (CONFIRMED) stale cross-reference in
  `services/orion-cortex-exec/app/endogenous_runtime.py` still called the
  origination module "NO-GO-verdicted, mostly-dead" after this cycle flipped it live.
  - Fix: cross-reference rewritten to point at the module's STATUS docstring and
    design doc ("do NOT assume it is dead").
- Finding: (PLAUSIBLE) the two histogram parsers diverged — Postgres skips negative
  rows, the SPARQL parser still clamped them into bucket 0.
  - Fix: SPARQL parser now skips negatives too, with a regression test.
- Finding: (CONFIRMED) `drive_source` printed twice in the same report section and
  the fallback fact duplicated between note and label.
  - Fix: one render site (the bullet), short labels, fallback stated once in caveats.
- Findings accepted without code change: docstring runtime claims (informational,
  accepted above); strict-validation dead-lettering silently shrinking the gate
  denominator (house pattern for every schema-validated kind — tracked in
  Follow-ups); design-rationale prose triplicated across model/migration/README
  (matches the sibling `action_outcomes` convention exactly — not churning it in
  this PR).

## Follow-ups (tracked issues per CLAUDE.md §11)

1. **Eval harness gap (orion-sql-writer + gate script)**: neither has an eval lane;
   the gate script is itself a measurement instrument but has no eval asserting its
   verdict behavior end-to-end against a seeded database. Smallest useful eval: seed
   a scratch Postgres with known audit distributions and assert GO/NO-GO/UNMEASURABLE
   verdicts. File as an issue when opening this PR.
2. **Dead-lettered drive audits are invisible to the gate**: a schema-drifted
   producer's rows land in `bus_fallback_log`, silently shrinking verdict (b)'s
   denominator. Consider a gate caveat sourced from `bus_fallback_log` counts for
   this kind.

## Restart required

```bash
# 1) sql-writer: pick up new code (after merge) + live .env channel/route additions
docker compose --env-file .env --env-file services/orion-sql-writer/.env \
  -f services/orion-sql-writer/docker-compose.yml up -d --build

# 2) concept-induction: pick up the P7 flag flip (env only, no code change needed pre-merge)
docker compose --env-file .env --env-file services/orion-spark-concept-induction/.env \
  -f services/orion-spark-concept-induction/docker-compose.yml up -d

# First verification (within ~1h of sql-writer restart — DriveEngine ticks continuously):
psql -h localhost -p 55432 -U postgres -d conjourney \
  -c "SELECT count(*), max(created_at) FROM drive_audits;"

# 48h watch:
docker logs orion-spark-concept-induction 2>&1 | grep endogenous_origination_fired
python scripts/analysis/measure_autonomy_gate.py --window-hours 24   # after a day of rows
```

## Risks / concerns

- Severity: MEDIUM. Concern: P7 enable could ring (endogenous tension → goal → action →
  new tension). Mitigation: exogenous-silence suppression, origination band, capability
  budgets, 88/day dispatch cap, P3 relief damping (−0.10 cap, idempotent per goal); 48h
  watch on `endogenous_origination_fired` rate; rollback is one env flip + restart.
- Severity: LOW. Concern: drive_audits volume (one row per DriveEngine tick). Mitigation:
  slim columns only; PK upsert; retention/pruning deferred until observed volume warrants.
- Severity: LOW. Concern: live Postgres path UNVERIFIED until restart. Mitigation:
  verification query above; table-missing degrade in the gate keeps it honest meanwhile.

## PR link

(filled after push)
