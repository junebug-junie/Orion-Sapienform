# grammar_traces retention + the Atlas listing index

Branch: `fix/grammar-traces-retention`

## Summary

- `grammar_traces` was the one grammar table with **no retention at all**, while its
  children (`grammar_events` / `grammar_atoms` / `grammar_edges`) have been pruned at
  3 days since PR #1759. The parent rows never expired.
- Measured live before this patch: **487,970 trace rows, oldest 2026-07-23, and 205,465
  of them (42%) already had zero atoms.** The Grammar Atlas was not "at risk of" listing
  hollow traces — it already was. That is CLAUDE.md's "UI panel rendered with no real
  backing artifact", an invalid success state, not merely wasted disk.
- Added `apply_grammar_traces_retention` (window `GRAMMAR_TRACES_RETENTION_DAYS=3`,
  matching the children) and registered it **last** in `GRAMMAR_RETENTION_TABLES`.
- Found and fixed a second, unrelated live problem in the same table: the Atlas's own
  listing query `ORDER BY started_at DESC LIMIT 50` had **no usable index** — the table's
  only index was the `trace_id` primary key — so every Atlas page load sequentially
  scanned all 487,970 rows.
- Hardened `_other_retention_truth_blocks`, which hand-listed retention windows in a dict
  and `KeyError`'d the entire `/health` snapshot when a table was added to
  `_EXTRA_RETENTION_TABLES` and not to it.

## Outcome moved

| | before | after |
|---|---|---|
| Atlas trace listing (`ORDER BY started_at DESC LIMIT 50`) | Parallel Seq Scan, 487,970 rows, **11,523 blocks, 58.8 ms** | Index Scan, **47 blocks, 0.35 ms** |
| Atlas listing *filtered by `session_id`* | Seq Scan, 11,524 blocks, 40.3 ms | **unchanged** — no `session_id` index; `idx_grammar_traces_started_at` cannot serve the filter |
| Retention's batched delete probe on `grammar_traces` | no index; would seq-scan per batch | Index Only Scan, 13 blocks, 0.85 ms |
| Traces past the retention window | never deleted (unbounded since 2026-07-23) | pruned every 60s alongside their children |
| Hollow traces (trace row present, zero atoms) | 205,465 / 487,970 (42%) | drains with the retention backlog |

Both plans measured with `EXPLAIN (ANALYZE, BUFFERS)` against the real live table, not
estimated. The unfiltered listing is the default the Atlas UI loads;
`/api/substrate/atlas/traces?session_id=...`
(`services/orion-hub/scripts/grammar_atlas_routes.py:153-159`) still seq-scans and is
**not** improved by this patch.

## Current architecture

`services/orion-sql-writer` runs a bounded retention pass every
`GRAMMAR_RETENTION_INTERVAL_SEC` (60s, PR #1759) over the tables listed in
`grammar_truth.GRAMMAR_RETENTION_TABLES`. Each table gets a batched
`DELETE ... WHERE created_at < :cutoff ORDER BY created_at, <id> LIMIT :batch_size`,
an FK-safety check, batch/elapsed caps, and a non-fatal debt count.

`grammar_events` additionally respects a **cursor floor**: it refuses to delete rows any
of the five reducer lanes still has unconsumed below the cutoff
(`_grammar_events_cursor_floor`).

`grammar_traces` was not in that list. `orion/grammar/ledger.py` upserts trace rows;
`orion/grammar/query.py::list_traces` (behind
`services/orion-hub/scripts/grammar_atlas_routes.py`, `/api/substrate/atlas/traces`)
lists them newest-first and then expands one via `get_trace`.

## Architecture touched

- `services/orion-sql-writer` — one more table in the existing periodic retention cycle.
  No new service, no new loop, no new abstraction.
- Live Postgres (`conjourney`) — two additive indexes, created `CONCURRENTLY`.
- No bus channel, schema, or API contract changed.

## Files changed

- `services/orion-sql-db/manual_migration_grammar_traces_retention.sql` (new): the two
  indexes, with the measured reason for each.
- `services/orion-sql-writer/app/grammar_truth.py`: `apply_grammar_traces_retention`;
  `grammar_traces` added to `_EXTRA_RETENTION_TABLES` and last in
  `GRAMMAR_RETENTION_TABLES`; `_other_retention_truth_blocks` derived instead of
  hand-listed; two `known_risks` entries corrected (they asserted grammar_traces has no
  retention, which this patch makes false).
- `services/orion-sql-writer/app/grammar_retention_loop.py`: `retention_days_for` now
  returns a window for `grammar_traces` — without this the table is registered but
  silently skipped (`days <= 0` means disabled), which looks identical to working
  retention in the logs.
- `services/orion-sql-writer/app/settings.py`: `grammar_traces_retention_days` (3); also
  corrected a stale comment that justified the 3-day window with a "~20,000x margin"
  review had already disproved with live data.
- `services/orion-sql-writer/.env_example`: `GRAMMAR_TRACES_RETENTION_DAYS=3`.
- `services/orion-sql-writer/tests/test_grammar_retention_periodic.py`: new
  `TestTheParentTraceRowIsPrunedToo`; `test_only_grammar_events_opts_in` renamed and
  widened to `test_exactly_the_cursor_coupled_tables_opt_in`.
- `services/orion-sql-writer/tests/test_grammar_truth.py`: `_mock_settings` derives its
  retention-days attributes from `_EXTRA_RETENTION_TABLES`.

## Why the cursor floor applies to the parent too — and what it does *not* buy

`respect_cursor_floor=True` on `grammar_traces`, same as `grammar_events`.

**My first justification for this was wrong, and review falsified it on live data.** I
wrote that a trace's `created_at` is `<=` the `created_at` of every event under it, so the
same floor keeps the parent alive at least as long as its owed children. That claim is
100% false (3,627 of 3,627 sampled traces violate it) and self-contradictory even on its
own terms — a *smaller* `created_at` is deleted *sooner*, so if it were true the parent
would die first.

The two columns are on different clocks. `orion/grammar/ledger.py:71` stamps
`grammar_traces.created_at` with wall-clock **write** time;
`ledger.py:117` stamps `grammar_events.created_at` with **occurrence** time
(`observed_at or emitted_at`). Occurrence precedes write, so a trace row is consistently
*newer* than its own earliest event — measured lag 0.15s to 9.4s.

What is actually true, stated precisely:

- A trace outlives its **earliest** event by construction.
- It does **not** outlive every *later* unconsumed event, and the floor does not close
  that gap: the floor is `MIN(created_at)` over unconsumed events, so a trace whose first
  event was already consumed can sit below the floor and be deleted while a later sibling
  survives. Measured at the live 3-day cutoff: **4 such events, 8.6s of lag**, gone on the
  next 60s cycle.

The floor stays on for the case it was built for — a genuinely stalled lane, where it
holds trace rows back by hours or days rather than seconds. Nothing reads across the seam
in the meantime: reducers key on `grammar_events` and never join `grammar_traces`, and the
Atlas keys on `grammar_traces`. The docstring now says this instead of the inverted claim.

**Also checked, because it would have been the real bug:** `orion/grammar/ledger.py`
upserts trace rows, so if `created_at` were refreshed on conflict a long-running trace
could be deleted while still live. It is not — `created_at` appears only in `values`
(`:71`), never in either `on_conflict_do_update` `set_` (`:74-83`, `:86-89`), corroborated
live by 2,716 rows whose `started_at` (which *is* refreshed) is up to 2h40m later than
their `created_at`. And the risk is not reachable anyway: across 21,219 traces, lag from
trace-row creation to its last event is p50 −0.02s, p999 1m27s, max 8m59s, **zero over an
hour**, against a 3-day window.

## Why order matters, and why nothing enforces it

`grammar_traces` runs **last** in `GRAMMAR_RETENTION_TABLES`, after every table that hangs
off it.

`services/orion-sql-db/manual_migration_grammar_atlas.sql` declares
`references grammar_traces(trace_id)` on six child tables — but the live database has
**zero** foreign key constraints touching any `grammar_*` table (verified against
`pg_constraint`, 2026-08-20). The tables were created from SQLAlchemy metadata first, so
the migration's FK clauses never took effect. Ordering is therefore a correctness
argument this code has to make for itself, not one the database will make for it. Pinned
by `test_the_parent_is_pruned_after_its_children`.

## Schema / bus / API changes

- Added: two indexes on `grammar_traces` (`idx_grammar_traces_created_at`,
  `idx_grammar_traces_started_at`). Additive, `CONCURRENTLY`, no lock.
- Removed / renamed / behavior changed: none.
- Compatibility: `grammar_traces` rows older than the window now disappear. Any consumer
  that expected traces to live forever will see fewer of them — none found; the Atlas is
  the only reader and it is already the thing being fixed.

## Env/config changes

- Added keys: `GRAMMAR_TRACES_RETENTION_DAYS=3`
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: yes (`services/orion-sql-writer/.env_example`)
- local `.env` synced: **by hand**, verified at
  `/mnt/scripts/Orion-Sapienform/services/orion-sql-writer/.env:64`.
  `scripts/sync_local_env_from_example.py` reads `.env_example` from the *primary*
  checkout, so a key added in a worktree is invisible to it.
- Skipped keys requiring operator action: none

## Tests run

```text
$ pytest services/orion-sql-writer/tests -q
11 failed, 390 passed, 3 skipped, 34 warnings in 12.72s
```

The same 11 failures reproduce on `main` at the merge base (377 passed there), so this
branch adds 13 passing tests and no new breakage. Pre-existing failures, unrelated to
retention: `test_biometrics_summary_sql_shape`, `test_chat_history_response_identity_merge`
(4), `test_journal_entry_payload_boundary`, `test_notify_attention_ack` (3),
`test_notify_attention_escalate` (2).

New tests, all in `services/orion-sql-writer/tests/`:

- `test_grammar_retention_periodic.py::TestTheParentTraceRowIsPrunedToo` — 6 tests: the table
  is in the retention cycle, the parent is pruned after its children, the delete keys on
  `trace_id`, the loop passes a real window, a stalled lane holds the parent back, and both
  the Atlas listing and the delete probe have an index.
- `test_grammar_retention_periodic.py::test_exactly_the_cursor_coupled_tables_opt_in` —
  parametrized over all five managed tables; asserts the real `respect_cursor_floor` kwarg
  reaching `_apply_bounded_table_retention`, not its source text.
- `test_every_managed_table_has_a_real_settings_window` — asserts against real
  `Settings.model_fields`, so a `MagicMock` fixture cannot hide config drift.
- `test_a_missing_window_degrades_health_instead_of_500ing_it`.
- `test_startup_only_covers_the_pre_periodic_tables_on_purpose` — pins the deliberate
  `periodic_only = {"grammar_traces"}` exemption against `main.py`.

## Evals run

`services/orion-sql-writer/` has no `evals/` directory. No eval harness exists for this
service; the live runtime evidence below stands in its place. Follow-up worth filing:
the service has retention, cursor, and truth-block behavior that would suit a small eval
harness, and today has none.

## Docker/build/smoke checks

```text
$ ./scripts/safe_docker_build.sh orion-sql-writer up -d --build
Container orion-athena-sql-writer  Recreated
Container orion-athena-sql-writer  Started
```

Live runtime evidence, `docker logs orion-athena-sql-writer`:

```text
grammar_traces_retention_batch batch=1 deleted=18 cutoff=2026-08-17T16:13:20.563369+00:00
grammar_traces_retention_complete cutoff=2026-08-17T16:13:20.563369+00:00 rows_pruned=18 \
  batches=1 elapsed_sec=0.02 remaining_debt=0 capped_batches=False capped_elapsed=False
grammar_retention_cycle pruned=6273 remaining_debt={'grammar_events': 4577962, \
  'grammar_edges': 266085} floored=[] skipped=[]
```

Table state, before the branch was deployed and after it reached steady state:

```text
                    before            after
rows                488,742           81,478
oldest created_at   2026-07-23        2026-08-17 16:13:23   (exactly the 3-day window)
hollow traces       219,702           1
total relation size ~1.1 GB           219 MB
```

"Hollow" = a trace row with zero surviving `grammar_atoms` children. The 219,702 hollow
rows were the accumulated pruning artifact this patch exists to stop; the single remaining
one is a trace whose atoms were pruned moments before the parent's own cycle ran, and it
clears on the next pass. Steady state is now ~18-20 rows per 60s cycle with
`remaining_debt=0` — the table tracks its window instead of growing without bound.

Index evidence (`EXPLAIN (ANALYZE, BUFFERS)`), captured before and after the manual
migration was applied live:

```text
Atlas listing  (ORDER BY started_at DESC LIMIT 50)
  before: Parallel Seq Scan  11,523 blocks   58.8 ms
  after:  Index Scan             47 blocks    0.35 ms

Retention delete probe (created_at < cutoff)
  after:  Index Only Scan        13 blocks    0.85 ms   Heap Fetches: 0
```

Both indexes confirmed `indisvalid = t` in `pg_index`.

## Review findings fixed

- Finding: **My cursor-floor rationale was exactly inverted.** The docstring claimed a
  trace's `created_at` is `<=` the `created_at` of every event under it, so one floor keeps
  parent and children together. That is false in both directions: the two columns are on
  different clocks (`grammar_traces.created_at` is wall-clock *write* time,
  `orion/grammar/ledger.py:71`; `grammar_events.created_at` is *occurrence* time,
  `ledger.py:117`), and even if it held, a *smaller* `created_at` deletes *sooner*, so the
  parent would die first. 3,627 of 3,627 sampled traces violated the claimed inequality.
  - Fix: rewrote the docstring and the "cursor floor" section of this report to state the
    true, narrower property — a trace outlives its *earliest* event by construction, but
    not its later unconsumed siblings.
  - Evidence: measured residual — 4 events were briefly parentless, for 8.6 seconds,
    self-healing on the next cycle. That is the real bound, and it is now written down.

- Finding: **One of my new tests could not fail.**
  `assert "respect_cursor_floor=True" in inspect.getsource(fn)` was satisfied by the new
  function's own docstring.
  - Fix: replaced with the parametrized behavioral test that monkeypatches
    `_apply_bounded_table_retention` and asserts the kwarg actually received.
  - Evidence: mutation-tested both ways. Flipping the real kwarg to `False` left the old
    test green (29 passing); against the new test it fails exactly one parametrization.

- Finding: `_other_retention_truth_blocks` hand-listed its tables and raised `KeyError`
  (then, after my first fix, `AttributeError`) on an unknown one — which 500s `/health`,
  the same blast radius as the bug it replaced.
  - Fix: derive the window map from `_EXTRA_RETENTION_TABLES` and *degrade* rather than
    raise — log an error, record the table in `_retention_window_config_missing`, and emit a
    `{table}_retention_window_unconfigured` degraded reason.
  - Evidence: `test_a_missing_window_degrades_health_instead_of_500ing_it`.

- Finding: the `_mock_settings` fixture returns a `MagicMock`, which answers `hasattr()` for
  any attribute name, so a missing real settings field would never be caught.
  - Fix: the fixture now derives its retention-days attrs from `_EXTRA_RETENTION_TABLES`,
    and a new test asserts against real `Settings.model_fields`.
  - Evidence: `test_every_managed_table_has_a_real_settings_window`.

- Finding: `main.py` still hand-lists four startup retention tables; `grammar_traces` is not
  among them.
  - Fix: this is deliberate — the startup pass already blocks the event loop for ~260s and
    should not grow. Pinned the exemption in a test instead of adding a fifth blocking
    block, so the omission is now an asserted decision rather than an oversight.
  - Evidence: `test_startup_only_covers_the_pre_periodic_tables_on_purpose`.

- Finding: the ordering comment claimed a microsecond-scale safety margin against a window
  that is really ~1.8 days.
  - Fix: replaced with the measured convergence table (traces ~2.4h, atoms ~0.45d,
    edges ~0.5d, events ~1.8d).
  - Evidence: in `grammar_truth.py`, alongside `GRAMMAR_RETENTION_TABLES`.

- Finding: two `known_risks` entries and the service README both asserted that
  `grammar_traces` has no retention. The README was additionally stale from PR #1759 (said
  four tables, "startup-run", default 15).
  - Fix: both corrected; README rewritten to five tables, default 3, 60s timer, cursor
    floor, and `grammar_traces`' deliberate periodic-only status.

## Restart required

Already applied on athena — the branch is deployed and verified live. To reproduce
elsewhere, from a worktree:

```bash
./scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

The SQL migration is already applied live and is idempotent
(`create index concurrently if not exists`). On a fresh database:

```bash
psql -h localhost -p 55432 -U postgres -d conjourney \
  -f services/orion-sql-db/manual_migration_grammar_traces_retention.sql
```

Note it cannot run inside a transaction block — `CREATE INDEX CONCURRENTLY` forbids it, and
an interrupted build leaves an `INVALID` index that must be dropped and rebuilt.

## Risks / concerns

- Severity: low. Concern: a trace can briefly outlive nothing — that is, its atoms are
  pruned in the same cycle a moment before the parent's own pass, leaving it hollow for up
  to one 60s cycle. Mitigation: measured at 4 rows / 8.6s, self-healing; the ordering in
  `GRAMMAR_RETENTION_TABLES` puts `grammar_traces` last precisely to minimize it.
- Severity: low. Concern: three days is short for forensic work on old traces. Mitigation:
  `GRAMMAR_TRACES_RETENTION_DAYS` is a plain env knob; raising it costs ~70 MB/day.
- Severity: low. Concern: the delete reclaims space *inside* the table, not to the OS, so
  the 219 MB figure will not shrink further without `VACUUM FULL`. Mitigation: not worth
  it — the volume has 619 GB free, and the table is now bounded rather than growing.
- Severity: informational. Concern: `orion/memory/crystallization/sources.py:27-38`, the
  only non-Atlas reader of this table, queries
  `WHERE trace_id = $1 OR event_id = $1` — but `grammar_traces` has no `event_id` column, so
  every call throws into a bare `except Exception: pass`, and its fallback table
  `substrate_grammar_events` does not exist. Dead since written, unrelated to this patch,
  not fixed here. Worth its own issue.

## PR link

<link>
