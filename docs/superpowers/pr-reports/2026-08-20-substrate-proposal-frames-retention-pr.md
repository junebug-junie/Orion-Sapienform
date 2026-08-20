# substrate_proposal_frames retention + autovacuum tuning on the high-churn tables

## Summary

- `substrate_proposal_frames` had **no retention at all**. Live 2026-08-20: 474,230 rows /
  1,758 MB, oldest 2026-07-23, growing ~27k rows and ~105 MB a day with nothing to stop it.
- It cannot use the existing grammar cursor floor, so this adds `_substrate_chain_floor()`,
  which floors retention on the substrate pipeline's **pending markers** rather than on the
  clock.
- Generalises `_apply_bounded_table_retention()` with an optional `floor_resolver` callable.
  Every existing call site and its tests are unchanged.
- Adds per-table autovacuum settings on the eight high-churn tables. The cluster default
  `autovacuum_vacuum_scale_factor = 0.2` meant `grammar_events` had to reach ~1.21M dead
  tuples before autovacuum would look at it.
- Fixes a stale README paragraph that survived the `grammar_traces` patch and still claimed
  `grammar_traces` has no retention.
- Records the ~13 GB of substrate frame tables that remain unbounded, as an explicit
  deliberately-not-done decision rather than an oversight.

## Outcome moved

A 1.76 GB table that grew without bound now tracks a window. And the instrument that was
supposed to reclaim the resulting dead tuples — autovacuum — was mistuned by an order of
magnitude for exactly these tables.

## Current architecture

The substrate pipeline is four stages, each writing a frame table and reaching back into
the previous one by `frame_id`:

```text
proposal -> policy decision -> execution dispatch -> feedback
```

Each hop has a pending marker and a partial index, added 2026-08-19 by
`services/orion-sql-db/manual_migration_substrate_pending_markers.sql`. That migration
replaced an unbounded anti-join per stage (106,052 blocks read + 465 MB spilled to temp,
per execution, every 2 seconds) and its header explicitly records that a *time*-bounded
version was attempted first and reverted, because the dispatch→feedback hop legitimately
ran p50 34.6 hours and max 11.3 days behind.

`services/orion-sql-writer/app/grammar_truth.py` already had bounded retention for five
tables, on a 60s timer, with a cursor floor for the grammar lane. `substrate_proposal_frames`
was not among them.

## Architecture touched

- `services/orion-sql-writer` — one new retention function, one new floor resolver, one new
  setting. No new service, no new contract, no bus/schema change.
- `services/orion-sql-db` — one new manual migration (autovacuum storage parameters only).
- Read-only dependency on `substrate_policy_decision_frames` and
  `substrate_execution_dispatch_frames` (their pending markers), both owned by other
  services. No writes to either.

## Files changed

- `services/orion-sql-writer/app/grammar_truth.py`: `_SUBSTRATE_CHAIN_PENDING`,
  `_substrate_chain_floor()`, `apply_substrate_proposal_frames_retention()`, the
  `floor_resolver` parameter, and registration in `_EXTRA_RETENTION_TABLES` /
  `GRAMMAR_RETENTION_TABLES`.
- `services/orion-sql-writer/app/settings.py`: `substrate_proposal_frames_retention_days`.
- `services/orion-sql-writer/app/grammar_retention_loop.py`: pass the window to the loop.
- `services/orion-sql-writer/.env_example`: `SUBSTRATE_PROPOSAL_FRAMES_RETENTION_DAYS=10`.
- `services/orion-sql-writer/README.md`: retention section, the stale known-gap paragraph,
  the autovacuum note, and the unbounded-substrate-family table.
- `services/orion-sql-writer/tests/test_grammar_retention_periodic.py`: new tests, plus two
  existing tests corrected (see below).
- `services/orion-sql-db/manual_migration_autovacuum_high_churn_tables.sql`: new.

## Why the floor asks the markers and not the clock

The grammar cursor floor deliberately does **not** ask "where is the oldest cursor", because
a cursor stops moving when its *source* goes quiet, which is indistinguishable from a stall.

A pending marker has no such ambiguity. It is set at insert and cleared in the same
transaction as the downstream write, so a pending row is unconsumed work by construction,
and a quiet source produces no pending rows at all rather than a stuck floor. Flooring
directly on the oldest pending row is therefore correct here and would have been wrong for
grammar. The two are not interchangeable, and a test pins that they are not swapped.

The residual failure mode is a **leaked marker**: a row stuck pending forever would pin
retention forever. That is why a binding floor logs at WARNING with how far back it reaches,
and why `remaining_debt` is measured against the retention window rather than the clamped
cutoff — "a stage is stuck" and "retention is caught up" cannot look the same in the logs.

Live 2026-08-20, all three stages were current: 1 / 96 / 15 pending rows, all under four
minutes old.

## The two-clocks trap, avoided on purpose

The floor probes `MIN(created_at)`, **not** `MIN(generated_at)`, even though the partial
indexes are on `generated_at`.

`generated_at` is when the stage produced the frame; `created_at` is when this row was
written. Measured live 2026-08-20:

```text
table                                 rows where created_at < generated_at   max skew
substrate_proposal_frames                                     0 / 474,280     0.167 s
substrate_policy_decision_frames                             24 / 474,280     0.006 s
substrate_execution_dispatch_frames                           0 / 474,165   724.745 s
```

The delete predicate keys on `created_at`. A floor read off one clock and compared against a
cutoff on the other deletes rows it meant to keep. The pending set is tiny, so the partial
index still does the narrowing and the heap fetch for `created_at` is negligible — measured
below.

## Schema / bus / API changes

- Added: none. No new table, column, channel, or schema.
- Removed / renamed: none.
- Behavior changed: `substrate_proposal_frames` rows older than 10 days are now deleted,
  subject to the pending-marker floor. `/grammar/truth` gains a
  `other_table_retention.substrate_proposal_frames` block and can now emit
  `substrate_proposal_frames_retention_debt_remaining` as a degraded reason.
- Compatibility: `_apply_bounded_table_retention()` gained an optional keyword-only
  parameter with a default; no existing call site changes.

## Env/config changes

- Added keys: `SUBSTRATE_PROPOSAL_FRAMES_RETENTION_DAYS` (default 10).
- Removed keys: none. Renamed keys: none.
- `.env_example` updated: yes, with the rationale and the ~105 MB/day cost of raising it.
- local `.env` synced: **by hand**, in both the primary checkout and this worktree.
  `scripts/sync_local_env_from_example.py` reads `.env_example` from the *primary* checkout,
  so a key added in a worktree is invisible to it — running the script would have been a
  silent no-op here. Verified `services/orion-sql-writer/.env` is still gitignored.
- Skipped keys requiring operator action: none.

## Tests run

```text
$ pytest services/orion-sql-writer/tests -q
11 failed, 406 passed, 3 skipped, 34 warnings in 11.70s
```

The 11 failures pre-exist on `main` (377 passed there) and are unrelated to retention:
`test_biometrics_summary_sql_shape`, `test_chat_history_response_identity_merge` (4),
`test_journal_entry_payload_boundary`, `test_notify_attention_ack` (3),
`test_notify_attention_escalate` (2). This branch adds no new failures.

New tests in `tests/test_grammar_retention_periodic.py`:

- `TestSubstrateProposalFramesIsBounded` — registered in the cycle; floored on the pipeline
  markers and NOT on the grammar cursor floor; uses the plain engine, not the grammar engine;
  the loop passes a real window; a real `Settings` field exists.
- `TestTheSubstrateChainFloor` — covers every stage; **every probe returns a proposal
  timestamp, not the pending row's own**; probes `created_at` not `generated_at`; **every
  probe fences the min-aggregate with `OFFSET 0`**; oldest-across-stages wins; caught-up
  imposes no floor; a naive timestamp is normalised; a failed probe reports unresolved and
  never "no floor"; an unresolved floor refuses to prune; a binding floor clamps the cutoff.
- `test_the_grammar_cursor_floor_also_normalises_a_naive_timestamp` — a **pre-existing** gap,
  found by mutation-testing the new floor and hitting the old one instead. Deleting that line
  left the entire suite green.

Every new test was mutation-tested. Eight mutations, each failing exactly the intended test:

```text
M1 drop floor_resolver from the call site                    -> 1 failed
M2 probe generated_at instead of created_at                  -> 1 failed
M3 failed probe reports resolved=True                        -> 1 failed
M4 drop tz normalisation in _substrate_chain_floor           -> 1 failed
M5 take the newest pending row instead of the oldest         -> 2 failed
M6 drop tz normalisation in _grammar_events_cursor_floor     -> 1 failed (0 before this patch)
M7 drop OFFSET 0 from all three probes                       -> 1 failed
M8 floor on the child's own timestamp instead of the parent  -> 1 failed
M9 drop one stage from the chain                             -> 1 failed
```

M4 is worth calling out: the first attempt appeared to prove the test was inert, because the
identical two lines exist in `_grammar_events_cursor_floor` earlier in the file and the
mutation hit that copy instead. Retargeting it to the new function failed correctly — and the
misfire is what turned up M6.

## Evals run

```text
$ pytest services/orion-sql-writer/evals/test_substrate_retention_integrity_eval.py -q
6 passed in 1.90s
```

New file: `services/orion-sql-writer/evals/test_substrate_retention_integrity_eval.py`. It is
read-only and runs against the live database, because the two most serious bugs on this patch
were both unreachable from a mock:

- `test_no_pending_stage_has_lost_its_parent_proposal` (x2 stages) — the property the whole
  floor exists to hold, checked against live rows. This is the check that would have caught
  the parent-before-child bug.
- `test_a_backlog_is_always_explained_by_either_the_floor_or_active_pruning` — a backlog is
  acceptable only when a stage is genuinely behind, or the service reports it is actively
  pruning. A backlog with neither explanation is what a silently stopped retention loop looks
  like. (The first draft of this asserted on the floor alone and failed on a perfectly healthy
  mid-drain database — corrected.)
- `test_each_floor_probe_stays_cheap_on_live_data` (x3 stages) — asserts on real
  `EXPLAIN (ANALYZE, BUFFERS)` output: under 100 ms and no sequential scan. Whether the probe
  uses the partial index is a *planner* decision made from live statistics, and it had already
  flipped once. No SQL-string assertion can see that.

Before this patch the service had one eval file. The gap is now closed for this path.

## Docker/build/smoke checks

```text
$ ./scripts/safe_docker_build.sh orion-sql-writer up -d --build
Container orion-athena-sql-writer  Recreated
Container orion-athena-sql-writer  Started
```

Live retention loop, `docker logs orion-athena-sql-writer`:

```text
grammar_retention_loop starting interval_sec=60 max_batches=3 max_elapsed_sec=20 \
  days={'grammar_events': 3, ..., 'substrate_proposal_frames': 10}
substrate_proposal_frames_retention_complete cutoff=2026-08-10T16:56:02.146874+00:00 \
  rows_pruned=3000 batches=3 elapsed_sec=1.61 remaining_debt=224585 \
  capped_batches=True capped_elapsed=False
```

`GET /grammar/truth` → `other_table_retention.substrate_proposal_frames`:

```text
configured_days      = 10
cutoff_at            = 2026-08-10T16:56:02.146874+00:00
rows_pruned_last_run = 3000
remaining_debt       = 224585
failure_reason       = None
cursor_floor_at      = 2026-08-20T16:51:59.001758+00:00   (current pending row)
cursor_floor_applied = False                              (correct: nothing is behind)
fk_delete_verified   = True
```

The backlog drains at 3,000 rows per 60s cycle against ~27k rows/day of arrival, so it
converges in roughly 75 minutes rather than in one spike — deliberate, given the ~1 GB of
TOAST churn it implies.

Autovacuum migration applied live (`docker exec -i orion-athena-sql-db psql ... < file`), all
eight tables plus the four TOAST relations verified in `pg_class.reloptions`. Effect within
minutes, before the milder second version was applied:

```text
table                      dead before   dead after   last autovacuum before
grammar_events                 572,758      119,959   ~3h stale
grammar_atoms                  125,744        1,559   —
substrate_proposal_frames       51,417        6,153   34h stale
```

Query plans, live `EXPLAIN (ANALYZE, BUFFERS)`:

```text
floor probe: policy->dispatch, unfenced   490 ms   102,343 buffers   474,708 rows discarded
floor probe: policy->dispatch, fenced     5.4 ms
floor probe: proposal->policy, fenced     0.18 ms
floor probe: dispatch->feedback, fenced   1.6 ms
retention delete probe (2000 rows)        4.97 ms  Incremental Sort over backward index scan
debt count                                45 ms    parallel index-only scan
```

## Review findings fixed

An adversarial review ran in a subagent against live data and returned 12 findings, 3 HIGH.
All three HIGHs were real; I verified each myself against the database before fixing.

- **HIGH — one floor probe was a ~490 ms near-full-table scan every 60 seconds, and the
  docstring asserted the opposite.** I wrote "three tiny index-only scans". Postgres rewrites
  a bare `SELECT MIN(created_at) ... WHERE dispatch_pending` into `ORDER BY created_at LIMIT 1`
  over the *full* `created_at` index with the marker as a filter; since pending rows are always
  the newest rows, the ascending scan discards the entire table first.
  - Fix: `OFFSET 0` optimisation fence on all three probes.
  - Evidence: reproduced myself — 490 ms / 102,343 buffers / "Rows Removed by Filter: 474,708"
    versus 0.37 ms / 28 buffers fenced. A 1,340x difference. Now pinned by both a unit test
    (`test_every_probe_fences_the_min_aggregate_with_offset_0`) and an eval that asserts on
    real EXPLAIN output. The nastiest part: this cost was *highest when the pipeline was caught
    up* and would have fallen as a backlog developed, so it would never have looked like a
    backlog problem. It would also eventually have hit the 30s `statement_timeout` and turned
    the fail-safe into a fail-stuck.

- **HIGH — the floor protected the wrong row for 2 of its 3 stages.** Two probes read the
  *pending row's own* `created_at`, but those rows live in downstream tables. A child is always
  written after its parent, so flooring at the child's timestamp leaves the parent below the
  floor and deletable — backwards from the safety being claimed. The commit message's "any row
  still owed work by any stage survives regardless of age" was simply not what the code did.
  - Fix: both downstream probes now `JOIN substrate_proposal_frames p ON p.frame_id =
    d.source_proposal_frame_id` and take `MIN(p.created_at)`.
  - Evidence: measured live over 3 days — dispatch rows are created a mean **123.2s** and max
    **920.2s** after their parent proposal, and **0 of 55,573** before it. Pinned by
    `test_every_probe_returns_a_proposal_timestamp_not_the_pending_rows_own` and by the eval's
    live orphan check. This is the second parent/child clock inversion in two days; noted as
    such in the code comment.

- **HIGH — a real reader needed 306,615 rows the patch would delete.**
  `scripts/analysis/measure_proposal_feedback_correlation.py::fetch_chain_completeness` joined
  **every** feedback frame ever written against the upstream tables with no `WHERE` clause.
  `substrate_feedback_frames` has no retention, so its headline would have flipped to
  `INCOMPLETE` permanently — for the entirely expected reason that pruned rows do not resolve.
  Not coverable by any marker: a feedback frame only exists once the chain has fully drained.
  - Fix: bounded that query to the caller's `--window-hours`, and it now reports
    `rows_outside_window` explicitly so a shrinking denominator cannot be mistaken for a clean
    result.

- **MEDIUM — the 7-day window tied exactly to a consumer's window.**
  `run_attention_bound_proposal_eval.py` uses `WINDOW_DAYS = 7` and
  `measure_proposal_feedback_correlation.py` defaults to 200h (8.33 days). Retention would have
  raced both at their oldest edge, and both degrade quietly to "insufficient data".
  - Fix: window is **10 days**, clearing the longer of the two by ~1.7 days.
  - Evidence: live, 10 days keeps 202,276 of 474,861 rows; 7 days would have kept 113,060.

- **MEDIUM — "17x more often, same total work, spread out" was wrong,** and made athena's known
  I/O ceiling worse rather than neutral. Heap work scales with dead tuples; **index** vacuum
  cost scales with index *size*. These tables carry 3442 / 2756 / 1184 MB of indexes, so 17x
  the passes is ~17x the index-scan I/O.
  - Fix: backed off from `0.01 + 10000` to `0.05 + 3000`, a measured **2.3x–4.0x** against live
    row counts, and said plainly in the file that this is a compromise and not a measured
    optimum. Cost limits still untouched.

- **MEDIUM — 84% of this table's bytes are TOAST, and the migration did not touch TOAST.**
  TOAST relations have their own `toast.autovacuum_*` settings and inherit nothing; all four
  substrate TOAST relations were on `reloptions = NULL`.
  - Fix: added `toast.autovacuum_*` to the four TOAST-bearing tables, and fixed the migration's
    own verification query, which checked only `pg_class.reloptions` for the main table and so
    would have reported the TOAST settings as missing when they had applied fine.
  - Evidence: verified via `reltoastrelid` join — all four now carry the settings.

- **MEDIUM — six specific numeric claims were wrong.** All corrected against live data:
  "keeps ~190k rows (~650 MB)" → 113,060 at 7d / 202,276 at 10d; "loses 60% of its rows" →
  76.2% at 7d; "idx_*_pending, 32 kB" → 32 kB / **9312 kB** / 48 kB; "these seven tables" → the
  file issues **eight** `ALTER TABLE`s; "roughly 17x" → per-table 2.3x–4.0x; "~13 GB of
  substrate history" → **8.3 GB** (the 13 GB wrongly counted `substrate_organ_emissions`, which
  the same table labels *bounded*).

- **MEDIUM — `reconcile_policy_pending` is a third writer of the marker,** not just
  insert-and-clear as the docstring claimed. It re-sets `policy_pending = true` on a 900s timer
  with no time bound. Harmless today; a trap the moment `substrate_policy_decision_frames` gets
  retention — pruning a decision frame makes its proposal look unprocessed, reconcile re-flags
  it, and the floor pins forever. The README explicitly invited that next step while leaving
  the trap unmentioned.
  - Fix: documented in both the code comment and the README's known-gap section.

- **MEDIUM — `_verify_delete_safe` gives false assurance on this table.** It reports
  `delete_is_child_safe; no incoming FK constraints found`, which is literally true, but that
  check was written for the grammar tables, which have no children. This table has three
  referencing tables storing `source_proposal_frame_id` as plain `text` with no FK — so "no
  incoming FK" means the database will not *tell* you when you orphan something.
  - Fix: caveat in the docstring and README; the eval's live orphan check is the real guard.

- **MEDIUM — retention reclaims no disk, and nothing said so.** `DELETE` returns space for
  reuse inside the relation, not to the OS.
  - Fix: stated plainly in README, `.env_example` and settings, with the heap/index/TOAST split.

- **LOW — one assertion was trivially true.**
  `assert not any("DELETE" in sql ...)` in `test_an_unresolved_floor_refuses_to_prune`: when the
  floor is unresolved the function returns before touching the connection, so `deleted` is
  always `[]`.
  - Fix: replaced with `state.cutoff_at is None` plus `deleted == []`, and dropped a redundant
    `monkeypatch` in the same test.

- **LOW — PEP 8 blank lines; missing eval.** Both fixed (the eval is described above).

Two review claims I checked and did **not** act on: `reconcile_policy_pending` gets *cheaper*
after retention, not more expensive (confirmed — fewer rows in the anti-join); and the
DESC-only `created_at` index serves the delete fine (confirmed, 4.97 ms).

## Restart required

Already applied on athena. To reproduce elsewhere, from a worktree:

```bash
./scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

The autovacuum migration is applied live and is idempotent:

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_autovacuum_high_churn_tables.sql
```

## Risks / concerns

- Severity: medium. Concern: the autovacuum change is a judgement call, not a measurement.
  ~3.5x more vacuum passes means ~3.5x more index-scan I/O on a host already I/O-stalled ~22%
  of wall time. Mitigation: the migration says so in its own header, and the `RESET` rollback
  is one statement per table. Worth checking `pg_stat_user_tables.n_dead_tup` and host I/O in a
  day and adjusting in either direction.
- Severity: medium. Concern: a leaked pending marker pins retention forever, and the
  `reconcile_policy_pending` interaction above turns that from theoretical into likely the
  moment anyone adds retention to `substrate_policy_decision_frames`. Mitigation: a binding
  floor logs at WARNING with how far back it reaches; `remaining_debt` is measured against the
  retention window, not the clamped cutoff, so "stuck" and "caught up" cannot look the same;
  the eval fails if a backlog has neither explanation.
- Severity: low. Concern: 10 days of proposal history may be short for deep forensic work.
  Mitigation: plain env knob, ~105 MB/day.
- Severity: low. Concern: no disk is returned to the OS; the table stays ~1.76 GB. Mitigation:
  not worth `VACUUM FULL` — 619 GB free on the volume — and the point was bounding growth.
- Severity: informational. Concern: the substrate frame family is ~8.3 GB of still-unbounded
  tables (`substrate_execution_dispatch_frames` 2,126 MB, `substrate_field_state` 1,745 MB,
  `substrate_feedback_frames` 1,641 MB, `substrate_policy_decision_frames` 1,485 MB,
  `substrate_attention_frames` 1,055 MB, `substrate_perception_embedding_baseline` 477 MB).
  Extending retention is mostly mechanical now, but deliberately **not** done here — that is
  the substrate's own record of what it proposed, decided and did, and pruning it should be an
  explicit decision rather than a side effect. Documented as such in the README.
- Severity: informational. Concern: `scripts/check_env_template_parity.py`,
  `check_schema_registry.py` and `check_bus_channels.py` from CLAUDE.md §17 do not exist in
  this repo (confirmed 2026-07-12 in the Makefile's own note). Env parity was verified by hand.

## PR link

<link>
