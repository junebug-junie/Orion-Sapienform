# Run grammar retention on a timer, cut the window to 3 days, and make the floor lane-aware

## Summary

- Retention already existed and each run was correct. Its only trigger was **process start**, so it deleted ~365,000 rows per restart against 1,117,440 rows/day of arrival. It could not converge, and no cap tuning reaches it.
- Moved it to a bounded 60-second loop. Live: `pruned=12000` per cycle, debt declining monotonically, zero errors.
- Window 15 → 3 days (Juniper's call). 15 days was not a fix either — at the measured 2.42 GB/day it settles at ~36 GB, exactly where the tables already were.
- `grammar_events` is a **cursor-driven consumption queue**, not an archive. Added a floor so retention cannot delete rows a reducer still owes — and, after review, made that floor ask the *right* question.
- Made the debt count non-fatal and measured against the retention window, so it can no longer report calm while retention is pinned.

## Outcome moved

Before: retention ran once per process start and stopped.

```
arrival        1,117,440 rows/day across four tables (2.42 GB/day)
deletion         365,000 rows per process start, then exactly 0
standing debt  5,352,878 rows past the cutoff, and growing
```

After, observed live over three consecutive cycles:

```
grammar_retention_loop starting interval_sec=60 max_batches=3 max_elapsed_sec=20 days={...: 3}
grammar_retention_cycle pruned=12000 remaining_debt={'grammar_events': 6956050, ...} floored=[] skipped=[]
grammar_retention_cycle pruned=12000 remaining_debt={'grammar_events': 6953555, ...} floored=[] skipped=[]
grammar_retention_cycle pruned=12000 remaining_debt={'grammar_events': 6951046, ...} floored=[] skipped=[]
```

Host I/O stall during the drain: `full avg60 = 2.33`, versus ~21% when the startup pass ran at full speed.

**Honest drain estimate, from observed rate rather than my earlier arithmetic.** `grammar_events` nets ~2,502 rows/cycle (3,000 pruned minus ~500 arriving) against 6.95M debt — roughly **2.5-3 days** to converge, with `grammar_events` the long pole. The other three finish sooner, and their cycles go near-instant once drained, which shortens the period and speeds events up. My in-code comment says ~1.65 days; that was derived, this is measured, and the measured number is the one to trust. Raising `GRAMMAR_RETENTION_PERIODIC_MAX_BATCHES` shortens it — there is I/O headroom — but it is not urgent, because **draining faster does not free disk any sooner** (see Risks).

## Current architecture

`services/orion-sql-writer/app/main.py` called `apply_grammar_events_retention()` and three siblings once, inline, during lifespan startup. Each did a batched delete bounded by `max_batches_per_startup` (100) and `max_elapsed_sec` (120). Each logged `remaining_debt` accurately. Nothing ever read it, and nothing ever ran again.

## Architecture touched

One service. No bus channel, schema, or API contract changed. A new background task, a new lane table, and four config defaults.

## Files changed

- `services/orion-sql-writer/app/grammar_retention_loop.py` (NEW): the 60s loop, `asyncio.to_thread` + cooperative stop.
- `services/orion-sql-writer/app/grammar_truth.py`: `GRAMMAR_LANES`, lane-aware `_grammar_events_cursor_floor()`, `run_one_retention_cycle()`, non-fatal debt count, floor fields surfaced on `/health`.
- `services/orion-sql-writer/app/main.py`: creates and cancels `retention_task`.
- `services/orion-sql-writer/app/settings.py`, `.env_example`: 3-day window, three periodic knobs.
- `services/orion-sql-writer/tests/test_grammar_retention_periodic.py` (NEW, 24 tests), `tests/test_grammar_truth.py` (updated for the new contract).

## The floor, and the mistake in my first version

`grammar_events` is walked forward by five reducer lanes (`services/orion-substrate-runtime/app/store.py:352-390`), each with its own `source_service`/`trace_id` filter and its own cursor. Deleting ahead of a lagging cursor skips events silently — CLAUDE.md's "reducers alive but cursors stale".

**My first floor clamped the cutoff to `MIN(last_event_created_at)` across all cursors. That is the wrong question, and review caught it before deploy.**

A cursor advances only when its lane consumes a *matching* row. A still cursor therefore means "this lane's source emitted nothing lately", which is indistinguishable from "this reducer is stalled". `chat_grammar_consumer` tracks Juniper talking to Orion — **178 rows/day**, and measured silent for **1 day 19.6 hours** while the system was healthy and ingesting ~500k grammar_events/day. Against a 3-day window that is a 1.65× margin on *ordinary behaviour*: a quiet weekend would have pinned retention and stopped pruning entirely.

I had justified the window with "all five cursors are within 13 seconds, a ~20,000× margin". That was a point-in-time sample presented as a distribution, and it was the wrong statistic for the job.

The floor now asks the right question: **does this lane have unconsumed rows below the cutoff?** A silent lane has none — everything older than its cursor is consumed and nothing newer exists — so it imposes no floor. A stalled lane has real rows above its cursor and below the cutoff, and those survive.

## A second error, found in live data rather than code

I hand-wrote the lane table and guessed the execution lane's filter as `("orion-cortex-exec",)`. The real `EXECUTION_SOURCE_SERVICES` is a three-service frozenset also covering `orion-harness-governor` and `orion-hub` — and live `grammar_events` confirms `orion-harness-governor` writes `cortex.exec:` traces at 528 rows/day.

That would have **under-protected the exact lane the floor exists to protect**: the probe would have missed unconsumed governor/hub rows and reported the lane clear. My drift test only spot-checked two lanes, so it did not catch it. The test now asserts all five including sources, and fails on precisely that mutation.

The lane table stays duplicated rather than imported: `orion/substrate/*/constants.py` executes `orion/substrate/__init__.py`, which drags the graph-DB store and materializer into this thin writer — the mistake that crash-looped two services on 2026-08-19.

## Schema / bus / API changes

None. `/health`'s retention block gains `cursor_floor_at`, `cursor_floor_applied`, `debt_count_failed_reason` (additive).

## Env/config changes

- Changed: `GRAMMAR_{EVENTS,EDGES,ATOMS}_RETENTION_DAYS`, `SUBSTRATE_ORGAN_EMISSIONS_RETENTION_DAYS` 15 → 3.
- Added: `GRAMMAR_RETENTION_INTERVAL_SEC=60`, `GRAMMAR_RETENTION_PERIODIC_MAX_BATCHES=3`, `GRAMMAR_RETENTION_PERIODIC_MAX_ELAPSED_SEC=20`.
- `.env_example` updated: yes.
- Local `.env` synced: **yes, by hand.** `scripts/sync_local_env_from_example.py` reads `.env_example` from the primary checkout, so keys added in a worktree are invisible to it. Verified `services/orion-sql-writer/.env` carries all seven values and is still gitignored.

## Tests run

```text
pytest services/orion-sql-writer/tests -q
  11 failed, 377 passed, 3 skipped

The 11 failures are PRE-EXISTING and fail identically on unmodified main
(notify_attention_*, journal_entry_payload_boundary). Verified by diffing the
FAILED list against a baseline run in the primary checkout: zero new failures.
```

Mutation tests, all against the real file, not a fixture:

```text
floor reverted to MIN(cursor position)  -> 6 FAILED, incl. the silent-lane regression by name
debt measured against clamped cutoff    -> 1 FAILED
unresolved floor falls through          -> 2 FAILED (both SKIP tests)
execution lane drifted to 1 service     -> 1 FAILED (the drift test)
```

## Evals run

None. `orion-sql-writer` has no eval harness, and this changes no cognitive behaviour — it changes when a delete runs and how far back it reaches. The quality signal is the live cycle log and debt series above, reported rather than asserted.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-sql-writer up -d --build
  Container orion-athena-sql-writer Started

live: cutoff=2026-08-17T01:03 (exactly 3 days back -- floor did not pin, as expected
      with all lanes caught up)
live: grammar_retention_loop starting interval_sec=60 max_batches=3 ...
live: 3x grammar_retention_cycle pruned=12000 floored=[] skipped=[]
live: retention_failed / Traceback count = 0
```

## Review findings fixed

Review ran in a subagent before deploy and found ten issues. All acted on.

- **HIGH — floor pinned on lane silence, not stall.** Fix: lane-aware floor (above). Evidence: `test_a_silent_lane_with_nothing_unconsumed_imposes_no_floor`, and reverting the code fails 6 tests.
- **HIGH — a pinned floor reported healthy.** Debt was counted against the *clamped* cutoff, so `/health` showed `remaining_debt: 0` while millions sat above the floor. Fix: count against the retention window. Evidence: `test_debt_is_measured_against_the_window_not_the_clamped_cutoff`.
- **HIGH — three new state fields were never surfaced.** `cursor_floor_at`/`cursor_floor_applied`/`debt_count_failed_reason` existed but `_retention_block()` did not emit them, so "pinned" and "healthy, nothing to do" looked identical. Fix: added, with a test asserting they are surfaced.
- **HIGH — `reset_grammar_cursor(mode="earliest")` writes 1970.** Under the position-based floor, one HTTP call would have disabled retention permanently and silently. The lane-aware floor removes this by construction: an epoch cursor floors only if unconsumed rows actually exist, which is correct behaviour for a deliberate replay.
- **MEDIUM — SIGKILL on every restart.** `asyncio.to_thread` is not cancellable once started, so shutdown reported clean while the worker kept deleting; Python joins executor threads at exit, blocking past Docker's 10s grace. Fix: `threading.Event` checked between tables, set on cancellation, with a bounded settle.
- **MEDIUM — sizing arithmetic ~1.8× optimistic.** I treated `interval` as a fixed period; the loop sleeps *then* works. Corrected in the comment, and superseded by the measured number above. Review also confirmed there is **no** cycle-overlap bug — cycles are strictly sequential.
- **MEDIUM — the cycle summary went silent exactly when pinned.** It logged only when `pruned or debts` was non-zero. Fix: unconditional line every cycle plus a separate warning when the floor binds.
- **LOW — `MIN()` skips NULL cursor rows** and the docstring's stated contract was wrong. Fix: per-lane iteration handles NULL explicitly, documented as "tail-seeded, wants no history".
- **LOW — untested branches.** The unresolved-floor skip, the unexpected-type branch, and the boundary row had no coverage. Fix: all covered; 12 tests → 24.
- **LOW — stale `known_risks` on `/health`** still promised "a future background pruner". Fix: rewritten, and it now names the real remaining gap (`grammar_traces` has no retention).

Review also cleared, with reasoning I verified: the strict `<` delete boundary, cursor-store completeness, engine thread-safety, `retention_task` cancellation, and the startup path.

## Restart required

Already deployed and verified. No further restart.

## Risks / concerns

- **Severity: medium. Draining does not free disk.** `DELETE` returns space for reuse inside the table, not to the OS. The tables will stay ~36 GB until they are rewritten (`VACUUM FULL`, exclusive lock, or `pg_repack`, which is not in the image). Juniper deferred this until retention is converging — it now is, so this is the next decision. Until then the win is "growth stops", not "disk returns".
- **Severity: medium. `grammar_traces` has no retention.** The Grammar Atlas (`orion/grammar/query.py`, `services/orion-hub/scripts/grammar_atlas_routes.py`) lists traces and then loads their atoms/edges with no time bound. Once the drain passes 3 days, the Atlas will list traces whose events are gone and render empty graphs — CLAUDE.md's "UI panels with no real backing artifact". Found by review; **not fixed here** and it needs its own patch. Parked.
- **Severity: low. Memory consolidation reads grammar by `trace_id` with no time bound** (`orion/memory/consolidation_grammar.py:41-48`). Safe today — live check showed one open window, one day old — but a consolidation backlog older than 3 days would silently get empty evidence lists. Worth a bound.
- **Severity: low. The startup pass blocks the event loop** for minutes (observed ~260s across four tables) before the periodic loop starts. Pre-existing, not introduced here, but more visible now that the periodic loop makes the startup pass largely redundant. Candidate for removal in a follow-up.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1759
