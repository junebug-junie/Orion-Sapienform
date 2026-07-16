# PR report — verdict-aware selection for rung-3 attention broadcast

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1078
Branch: `fix/attention-verdict-aware-selection`
Status: **DONE**

## Summary

- Implemented the "verdict-aware selection" fix explicitly deferred as its
  own scoping decision by PR #1061 (see
  `docs/notes/2026-07-14-attention-salience-decay-bypass-investigation.md`'s
  Non-goals section): a loop a human explicitly Resolved or Dismissed via
  the Hub could still win rung-3's continuous workspace competition
  (`orion/substrate/attention_broadcast.py::build_substrate_attention_frame`,
  every 30s) forever, because `build_open_loops()` never consulted
  `attention_loop_outcome` verdicts at all.
- New module `orion/substrate/attention/verdicts.py`:
  `load_terminal_verdict_loop_ids(loop_ids)` -- a bounded (never
  whole-table), read-only, fail-open direct-SQLAlchemy lookup against
  `attention_loop_outcome`, mirroring the exact connection/query pattern
  already shipped twice for the same table
  (`services/orion-hub/scripts/attention_loops_store.py`,
  `services/orion-thought/app/store.py::load_recent_loop_outcomes`).
- `orion/substrate/attention/scoring.py::build_open_loops()` gained an
  optional `verdict_lookup: Callable[[list[str]], set[str]] | None = None`
  parameter. Excluded loops are genuinely absent from `open_loops` -- not
  down-weighted -- so they cannot become `selected_action`. Default `None`
  preserves prior behavior byte-for-byte for the chat-scoped caller
  (`attention_frame.py`), which does not pass it.
- `orion/substrate/attention_broadcast.py::build_substrate_attention_frame`
  wires in `verdict_lookup=load_terminal_verdict_loop_ids` -- scoped to only
  this path, matching the original incident and this fix's stated scope.
- 12 new regression tests
  (`orion/substrate/tests/test_attention_verdict_exclusion.py`) plus one
  end-to-end live-data confirmation against the real
  `open-loop-9d84d08cddf5` verdict rows from the original investigation.

## Outcome moved

The rung-3 workspace-competition path that let a closed loop dominate
indefinitely (the live incident PR #1061 investigated and partially fixed)
now has a second, independent gate: even if a loop's salience recovers for
any reason (recency reset, a fresh evidence write, a future scoring change),
a human's explicit Resolve/Dismiss verdict permanently removes it from
competing, not just down-weights it.

## Current architecture

`build_open_loops()` (`orion/substrate/attention/scoring.py`) is the single
shared loop-construction function used by both the chat-scoped, per-turn
`attention_frame.py` and the substrate-broadcast, continuous-tick
`attention_broadcast.py::build_substrate_attention_frame`. Neither caller
had any visibility into `attention_loop_outcome` (the table
`services/orion-hub/scripts/attention_loops_store.py::persist_loop_outcome`
writes when a human clicks Resolve/Dismiss in the Hub) -- that table was
previously only read by `orion-hub` itself (surfacing pending cards) and by
`orion-thought`'s reverie narration layer (PR #1055, prompt-level
truthfulness only, no effect on selection).
`services/orion-substrate-runtime/app/worker.py::_attention_broadcast_tick`
runs `build_substrate_attention_frame` every
`ORION_ATTENTION_BROADCAST_INTERVAL_SEC` (default 30.0s, confirmed live in
this environment's `.env`), each tick building `open_loops` capped at
`max_open=5` from a `max_open*3`-wide deduped signal pool passed in as
headroom -- headroom that existed already but, before this patch, had no
consumer that could actually use it for backfill.

## Architecture touched

`orion/substrate/attention/scoring.py` (shared, imported by both chat-scoped
and substrate-broadcast callers), `orion/substrate/attention_broadcast.py`
(substrate-broadcast caller only), one new pure module
`orion/substrate/attention/verdicts.py`. No schema/bus/API contract changes,
no new env keys, no Docker/dependency changes (`sqlalchemy` is already a
dependency of `orion-substrate-runtime`).

## Files changed

- `orion/substrate/attention/verdicts.py` (new): read-only, bounded,
  fail-open `attention_loop_outcome` lookup. `TERMINAL_VERDICTS =
  {"resolved", "dismissed"}` -- deliberately excludes the table's third
  valid verdict, `decayed_unattended`, which is implicit non-engagement
  (an automatic timeout label), not an explicit human closure, and stays
  eligible to compete.
- `orion/substrate/attention/scoring.py`: `build_open_loops()` gained
  `verdict_lookup`. Refactored the candidate pipeline so `loop_id` is
  computed exactly once per deduped signal (previously two independent call
  sites computed the identical `stable_id(...)` expression -- a latent
  drift risk flagged by code review) and so verdict exclusion is applied to
  the full deduped signal pool *before* the `max_open` truncation, not
  after (also flagged by code review -- see below).
- `orion/substrate/attention_broadcast.py`: `build_substrate_attention_frame`
  passes `verdict_lookup=load_terminal_verdict_loop_ids` to `build_open_loops`,
  with an inline comment documenting why `attention_frame.py`'s chat-scoped
  call site is deliberately left unwired.
- `orion/substrate/tests/test_attention_verdict_exclusion.py` (new): 12
  tests -- see Tests run below.

## Schema / bus / API changes

None. `attention_loop_outcome` is an existing table (migration:
`services/orion-sql-db/manual_migration_attention_loop_outcome.sql`,
already applied and in live use by `orion-hub`); this patch only adds a
third, bounded, read-only reader of it. No new bus channel, no new schema
registration, no payload-shape change to any existing event.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: not applicable -- `verdicts.py` reuses the
  existing `POSTGRES_URI` env key, already present in
  `services/orion-substrate-runtime/.env_example` and `.env` (used by
  `worker.py::_get_sql_engine` for turn-referent persistence).
- local `.env` synced: not applicable, no template change.
- skipped keys requiring operator action: none.

## Tests run

```text
.venv/bin/python -m pytest orion/substrate/tests/ -q
=> 218 passed (12 new)

.venv/bin/python -m pytest orion/substrate/tests/test_attention_verdict_exclusion.py -q
=> 12 passed

.venv/bin/python -m pytest services/orion-substrate-runtime/tests/test_worker_attention_broadcast_tick.py -q
=> 6 passed
```

New test coverage:
- a loop with no verdict competes normally
- a `resolved` verdict excludes the loop even at max salience
- a `dismissed` verdict excludes the loop
- exclusion frees the slot for the next-best candidate instead of just
  shrinking the field (regression test for the code-review finding below)
- `verdict_lookup` is called exactly once per tick, bounded to that tick's
  candidate loop_ids only
- `verdict_lookup=None` (the default) preserves prior behavior exactly
- a raising `verdict_lookup` callable doesn't crash `build_open_loops`
- `load_terminal_verdict_loop_ids`: empty-input short circuit,
  verdict-filtering logic (mocked engine), fail-open on a mocked
  connection failure
- end-to-end through `build_substrate_attention_frame`: exclusion works
  with real node/signal inputs; a DB failure during the lookup doesn't
  crash frame-building and the loop stays eligible (fail-open, verified at
  the top-level entry point, not just the lookup function in isolation)

## Evals run

None applicable -- no eval harness exists for
`orion/substrate/attention_broadcast.py` or `scoring.py` (same gap noted in
PR #1061's own report). The live Postgres query below is the closest thing
to an eval for this specific fix, run by hand.

## Docker/build/smoke checks

Not run as a container build -- pure Python logic change, no new
dependency (`sqlalchemy==2.0.36` already in
`services/orion-substrate-runtime/requirements.txt`), no compose/Dockerfile
change. Instead ran a direct, read-only smoke against the live database
from the worktree (no restart needed for this, since it calls the new
function directly against real data rather than through the running
container):

```text
POSTGRES_URI="postgresql://postgres:postgres@localhost:55432/conjourney" \
  .venv/bin/python -c "
from orion.substrate.attention.verdicts import load_terminal_verdict_loop_ids
print(load_terminal_verdict_loop_ids(['open-loop-9d84d08cddf5', 'open-loop-does-not-exist']))
"
=> {'open-loop-9d84d08cddf5'}
```

This confirms the query correctly identifies the exact loop from the
original PR #1061 investigation (verdicted `resolved` 2026-07-08 20:32,
`dismissed` 2026-07-10 21:49, confirmed live via `psql`) as terminal-
verdicted, against the real `attention_loop_outcome` table.

**Full end-to-end live verification (through the running container) was
NOT performed** -- see "Live-data findings" below for why, and what would
be needed.

## Review findings fixed

Code-review skill (subagent, high effort) run against the full diff before
this report. Two material findings, both fixed and re-verified:

- **Finding**: exclusion was applied to the signal pool *after* it was
  already truncated to `max_open` (`candidates = merge_signals(signals,
  limit=max_open)` ran before `verdict_lookup`). `attention_broadcast.py`
  deliberately passes a `max_open*3`-wide pool as headroom specifically so
  an excluded loop's slot can be backfilled by the next-best candidate, but
  that headroom was discarded by `build_open_loops`'s own internal
  truncation before exclusion ever ran -- so excluding a loop just shrank
  the competing field by one instead of freeing a slot.
  - **Fix**: restructured to dedupe the full signal pool first
    (`merge_signals(signals, limit=len(signals))` -- a no-op cap), apply
    `verdict_lookup` to the full deduped pool, and *then* truncate to
    `max_open`.
  - **Evidence**: new
    `test_excluded_loop_frees_its_slot_for_the_next_best_candidate` (3
    signals, `max_open=2`, excludes the top-ranked signal, asserts the
    3rd-ranked signal -- previously truncated away before it could even be
    considered -- now appears).
- **Finding**: `loop_id` was computed twice from the same expression
  (`stable_id("open-loop", compact(s.target_text, 120).lower())`) at two
  independent call sites -- correct today (both pure/deterministic), but a
  latent drift risk: if either call site were edited alone in a future
  patch, the exclusion check would silently stop matching with no
  structural guarantee of a test catching it.
  - **Fix**: `loop_id` is now computed exactly once per deduped signal
    (`id_signal_pairs = [(stable_id(...), s) for s in deduped]`) and
    threaded through the rest of the function as the single source of
    truth.
  - **Evidence**: full test suite re-run after the refactor, all 218 pass
    (12 new + 206 pre-existing), confirming byte-identical behavior for the
    `verdict_lookup=None` case.

Not fixed, accepted as documented (per reviewer, non-blocking):

- `attention_loop_outcome` has no index on `loop_id` itself (only
  `created_at` and `(verdict, created_at)`), so `WHERE loop_id IN (...)` is
  a sequential scan -- identical to the pre-existing pattern in
  `services/orion-thought/app/store.py::load_recent_loop_outcomes`. The
  table is sparse (human Resolve/Dismiss clicks only), so this is cheap
  today; worth a follow-up index if the table grows materially.
- A second, independent SQLAlchemy engine/pool in `verdicts.py` alongside
  `worker.py::_get_sql_engine`'s own cached engine for the same Postgres
  instance -- reviewer confirmed this matches (and is more disciplined
  than) existing precedent in `orion/substrate/policy_profiles.py` and
  `review_telemetry.py`, which `create_engine()` fresh on every call with
  no caching at all. Given the 30s tick cadence, not a real exhaustion
  risk.
- The new test suite doesn't exercise the multi-tick dwell/coalition-decay
  transition specifically (only single-tick scenarios) -- reviewer traced
  the decay code path by hand and confirmed no special-case bug exists (an
  excluded loop just triggers the same decay path any naturally-losing loop
  already goes through), so this is a minor coverage gap, not a defect.

## Live-data findings

Confirmed live in this environment before writing code:

```sql
SELECT loop_id, verdict, created_at FROM attention_loop_outcome
WHERE loop_id LIKE '%9d84d08cddf5%' ORDER BY created_at;

        loop_id         |  verdict  |          created_at
------------------------+-----------+-------------------------------
 open-loop-9d84d08cddf5 | resolved  | 2026-07-08 20:32:35.513266+00
 open-loop-9d84d08cddf5 | dismissed | 2026-07-10 21:49:16.812265+00
```

Checked whether `open-loop-9d84d08cddf5` still appears in the live
substrate-broadcast projection (`substrate_attention_broadcast_projection`,
a single-row-upserted table other organs query) as of this session
(2026-07-16 02:37 UTC): it does not, and in fact **no loop is currently
competing at all** --
`docker logs orion-athena-substrate-runtime` shows
`substrate_attention_broadcast_completed selected=none loop=None
open_loops=0` on the most recent tick, and `ORION_ATTENTION_BROADCAST_ENABLED=true`
with the container actively running confirms the tick loop is live, not
just configured.

This is consistent with PR #1061's fix (recency + dwell-scoping, already
deployed since 2026-07-14) having already worked as intended: the
`substrate.transport` node's `dynamic_pressure` has apparently decayed
below `ORION_ATTENTION_BROADCAST_MIN_SALIENCE` (0.2) over the intervening
~8 days, so it no longer generates a competing signal at all, independent
of this patch's verdict exclusion. **This means a full restart-based,
end-to-end live verification of this specific fix (watching a verdicted
loop get excluded from a live tick) is not currently possible in this
environment** -- there is no verdicted loop presently in play to exclude.
I did not restart `orion-athena-substrate-runtime`, both because it would
not currently demonstrate anything (no candidate loop exists to test
against) and per this task's own instruction that a restart-based
verification is a bonus, not a requirement, when unit tests already prove
the exclusion logic. The direct query smoke test above (against real data,
no restart) is the live verification actually performed.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-substrate-runtime/.env \
  -f services/orion-substrate-runtime/docker-compose.yml up -d --build
```

Required for this fix to take effect on the live `orion-athena-substrate-runtime`
container. **Not run this session** (see Live-data findings above for why).

## Risks / concerns

- Severity: low
- Concern: `attention_loop_outcome` has no index on `loop_id`, so the new
  query is a sequential scan (bounded by table size, which is small and
  human-verdict-only). Matches existing precedent elsewhere; noted as a
  follow-up, not a blocker.
- Severity: low
- Concern: because no loop is currently competing live in this environment
  (see above), this specific fix has only been verified at the code level
  (12 new tests covering the exclusion logic, the DB lookup, and fail-open
  behavior end-to-end through `build_substrate_attention_frame`) plus a
  direct real-data query smoke test -- not through an actual running-tick,
  post-restart observation. Low risk given the thin, well-isolated,
  fail-open nature of the change, but worth a follow-up live check once the
  restart above happens and/or once a new loop accumulates a verdict while
  still salient enough to compete.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1078
