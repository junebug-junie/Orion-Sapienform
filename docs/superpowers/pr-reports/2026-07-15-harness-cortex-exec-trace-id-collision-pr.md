# PR report — harness governor / cortex-exec trace_id collision on the grammar ledger

PR: (create at https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/trace-id-collision)
Branch: `fix/trace-id-collision`
Status: **DONE**

## Summary

- `HarnessGrammarCollector.trace_id` (`orion/harness/grammar_emit.py`) computed
  the bare, unlaned `cortex_exec_trace_id(node, correlation_id)` — the exact
  same string a real `CortexExecGrammarCollector` root call (`trace_lane=None`)
  also produces for a call sharing the same `correlation_id`.
  `orion/grammar/ledger.py`'s `_upsert_trace()` upserts `grammar_traces` keyed
  on `trace_id` alone (no `source_service` in the conflict key), so whichever
  producer's `trace_started`/`trace_ended` landed last in the ledger silently
  overwrote the other's `started_at`/`ended_at`.
- Confirmed live via Postgres before touching code, correlation_id
  `99487e95-709c-4340-8bf5-c4c9840a247b` (2026-07-15): `orion-harness-governor`
  wrote a real ~30s trace (9 real tool steps) at `15:58:20`–`15:58:50`;
  `orion-cortex-exec` separately wrote an unrelated ~0.7s trace at
  `15:59:27`–`15:59:28` to the *same* `trace_id`; `grammar_traces` ended up
  showing only the 0.7s blip, with the real 30s harness run completely
  erased from the ledger's summary table.
- Fix: `HarnessGrammarCollector.trace_id` now passes `lane="harness_motor"` to
  the existing `cortex_exec_trace_id(..., lane=...)` mechanism
  (`orion/substrate/execution_loop/ids.py`) — the same mechanism
  `CortexExecGrammarCollector` already uses to isolate its own auxiliary
  sub-call lanes (`stance_react`, `harness_finalize_reflect`,
  `orion_voice_finalize`). No change to `orion/grammar/ledger.py` or the
  `grammar_traces` schema/conflict key.
- Code review (high effort) found a real second-order regression the fix
  introduced: `orion/substrate/execution_loop/grammar_extract.py`'s
  `extract_execution_state_from_events()` parses `correlation_id` out of
  `trace_id` via `parse_execution_trace_id()`, which only splits on the
  first two colons — so for any lane-suffixed `trace_id` the parsed
  "correlation_id" actually contains `"{correlation_id}:{lane}"`. Once
  `HarnessGrammarCollector.trace_id` always carries `:harness_motor`, every
  harness-governor execution run's `ExecutionRunStateV1.correlation_id`
  (exposed as-is on `orion-substrate-runtime`'s
  `GET /projections/execution_trajectory` debug endpoint) would have been
  silently suffixed. Fixed by preferring the event's own clean
  `correlation_id` field over the parsed value.
- Review also flagged (accepted, not fixed — see Risks) that this closes one
  instance of the collision class via convention, not a structural fix to
  the ledger's conflict key, and noted the fix incidentally also prevents a
  second, previously-unverified collision instance in
  `orion/substrate/execution_loop/reducer.py`'s `runs` dict (which also keys
  by `trace_id` and would have merged the two unrelated runs' step
  counts/status together).

## Outcome moved

`grammar_traces` no longer loses a real harness-governor turn's
`started_at`/`ended_at` to an unrelated, unlaned cortex-exec root call
sharing the same `correlation_id`. `ExecutionTrajectoryProjectionV1.runs`
(the live execution-loop reducer's projection) also no longer risks merging
two unrelated runs' step counts/status/reasoning flags into one entry for
the same reason.

## Current architecture

`cortex_exec_trace_id(node_name, correlation_id, *, lane=None)`
(`orion/substrate/execution_loop/ids.py`) builds `cortex.exec:{node}:{corr}`,
optionally suffixed `:{lane}`. `CortexExecGrammarCollector`
(`services/orion-cortex-exec/app/grammar_emit.py`) already used `lane` for
three known auxiliary verbs via `trace_lane_for_verb()`
(`CORTEX_EXEC_ISOLATED_TRACE_LANES`), leaving its own root/default call
(`trace_lane=None`) unlaned. `HarnessGrammarCollector`
(`orion/harness/grammar_emit.py`) — a structurally independent sibling class,
no shared base — also called `cortex_exec_trace_id` unlaned, for what the
`ids.py` docstring already calls "the primary unified-turn motor trace."
`orion/grammar/ledger.py::_upsert_trace()` upserts `grammar_traces` on
`on_conflict_do_update(index_elements=["trace_id"])` /
`on_conflict_do_nothing(index_elements=["trace_id"])`, with `trace_id` as the
sole conflict key. `orion/substrate/execution_loop/reducer.py` and
`grammar_extract.py` (a separate, in-memory execution-trajectory reducer
distinct from the ledger) also key `ExecutionTrajectoryProjectionV1.runs` by
`trace_id`, and `grammar_extract.py::extract_execution_state_from_events()`
parses `node_id`/`correlation_id` back out of `trace_id` via
`parse_execution_trace_id()` when building each run's state.

## Architecture touched

- `orion/harness/grammar_emit.py` — `HarnessGrammarCollector.trace_id`
- `orion/substrate/execution_loop/grammar_extract.py` —
  `extract_execution_state_from_events()`'s `correlation_id` precedence
- `services/orion-cortex-exec/app/grammar_emit.py` — reserved-lane-name
  comment only, no behavior change
- Tests: `orion/harness/tests/test_harness_grammar_emit.py`,
  `orion/grammar/tests/test_ledger.py`, `tests/test_execution_substrate_reducer.py`
- No schema, bus, env, or Docker changes.

## Files changed

- `orion/harness/grammar_emit.py`: `HarnessGrammarCollector.trace_id` now
  passes `lane="harness_motor"` to `cortex_exec_trace_id()`.
- `orion/harness/tests/test_harness_grammar_emit.py`: updated
  `test_trace_id_matches_cortex_exec_shape` to assert the new lane-suffixed
  shape (and that it now differs from the old unlaned shape); added
  `test_harness_trace_id_distinct_from_unlaned_cortex_exec_root`, a direct
  live-shaped collision-reproduction test.
- `orion/grammar/tests/test_ledger.py`: added
  `test_harness_and_unlaned_cortex_exec_trace_ids_no_longer_share_ledger_key`,
  which builds two `trace_started` `GrammarEventV1`s shaped exactly like the
  live-confirmed collision (same node/correlation_id, one harness-shaped, one
  unlaned-cortex-exec-root-shaped) and asserts `_upsert_trace()` compiles
  them to distinct `trace_id` SQL params.
- `orion/substrate/execution_loop/grammar_extract.py`: `correlation_id`
  precedence fixed to prefer `event.correlation_id` over the value parsed out
  of a (possibly lane-suffixed) `trace_id`.
- `tests/test_execution_substrate_reducer.py`: added
  `test_extract_correlation_id_not_polluted_by_harness_motor_lane_suffix`,
  reproducing the real `harness_motor`-laned shape and asserting
  `run.correlation_id` stays clean.
- `services/orion-cortex-exec/app/grammar_emit.py`: added a comment
  reserving `"harness_motor"` in `CORTEX_EXEC_ISOLATED_TRACE_LANES` so a
  future cortex-exec verb of that name can't silently recreate the collision
  in a new lane-vs-lane pairing.

## Schema / bus / API changes

None. `HarnessGrammarCollector.trace_id`'s string *value* changes (gains a
`:harness_motor` suffix) but its type/shape contract
(`cortex.exec:{node}:{correlation_id}[:{lane}]`) is unchanged and already
handled by every consumer swept (see Review findings below). No
`grammar_traces`/`grammar_events` schema change; no bus channel change.

## Env/config changes

None.

## Tests run

```text
.venv/bin/python -m pytest orion/harness/tests/ -q
=> 134 passed, 3 failed (pre-existing on main, unrelated: test_grounding_capsule_consumers.py x2,
   test_harness_runner.py x1 -- confirmed identical failures on main before this branch,
   via git stash -u/pop round-trip in the shared checkout, restored cleanly)

.venv/bin/python -m pytest orion/grammar/tests/test_ledger.py -q
=> 8 passed (6 pre-existing + 2 new)

.venv/bin/python -m pytest tests/test_execution_substrate_reducer.py tests/test_execution_loop_ids.py \
  tests/test_execution_projection_schemas.py tests/test_execution_substrate_pipeline.py \
  tests/test_route_substrate_reducer.py -q
=> 51 passed (50 pre-existing + 1 new)

PYTHONPATH=services/orion-cortex-exec:. .venv/bin/python -m pytest services/orion-cortex-exec/tests/test_exec_grammar_emit.py -q
=> 16 passed (untouched, confirmed still green)

git diff --check
=> clean, exit 0
```

## Evals run

None applicable — no eval harness exists for `orion/harness/grammar_emit.py`,
its cortex-exec sibling, or the execution-loop reducer package. The live
Postgres query against `grammar_events`/`grammar_traces` (grouped by
`trace_id`, comparing `source_service`) used to confirm the collision before
touching code is the closest thing to an eval — informal, by hand, not
automated. Same gap flagged in this session's other PR reports
(2026-07-15-harness-grammar-atom-wall-clock-pr.md, and earlier).

## Docker/build/smoke checks

Not run — pure Python logic change inside shared package code
(`orion/harness/`, `orion/substrate/execution_loop/`) and one comment-only
change in `orion-cortex-exec`'s existing code path. No config/dependency
changes to validate. `orion-harness-governor` and `orion-cortex-exec` both
publish/consume `GrammarEventV1`s carrying the new `trace_id` shape at
runtime, so a restart is required for the fix to take effect in production
(see Restart required below) even though no build/config surface changed.

## Review findings fixed

Code-review skill (subagent, high effort) against the first commit
(`d2e311a8`), 8 finder angles including a dedicated live-code-execution
verification of the correlation_id claim:

- **Finding (HIGH)**: `orion/substrate/execution_loop/grammar_extract.py`'s
  `extract_execution_state_from_events()` preferred the `correlation_id`
  parsed out of `trace_id` (via `parse_execution_trace_id()`, which only
  splits on the first two colons) over the event's own clean
  `correlation_id` field. Once `HarnessGrammarCollector.trace_id` always
  carries `:harness_motor`, this silently suffixed every harness-governor
  execution run's `ExecutionRunStateV1.correlation_id` — a field exposed
  as-is on `orion-substrate-runtime`'s live
  `GET /projections/execution_trajectory` debug endpoint. Reviewer
  reproduced this directly against the patched code (`run.correlation_id`
  showing `...:harness_motor`) before reporting it.
  - **Fix**: precedence swapped to `events[0].correlation_id or (parsed[1] if
    parsed else "unknown")`, matching the pattern of preferring the explicit
    field, without touching `parse_execution_trace_id()`'s own
    lane-suffix-inclusive contract (other tests already pin that behavior
    intentionally for the three pre-existing cortex-exec lanes).
  - **Evidence**: new `test_extract_correlation_id_not_polluted_by_harness_motor_lane_suffix`
    (`tests/test_execution_substrate_reducer.py`) reproduces the real
    `harness_motor`-laned shape end-to-end through
    `extract_execution_state_from_events()` and asserts `run.correlation_id
    == "corr-abc"` (not `"corr-abc:harness_motor"`).
- **Finding (LOW)**: `"harness_motor"` lives in the same flat lane-string
  namespace as `CORTEX_EXEC_ISOLATED_TRACE_LANES`; nothing stops a future
  cortex-exec verb literally named `harness_motor` from being added there,
  silently recreating a lane-vs-lane collision.
  - **Fix**: added a reserved-name comment directly on
    `CORTEX_EXEC_ISOLATED_TRACE_LANES` in
    `services/orion-cortex-exec/app/grammar_emit.py`.
  - **Evidence**: verified the frozenset's actual runtime contents are
    unchanged (`{'harness_finalize_reflect', 'orion_voice_finalize',
    'stance_react'}` — comment-only addition, no phantom set member) and all
    16 `test_exec_grammar_emit.py` tests still pass.

Not fixed, accepted and documented — see Risks / concerns below:

- **Finding (MEDIUM)**: the fix closes this specific collision instance via
  convention (choosing a lane string no real caller currently uses), not a
  structural fix — `orion/grammar/ledger.py`'s conflict key is still
  `trace_id` alone, with no `source_service` component. A future
  producer/lane pairing that happens to reuse the same
  `(node, correlation_id, lane-or-none)` tuple would reproduce this bug
  class in a new pairing.
- **Finding (LOW/nit)**: `test_harness_trace_id_distinct_from_unlaned_cortex_exec_root`
  in `orion/harness/tests/test_harness_grammar_emit.py` asserts substantially
  the same fact as the second assertion already added to
  `test_trace_id_matches_cortex_exec_shape` in the same file. Kept both —
  the dedicated, heavily-documented regression test has standalone
  discoverability value even though it isn't meaningfully distinct test
  coverage.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-harness-governor/.env \
  -f services/orion-harness-governor/docker-compose.yml up -d --build

docker compose --env-file .env --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml up -d --build

docker compose --env-file .env --env-file services/orion-substrate-runtime/.env \
  -f services/orion-substrate-runtime/docker-compose.yml up -d --build
```
Not run this session.

## Risks / concerns

- Severity: medium
- Concern: `orion/grammar/ledger.py`'s `grammar_traces` conflict key remains
  `trace_id` alone (no `source_service`), so this fix closes the specific
  confirmed collision (`HarnessGrammarCollector` vs. unlaned
  `CortexExecGrammarCollector` root call) by convention — picking a lane
  string no real caller uses — rather than making the collision class
  structurally impossible. Per the task scope, a compound-conflict-key
  change to the ledger was explicitly out of scope unless the
  lane-disambiguation approach proved unworkable; it did not, so that larger
  change was not undertaken here.
- Mitigation: `"harness_motor"` is now documented as reserved in
  `CORTEX_EXEC_ISOLATED_TRACE_LANES`. Any future addition of a fourth
  grammar-collector class, or a `trace_id`-namespace scheme beyond
  `cortex.exec:`, should re-examine whether `source_service` belongs in the
  ledger's conflict key.
- Severity: low
- Concern: `services/orion-bus/app/grammar_emit.py`'s
  `BusTransportGrammarCollector` (a third sibling collector, distinct
  `trace_id` namespace) was not audited as part of this collision sweep —
  out of scope for this specific `cortex.exec:` namespace collision, but
  worth a future audit if a similar report ever surfaces for it.

## PR link

`gh` is authenticated in this environment.
