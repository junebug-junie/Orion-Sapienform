# PR report — harness grammar atoms never got real wall-clock observed_at

PR: (create at https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/harness-grammar-atom-wall-clock)
Branch: `fix/harness-grammar-atom-wall-clock`
Status: **DONE**

## Summary

- Confirmed and fixed a real, non-speculative sibling bug flagged in memory
  (`project_grammar_collector_shared_timestamp_bug_siblings`): `HarnessGrammarCollector`
  (`orion/harness/grammar_emit.py`) shared the exact bug already fixed in
  `CortexExecGrammarCollector` (PR #1042) — every `GrammarAtomV1`/edge/
  `trace_ended` event in a harness trace got one shared `observed_at`
  captured once at collector construction, regardless of how much real
  elapsed time separated request-received from step-started from
  step-completed from result-assembled.
- Live-verified before touching code: harness traces with 7–55 atoms,
  spanning up to 2m15s of real wall-clock time (per the sibling
  bare-correlation-id trace's `created_at` range), all showed exactly 1
  distinct `observed_at` value across every atom.
- Fixed with the same shape as the cortex-exec patch: a new
  `_atom_observed_at` dict populated in `_put_atom()`, alongside
  `GrammarAtomV1.time_range` (previously always `None` — a second location
  the same "timing is fake" symptom survived).
- Found and fixed a **second, distinct bug** while writing the regression
  tests: `record_step_started`/`record_step_completed`/`record_step_failed`
  — the three most timing-sensitive atom types — bypassed `_put_atom()`
  entirely (`self._atoms[key] = atom` direct assignment, pre-dating this
  patch), so even with the collector-level fix in place these three would
  have kept falling back to the trace-start timestamp.
- Code review (high effort, 8 angles) found and fixed a **third bug**,
  independently flagged by 2 of 8 angles from tracing the real production
  call graph: `record_result_assembled` is genuinely re-invoked on the same
  collector instance across two production phases (motor publish, then
  finalize publish), and since its `atom_id` is deterministic, the naive
  fix would have published the same identity twice with contradictory
  timestamps. Fixed by making `_put_atom()` idempotent per `atom_id`. The
  identical fix was then backported to the sibling `CortexExecGrammarCollector`,
  confirmed live-relevant there too (not hypothetical — traced the real
  4-call-site production pattern in `router.py`).

## Outcome moved

Harness execution traces now carry real per-atom timestamps. `orion/grammar/ledger.py`'s
existing `_upsert_trace()` (already fixed in PR #1042 to prefer `observed_at`
over `emitted_at`) now gets real, non-collapsed data from the harness
producer too — `grammar_traces.started_at`/`ended_at` for harness traces
should stop showing zero-duration.

## Current architecture

`HarnessGrammarCollector` (`orion/harness/grammar_emit.py`) is one of three
independently-duplicated grammar-collector classes (siblings:
`CortexExecGrammarCollector` in `services/orion-cortex-exec/app/grammar_emit.py`,
`BusTransportGrammarCollector` in `services/orion-bus/app/grammar_emit.py`) —
no shared base class exists; each accumulates `GrammarAtomV1`s/edges via
`record_*()` methods, then `build_*_grammar_events()` assembles the final
`GrammarEventV1` list for publish. `HarnessRunner.run()` constructs one
collector per turn; the same instance is threaded through the motor-publish
path (`runner.py`) and, if `final_text` is present, a second finalize-publish
path (`services/orion-harness-governor/app/bus_listener.py`).

## Architecture touched

`orion/harness/grammar_emit.py` (shared package code), `services/orion-cortex-exec/app/grammar_emit.py`
(separate service, backported fix), plus 2 test files. No schema/env/Docker
changes.

## Files changed

- `orion/harness/grammar_emit.py`: `_atom_observed_at` dict + `time_range`
  stamping in `_put_atom()` (made idempotent per `atom_id` after review);
  `record_step_started`/`record_step_completed`/`record_step_failed` routed
  through `_put_atom()` instead of direct dict assignment;
  `build_harness_grammar_events()`/`build_harness_grammar_finalize_events()`
  use real per-atom timestamps via a new `observed_at_for()` helper;
  `trace_ended` uses the real max recorded timestamp instead of reusing
  trace-start.
- `orion/harness/tests/test_harness_grammar_emit.py`: 8 new regression
  tests, mirroring `CortexExecGrammarCollector`'s test suite, adapted to
  `HarnessGrammarCollector`'s actual API.
- `services/orion-cortex-exec/app/grammar_emit.py`: backported the
  idempotency-per-`atom_id` guard to `_put_atom()` (the timestamp/`time_range`
  fix itself already shipped in PR #1042; this patch only adds the
  re-invocation guard, newly proven necessary).
- `services/orion-cortex-exec/tests/test_exec_grammar_emit.py`: 1 new
  regression test mirroring the harness one.

## Schema / bus / API changes

None. `HarnessGrammarCollector._atom_observed_at` and
`CortexExecGrammarCollector`'s idempotency change are both purely additive/
internal — no consumer-visible shape change.

## Env/config changes

None.

## Tests run

```text
.venv/bin/python -m pytest orion/harness/tests/test_harness_grammar_emit.py -q
=> 15 passed

.venv/bin/python -m pytest orion/harness/ -q
=> 145 passed, 3 failed (pre-existing on main, unrelated: test_grounding_capsule_consumers.py x2,
   test_harness_runner.py x1 -- confirmed identical failures on main before this branch)

PYTHONPATH=services/orion-harness-governor:. .venv/bin/python -m pytest services/orion-harness-governor/tests -q
=> 13 passed

PYTHONPATH=services/orion-cortex-exec:. .venv/bin/python -m pytest services/orion-cortex-exec/tests/test_exec_grammar_emit.py -q
=> 16 passed (15 pre-existing + 1 new)
```

## Evals run

None applicable — no eval harness exists for `orion/harness/grammar_emit.py`
or its cortex-exec sibling. The live Postgres query (`grammar_events`
grouped by `trace_id`, filtered `source_service='orion-harness-governor'`,
comparing `count(DISTINCT observed_at)` against `wall_clock_spread`) used to
confirm this bug before touching code is the closest thing to an eval —
informal, by hand, not automated. Same gap flagged in PR #1052/#1061's
reports this session.

## Docker/build/smoke checks

Not run — pure Python logic change inside `orion-harness-governor`'s shared
package code and `orion-cortex-exec`'s existing code path, no config/
dependency changes to validate.

## Review findings fixed

Code-review skill, high effort (3 correctness + 3 cleanup + altitude +
conventions = 8 finder angles, verified):

- **Finding** (2 of 8 angles, independently, both tracing the real
  production call graph): `record_result_assembled` is genuinely
  re-invoked on the same collector across the motor-publish and
  finalize-publish phases (`orion/harness/runner.py` → `services/orion-harness-governor/app/bus_listener.py:344-354`).
  Since `atom_id` is deterministic, the second call would advance
  `_atom_observed_at` and publish the same `atom_id`/`event_id` twice with
  contradictory `observed_at`. `on_conflict_do_nothing` in
  `orion/grammar/ledger.py` masks this at the DB layer, but a raw bus
  consumer of `orion:grammar:event` would see it directly.
  - **Fix**: `_put_atom()` made idempotent per `atom_id` — first-seen
    timestamp wins; atom content still refreshes on every call.
  - **Evidence**: new `test_result_assembled_reinvoked_keeps_first_observed_at`
    reproduces the real double-call sequence and asserts pinned timestamp +
    refreshed content.
- **Finding** (altitude angle, backed by concrete `router.py` line
  citations): `CortexExecGrammarCollector`'s `_put_atom()` has the identical
  `_atom_observed_at` shape but never got an idempotency guard, and the
  same re-invocation pattern is confirmed live there too
  (`get_or_create_collector()` caches one collector per `ctx`;
  `record_assembled_grammar()` is called from 4 separate sites in
  `router.py`).
  - **Fix**: backported the identical guard.
  - **Evidence**: all 16 cortex-exec grammar_emit tests pass (15
    pre-existing + 1 new, mirroring the harness regression test).
- **Finding** (simplification angle): the `observed_at` fallback lookup
  (`collector._atom_observed_at.get(id, observed_at)`) was copy-pasted 4
  times across the two harness builder functions.
  - **Fix**: extracted into one `observed_at_for()` method.
  - **Evidence**: full suite still passes (behavior-neutral refactor).
- **Finding** (conventions angle): a code comment cited "PR #1039" for the
  cortex-exec grammar-atom-wall-clock work; the actual PR was #1042 (#1039
  is unrelated — `fix/execution-dispatch-real-send-bugs`).
  - **Fix**: corrected the comment.
  - **Evidence**: `git log` cross-check.

Not fixed, accepted and documented — would expand scope beyond the
confirmed bug class:

- `orion/grammar/ledger.py`'s `on_conflict_do_nothing` means a re-invoked
  `record_result_assembled`'s refreshed *content* (not just timestamp)
  never actually reaches Postgres — only the first-arriving row persists.
  Whether atom rows should be upsertable-on-conflict instead of insert-only
  is a real, separate persistence-semantics question.
- `services/orion-bus/app/grammar_emit.py`'s `BusTransportGrammarCollector`
  still has neither the timestamp fix nor the idempotency guard —
  confirmed still true, matches prior memory, lower priority per that same
  memory (sub-millisecond real timescale for that collector's atoms).
- `observed_at_for()`'s fallback to trace-start when an atom bypassed
  `_put_atom()` is currently dead code in production (all 9 `record_*`
  methods route through it) — a candidate for louder failure (log/raise)
  instead of a silent default, but guards a hypothetical future bug, not a
  live one; left matching the already-accepted sibling pattern.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-harness-governor/.env \
  -f services/orion-harness-governor/docker-compose.yml up -d --build

docker compose --env-file .env --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml up -d --build
```
Not run this session.

## Risks / concerns

- Severity: low
- Concern: `services/orion-bus/app/grammar_emit.py`'s `BusTransportGrammarCollector`
  is now the last of the three sibling collectors with neither fix. Per
  existing memory this is intentionally lower priority (sub-millisecond
  real timescale, genuinely less consequential) — not a new concern, just
  confirmed still open.
- Severity: low
- Concern: no shared base class/mixin exists across the three collectors,
  so a future 4th instance of this bug class (or the next timing-semantics
  fix) requires another independent patch. Flagged by the review's reuse
  angle as worth a future architecture decision, not undertaken here to
  keep this patch thin and scoped to the confirmed bug.

## PR link

`gh` is authenticated in this environment.
