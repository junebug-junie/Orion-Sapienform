# PR report — bus-transport grammar atoms never got real wall-clock observed_at

PR: (create at https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/bus-transport-grammar-atom-wall-clock)
Branch: `fix/bus-transport-grammar-atom-wall-clock`
Status: **DONE**

## Summary

- Fixed the third and final sibling of the shared-`observed_at` bug
  (memory `project_grammar_collector_shared_timestamp_bug_siblings`):
  `BusTransportGrammarCollector` (`services/orion-bus/app/grammar_emit.py`)
  had the identical bug already fixed in `CortexExecGrammarCollector`
  (PR #1042) and `HarnessGrammarCollector` (PR, commit `4620c21b`) — every
  `GrammarAtomV1`/edge/`trace_ended` event in a trace shared one
  collector-construction-time `observed_at` instead of each atom's own real
  capture time.
- Same fix shape: new `_atom_observed_at: dict[str, datetime]`, populated
  by `_put_atom()` alongside `GrammarAtomV1.time_range` (previously always
  `None`); a new `observed_at_for()` helper used by
  `build_bus_transport_grammar_events()` for both atoms and edges;
  `trace_ended` now uses `max(_atom_observed_at.values(), default=observed_at)`
  instead of reusing trace-start.
- Investigated (not assumed) whether this collector needed the
  idempotency-per-`atom_id` guard the other two siblings needed —
  **it does not**. Every real call site constructs a brand-new collector
  per observer tick and calls each `record_*()` method at most once per
  instance. See "Re-invocation investigation" below.
- Audited all 7 `record_*()` methods for the harness sibling's second bug
  class (methods bypassing `_put_atom()` with a direct
  `self._atoms[key] = atom` assignment) — **none bypass it**, so there was
  no second bug to fix here.
- Live-verified before touching code: this collector's real-world timing
  spread is genuinely negligible, not just theoretically nonzero. See
  "Live-data findings" below — this is the honest, calibrated version of
  the "Outcome moved" claim for this specific sibling.

## Outcome moved

Bus-transport observer-tick traces now carry per-atom `time_range` and, in
principle, per-atom `observed_at` distinct from trace-start. In practice
(see live-data findings below), every atom in a real observer tick is
recorded within the same event-loop tick with no `await` between
`record_*()` calls, so the real-world timestamp spread this fix produces is
sub-millisecond — the fix removes the structural bug (a shared constant
masquerading as per-atom timing, which `orion/grammar/ledger.py`'s
`time_range`-reading Grammar Atlas API consumers should never see as
identical-by-construction), but it does **not** meaningfully change
`grammar_traces.started_at`/`ended_at` duration for this producer the way
the harness fix did (harness traces spanned up to 2m15s; bus-transport
traces span microseconds). Calibrated claim, not oversold.

## Current architecture

`BusTransportGrammarCollector` (`services/orion-bus/app/grammar_emit.py`)
is the last of three independently-duplicated grammar-collector classes
(siblings: `CortexExecGrammarCollector` in
`services/orion-cortex-exec/app/grammar_emit.py`, `HarnessGrammarCollector`
in `orion/harness/grammar_emit.py`) — no shared base class exists; each
accumulates `GrammarAtomV1`s/edges via `record_*()` methods, then a
`build_*_grammar_events()` function assembles the final `GrammarEventV1`
list for publish. `services/orion-bus/app/bus_observer.py`'s
`run_bus_observer_loop()` calls `run_observer_tick()` once per poll
interval; `run_observer_tick()` builds a fresh `ObserverRollup` from a
Redis snapshot, calls `rollup.to_collector()` to get a brand-new
`BusTransportGrammarCollector`, then `build_bus_transport_grammar_events()`
and publishes via `publish_bus_transport_grammar_trace()`. The
except-path constructs a second, independent `fail_collector` for the
same tick.

## Re-invocation investigation (step 3)

Traced the real call graph in `services/orion-bus/`:

- `ObserverRollup.to_collector()` (`bus_observer.py:58-79`) constructs one
  `BusTransportGrammarCollector` and calls `record_tick_started()`,
  `record_health_observed()`, `record_stream_depth()` (once per stream in a
  loop, but each stream key is a distinct `role`/`atom_id`, not a repeat
  call for the same atom), `record_backpressure()` (same, per stream key),
  `record_uncataloged_stream()` (same), and `record_tick_completed()` —
  each of the fixed-role methods (`record_tick_started`,
  `record_health_observed`, `record_tick_completed`) called exactly once.
- `run_observer_tick()`'s except-path (`bus_observer.py:163-170`)
  constructs a second, independent `fail_collector` and calls
  `record_tick_started()` + `record_tick_failed()`, each once.
- `run_observer_tick()` itself is called once per `while True` iteration in
  `run_bus_observer_loop()` (`bus_observer.py:181-195`) — no retry/backoff
  wrapper, no shared/cached collector across iterations, every iteration
  builds a brand-new `ObserverRollup`/collector.

No path exists where the same `record_*()` method is invoked twice on the
same collector instance — unlike `HarnessGrammarCollector`'s
`record_result_assembled`, which is genuinely re-invoked across the
motor-publish and finalize-publish phases. **Conclusion: no idempotency
guard needed here.** Independently re-verified by a code-review subagent
that grepped every `BusTransportGrammarCollector(` construction site
(production + tests) and confirmed the same finding.

## Live-data findings

Query (per the mandatory pre-work check, `source_service='orion-bus'`
confirmed via `_provenance()`):

```text
SELECT trace_id, count(*) AS atoms, count(DISTINCT observed_at) AS distinct_observed_at,
  max(observed_at) - min(observed_at) AS observed_at_spread,
  max(created_at) - min(created_at) AS wall_clock_spread
FROM grammar_events
WHERE source_service = 'orion-bus' AND event_kind = 'atom_emitted'
GROUP BY trace_id HAVING count(*) > 2
ORDER BY max(created_at) DESC LIMIT 10;
```

Result: 10 real traces, all with 7 atoms, `distinct_observed_at = 1` for
every trace (confirming the bug) — but **also** `wall_clock_spread =
00:00:00` for every trace. A follow-up query with microsecond-precision
`EXTRACT(EPOCH ...)` confirmed `wall_clock_spread_ms = 0.000000` exactly,
i.e. `min(created_at)` and `max(created_at)` are bit-for-bit identical
timestamps for all 7 rows in a trace (e.g.
`2026-07-16 01:01:25.858311+00` for every row of that trace).

This confirms the flagged possibility explicitly: this collector's atoms
really do span one observer tick with genuinely negligible real elapsed
time, because `ObserverRollup.to_collector()` and `run_observer_tick()`
are pure synchronous Python with no `await`/I/O between `record_*()`
calls — unlike the harness/cortex-exec producers, which interleave
`record_*()` calls with real async work (LLM calls, tool execution,
recall). The fix is still correct to ship (it removes a structural bug:
`time_range` was always `None`, and `observed_at` was a shared constant by
construction rather than a genuinely-measured value that happened to be
close), but the "Outcome moved" claim for this specific sibling is
calibrated accordingly — do not expect
`grammar_traces.started_at`/`ended_at` duration to visibly change for
`bus.transport:*` traces the way it did for harness traces.

## Architecture touched

`services/orion-bus/app/grammar_emit.py` and its test file only. No
schema/env/Docker changes. No changes to `bus_observer.py` or any other
consumer.

## Files changed

- `services/orion-bus/app/grammar_emit.py`: `_atom_observed_at` dict +
  `time_range` stamping in `_put_atom()` (no idempotency guard — see
  investigation above); new `observed_at_for()` helper;
  `build_bus_transport_grammar_events()` uses real per-atom timestamps for
  atoms and edges; `trace_ended` uses the real max recorded timestamp
  instead of reusing trace-start.
- `services/orion-bus/tests/test_orion_bus_grammar_emit.py`: 7 new
  regression tests mirroring `HarnessGrammarCollector`'s test suite,
  adapted to `BusTransportGrammarCollector`'s actual API (no idempotency
  test — not applicable, see investigation above).

## Schema / bus / API changes

None. `BusTransportGrammarCollector._atom_observed_at` and
`observed_at_for()` are purely additive/internal — no consumer-visible
shape change. `GrammarAtomV1.time_range` was already part of the schema
(used by the other two siblings); this collector now populates it like
they do.

## Env/config changes

None.

## Tests run

```text
PYTHONPATH=services/orion-bus:. .venv/bin/python -m pytest services/orion-bus/tests/test_orion_bus_grammar_emit.py -q
=> 11 passed (4 pre-existing + 7 new)

PYTHONPATH=services/orion-bus:. .venv/bin/python -m pytest services/orion-bus/tests -q
=> 15 passed (full service test dir, no regressions)
```

## Evals run

None applicable — no `services/orion-bus/evals/` directory exists (same
gap as the harness/cortex-exec siblings' reports). The live Postgres query
above is the closest thing to an eval — informal, by hand, not automated.

## Docker/build/smoke checks

Not run — pure Python logic change inside `orion-bus`'s existing code
path, no config/dependency changes to validate.

## Review findings fixed

Code-review skill (equivalent, dispatched as a repo-aware subagent with
independent verification instructions), high effort, covering: correctness
of the re-invocation claim (independently re-traced the call graph),
`_put_atom()`-bypass audit, `observed_at_for()` extraction/duplication
check, edge-timestamp-direction check, altitude/architecture, conventions,
and a live before/after test-mutation check (reverted the timestamp
substitutions, confirmed 4 of the 6 timing-sensitive tests fail as
expected, then restored).

**No material findings.** All checks passed on first pass:

- Re-invocation claim independently confirmed safe (no repeat
  `record_*()` calls on any real collector instance).
- No `_put_atom()`-bypass bug found (unlike the harness sibling, all 7
  `record_*()` methods already routed through `_put_atom()` before this
  patch).
- `observed_at_for()` correctly extracted and used at both call sites, no
  duplicated inline fallback lookups.
- Edge `to_atom_id` confirmed to always be the just-recorded atom at all 4
  edge-append sites — timestamp direction is correct.
- New tests independently verified to catch the regression (reverting the
  fix locally made 4 of 6 new tests fail, as expected).

Nice-to-have, not fixed (out of scope, noted for follow-up):

- `CortexExecGrammarCollector`'s three inline `.get(id, observed_at)`
  lookups (`services/orion-cortex-exec/app/grammar_emit.py:562,594,620`)
  were never given the `observed_at_for()`-style extraction the harness
  sibling got in its own review round — this patch is actually ahead of
  that sibling on this specific cleanup, not behind it.
- No shared base class/mixin exists across the three sibling collectors,
  so this is now the third independent copy of the same
  `_atom_observed_at`/`_put_atom`/fallback pattern. A future 4th producer
  of this bug class would need another independent patch. Flagged by both
  this review and the harness PR's review; still not undertaken, to keep
  each patch thin and scoped.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-bus/.env \
  -f services/orion-bus/docker-compose.yml up -d --build
```
Not run this session.

## Risks / concerns

- Severity: low
- Concern: as documented in "Live-data findings," the real-world impact of
  this fix for this specific producer is negligible (sub-millisecond,
  genuinely simultaneous atom recording) — do not expect this PR to move
  any dashboard/duration metric for `bus.transport:*` traces the way the
  harness fix moved harness trace durations. The fix is still correct
  (removes a structural bug in `time_range`/`observed_at` semantics that a
  raw bus consumer or the Grammar Atlas API could observe), just low
  practical impact.
- Severity: low
- Concern: same as the harness/cortex-exec PRs' reports — no shared base
  class exists across the three sibling collectors. Not addressed here,
  consistent with keeping this patch thin.

## PR link

`gh` is authenticated in this environment.
