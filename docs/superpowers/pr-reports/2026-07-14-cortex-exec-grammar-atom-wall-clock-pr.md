## Summary

- Fixes a real, live-confirmed bug: every atom inside one `orion-cortex-exec`
  execution trace shared a single timestamp on `GrammarEventV1.observed_at`
  (captured once at trace-START, not "flush time" — see the correction note
  below), so intra-trace timing in the `grammar_events` Postgres corpus was
  fake. `CortexExecGrammarCollector` now captures a real wall-clock moment
  per atom (`_put_atom()`) and threads it into each event's `observed_at`
  field. `emitted_at` deliberately stays the shared flush-time value — it
  was never wrong (every event genuinely is published to the bus in the same
  flush batch).
- **Second round, following a `/code-review max` pass on the already-pushed
  first round** (see "Review findings fixed" below) — the first round's fix
  was correct but incomplete: it made `grammar_events.observed_at` real, but
  two adjacent, already-live consumers still showed fake timing:
  - `GrammarAtomV1.time_range` (an existing schema field, wired through by
    `orion/grammar/ledger.py` to persisted `grammar_atoms.time_start/time_end`
    and served by the live Grammar Atlas API) was never populated by this
    producer. `_put_atom()` now stamps it, from the same captured moment as
    `_atom_observed_at`.
  - `grammar_traces.started_at`/`ended_at` (same Atlas API) were derived
    from `event.emitted_at` (`orion/grammar/ledger.py`), not `observed_at` —
    so trace-level duration stayed fake even after the first round.
    `trace_ended.observed_at` now reflects the real last-recorded-atom
    time (was hardcoded to trace-start, identical to `trace_started`'s).
    `orion/grammar/ledger.py::_upsert_trace()` now prefers `observed_at`
    over `emitted_at` (via the already-existing `_created_at()` helper),
    safely for every grammar-event producer, not just cortex-exec.
- Also corrected: the first round's own "live-confirmed" evidence
  (`max(emitted_at)-min(emitted_at) = 0.00s`) was measured against the
  wrong field — `emitted_at` was never the buggy one. Re-verified live
  directly against `observed_at` (the actually-fixed field): also 0.0000s
  median pre-fix, so the underlying bug claim holds, but the field name/
  mechanism description in the source comment, spec, and this report were
  imprecise and are now corrected throughout.
- Test-file cleanup: removed an unnecessary `datetime` subclass in the
  fake-clock test helper (a plain callable works identically, verified
  empirically), removed its dead unused return value, and strengthened one
  test that previously passed trivially even if the whole fix were reverted.
- Design spec: `docs/superpowers/specs/2026-07-14-cortex-exec-grammar-atom-wall-clock-spec.md`.
- 10 new/updated regression tests total across both rounds.
- Confirmed (not speculated) that `HarnessGrammarCollector`
  (`orion/harness/grammar_emit.py`) has the identical bug — tracked as a
  follow-up, explicitly out of scope for this patch.

## Outcome moved

`grammar_events` (5M+ real rows, 679K from `orion-cortex-exec` alone) is the
strongest raw-cognitive-signal substrate identified this session for a future
felt-state/trajectory corpus. This patch makes its intra-trace timing real —
both at the event level (`observed_at`) and at the two adjacent already-live
consumer surfaces (`grammar_atoms.time_start/time_end`, `grammar_traces.
started_at/ended_at`) that the first round missed.

## Current architecture

Before this patch: `CortexExecGrammarCollector.record_*` methods (9 total)
built `GrammarAtomV1` objects as pure data with no timestamp capture, stored
in `self._atoms`. `build_cortex_exec_grammar_events()` ran once at trace end,
using one shared trace-start value for every atom/edge event's `observed_at`,
and hardcoding `trace_ended.observed_at` to the same trace-start value.
`orion/grammar/ledger.py::_upsert_trace()` derived `grammar_traces.started_at`/
`ended_at` from `emitted_at` unconditionally.

## Architecture touched

`services/orion-cortex-exec/app/grammar_emit.py` (producer fix, both rounds),
`orion/grammar/ledger.py` (shared package — `_upsert_trace()` now prefers
`observed_at`, benefits every grammar-event producer, safe fallback to
`emitted_at` for producers without real `observed_at` yet, no regression for
them). No schema change (`orion/schemas/grammar.py` untouched —
`GrammarEventV1.observed_at` and `GrammarAtomV1.time_range` both already
existed, just were populated wrong/not at all). No env/compose/registry
changes.

## Files changed

- `services/orion-cortex-exec/app/grammar_emit.py`: `_atom_observed_at: dict[str, datetime]`
  on `CortexExecGrammarCollector`, keyed by `atom.atom_id`, populated in
  `_put_atom()` — which now also stamps `atom.time_range = TimeRangeV1(start=now, end=now)`
  from the same captured moment. All 9 `record_*` methods route through
  `_put_atom()`. `build_cortex_exec_grammar_events()`'s atom/edge loops use
  the real per-atom/per-edge-target timestamp for `observed_at` only;
  `trace_ended.observed_at` now derives from `max(_atom_observed_at.values())`
  (falls back to trace-start for a zero-atom trace). Source comments
  corrected to accurately describe trace-start vs. flush-time.
- `orion/grammar/ledger.py`: `_upsert_trace()` now computes `trace_ts = _created_at(event)`
  (reusing the existing helper, which already prefers `observed_at` over
  `emitted_at` over `now()`) instead of hardcoding `event.emitted_at` for
  both `started_at` and `ended_at`.
- `services/orion-cortex-exec/tests/test_exec_grammar_emit.py`: fake-clock
  helper simplified (`SimpleNamespace` instead of a `datetime` subclass,
  dead return value removed); `test_edge_observed_at_matches_target_atom_real_timestamp`
  strengthened with a direct inequality check against the old shared
  constant (previously only checked internal self-consistency, which passes
  even fully reverted); 3 new tests for `time_range` population and real
  `trace_ended` behavior (including the zero-atom fallback path).
- `orion/grammar/tests/test_ledger.py`: 2 new tests for `_upsert_trace()`
  preferring `observed_at`, with an explicit fallback-to-`emitted_at` test
  for producers without real `observed_at`.
- `docs/superpowers/specs/2026-07-14-cortex-exec-grammar-atom-wall-clock-spec.md`:
  corrected mechanism description (trace-start vs. flush-time), corrected
  live evidence (now cites the right field), confirmed harness sibling bug.

## Schema / bus / API changes

None. `GrammarEventV1.observed_at` and `GrammarAtomV1.time_range`
(`orion/schemas/grammar.py`) both pre-existed — this patch only changes
which values get written into already-existing fields.

## Env/config changes

None.

## Tests run

```text
.venv/bin/python -m pytest services/orion-cortex-exec/tests/test_exec_grammar_emit.py services/orion-cortex-exec/tests/test_exec_grammar_publish_fail_open.py -q
=> 16 passed

PYTHONPATH=services/orion-sql-writer:. .venv/bin/python -m pytest orion/grammar/tests/test_ledger.py -q
=> 5 passed

PYTHONPATH=services/orion-sql-writer:. .venv/bin/python -m pytest services/orion-sql-writer/tests/test_grammar_ledger_sql_shape.py -q
=> 1 passed

PYTHONPATH=services/orion-sql-writer:. .venv/bin/python -m pytest services/orion-sql-writer/tests/test_grammar_ledger_integration.py -q
=> 3 errors, all psycopg2.OperationalError connecting to localhost:5432 (the
   live DB in this environment is on port 55432, not 5432) -- a pre-existing
   environment/fixture-config issue, not caused by this patch. The one
   non-DB-dependent test in the sibling file (test_grammar_ledger_sql_shape.py)
   passes.

.venv/bin/python -m pytest services/orion-cortex-exec/tests -q
=> 12 pre-existing collection errors ("Verb already registered"), confirmed
   identical on a clean main checkout (verified via a fresh clone) --
   unrelated to this patch.
```

## Evals run

None applicable. `services/orion-cortex-exec/` has no `evals/` directory at
all (confirmed via directory listing) — no eval harness exists for this
service. Per CLAUDE.md section 11, flagging this explicitly rather than
silently: no eval harness added here (out of proportionate scope for a bug
fix), no follow-up issue filed yet — worth doing if/when this service gets
non-trivial cognition-relevant logic beyond grammar-event bookkeeping.

## Docker/build/smoke checks

Not run this session (no Docker access exercised). This patch changes
runtime behavior (what gets computed and persisted at execution time) but
not deployment topology (no env, port, volume, or dependency changes) — no
`docker compose build`/`config` check was run to validate the container
image itself; that layer is **UNVERIFIED**, distinct from the code-level
test coverage above, which is real and passing. Restart commands below are
required to actually pick up the change; the restart itself will be the
first real verification at that layer.

## Review findings fixed

**Round 1** (8-angle review before initial push): 7 of 8 angles returned zero
findings. The altitude angle found the confirmed `HarnessGrammarCollector`
sibling bug (see Risks/concerns) and a spec-accuracy gap, both fixed in the
spec doc.

**Round 2** (`/code-review max`, 10 angles + verify + sweep, run against the
already-pushed round 1 diff): 6 of 10 angles returned zero findings
(removed-behavior, cross-file, language-pitfall, wrapper/proxy, reuse,
efficiency). 4 angles found real, fixed issues:

- Finding (altitude angle, CONFIRMED): `orion/grammar/ledger.py`'s
  `grammar_traces.started_at`/`ended_at` derive from `emitted_at`, still
  fake after round 1's fix.
  - Fix: `_upsert_trace()` now prefers `observed_at` via the existing
    `_created_at()` helper; `trace_ended.observed_at` now reflects real
    last-atom time instead of trace-start.
  - Evidence: `test_trace_started_prefers_observed_at_over_emitted_at`,
    `test_trace_ended_observed_at_reflects_last_atom_not_trace_start`.
- Finding (altitude angle, CONFIRMED): `GrammarAtomV1.time_range` (existing
  field, wired to persisted columns and a live API) never populated.
  - Fix: `_put_atom()` now sets `atom.time_range` from the same captured
    timestamp as `_atom_observed_at`.
  - Evidence: `test_atom_time_range_populated_with_real_timestamp`.
- Finding (conventions angle, CONFIRMED): source comment, spec, and PR
  report described the pre-fix bug as "flush-time" and cited
  `max(emitted_at)-min(emitted_at)=0.00s` as proof — `emitted_at` was never
  the buggy field; the actually-fixed field (`observed_at`) was captured at
  trace-start, not flush-time.
  - Fix: comments/spec/this report corrected; re-verified live directly
    against `observed_at` (also 0.0000s median pre-fix on real data,
    confirming the underlying claim while fixing the mechanism description).
- Finding (test-quality angle, CONFIRMED):
  `test_edge_observed_at_matches_target_atom_real_timestamp` passed
  trivially even with the whole fix reverted (only checked internal
  self-consistency).
  - Fix: added a direct inequality assertion against the old shared
    constant.
- Finding (simplification angle, CONFIRMED, x2): unnecessary `datetime`
  subclass in the test fake-clock helper (verified empirically that a plain
  callable works identically); dead unused return value on the same helper.
  - Fix: both removed.
- Finding (conventions angle, PLAUSIBLE): PR report's Docker section
  self-contradicted its own Summary; evals-gap not disclosed.
  - Fix: both sections rewritten above.
- Finding (correctness angle, PLAUSIBLE, not fixed): a backward wall-clock
  step (NTP correction) between two `_put_atom()` calls in the same trace
  could theoretically produce non-monotonic `observed_at` ordering.
  - Disposition: not fixed — low realistic likelihood, low blast radius for
    a training-data/observability corpus (not a safety-critical path),
    added guard complexity not proportionate. Noted here for the record.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml up -d --build
```
Also restart `orion-sql-writer` (reads `orion/grammar/ledger.py`, a shared
module, so its running container needs the updated code too):
```bash
docker compose --env-file .env --env-file services/orion-sql-writer/.env \
  -f services/orion-sql-writer/docker-compose.yml up -d --build
```
Neither run this session — not deployed.

## Risks / concerns

- Severity: low
- Concern: `HarnessGrammarCollector` has the same event-level bug, confirmed,
  not fixed here; its `GrammarAtomV1.time_range` is presumably also
  unpopulated (not independently re-checked this round).
- Mitigation: tracked explicitly in project memory as a near-term follow-up.
- Severity: very low
- Concern: theoretical backward-clock non-monotonicity (see Review findings).
- Mitigation: accepted as-is, disproportionate to guard against for this
  corpus's actual use case.

## PR link

`gh` is unauthenticated in this environment — open manually at:
https://github.com/junebug-junie/Orion-Sapienform/pull/new/docs/grammar-atom-wall-clock-spec
