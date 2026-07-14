## Summary

- Fixes a real, live-confirmed bug: every atom inside one `orion-cortex-exec`
  execution trace shared a single flush-time timestamp, so intra-trace timing
  in the `grammar_events` Postgres corpus was fake (median intra-trace
  duration across 35,994 real traces was 0.00s). `CortexExecGrammarCollector`
  now captures a real wall-clock moment per atom (`_put_atom()`) and threads
  it into each event's `observed_at` field.
- `emitted_at` deliberately stays the shared flush-time value — it was never
  wrong (every event genuinely is published to the bus in the same flush
  batch); this mirrors the pre-existing, already-correct `trace_started`/
  `trace_ended` pattern rather than inventing a new convention.
- Design spec: `docs/superpowers/specs/2026-07-14-cortex-exec-grammar-atom-wall-clock-spec.md`.
- 5 new regression tests; 8-angle code review found zero correctness bugs.
- Confirmed (not speculated) that `HarnessGrammarCollector`
  (`orion/harness/grammar_emit.py`) has the identical bug — tracked as a
  follow-up, explicitly out of scope for this patch.

## Outcome moved

`grammar_events` (5M+ real rows, 679K from `orion-cortex-exec` alone) is the
strongest raw-cognitive-signal substrate identified this session for a future
felt-state/trajectory corpus. This patch makes its intra-trace timing real,
which was a hard blocker for building anything on top of it that needs actual
event ordering/timing within an execution, not just across executions.

## Current architecture

Before this patch: `CortexExecGrammarCollector.record_*` methods (9 total)
built `GrammarAtomV1` objects as pure data with no timestamp capture, stored
in `self._atoms`. `build_cortex_exec_grammar_events()` ran once at trace end,
computing a single flush-time value used for every atom/edge event's
`observed_at`.

## Architecture touched

`services/orion-cortex-exec/app/grammar_emit.py` only. No schema change
(`orion/schemas/grammar.py` untouched — `GrammarEventV1.observed_at` already
existed, just was populated wrong). No other service, no `.env`/compose/
registry changes.

## Files changed

- `services/orion-cortex-exec/app/grammar_emit.py`: `_atom_observed_at: dict[str, datetime]`
  added to `CortexExecGrammarCollector`, keyed by `atom.atom_id` (not the
  internal `_atoms` string key, so edge lookups — which reference `atom_id`
  — can use it directly). Populated in `_put_atom()`. Three `record_*`
  methods (`record_step_started`/`completed`/`failed`) switched from direct
  `self._atoms[key] = atom` to `self._put_atom(key, atom)` so all 9 emitters
  now route through the same capture point. `build_cortex_exec_grammar_events()`'s
  atom and edge loops use the real per-atom/per-edge-target timestamp for
  `observed_at` only.
- `services/orion-cortex-exec/tests/test_exec_grammar_emit.py`: 5 new tests
  — distinct real `observed_at` per atom in recording order, `emitted_at`
  stays uniformly shared, edge `observed_at` matches its target atom's real
  time, and a defensive fallback test for an atom that bypasses `_put_atom`.
- `docs/superpowers/specs/2026-07-14-cortex-exec-grammar-atom-wall-clock-spec.md`
  (new): full design spec, corrected during implementation (see Review
  findings below) and updated post-review with the confirmed harness sibling
  bug.

## Schema / bus / API changes

None. `GrammarEventV1`/`GrammarAtomV1` (`orion/schemas/grammar.py`) untouched
— this patch only changes which values get written into an already-existing
field.

## Env/config changes

None.

## Tests run

```text
.venv/bin/python -m pytest services/orion-cortex-exec/tests/test_exec_grammar_emit.py services/orion-cortex-exec/tests/test_exec_grammar_publish_fail_open.py -q
=> 13 passed

.venv/bin/python -m pytest services/orion-cortex-exec/tests -q
=> 12 pre-existing collection errors ("Verb already registered"), confirmed
   identical on a clean main checkout (verified via a fresh clone) --
   unrelated to this patch, a pre-existing test-isolation issue when the
   whole tests/ directory collects together.
```

## Evals run

None applicable — bug fix + tests, no model/training component. The real
"eval" is the live check below.

## Docker/build/smoke checks

Not deployed this session. No runtime/config changes — a code-only fix
inside `orion-cortex-exec`'s existing Python path, picked up on next
container rebuild/restart.

## Review findings fixed

8-angle code review run against the diff (correctness x3, reuse,
simplification, efficiency, altitude, CLAUDE.md conventions). 7 of 8 angles
returned zero findings. The altitude angle found one real, actionable gap —
not a bug in this diff, but a spec-accuracy issue:

- Finding (altitude angle): the spec's "Missing questions" section framed
  "do sibling collectors share this bug?" as an open question, when direct
  code reading during review confirmed `HarnessGrammarCollector`
  (`orion/harness/grammar_emit.py:92-94,400-417`) has the **identical** bug
  — not even the old flush-time-only mitigation cortex-exec had, its
  `_put_atom` never captures any timestamp at all.
  - Fix: spec updated to state this as confirmed, not speculative; tracked
    as a near-term follow-up in project memory
    (`project_grammar_collector_shared_timestamp_bug_siblings`), explicitly
    out of scope for this patch (no shared base class exists across the
    three sibling collectors to generalize into without a separate
    cross-service architecture decision).
  - Evidence: `docs/superpowers/specs/2026-07-14-cortex-exec-grammar-atom-wall-clock-spec.md`'s
    "Missing questions" section, corrected post-review.

Two low-severity nits from the simplification angle (parallel `_atoms`/
`_atom_observed_at` dicts keyed differently; the `.get(id, observed_at)`
fallback inlined twice rather than a helper) were explicitly assessed as
"a wash" / not worth the abstraction for two call sites — not changed.

## Restart required

```text
No restart required this session (not deployed). Once merged, orion-cortex-exec
needs a normal rebuild/restart to pick up the code change:
docker compose --env-file .env --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: low
- Concern: `HarnessGrammarCollector` has the same bug, confirmed, not fixed here.
- Mitigation: tracked explicitly in project memory as a near-term follow-up,
  not silently deferred; documented in the spec doc's corrected "Missing
  questions" section.

## PR link

`gh` is unauthenticated in this environment — open manually at:
https://github.com/junebug-junie/Orion-Sapienform/pull/new/docs/grammar-atom-wall-clock-spec
