# cortex-exec grammar-atom real wall-clock timing — spec

Status: design only, nothing in this document is implemented.

## Arsonist summary

`grammar_events` (Postgres, `orion-sql-writer`) holds 5,067,616 real rows across
30 days — 679,059 from `orion-cortex-exec` alone (293K atoms, 259K edges, 63K
trace start/end pairs), causally chained via `parent_event_id`/`root_event_id`,
semantically labeled (`atom_type`, `semantic_role`, `salience`, `confidence`).
This is the strongest raw-cognitive-signal substrate found this session —
categorically richer than anything built from hardware telemetry today. But
every atom inside one execution trace currently shares a single flush-time
timestamp: querying real data, median intra-trace duration (`max(emitted_at) -
min(emitted_at)` per `trace_id`) across 35,994 real cortex-exec traces is
**0.00 seconds**. The timing is fake. Real per-step timing already exists in
memory during execution (`executor.py`'s `t0 = time.time()` / `latency_ms`
pattern, confirmed at `executor.py:2394` and 9 other call sites) — it's
measured, then thrown away before it reaches the grammar-event record. This
spec fixes that, and only that.

## Current architecture

- `orion/schemas/grammar.py`: `GrammarAtomV1` (`:91-109`) has **no timestamp
  field at all**. `GrammarEventV1` (`:176-197`) wraps an atom and already has
  `emitted_at: datetime` (required) and `observed_at: datetime | None`.
- `services/orion-cortex-exec/app/grammar_emit.py`:
  - `CortexExecGrammarCollector` (dataclass, `:47-58`) is instantiated once per
    plan-execution trace, at real trace-start time
    (`new_cortex_exec_collector()`, `observed_at=datetime.now(timezone.utc)`
    at `:476`).
  - Its `record_step_started`/`record_step_completed`/`record_step_failed`/
    `record_result_assembled`/`record_request_received`/`record_plan_started`/
    `record_recall_gate_observed`/`record_result_emitted` methods (`:185-360`
    roughly) build `GrammarAtomV1` objects as pure data and store them in
    `self._atoms: dict[str, GrammarAtomV1]`, keyed by a stable string. **None
    of these methods call `datetime.now()` or capture any timing.**
  - `record_step_completed` (`:226-258`) receives `latency_ms: int | None` as
    a parameter — a real duration, already measured by the caller — but it
    only ever lands in the atom's free-text `summary` string
    (`f"...latency_ms={latency_ms or 0}..."`), never a structured field.
  - `build_cortex_exec_grammar_events()` (`:483-570`) runs once, at the end of
    the trace: computes a single `emitted_at = datetime.now(timezone.utc)`
    (`:487`) and wraps every accumulated atom into a `GrammarEventV1` using
    that SAME `emitted_at`/`observed_at` pair for all of them (loop at
    `:506-523`). Only `trace_ended` gets its own fresh `datetime.now()`
    (`:558`) — everything else in between, i.e. all the real step-level
    activity, is flush-time-identical.
- `services/orion-cortex-exec/app/executor.py`: real per-step timing already
  exists here — `latency_ms=int((time.time() - t0) * 1000)` appears at 10
  call sites (`:2394,2419,2619,2727,2760,2775,2788,4332,4487,4503`), meaning
  `t0 = time.time()` is captured at real step-start moments elsewhere in this
  file. This is the source of truth `record_step_completed`'s `latency_ms`
  parameter already draws from.

## Missing questions

- Exact call sites where `t0` itself gets set, and whether threading the real
  `t0`-based start time through to `record_step_started` (in addition to
  `record_step_completed` already getting `latency_ms`) is worth the extra
  plumbing versus just capturing a fresh `datetime.now(timezone.utc)` at each
  `record_*` call. The gap between "the executor's real event" and "the
  `record_*` call synchronously following it" should be sub-millisecond
  either way — a fresh capture at the `record_*` site is almost certainly
  precise enough and is far less invasive (no `executor.py` changes needed).
  Default position below: fresh capture, not threaded `t0`. Revisit only if
  the acceptance checks show a real discrepancy.
- ~~Do the sibling grammar-event producers... share this exact
  flush-batching pattern?~~ **Resolved during code review (2026-07-14),
  confirmed not hypothetical**: `HarnessGrammarCollector`
  (`orion/harness/grammar_emit.py:92-94,400-417`) has the identical bug —
  its `_put_atom` never captures a timestamp at all, and
  `build_harness_grammar_events` wraps every atom with one
  collector-construction-time `observed_at`. Given harness governs the
  unified-turn motor path (real LLM/tool-step latencies, same stakes as
  cortex-exec), this is a real, confirmed, near-term follow-up, not a maybe.
  `BusTransportGrammarCollector` (`services/orion-bus/app/grammar_emit.py:295-309`)
  has the same code shape but lower practical stakes (its atoms span one
  observer tick, plausibly genuinely sub-millisecond). Still out of scope
  for THIS patch — no shared base class exists to generalize into without
  first making an undiscussed cross-service architecture decision (these
  three collectors are independently duplicated peer code across service
  boundaries, not one mechanism with a special case bolted on) — but no
  longer an open question, and harness specifically should be scheduled as
  a near-term follow-up patch, not left indefinitely deferred.
- Existing test coverage for `grammar_emit.py` — needs to be located and read
  before writing the acceptance-check tests below, so new tests don't
  duplicate or fight existing ones asserting on the (currently fake) shared
  timestamp behavior.

## Proposed schema / API changes

**No `orion/schemas/grammar.py` change.** `GrammarEventV1.emitted_at`/
`observed_at` already exist as per-event fields — they are simply populated
with one broadcast value today instead of real per-atom values. This is
smaller blast radius than the version of this idea first floated in-session
(adding a field to `GrammarAtomV1`) — no schema migration, no backward-compat
concern for existing rows, nothing new to add to a registry.

Concrete changes, all inside `services/orion-cortex-exec/app/grammar_emit.py`:

1. `CortexExecGrammarCollector`: add
   `_atom_observed_at: dict[str, datetime] = field(default_factory=dict)`,
   parallel to the existing `_atoms` dict, keyed identically.
2. Every `record_*` method: add one line at the point the atom is stored —
   `self._atom_observed_at[key] = datetime.now(timezone.utc)`.
3. `build_cortex_exec_grammar_events()`'s atom-wrapping loop (`:506-523`):
   set **`observed_at` only** to `collector._atom_observed_at.get(atom.atom_id,
   observed_at)` for each atom's event — keep `emitted_at` as the shared
   flush-time value, unchanged. Refined during implementation: `emitted_at`
   was never actually wrong — every event genuinely is published to the bus
   in the same flush batch, so a shared `emitted_at` is honest. `observed_at`
   is the field meant to carry "when did this really happen," and that's the
   one that was broken (silently defaulting to the same flush-time value).
   This mirrors the existing, already-correct `trace_started` pattern
   (`emitted_at`=flush time, `observed_at`=real trace-start) rather than
   inventing a new convention. Key by `atom.atom_id` (not the internal
   `_atoms` string key) so edge lookups, which reference `atom_id`, can use
   the same dict directly.
4. `edge_emitted` events (`:525-552`): same `observed_at`-only treatment,
   using the **target** atom's (`to_atom_id`) real captured time — an edge
   semantically becomes true when its second endpoint happens.
5. `trace_started`/`trace_ended`: unchanged — already correctly distinct
   (trace-start and flush-completion moments respectively).

## Files likely to touch

- `services/orion-cortex-exec/app/grammar_emit.py` — all real work.
- `services/orion-cortex-exec/tests/` (exact file TBD, see missing questions)
  — likely needs new/updated assertions; existing tests may currently assert
  the shared-timestamp behavior and need correcting, not just extending.
- Nothing else. No `orion/schemas/`, no other service, no `.env`/compose/
  registry changes. Contained entirely within `orion-cortex-exec`.

## Non-goals

- Not touching the other grammar-event producers (hub/biometrics/route/
  transport-loop) in this patch.
- Not building the actual trajectory corpus/model over `grammar_events` yet —
  this patch only makes the underlying timing real; extracting/windowing a
  corpus from it is separate, future work.
- Not changing `edge_emitted`'s `evidence_event_ids`/`relation_type`
  semantics beyond the timestamp fix.
- **Explicitly out of scope but tracked, not forgotten** (see
  `project_biometrics_channel_defects_to_fix` in this session's memory):
  the `expected_offline_suppression` one-way-ratchet bug, the three
  folded-into-`strain` dead channels (`thermal_pressure`/`memory_pressure`/
  `disk_pressure`), and the `cpu_pressure`/`gpu_pressure` accumulator
  oscillation on the non-cognitive/biometrics side of this same roadmap.

## Acceptance checks

- Unit test: run a synthetic multi-step plan through the collector, assert
  distinct `record_*` calls produce events with strictly increasing (or at
  minimum non-identical) `emitted_at` values — not one shared timestamp.
- Unit test: assert ordering — `trace_started.emitted_at <=
  atom_emitted[0].emitted_at <= ... <= trace_ended.emitted_at`.
- Unit test: `edge_emitted` events use their target atom's real timestamp,
  not the trace-start fallback, for any edge whose target atom was actually
  recorded (fallback path only exercised for a genuinely missing key).
- Live check (post-deploy): query real `grammar_events` for `orion-cortex-exec`
  traces with `created_at` after deploy, confirm `max(observed_at) -
  min(observed_at)` per `trace_id` is now nonzero and plausible (roughly
  matching real step latencies visible in the atom's own `summary` text, not
  0.00s and not absurdly large). `emitted_at` is expected to stay 0-spread —
  that's correct, unchanged, honest flush-time behavior, not a regression.
- Full existing `services/orion-cortex-exec/tests` suite still passes.

## Recommended next patch

Implement exactly the "Proposed schema / API changes" section, scoped to
`services/orion-cortex-exec/app/grammar_emit.py` and its test file only.
Resolve the `t0`-threading missing question by defaulting to a fresh
`datetime.now(timezone.utc)` capture at each `record_*` call site unless the
acceptance checks reveal a real precision problem.
