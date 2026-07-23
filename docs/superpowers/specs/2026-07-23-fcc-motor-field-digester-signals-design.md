# FCC-motor signals for orion-field-digester — design

Status: **proposed, design-only**. No code in this patch. Two implementation patches are
scoped below (Patch A: 4 new signals sharing one producer chain; Patch B: turn-incompletion
liveness signal) plus an appendix of findings on existing broken/crude channels that are
explicitly **out of scope** for whoever implements Patch A/B — those are handed off separately.

**Cross-reference:** Appendix item 3 below (`transport_pressure`/`bus_health` scope dishonesty)
overlaps with `docs/superpowers/specs/2026-07-22-transport-bus-signal-quality-measurement-design.md`
(PR #1278) — that doc is the fuller writeup of the same finding from a different investigation
thread; this doc's appendix entry is a condensed cross-reference, not a duplicate effort.

---

## Arsonist summary

Orion's field-digester has a "thinking/pressure" channel category (`reasoning_load`,
`execution_load`, `execution_friction`, `failure_pressure`, `repair_pressure`) but none of it
describes what the FCC motor (the harness/agent motor driving Orion's unified Hub chat turn)
actually did during a turn — how much real step-work it did, whether it got stuck repeating
the same tool failure, how verbose its tool output was, or whether the turn even completed.
The raw substrate for most of this already exists in the grammar-event stream produced by
`orion/harness/grammar_emit.py` and consumed by
`orion/substrate/execution_loop/grammar_extract.py`, either unused (`HarnessRunV1.step_count`,
`.compliance_verdict`) or genuinely absent (per-step tool-failure/verbosity data). One raw
event type (`harness_fcc_step`) that looked like the substrate for the failure-repetition
signal is explicitly discarded by the reducer — but tracing it further showed a *better*
substrate already flows through an existing, already-parsed event (`tool_result.is_error`
blocks inside step summaries), avoiding the need to un-discard the high-volume stream at all.

## Current architecture

Unified Hub turn: `orion/hub/turn_orchestrator.py` builds `HarnessRunRequestV1` and RPCs it to
`services/orion-harness-governor`, which runs the FCC motor (`orion/harness/fcc_motor.py` →
`orion/harness/runner.py`) and returns `HarnessRunV1` — already carrying `step_count` and
`compliance_verdict` (`completed`/`partial`/`failed`/`refused`), neither of which reaches any
downstream signal today. The governor also emits `GrammarEventV1` atoms via
`HarnessGrammarCollector`, buffered in memory and flushed once per run
(`runner.py:343-370`, `_publish_motor_lifecycle`). These land in
`grammar_extract.py::extract_execution_state_from_events()`, scoped to
`EXECUTION_SOURCE_SERVICES = {"orion-cortex-exec", "orion-harness-governor"}`
(`orion/substrate/execution_loop/constants.py`), building `ExecutionRunStateV1`, merged across
batches by `orion/substrate/execution_loop/merge.py` via `max()` on counters (never sum — this
is why the well-known `evidence_event_ids`-unbounded-list bug class, already hit three times in
this codebase, does not threaten new *scalar* counter fields, only new list-valued ones).
`compute_pressure_hints()` turns run state into a dict consumed by
`services/orion-field-digester/app/ingest/state_deltas.py`, mapped onto
`FieldStateV1.node_vectors` keys defined in
`services/orion-field-digester/app/tensor/channels.py::NODE_CHANNELS`. A channel only decays if
also listed in `app/digestion/decay.py::NODE_DECAY_CHANNELS`; channels written through the
normal `state_deltas.py` → `apply_perturbations()` path get `node_vector_updated_at` stamped
automatically (manual stamping is only a concern for bypass writes, e.g. `worker.py`'s
`field_coherence_warning`).

`compliance_verdict` and per-step char/tool-failure data are confirmed genuinely absent from
the grammar stream today (zero grep hits — not mis-parsed, never emitted). `repair_pressure`
(an existing channel) is confirmed unrelated to any of this — it's conversational/relational
repair pressure from a separate `chat_loop`/`repair_pressure_v2` pipeline gated on
`delta.target_kind == "chat_turn"`, not `"execution_run"`, so none of the new channels below
duplicate it.

`NODE_CHANNELS` additions are not covered by any automated contract-check gate —
`scripts/check_schema_registry.py`/`check_bus_channels.py` (cited in root `CLAUDE.md` §6) do
not exist in this repo (`Makefile` has an explicit note acknowledging this). This is a
lighter-weight, internal-schema-only change with no mechanical enforcement — documentation
discipline (README channel glossary + this spec doc) is the real gate here.

## Missing questions

- Does `short_error_kind()` (`orion/harness/grammar_emit.py`) bucket realistic tool-result
  error text (e.g. "Permission to use Bash has been denied") stably and distinctly from
  unrelated failures? Needs a live-data sample before finalizing the streak threshold in
  Patch A signal 2.
- What's a realistic `avg_step_chars_pressure` saturation point? The `4000`-char anchor below
  is a starting guess, not measured.
- Do existing downstream diffusion/coherence consumers of `NODE_CHANNELS` assume every channel
  is bounded to `[0,1]`? `harness_step_load` (Patch A signal 1) is `log1p`-scaled and NOT
  naturally bounded — needs confirming before shipping unclamped.
- Should Patch A and Patch B ship as one PR or two? Recommended: two (different risk profile —
  Patch B widens a producer allow-list, a real contract-surface touch; Patch A does not).

## Proposed schema / API changes

### Patch A — signals 1–4, one coordinated diff through the shared producer chain

**1. `harness_step_load`** — FCC-motor-only step count, split out of the existing blended
`execution_load` (currently `min(1.0, started/8.0)` over BOTH cortex-exec and harness-governor
steps, `grammar_extract.py:39`). Add `ExecutionRunStateV1.harness_started_step_count`,
incremented in the existing `exec_step_started` branch only when
`event.provenance.source_service == "orion-harness-governor"` (already dereferenced in the same
loop) — no new grammar-emit role, no producer-side change, one added line. Transform:
`log1p(harness_started_step_count)`, matching the existing `log1p(thinking_tokens_sum)`
convention in `services/orion-athena-spark-introspector/app/inner_state.py`. Leave
`execution_load` itself untouched in this patch (see Appendix item 1 for why that's a
deliberate deferral, not an oversight).

**2. `tool_failure_streak_pressure`** — repeated-identical-tool-failure detector. New helper
`_extract_tool_result_errors(step)` in `orion/harness/fcc_motor.py` (mirroring the existing
`_extract_tool_name`), returning error text for each `tool_result` content block where
`block.get("is_error")` is true — confirmed live (`fcc_motor.py:185-190`,
`_summarize_content_blocks`) that this flag is already computed per-block on every step,
independent of the once-per-turn `exec_step_failed`/subprocess-error branch that looked like
the obvious substrate but isn't (that branch only fires on FCC-subprocess-level failure, not
per individual tool-call denial — confirmed via direct read of `runner.py:302-326`, which is
inside an `elif etype == "error":` branch that immediately breaks the stream loop, structurally
separate from the `if etype == "step":` branch where per-tool-call `tool_result` blocks
actually live). In `runner.py`'s step loop, bucket each error via the existing
`short_error_kind()`; track two bounded scalars only —
`tool_failure_streak` (resets on a different bucket) and `tool_failure_streak_max` — never a
growing list. Transform: `min(1.0, streak_max / 3.0)`.

**3. `avg_step_chars_pressure`** — per-step verbosity. `measure_step_payload_chars()`
(`orion/fcc/context_budget.py`, already computed in `fcc_motor.py:589` for context-overflow
budgeting) summed (`step_char_sum`) and max-tracked (`step_char_max`) locally in `runner.py`'s
step loop. Average computed at read time in `compute_pressure_hints()` as
`step_char_sum / max(1, completed_step_count)`, reusing the already-merged
`completed_step_count` denominator instead of a second counter that could desync under
independent `max()` merging.

**4. `compliance_deficit`** — `runner.py` already computes a local `compliance_verdict`;
thread it into the existing `record_result_assembled()` call as a new kv. Rank
`{unknown:0, completed:0, partial:1, failed:2, refused:3} / 3.0`. Kept as its own channel, not
folded into `failure_pressure` — a turn can resolve `partial` (context overflow, empty draft)
with zero individual tool failures, so the two measure genuinely different things. Note:
`"refused"` will never actually reach this pipeline in practice (set by a caller that
short-circuits before the FCC motor ever runs) — document as a known coverage gap, not a bug.

**Files:** `orion/harness/runner.py`, `orion/harness/fcc_motor.py`,
`orion/harness/grammar_emit.py`, `orion/schemas/execution_projection.py` (5 new scalar fields,
no new list-valued fields), `orion/substrate/execution_loop/grammar_extract.py` (+`import math`),
`orion/substrate/execution_loop/merge.py` (max()/rank-max merge, matching the existing
`_pick_status()` pattern), `services/orion-field-digester/app/ingest/state_deltas.py` (4 new
tuples in the `execution_run` mapping block, attributed to `node_key` not
`reasoning_node_key`), `services/orion-field-digester/app/tensor/channels.py` (4 new
`NODE_CHANNELS` entries — no `CAPABILITY_CHANNELS`/diffusion wiring yet, an explicit documented
follow-up), `services/orion-field-digester/app/digestion/decay.py` (same 4 names added to
`NODE_DECAY_CHANNELS`).

**Commit sequencing:** (1) schema + merge + grammar_extract parsing/pressure_hints, lands inert
with tests; (2) runner.py + grammar_emit.py producer changes, kv actually flows; (3)
field-digester wiring, channels go live; (4) docs.

### Patch B — signal 5: turn incompletion (`harness_rpc_timeout`)

Confirmed the hard case: if the governor RPC never returns, **zero bus-observable trace
exists** before that point today — `record_request_received()`/`record_plan_started()` are
only buffered in-memory, flushed at the end of a run that in this failure mode never happens.
Cheapest real fix: Hub already knows exactly when it gives up
(`orion/hub/turn_orchestrator.py`, the `if run is None:` branch, ~line 462-471,
`correlation_id` in hand) — make it publish a real grammar event there instead of only
returning a WS error frame to the client.

One genuine contract-surface touch: the timeout atom needs its own trace lane under Hub's own
node identity (`cortex_exec_trace_id(hub_node_name, correlation_id, lane="hub_turn_timeout")`,
reusing the existing lane-isolation helper from `orion/substrate/execution_loop/ids.py`)
because Hub cannot reliably know the governor's own `NODE_NAME` when the RPC never returned.
This requires widening `EXECUTION_SOURCE_SERVICES` (`constants.py`) to include `"orion-hub"` —
flag explicitly in review since it's a producer-allow-list widening, not purely additive.

**Files:** `orion/substrate/execution_loop/constants.py` (widen `EXECUTION_SOURCE_SERVICES`),
`services/orion-hub/scripts/grammar_emit.py` (new `build_turn_timeout_grammar_events()`,
mirroring `build_chat_turn_grammar_events`, `semantic_role="exec_turn_timeout"`),
`orion/hub/turn_orchestrator.py` (new `_publish_turn_timeout_grammar()` helper next to the
existing `_publish_unified_turn_chat_grammar`, same fail-open/try-except/settings-gated shape,
called from the `run is None` branch), `orion/schemas/execution_projection.py` (new
`turn_timed_out: bool` field), `grammar_extract.py` (new `exec_turn_timeout` role branch, new
`turn_incompletion` pressure-hint), `merge.py` (OR-merge, matching the existing
`recall_observed` pattern), field-digester wiring (same pattern as Patch A, new
`turn_incompletion` channel, attributed to Hub's own node — document that this measures "Hub's
view of governor unresponsiveness," not the governor's own node health, since they may run on
different physical nodes).

**Commit sequencing:** (1) constants + schema/merge/extract wiring with reducer tests; (2)
Hub-side builder + orchestrator publish call with Hub tests; (3) field-digester channel wiring;
(4) docs.

## Files likely to touch

`orion/harness/runner.py`, `orion/harness/fcc_motor.py`, `orion/harness/grammar_emit.py`,
`orion/schemas/execution_projection.py`, `orion/substrate/execution_loop/grammar_extract.py`,
`orion/substrate/execution_loop/merge.py`, `orion/substrate/execution_loop/constants.py`,
`orion/hub/turn_orchestrator.py`, `services/orion-hub/scripts/grammar_emit.py`,
`services/orion-field-digester/app/ingest/state_deltas.py`,
`services/orion-field-digester/app/tensor/channels.py`,
`services/orion-field-digester/app/digestion/decay.py`,
`services/orion-field-digester/README.md` (channel glossary).

## Non-goals

- Not touching `execution_load`, `reasoning_load`, or `transport_pressure`/`bus_health`/
  `delivery_confidence` — see Appendix, explicitly deferred to a separate follow-up effort.
- Not adding `CAPABILITY_CHANNELS`/diffusion-edge wiring for the 5 new channels in this patch —
  they land readable via `node_vectors` first; diffusion targets are a documented follow-up
  decision, not silently defaulted.
- Not building a live/mid-run version of the tool-failure streak (per-step emission instead of
  end-of-run flush) — this codebase's single-flush-per-run batching makes "streak at flush
  time" and "max streak" the same observation; a live/mid-run streak is a genuinely different
  producer shape, future work if ever needed.
- Not fixing `compliance_verdict="refused"` never reaching the pipeline — documented gap, not
  in scope.

## Acceptance checks

- `pytest orion/harness/tests -q`, `pytest tests/test_execution_substrate_reducer.py -q`,
  `pytest services/orion-field-digester/tests -q` all pass, extended with new cases per signal
  (streak counting, char accumulation, compliance threading, source-service-scoped step
  counting, turn-timeout event handling).
- Live sanity check before calling either patch done: deploy, run a real Orion-mode Hub turn
  that includes at least one tool-call failure, and read field-digester's live `node_vectors`
  to confirm all 5 new channels are non-degenerate (not flat/always-zero) — this repo's own
  metric-quality-gate "live-data sanity check" requirement, not optional polish.
- Code-review skill run in a subagent on both patches, material findings fixed, before either
  is called done.

## Recommended next patch

Implement Patch A first (self-contained within the harness-governor → field-digester chain, no
contract-surface widening). Patch B follows once Patch A's field-digester wiring pattern is
live and confirmed working, since Patch B reuses that exact same wiring shape for its one new
channel.

---

## Appendix: known-broken existing signals — findings only, NOT in scope for Patch A/B

Documented here for a separate follow-up effort to pick up. Nothing in this appendix is
touched by Patch A or Patch B.

### 1. `execution_load` — blended, hard-capped, and (after Patch A) partially redundant

**File:line:** `orion/substrate/execution_loop/grammar_extract.py:39`,
`execution_load = min(1.0, started / 8.0)`.

**Why it's broken:** Two separate problems.
- **Hard cap discards magnitude.** Any run past 8 started steps reads identically to one at
  exactly 8 — a 40-step run and an 8-step run both saturate to `1.0`.
- **Blended producers.** `started` counts `exec_step_started` events from BOTH
  `orion-cortex-exec` and `orion-harness-governor` into one undifferentiated counter, despite
  them being structurally different processes.
- **New wrinkle from Patch A:** Patch A adds `harness_step_load` as a *sibling* channel without
  narrowing `execution_load` itself — after Patch A ships, `execution_load` still
  double-counts harness-governor steps that `harness_step_load` now also measures separately.
  Left untouched in Patch A only because its existing downstream diffusion-edge consumers
  (`services/orion-field-digester/README.md:798-808`) were out of scope to re-verify there —
  that makes this a real follow-up, not a resolved concern.

**Likely fix shape to evaluate (not decided here):** narrow `execution_load` to
cortex-exec-only steps (true complement to `harness_step_load`), or retire it entirely if
nothing downstream needs a cortex-exec-specific magnitude once `harness_step_load` exists.
Requires auditing every consumer (diffusion edges, self-state-adjacent policy YAMLs) first.

### 2. `reasoning_load` — a boolean wearing a magnitude's name

**File:line:** `orion/substrate/execution_loop/grammar_extract.py:41`,
`reasoning_load = 0.35 if run.reasoning_present else 0.05`.

**Why it's broken:** This is the signal that prompted the original conversation behind this
whole spec ("we rarely run any thinking models... what about a proxy for thinking"). It is not
a magnitude — a two-valued flag derived from `reasoning_present`, itself a boolean parsed from
`exec_result_assembled`'s `reasoning_present=` kv (`grammar_extract.py:136`). Every turn that
used any reasoning at all reads identically, regardless of how much. Despite this, it's mapped
to the `reasoning_pressure` capability channel via `config/field/orion_field_topology.v1.yaml`,
`config/field/biometrics_lattice.yaml`, and (now-partially-deleted per the 2026-07-22
SelfStateV1 removal) `config/self_state/self_state_policy.v1.yaml`. Separately,
`services/orion-athena-spark-introspector/app/inner_state.py:309` already computes a real
magnitude for a differently-scoped reasoning signal —
`reasoning_load_raw = log1p(thinking_tokens_sum)` — proving the raw ingredient (actual token
counts) is available elsewhere in the codebase; this near-boolean version simply never adopted
it.

**Likely fix shape to evaluate (not decided here):** replace with a real magnitude sourced
from actual reasoning/thinking token counts (following the `inner_state.py` `log1p`
precedent) if that data can be threaded into the execution_loop pipeline; otherwise rename to
honestly reflect that it's presence-only, not a magnitude.

### 3. `transport_pressure` / `bus_health` / `delivery_confidence` — scope dishonesty

**File:line:** `orion/substrate/transport_loop/extract.py::compute_transport_pressures()`, fed
exclusively by `TransportBusStateV1`, fed exclusively by `BUS_OBSERVER_STREAMS`.

**Why it's broken:** Full writeup in
`docs/superpowers/specs/2026-07-22-transport-bus-signal-quality-measurement-design.md`
(PR #1278) — condensed here per the same "reuse existing signal" honesty problem this spec's
own Missing Questions surfaced. Redis Streams (the only data type with an `XLEN`/backlog
concept) exist exactly once in the entire architecture: world_pulse's result queue + DLQ.
Every other service's real traffic runs over pub/sub, which has no depth primitive. So these
channels structurally can only ever reflect "is world_pulse's one queue backing up" — never
general cross-service bus health, no matter how correctly wired. The name ("transport domain,"
"bus health") implies far broader coverage than the channel can ever measure.

**Likely fix shape to evaluate (not decided here):** either invent a genuine depth/backlog
proxy for pub/sub channels, or rename/redocument honestly as "world_pulse queue health" and
treat general cross-service transport stress as a genuinely unmeasured gap.
