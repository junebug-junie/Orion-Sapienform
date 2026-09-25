# Route lane field digestion — metric gate result (no channel built)

## Summary

- Ran the CLAUDE.md §0A metric quality gate on every candidate pressure channel the route
  lane (`orch.route:` traces, `route_arbitration_reducer`) could feed into the field.
- **Every candidate failed.** No `pressure_hints`, no `route_arbitration_run` digester branch,
  no channel declaration, no topology edge were built. This PR ships only this gate record.
- `topic_coherence` (chat lane): **not wired** (fails the independence check). **Not deleted in
  this PR either**: it has a live consumer, and deleting it changes a live metric's definition.
  That is Juniper's call; options below.
- Two side findings recorded as follow-ups: the route atom's uncertainty rule checks for a lane
  reason string the router never emits, and the existing `node:substrate.route`
  prediction-error reading has sat at exactly 0.0003 for at least 8 hours.

## Outcome moved

A decision, backed by live numbers: the route lane has nothing in it today that can
tell "orchestration is struggling" apart from "orchestration is fine". Feeding it into
`capability:orchestration` would have put a constant into the field. The next person who
tries this starts from these numbers.

## Current architecture

- Producer: `services/orion-cortex-orch/app/orchestrator.py:640-690` builds `route_metadata`
  and `services/orion-cortex-orch/app/grammar_emit.py` serializes it as one
  `route_arbitration_decided` atom per call. The atom's summary carries exactly five
  categorical keys: `lane`, `lane_reason`, `mind_requested`, `mind_skip_reason`,
  `output_mode`. There is **no** latency, retry, timeout, degraded-route or outcome field.
- The lane is picked by `services/orion-cortex-orch/app/execution_lanes.py::resolve_execution_lane`,
  a fixed lookup on verb and mode. It is a classification of *what work arrived*, not a
  measurement of *how orchestration handled it*.
- Reducer: `orion/substrate/route_loop/reducer.py` emits
  `StateDeltaV1(target_kind="route_arbitration_run")` with raw `merged.model_dump()` in `after`.
- Field: `services/orion-field-digester/app/ingest/state_deltas.py::delta_to_perturbations`
  has no `route_arbitration_run` branch. Route reaches the field only as
  `node:substrate.route` `prediction_error` (`orion/substrate/prediction_error.py::route_prediction_error`,
  a categorical mismatch rate between successive runs).
- `capability:orchestration` is fed by `node:athena` (`cpu_pressure`, `cortex_exec_step_load`,
  `execution_friction`, `failure_pressure`, `stream_backlog_pressure`) plus
  `capability:transport` and `capability:llm_inference` (`config/field/orion_field_topology.v1.yaml:176-243`).

## Live data (pulled 2026-09-25 ~00:10 UTC)

`grammar_events` where `source_service='orion-cortex-orch'`: retention reaches back to
2026-09-22 00:10, so this is **all retained history**: 5,680 atoms (4,832 / 4,872 / 1,658 events
per day on 09-22/23/24; volume fell through 09-24, from ~200/h to ~20/h, not investigated here).

| lane | lane_reason | mind_requested | output_mode | atom uncertainty | count |
|---|---|---|---|---|---|
| background | verb_background | false | direct_answer | 0.42 | 5,663 |
| chat | mode_chat | false | direct_answer | 0.42 | 16 |
| background | verb_background | false | code_delivery | 0.42 | 1 |

- 99.7% of atoms are byte-identical in every decision field.
- `mind_skip_reason = mind_enabled_not_true` on 5,680 / 5,680.
- `lane_reason` in {`fallback_background`, `lane_routing_disabled`, `unknown`, `explicit_options`}: 0 / 5,680.
- `verb_chat` (ordinary Hub chat): 0 / 5,680. The route lane is mostly observing background verbs.
- Live projection `substrate_route_arbitration_projection`: 822 runs, 815 identical
  (`background/verb_background/direct_answer/false`).
- `node:substrate.route.prediction_error` from `substrate_field_state` over the last 24h:
  40,324 ticks, min 0.0001, max 0.0006, median 0.0003, zero ticks above 0.01. For the last
  8 hours, every hour held exactly one distinct value: 0.0003.
- `capability:orchestration` right now: `pressure` 0.359, `execution_pressure` 0.241,
  `reasoning_pressure` 0.9, `reliability_pressure` 0.0.

## Metric quality gate, per candidate channel

### 1. `route_fallback_pressure`: share of calls the router could not classify (`fallback_background` / `lane_routing_disabled` / `unknown`)

1. Provenance: `execution_lanes.py:66` (`fallback_background`), `orchestrator.py:517`
   (`lane_routing_disabled`), `orchestrator.py:647,655` (`unknown` on exception).
2. Independence: independent of `node:substrate.route` PE (that one measures change between
   runs, not the fallback level) and of the execution channels. Passes.
3. Theory anchor: weak. A fallback means a verb arrived that the lookup table does not know.
   That is a gap in the code's coverage, and it shows up when code is deployed, not when load
   changes. It says nothing about orchestration pressure at run time. **Fails**: a static test
   ("every registered verb resolves to a non-fallback lane") would catch this properly.
4. Live data: 0 / 5,680. It has never read anything other than calm, so there is no evidence it can.
   **Fails.** It is exactly the "suspiciously clean 0.0" case §0A warns about.
5. Existing mechanism: `execution_lane_fallback_background` debug log only.
6. Reversibility: cheap, but nothing to reverse.

**Verdict: FAIL (3, 4).**

### 2. `mind_skip_pressure`: share of calls where mind escalation was skipped

1. Provenance: `orchestrator.py:635` sets `mind_skip_reason="mind_enabled_not_true"`.
2. Independence: this is a config flag's echo, not state.
3. Theory anchor: none. It reads "the mind is switched off in config", not "orchestration had to shed escalation".
4. Live data: 5,680 / 5,680 = 1.0, **saturated**.

**Verdict: FAIL (3, 4).**

### 3. Route atom `uncertainty`

1. Provenance: `orion/grammar/atom_signals.py:64-74::uncertainty_from_route_arbitration`.
2. Independence: a deterministic function of candidates 1 and 2 above, so not independent.
3. Theory anchor: none beyond those two.
4. Live data: 0.42 on 5,680 / 5,680 (the `mind_skip_reason` floor), **flat**.
5. Side bug: line 72 checks `lane_reason in {"unknown", "fallback", "lane_routing_disabled"}`,
   but the router emits `fallback_background`, never `fallback`. So a real fallback would not
   raise uncertainty. Follow-up, not fixed here (see Risks).

**Verdict: FAIL (2, 4).**

### 4. Lane mix / route volume (background share, arbitrations per minute)

1. Provenance: count of `route_arbitration_decided` atoms by `lane`.
2. Independence: **redundant**. Every non-chat orch call goes out as one PlanExecution to
   cortex-exec (`orchestrator.py:693`, `use_direct_exec`). That produces the `execution_run`
   delta that already drives `cortex_exec_step_load` / `execution_pressure` into
   `capability:orchestration`. It is the same event counted a second time.
3. Theory anchor: none. The share of each lane describes the mix of work, not pressure.
4. Live data: 99.7% background, so a share is flat as well.

**Verdict: FAIL (2, 3, 4).**

### 5. Latency / retries / degraded route / arbitration outcome

Not carried by the route atom at all. Building it means changing the emitter contract.
cortex-exec already measures run friction and failure (`execution_friction`, `failure_pressure`),
which is the likely redundancy target. **Not a candidate without a contract change.** Out of scope.

## `topic_coherence` decision

- Provenance: `orion/substrate/chat_loop/grammar_extract.py:117`, `max(0.0, 1.0 - repair_pressure_level)`.
- Independence: **fails**. It is an affine transform of `repair_pressure`, which is already a hint.
- **Not wired.** `services/orion-field-digester/tests/test_field_chat_perturbations.py::test_chat_turn_does_not_emit_topic_coherence`
  already pins that it never reaches the lattice.
- **Not deleted in this PR.** It is not an orphan: `orion/substrate/prediction_error.py:656`
  (`chat_prediction_error`) diffs all three chat hints. `node:substrate.chat` is then read by
  the attention self-model (`attention_self_model.py:128`, `ACTIVE_INFERENCE_DOMAINS` includes `chat`).
  Deleting it moves `repair_pressure`'s weight in that mean from 2/3 to 1/2. It also shifts the
  distribution the persisted EWMA baseline (`prediction_error_baseline_ewma/_var/_n`) was
  calibrated on. That is an in-place definition change to a live metric with consumers.
  Juniper's standing rule says that choice is hers.
  - Option A: delete `topic_coherence` from `compute_chat_pressure_hints`, update
    `chat_prediction_error`'s key tuple and tests, accept a transient z-score skew while the
    EWMA re-baselines. This follows the docstring's own "fix upstream" guidance.
  - Option B: keep the status quo. The double weighting is already disclosed in
    `chat_prediction_error`'s docstring and in `docs/superpowers/specs/2026-07-21-chat-route-prediction-error-shadow-design.md`.
  - Recommendation: A, as its own small PR, with a before/after replay of
    `chat_prediction_error` over a retained window.

## Architecture touched

None. Docs only.

## Files changed

- `docs/superpowers/pr-reports/2026-09-25-route-lane-field-digestion-gate-pr.md`: this gate record.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: none
- Compatibility notes: n/a

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: n/a (no template change)
- skipped keys requiring operator action: none

## Tests run

```text
No code changed. git diff --check: clean.
```

## Evals run

```text
None applicable (docs-only). The live-data pulls above are the evidence.
```

## Docker/build/smoke checks

```text
None (no runtime change). Live reads only: grammar_events, substrate_route_arbitration_projection,
substrate_field_state via docker exec orion-athena-sql-db psql (read-only SELECTs).
```

## Review findings fixed

See PR conversation; filled in after the review subagent run.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low
  - Concern: `uncertainty_from_route_arbitration` checks for `"fallback"`, but the router emits `"fallback_background"`, so a real fallback is invisible to atom uncertainty.
  - Mitigation: follow-up fix plus a regression test. Live impact today is zero (0 fallbacks in all retained history).
- Severity: medium
  - Concern: `node:substrate.route.prediction_error` is flat at 0.0003 (8h, one distinct value per hour), and route sits in `ACTIVE_INFERENCE_DOMAINS`. It is a near-constant input to the attention self-model. This is not decay: the value holds rather than falling geometrically. It is the upstream near-identity of route decisions.
  - Mitigation: follow-up to decide whether route belongs in `ACTIVE_INFERENCE_DOMAINS` while lane choice stays a static lookup.
- Severity: low
  - Concern: no ordinary Hub chat (`verb_chat`) reached the route lane in 3 days, and orch route volume fell ~10x across 09-24.
  - Mitigation: not investigated here. Flagged.

## PR link

(filled after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
