# Route lane field digestion — metric gate result (no channel built)

## Summary

- Ran the CLAUDE.md §0A metric quality gate on every candidate pressure channel the route
  lane (`orch.route:` traces, `route_arbitration_reducer`) could feed into the field.
- **Every candidate failed.** No `pressure_hints`, no `route_arbitration_run` digester branch,
  no channel declaration, no topology edge were built. This PR ships only this gate record.
- `topic_coherence` (chat lane): **not wired** (fails the independence check). **Not deleted in
  this PR either**: it has a live consumer, and deleting it changes a live metric's definition.
  That is Juniper's call; options below.
- Side findings recorded as follow-ups. The main one: the existing `node:substrate.route`
  prediction-error value in the field is a stale copy of the last reading above zero. The
  instrument behind it also cannot mathematically rise above ~0.0012. So route's one existing
  path into the field is broken too (details under Risks).

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
  `execution_friction`, `failure_pressure`, `stream_backlog_pressure`, `reasoning_load`) plus
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
  `node_vector_updated_at` shows the last write at `2026-09-24T12:11:48Z`. It is stale, not live
  (see Risks).
- `capability:orchestration` right now: `pressure` 0.359, `execution_pressure` 0.241,
  `reasoning_pressure` 0.9, `reliability_pressure` 0.0.

## Metric quality gate, per candidate channel

### 1. `route_fallback_pressure`: share of calls the router could not classify (`fallback_background` / `lane_routing_disabled` / `unknown`)

1. Provenance: `execution_lanes.py:66` (`fallback_background`), `orchestrator.py:517`
   (`lane_routing_disabled`), `orchestrator.py:647,655` (`unknown` on exception). Blind spot: when
   `EXEC_LANE_ROUTING_ENABLED=false`, line 517 overwrites *every* reason with `lane_routing_disabled`,
   so a real fallback cannot be told apart from that setting.
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
   raise uncertainty. The rule takes the higher of its two bumps, and the mind-skip floor (0.42)
   sits above the fallback bump (0.35). So fixing the string alone changes nothing while mind is
   off. Follow-up, not fixed here (see Risks).

**Verdict: FAIL (2, 4).**

### 4. Lane mix / route volume (background share, arbitrations per minute)

1. Provenance: count of `route_arbitration_decided` atoms by `lane`.
2. Independence: **redundant**. Every non-chat orch call goes out as one PlanExecution to
   cortex-exec (`orchestrator.py:693`, `use_direct_exec`). Live over 24h (from review), 817 of 823
   orch correlation IDs have matching cortex-exec grammar events. That produces the `execution_run`
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
    `chat_prediction_error`'s key tuple and tests, check `orion/inner_state_registry.py:731`, accept a transient z-score skew while the
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

Reviewer: orion-repo-agent subagent, targeted at commit 585d9aac2. It re-ran the live SELECTs and verified every code citation.

- Finding (material): the flat 0.0003 was blamed on routing near-identity. The real causes are a stale field copy (receipt write gated on `error > 0.0`) and an instrument averaged over ~824 runs, capped near 0.0012.
  - Fix: rewrote the medium risk and its follow-up. Verified myself: `worker.py:3880` gate, `node_vector_updated_at = 2026-09-24T12:11:48Z`.
  - Evidence: the Risks section above.
- Finding (minor): the uncertainty string fix is masked by the 0.42 mind-skip floor.
  - Fix: noted in candidate 3 and in Risks.
- Finding (minor): "every non-chat call reaches exec" was stated without a number.
  - Fix: added the reviewer's live 817/823 overlap.
- Finding (minor): the orchestration input list omitted `reasoning_load`.
  - Fix: added.
- Finding (minor): candidate 1 blind spot under `lane_routing_disabled`.
  - Fix: added to candidate 1.
- Finding (minor): Option A should also check `orion/inner_state_registry.py:731`.
  - Fix: added.
- Gate verdicts and the topic_coherence decision were confirmed by the reviewer. No missed candidate was found.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low
  - Concern: `uncertainty_from_route_arbitration` checks for `"fallback"`, but the router emits `"fallback_background"`, so a real fallback is invisible to atom uncertainty.
  - Mitigation: follow-up fix plus a regression test that uses `mind_requested=True`. The 0.42 mind-skip floor masks the fix while mind is off. Live impact today is zero (0 fallbacks in all retained history).
- Severity: medium (the existing route signal is broken, beyond what this PR set out to check)
  - Concern: two defects together make `node:substrate.route.prediction_error` meaningless in the field. Route is in `ACTIVE_INFERENCE_DOMAINS` (`attention_self_model.py:128`).
    1. **Stale copy.** `services/orion-substrate-runtime/app/worker.py` `_route_tick` (~line 3880) saves the `prediction_signal` receipt only `if error > 0.0`. That receipt is what the digester uses to set the field channel (mode `replace`, `state_deltas.py:607-619`). The "write every tick" fix just below covers only the graph node. So the field holds the last value above zero. `node_vector_updated_at` is `2026-09-24T12:11:48Z`, 12h stale at the time of the read. This is the same stuck-high-water-mark disease the worker's own comment says was fixed.
    2. **Can never read "not calm".** `route_prediction_error` averages over every run in the projection (~824 live, capped at 2000). Old runs match themselves by `trace_id` and score 0. One run flipping one field reads 0.25/824 ≈ 0.0003. Two fields read 0.0006. Both match the live values exactly. The ceiling is about 0.0012 even if every field flips.
  - Mitigation: follow-up PR. Fix the instrument (score only runs that are new or changed this tick) and the gated field write. Only then decide whether route stays in `ACTIVE_INFERENCE_DOMAINS`.
- Severity: low
  - Concern: no ordinary Hub chat (`verb_chat`) reached the route lane in 3 days, and orch route volume fell ~10x across 09-24.
  - Mitigation: not investigated here. Flagged.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2322

🤖 Generated with [Claude Code](https://claude.com/claude-code)
