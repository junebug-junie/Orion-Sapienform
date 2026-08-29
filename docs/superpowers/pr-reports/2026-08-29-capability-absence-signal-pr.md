# PR: make node absence reachable as a capability impact (phase 1)

## Summary

- Adds `sweep_absent_nodes()`, the trigger the node-staleness rule never had:
  `invoke_biometrics_pressure()` resolves its subject from the *incoming* event, so a node
  that stops reporting entirely is never evaluated at all.
- Adds **Rule F**: an `expected_online` node that has gone stale now emits
  `node_capability_impact` alongside its availability concern. Rule E, the only prior
  emitter, fires on GPU *saturation*, which a dead node can never produce.
- Fixes a **pre-existing `trace_id` collision**: every rule firing for one node in one tick
  shared a trace_id, so `group_candidate_events_by_trace()` merged them and the reducer
  (one atom per trace) dropped all but the first. Reproduced against shipping rules C and E
  alone, with no new code involved.
- Fixes a **pre-existing reducer bug**: the `node_capability_impact` arm could only ever
  append the literal `"capability:capability"`. Never caught, because the arm had never
  executed.
- Both fixes are mutation-tested: reverting either makes the matching test fail.

## NOT live in this patch

`sweep_absent_nodes()` has **no caller**. Nothing in this branch detects absence at runtime;
the circe gap is **not closed**. Phase 1 makes the signal expressible, correct and tested.
Phase 2 wires it into the substrate-runtime tick, and must thread that service's configured
`biometrics_node_stale_after_sec` (env `BIOMETRICS_NODE_STALE_AFTER_SEC`) rather than
letting the sweep fall back to the module's 180 s default, or an operator override will make
the sweep and the event path disagree about "stale".

## Outcome moved

`node_capability_impact` had **0 rows in `grammar_atoms` for its entire lifetime**, and
`capability_impacts` was `[]` in **every** `substrate_active_node_pressure_projection` row
ever written. It had two independent reasons never to appear -- unreachable by absence, and
discarded when reached by saturation. Both are now closed, and the value it writes is real
capability names instead of a constant.

Motivating incident: circe (`expected_online: true`, 5 declared capabilities) dark
2026-08-29 00:01:16Z -> 00:47:04Z, ~45 min. Its
`last_accepted_at["availability"]` never moved off `2026-08-19T03:13:33Z`.

## Current architecture

`node_catalog.yaml` declares which capabilities each node provides. Biometrics knows who is
reporting. The pressure organ turns that into pressure. The field digester propagates node
channels to capability channels. **None of it could express "this node is gone."** Every
layer modelled load and never presence.

## Architecture touched

`orion/substrate/biometrics_loop/` only. No service boundary crossed.

## Files changed

- `orion/substrate/biometrics_loop/pressure_organ.py`: `sweep_absent_nodes()` + Rule F
- `orion/substrate/biometrics_loop/candidate_events.py`: `semantic_role` in `trace_id`
- `orion/substrate/biometrics_loop/pressure_reducer.py`: expand real capability names
- `tests/test_capability_absence_signal.py`: 14 tests (new)
- `docs/superpowers/specs/2026-08-29-capability-absence-signal-design.md`: design (new)

## Schema / bus / API changes

- Added: `node_capability_absent` role (+ `ROLE_TO_PRESSURE_KIND["capability_absence"]`,
  `ROLE_TO_OPERATION["update"]`, `ALLOWED_PRESSURE_ROLES`). Added on review: Rule F first
  reused `node_capability_impact`, and sharing one role with Rule E caused three distinct
  defects at once. It carries a producer, a reducer arm, a clearing path and 5 tests in this
  same patch.
- Reached, not added: `node_capability_impact` already existed everywhere and had simply
  never executed.
- Behavior changed: `trace_id` format is now
  `substrate.pressure:{node}:{semantic_role}:{ts}`.
- Compatibility: **read-backward-compatible.** `parse_pressure_trace_id()` recovers the node
  with `split(":", 2)[1]`, so old-format ids emitted by an in-flight process during a deploy
  still parse. Verified: `services/orion-hub/tests/test_substrate_biometrics_debug_api.py`
  feeds an old-format fixture and passes unchanged.

## Env/config changes

None. No env key added, removed, renamed, or changed meaning; `.env_example` untouched, so
no sync was required.

## Tests run

```text
pytest tests/test_capability_absence_signal.py -q            -> 14 passed
pytest tests/test_biometrics_pressure_organ.py \
       tests/test_node_pressure_reducer.py \
       tests/test_biometrics_pipeline_ilo_pressures.py \
       tests/test_peak_pressure.py \
       services/orion-hub/tests/test_substrate_biometrics_debug_api.py -q -> 67 passed
pytest tests/ -k "substrate or biometric or pressure or grammar or field" \
       --continue-on-collection-errors                       -> 696 passed, 9 failed
python scripts/check_inner_state_registry.py                 -> gate OK (15 entries)
```

The 9 failures are **pre-existing**. Verified by reverting all three source files in-place
and re-running: identical 9 failures on clean `main` (`git stash` deliberately avoided --
it is shared across worktrees). The 41 collection errors are environmental (tests expect
Postgres on :5432; the live DB is on :55432).

## Evals run

```text
No eval harness exists for orion/substrate/biometrics_loop/.
```

Phase 1 is a pure-function change with deterministic unit cover including two mutation
tests. The real eval is phase 2's live check, recorded as an unchecked acceptance box in
the spec: kill circe's llamacpp container, confirm `capability_impacts` becomes non-empty
for the first time in its history, restore, confirm Rule B' clears it.

## Docker/build/smoke checks

```text
Not run -- no service code, Dockerfile, compose file, dependency, or port changed.
scripts/safe_graphify_update.sh -> REFUSED and auto-restored (node count 28306 -> 2485,
  ~91.2% loss), the known unfixed graphify bug. graphify-out/ verified clean afterwards.
```

## Review findings fixed

Code review (high effort, subagent) returned 6 findings. All 6 fixed; the two reproduced
defects re-verified against the reviewer's own scenarios.

- Finding 1 (high): Rule E and Rule F both emitted `semantic_role="node_capability_impact"`,
  so putting the role in `trace_id` did not separate them. Reachable in exactly the
  motivating case, because `node_reducer.py` merges `pressure_hints` and never clears them
  -- a GPU-heavy node that dies stays stale *and* keeps `gpu >= 0.60`.
  - Fix: Rule F emits its own role, `node_capability_absent`.
  - Evidence: reviewer measured 16 events / 13 unique, traces `[4,8,4]`, one `event_id`
    published twice. Now: **16 events / 16 unique, 4 traces `[4,4,4,4]`, 4 published / 4
    unique**, every atom surviving grouping.
    `test_saturated_and_dead_node_emits_no_duplicate_event_ids`.
- Finding 2 (medium): `capability_impacts` had no removal path -- a one-way ratchet reaching
  the concept graph via `biometrics_ctx.py:102`, the identical failure Rule B' exists to
  undo for `availability`.
  - Fix: `node_availability_recovered` clears them; `node_pressure_decayed` clears the
    saturation subset.
  - Evidence: `test_recovery_clears_capability_impacts`.
- Finding 3 (medium): the expansion ignored which rule fired, so routine saturation marked
  every declared capability.
  - Fix: absence marks all declared capabilities; saturation marks only `LLM_CAPABILITIES`.
  - Evidence: reviewer's repro (healthy circe, `gpu=0.65`) listed all five. Now
    `training`/`dream_batch`/`batch_inference` are excluded.
    `test_saturation_alone_does_not_impact_non_llm_capabilities`.
- Finding 4 (low): both roles shared `pressure_kind="capability"`, so a saturation event
  within the 300 s merge window silently swallowed the first absence event -- and
  "saturated, then died" is the ordinary crash sequence.
  - Fix: `node_capability_absent` gets its own `capability_absence` bucket.
  - Evidence: `test_absence_and_saturation_use_separate_merge_buckets`.
- Finding 5 (low): `sweep_absent_nodes()` has no caller, and defaults to 180 s rather than
  the runtime's configured `biometrics_node_stale_after_sec`.
  - Fix: not wired (phase 2, deliberate) but no longer implied -- added a "NOT live in this
    patch" section above and an explicit warning in the function's own docstring that the
    phase-2 caller must thread the setting.
- Finding 6 (low): a spec acceptance criterion marked `[x]` claimed "one per declared
  capability" while the code and its test assert exactly one per node.
  - Fix: corrected the criterion and the two restatements to the actual
    one-event-plus-reducer-expansion design.

Self-caught before review, same round: my docstring cited `prometheus` as covered by the
sweep. It is not -- the sweep iterates the projection, built from *received* events, and the
live projection holds only `atlas`, `circe`, `athena`. Rewritten as a stated phase-2 gap.

## Restart required

```text
No restart required for this patch alone -- nothing calls sweep_absent_nodes() yet.
Phase 2 wires it into the substrate-runtime tick and will need:
  docker compose --env-file .env --env-file services/orion-substrate-runtime/.env \
    -f services/orion-substrate-runtime/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: low. Rule F is gated on `capabilities` being non-empty and rides the reducer's
  existing per-node+`pressure_kind` merge window, so a long outage cannot become an event
  storm.
- Severity: low. The `trace_id` change alters the id of *newly emitted* pressure traces.
  Nothing keys off the old exact string (only `parse_pressure_trace_id`, which still works);
  historical rows are untouched.
- Severity: medium, deferred. `sweep_absent_nodes()` has no caller in this patch. It is
  tested and dead until phase 2 wires it into the tick. Called out as deliberate scope, not
  an oversight.
