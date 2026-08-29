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
- `tests/test_capability_absence_signal.py`: 9 tests (new)
- `docs/superpowers/specs/2026-08-29-capability-absence-signal-design.md`: design (new)

## Schema / bus / API changes

- Added: none. `node_capability_impact` was already in `ALLOWED_PRESSURE_ROLES`, already in
  `ROLE_TO_PRESSURE_KIND`/`ROLE_TO_OPERATION`, and already had a reducer arm. This patch
  makes existing dead code reachable rather than minting a concept.
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
pytest tests/test_capability_absence_signal.py -q            -> 9 passed
pytest tests/test_biometrics_pressure_organ.py \
       tests/test_node_pressure_reducer.py \
       tests/test_biometrics_pipeline_ilo_pressures.py \
       tests/test_peak_pressure.py -q                        -> 59 passed
pytest services/orion-hub/tests/test_substrate_biometrics_debug_api.py -q -> 3 passed
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
