## Summary

- The substrate tick decided "dormant" for every kind of node. The graph store only saves that flag on concept nodes.
- So each tick, every cold non-concept node (entity, evidence, assertion) "went dormant", the save dropped the flag, and the next tick did it again.
- The log line counted those fake flips (~171 per tick on 2026-11-11), and the tick made one wasted graph write for each.
- Fix: one shared list of node kinds that carry dormancy, `DORMANCY_NODE_KINDS = {"concept"}` in `orion/substrate/store.py`. The tick and the Falkor write both read it.
- No new kind starts storing dormancy.

## Outcome moved

- `dormancy_transitions` in the tick log becomes a real count: small, varying, and matched by movement in the dormant node count.
- About 170 fewer graph writes per tick (the phantom-flip writes stop because the write guard no longer sees a "change" on those nodes).

## Current architecture

- `SubstrateDynamicsEngine.tick()` (`orion/substrate/dynamics.py`) set `dormant_new` from activation/recency thresholds for every node. A change in the flag both added a `DormancyTransitionV1` and forced a write.
- `FalkorSubstrateStore.update_node_dynamics()` (`orion/substrate/falkor_store.py`) wrote `dormant` only when `node_kind == "concept"`.
- The only consumer of `dormancy_transitions` is the log line in `services/orion-substrate-runtime/app/worker.py` (`substrate_dynamics_tick_completed`).

## Architecture touched

- `orion/substrate` only. No bus, schema, env, or API change.

## Files changed

- `orion/substrate/store.py`: adds `DORMANCY_NODE_KINDS`, the single source of truth.
- `orion/substrate/dynamics.py`: decides dormancy only for kinds in `DORMANCY_NODE_KINDS`.
- `orion/substrate/falkor_store.py`: persists `dormant` by the same constant instead of a separate `"concept"` literal.
- `orion/substrate/tests/test_dynamics_dormancy_kinds.py`: regression tests (new).
- `services/orion-substrate-runtime/tests/test_person_sighting_refresh.py`: this test asserted that an entity node ends with `dormant=True`. That was only ever true on the in-memory test store; production never saved it (live: 0 dormant entity nodes). The real claim, activation back at rest (<= 0.08), is still asserted.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: non-concept nodes no longer get a `dormant` flag in tick metadata or in-memory stores. Falkor never stored it for them.
- Compatibility notes: readers that use `metadata.get("dormant")` (brain frame producer, graph_cognition features) already saw False for non-concept nodes from Falkor.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced: not needed
- skipped keys requiring operator action: none

## Tests run

```text
orion/substrate/tests/test_dynamics_dormancy_kinds.py         2 passed (non-concept test FAILS on main: phantom transitions on tick 0)
orion/substrate/tests                                          1171 passed, 4 failed (felt_state lived_answers/self_definition_lane: same 4 fail on main)
services/orion-substrate-runtime/tests                         failure list identical to main except test_person_sighting_refresh (fixed above)
tests/test_cognitive_substrate_phase4_dynamics.py + runtime falkor_routed_store: 3 failures, identical on main (pre-existing)
```

## Evals run

```text
services/orion-substrate-runtime/evals/run_pe_writer_tick_lost_update_eval.py  passed: true
services/orion-substrate-runtime/evals (pytest)                                   passed
```

## Docker/build/smoke checks

```text
No build needed for review. Live "before" evidence, 2026-11-11 (orion-athena-substrate-runtime):
substrate_dynamics_tick_completed activation_updates=5892 pressure_updates=2 dormancy_transitions=173
substrate_dynamics_tick_completed activation_updates=5894 pressure_updates=3 dormancy_transitions=171
substrate_dynamics_tick_completed activation_updates=5890 pressure_updates=4 dormancy_transitions=171
Falkor graph orion_substrate, dormant=true by kind: concept 387 / 5904, entity 0, evidence 0, assertion 0.
```

Post-deploy live check:

1. `docker logs --since 10m orion-athena-substrate-runtime 2>&1 | grep dormancy_transitions | tail -5`
   Expect `dormancy_transitions` to be small (single or low double digits) and to vary tick to tick, not a flat ~171.
2. `docker exec orion-athena-falkordb redis-cli GRAPH.RO_QUERY orion_substrate "MATCH (n:SubstrateNode) RETURN n.node_kind, sum(CASE WHEN n.dormant = true THEN 1 ELSE 0 END)"`
   Run it a few minutes apart. The concept dormant count should move by roughly the transitions logged in between. Other kinds stay 0.

## Review findings fixed

Reviewer (subagent, hand review of the diff): no material findings. Two low-severity notes:

- Finding: the in-memory store and Falkor's local cache mirror decided dormancy by a separate `include_concept_only` flag, not the shared list. That could drift from the graph if the list grows, and a stale non-concept `dormant=True` could never clear in-memory.
  - Fix: `apply_node_dynamics_write` (`orion/substrate/store.py`) now gates dormancy on `DORMANCY_NODE_KINDS` for every backend.
  - Evidence: dynamics + sighting + brain-frame tests, 99 passed.
- Finding: the legacy GraphDB full-node path would keep a stale non-concept `dormant=True` forever.
  - Fix: not changed. Production runs Falkor, and no live non-concept node has the flag (0 in the graph).
  - Evidence: Falkor count by kind in the smoke section.

## Restart required

```bash
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```

(From the clean primary `main` checkout after merge; the post-pull rebuild normally does this.)

## Risks / concerns

- Severity: low
- Concern: pressure is also saved on concept nodes only, and the tick computes it for every kind. Live `pressure_updates` is 2-5 per tick, so there is no sign of the same loop today. Not changed here.
- Mitigation: if `pressure_updates` ever climbs and stays flat, check this first.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
