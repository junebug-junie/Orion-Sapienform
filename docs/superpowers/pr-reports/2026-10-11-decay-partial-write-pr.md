## Summary

- The substrate decay tick (`SubstrateDynamicsEngine.tick`) no longer rewrites whole graph nodes. It writes only the fields it owns, so a concurrent write to anything else (observed_at, label, provenance, foreign metadata) survives.
- Its decayed activation is written compare-and-set: only if the node still holds the activation and decay stamp the tick read. A fresh activation written meanwhile wins; the tick decays it next tick.
- New store method `update_node_dynamics(NodeDynamicsWriteV1)` on the in-memory, Falkor and routed stores. Falkor issues `MATCH` (never `MERGE`, so a deleted node stays deleted) + named `SET`s, CAS in the same statement.
- Decay math unchanged (existing since-last/legacy tests and evals unchanged in value).
- New trace: `activation_cas_skips` on the tick result and in the `substrate_dynamics_tick_completed` log line.

## Outcome moved

Before: any write landing between a tick's read and its write was reverted (live 2026-10-11: 238 of 244 corrected `observed_at` values reverted within two minutes, PR #2608). After: 0 reverted in the new eval (300 observed_at corrections + 180 fresh activations over 60 ticks); the old path reverts all 480 on the same fixture.

## Current architecture

One decay writer since 2026-10-06: `orion-substrate-runtime`'s `_dynamics_tick` runs the engine every tick against the Falkor `orion_substrate` graph. The engine snapshot()s the whole graph, computes pressure/activation/dormancy, then `upsert_node`s every node that moved: a full `MERGE ... SET <every property>` from the snapshot copy. Only `EXTERNALLY_OWNED_METADATA_KEYS` (prediction_error etc.) were skipped.

Other "decay" code checked, not affected:
- `services/orion-field-digester/app/digestion/decay.py` (`NODE_DECAY_CHANNELS`): decays the digester's own in-process field state; other producers reach it only as deltas consumed by the same loop, and it persists an append-only per-tick snapshot row (`substrate_field_state`, keyed by `tick_id`). Single writer, no read-modify-write race.
- `orion/memory/crystallization/dynamics.py::decayed_activation` and `orion/substrate/relation_classification.py`: pure reads, never persisted.

## Architecture touched

- `orion/substrate` store seam (new targeted write), dynamics engine, Falkor recording test double.
- No bus, schema-registry, env or compose change.

## Files changed

- `orion/substrate/store.py`: `NodeDynamicsWriteV1`, `apply_node_dynamics_write`, `node_dynamics_cas_matches`, `encode_decay_stamp`, `node_dynamics_writer`; in-memory `update_node_dynamics`.
- `orion/substrate/falkor_store.py`: Falkor `update_node_dynamics` (targeted Cypher, CAS, cache mirror, scan-journal replay; generation bump forces a re-read after a miss).
- `orion/substrate/routed_store.py`: delegates to primary; `supports_node_dynamics_write`.
- `orion/substrate/dynamics.py`: tick builds the targeted write; full-node path kept only for stores without one (SPARQL/GraphDB, not used in production).
- `orion/graph/falkor_client.py`: `RecordingFalkorClient` answers the targeted write as applied.
- `services/orion-substrate-runtime/app/worker.py`: log `activation_cas_skips`.
- `orion/substrate/tests/test_dynamics_lost_update.py`: the regression (in-memory + real FalkorDB).
- `orion/substrate/evals/run_dynamics_lost_update_eval.py`: race eval.
- Existing tests updated to the targeted write: `test_dynamics.py`, `test_dynamics_decay_since_last.py`, `test_dynamics_falkor_routed.py`, `services/orion-substrate-runtime/tests/test_worker_falkor_routed_store.py`.
- `.github/workflows/substrate-neighborhood.yml`: runs the new test (real FalkorDB lane, must not skip) and eval.

## Schema / bus / API changes

- Added: store method `update_node_dynamics` (Python seam only).
- Removed / Renamed: none.
- Behavior changed: the tick no longer re-SETs unowned properties, no longer recreates a node deleted since its read, and skips its activation write when the node's activation/stamp moved. Legacy decay mode now writes the stamp as observed_at explicitly (what the Falkor store filled in before).
- Compatibility notes: no stored-data migration.

## Env/config changes

- None. No `.env_example` touched; no sync needed.

## Tests run

```text
ORION_TEST_FALKOR_URI=redis://localhost:6380 pytest orion/substrate/tests/test_dynamics_lost_update.py -> 8 passed (incl. 2 real-FalkorDB tests on throwaway graphs)
Reload test with tolerance forced to 0 -> fails (5 skips): catches the freeze
Same two race tests with the targeted path disabled (old behavior) -> 2 failed (observed_at reverted): the tests catch the bug
pytest orion/substrate/tests orion/substrate/evals orion/graph -> 1213 passed, 48 skipped, 3 failed (test_felt_state_self_definition_lane.py, also fail on main)
services/orion-substrate-runtime tests -> same failure set as main (env-sensitive, pre-existing)
tests/test_cognitive_substrate_phase4_dynamics.py -> 2 failures, identical on main
orion-recall test_concept_region_collector, orion-hub test_world_pulse_read_assertion_links -> pass
scripts/check_definition_drift.py -> no definition changes
```

## Evals run

```text
python -m orion.substrate.evals.run_dynamics_lost_update_eval -> passed; targeted reverted 0/300 observed_at, 0/180 fresh activation; old full-node path 300/300, 180/180
python -m orion.substrate.evals.run_decay_since_last_eval -> passed (math unchanged)
python -m orion.substrate.evals.run_complete_hydration_eval -> equivalent
```

## Docker/build/smoke checks

```text
No build in the worktree. Live verify after deploy (UNVERIFIED until then):
1. Set a sentinel on one real node:
   GRAPH.QUERY orion_substrate "MATCH (n:SubstrateNode {node_id:'<id>'}) SET n.decay_race_sentinel = 'keep'"
2. Wait 2 dynamics ticks (> 2 x DYNAMICS_TICK_INTERVAL_SEC).
3. Read it back: the sentinel is still there AND n.activation_decayed_at has moved forward.
4. docker logs orion-substrate-runtime | grep substrate_dynamics_tick_completed -> activation_cas_skips present (normally 0 or small).
5. Remove the sentinel: ... REMOVE n.decay_race_sentinel
```

## Review findings fixed

- Finding (critical): an equality CAS on activation can never match after a reload. FalkorDB stores the full double but returns ~15 significant digits (verified live: 0.7127189745122715 stored, 0.712718974512272 returned, `WHERE n.a = <read-back>` matches 0 rows). Every node would miss every tick, so decay would freeze, and each miss would force a full rescan.
  - Fix: compare within an absolute tolerance of 1e-12 (`ACTIVATION_CAS_TOLERANCE`). Float rounding is about 1e-16; one 30 s decay step on a 30-day half-life is about 8e-6, so a real concurrent write is still detected. The stamp string is still matched exactly.
  - Evidence: new `test_real_falkor_decay_keeps_moving_after_reload` (real FalkorDB, reloads before each of 3 ticks) passes; with tolerance forced to 0 it fails with `5 == 0` skips.
- Finding (major): the live test seeded a 15-digit value that survives rounding, so it hid the bug.
  - Fix: it now seeds real decay outputs; the reload test was added.
- Finding (major): a durable NULL activation can never match.
  - Fix: `coalesce(n.activation, 0.0)`.
- Finding (minor): the cache wrote pressure/dormancy for non-concept nodes, which the graph never stores.
  - Fix: the Falkor cache mirror gates those on concept.
- Finding (minor): a shadow store without the targeted write silently stopped receiving decay.
  - Fix: the routed store logs a one-time warning.
- Finding (minor, not fixed, follow-up): `_write_prediction_error_node` still does a full-node upsert that carries forward the tick's pressure/dormancy from its cached copy, so it can revert the tick's fields. Listed under risks.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```

## Risks / concerns

- Severity: low. Concern: the SPARQL/GraphDB backend still uses the full-node path (no targeted write). Mitigation: production runs `SUBSTRATE_STORE_BACKEND=falkor`; comment marks it.
- Severity: low. Concern: other writers that re-save a node from a stale copy (e.g. `_write_prediction_error_node` carrying forward engine-owned keys) can still overwrite the tick's fields the other way. Not in scope; the tick no longer clobbers them.
- Severity: low. Concern: a dormancy transition is reported in the tick result even when its CAS skipped the write. Next tick recomputes.

## PR link

(this PR)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
