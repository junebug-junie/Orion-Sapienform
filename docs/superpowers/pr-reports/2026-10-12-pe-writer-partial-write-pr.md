## Summary

- The prediction-error writer (`_write_prediction_error_node`, substrate-runtime) no longer overwrites the dynamics tick's fields. It writes only what it owns: `prediction_error`, its extra channels, evidence ids, `contributing_turn_ids`, and its own stamps (`observed_at`, salience, label, provenance).
- Every other whole-node writer is covered too, at the store: `FalkorSubstrateStore.upsert_node` now issues `MERGE ... ON CREATE SET <all> ON MATCH SET <all minus skipped keys minus tick-owned pressure/dormancy>`. Pressure and dormancy on an existing node are written only by the tick's targeted write from PR #2617.
- `skip_metadata_keys` can now also name the activation state (`activation`, `recency_score`, `activation_decayed_at`) on the Falkor, in-memory and GraphDB stores. The PE writer uses it, so it stops resetting activation.
- Retired the PE writer's read-and-carry-forward and `DYNAMICS_ENGINE_OWNED_METADATA_KEYS` (its only consumer). A read-modify-write still lost any tick write landing between its read and its write.
- Waking a dormant node stays the tick's job: a PE write moves `observed_at` to now and salience to the error; the next tick reads that as fresh seed activation and its own revival threshold clears `dormant`. Pinned by a test.

## Outcome moved

Before (live, 2026-10-12 00:01 UTC, `orion_substrate`): `node:substrate.bus_synaptic` read activation 0.0118343 == its prediction_error 0.011834 and recency_score 0, paired with the tick's older decay stamp; `node:substrate.biometrics` the same (0.03287 == 0.032866). Every 30 s PE write reset the node's activation to its salience and its recency to 0, and made any in-flight tick's compare-and-set miss.

After (eval, 6 nodes x 60 ticks, 360 PE writes placed at random before / inside the tick / around the tick): 0 PE fields reverted, 0 tick fields reverted, 0 tick CAS misses. The old whole-node write on the same fixture reverts the tick's fields on 131/131 "PE read -> tick -> PE write" races and causes 103/103 CAS misses.

Live effect after deploy: UNVERIFIED until deployed. Check: the same query should show PE nodes' `recency_score` > 0 and `activation` no longer equal to `prediction_error` right after a PE write.

## Current architecture

PR #2617 made `SubstrateDynamicsEngine.tick` write only its fields (`update_node_dynamics`, compare-and-set on activation + stamp). Every other writer still went through `upsert_node`: one `MERGE ... SET <every encoded property>`. The codec always encodes `dynamic_pressure` (default 0.0) and `dormant` (default False), so any fresh-model write reset them; the PE writer worked around that by reading the node and carrying the values forward, and wrote validator-default activation/recency outright.

Other `upsert_node` writers on `orion_substrate` checked: concept-induction materialization, `SubstrateGraphMaterializer`, seed loader, recall's concept-region collector, town-perception ingest, ask-answered listener, Falkor legacy-payload migration. None of them owns pressure/dormancy; the store-level ON MATCH rule covers them all. They keep writing activation (a materializer reinforcing a node is a legitimate activation write the tick's CAS yields to).

## Architecture touched

- `orion/substrate` store seam (`upsert_node` Cypher shape and cache mirror), codec constants.
- substrate-runtime PE writer.
- No bus, schema-registry, env or compose change.

## Files changed

- `orion/substrate/falkor_codec.py`: `TICK_OWNED_NODE_PROPERTIES`, `ACTIVATION_STATE_KEYS`; removed `DYNAMICS_ENGINE_OWNED_METADATA_KEYS`.
- `orion/substrate/falkor_store.py`: `upsert_node` ON CREATE / ON MATCH split, `RETURN`s the written node and caches the durable row; `_apply_node_to_cache` mirrors the split as a fallback.
- `orion/substrate/dynamics.py`: comment on the full-upsert fallback.
- `orion/substrate/store.py`: `preserve_skipped_keys` (metadata + signal-level activation/recency); in-memory store uses it.
- `orion/substrate/graphdb_store.py`: uses `preserve_skipped_keys`.
- `services/orion-substrate-runtime/app/worker.py`: PE writer passes `_PE_WRITER_SKIPS_TICK_OWNED`, carry-forward removed, ownership documented.
- `services/orion-substrate-runtime/tests/test_prediction_error_node_lost_update.py`: both interleavings (in-memory, recording Cypher, real FalkorDB), dormant-wake, create-complete.
- `orion/substrate/tests/test_dynamics_lost_update.py`: blind-upsert store guard (recording Cypher + real FalkorDB), in-memory activation skip.
- `services/orion-substrate-runtime/evals/run_pe_writer_tick_lost_update_eval.py`: race eval.
- Existing tests updated to the ON MATCH clause / new kwarg: `test_falkor_store.py`, `test_falkor_direct.py`, `test_falkor_materialization.py`, `test_dynamics_decay_since_last.py`, `test_worker_prediction_error_node.py`, `test_post_turn_closure_listener.py`.
- `.github/workflows/substrate-neighborhood.yml`: runs the new service test in the real-FalkorDB must-not-skip step, the new eval, triggers on the test path; installs `fastapi` (app.worker imports the service routes).

## Schema / bus / API changes

- Added: `skip_metadata_keys` accepts `ACTIVATION_STATE_KEYS` names.
- Removed: `falkor_codec.DYNAMICS_ENGINE_OWNED_METADATA_KEYS`.
- Renamed: none.
- Behavior changed: Falkor `upsert_node` never sets `dynamic_pressure`, `dynamic_pressure_reason`, `dormant`, `dormancy_updated_at` on an existing node; skipped keys are now set when the call creates the node (previously never).
- Compatibility notes: no stored-data migration. The dynamics tick's full-upsert fallback (stores without `update_node_dynamics`) would no longer persist pressure/dormancy on a Falkor-backed wrapper; production's tick runs on the routed Falkor store, which has the targeted write.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
PYTHONPATH=. pytest orion/substrate/tests orion/substrate/evals orion/spark orion/graph -q
  1299 passed, 5 failed (all 5 also fail on main: felt_state_self_definition_lane x3,
  falkor_materialization write_through_cache / hub_store_sees_spark_writes)
services/orion-substrate-runtime: pytest tests evals (minus postgres-only integration file)
  failure set identical to main (pre-existing, incl.
  test_worker_falkor_routed_store::test_write_prediction_error_node_preserves_dynamics_state_on_rewrite)
ORION_TEST_FALKOR_URI=redis://localhost:6380 (throwaway graphs, deleted after):
  test_prediction_error_node_lost_update.py 8 passed
  test_dynamics_lost_update.py 11 passed
  test_falkor_direct.py test_falkor_anchor_store.py test_neighborhood_falkor_live.py 43 passed
CI "Store contract" file list in a clean venv with the workflow's pip set: 315 passed, 3 skipped
Mutation check: old worker.py restored -> 4 of 7 new service tests fail (both interleavings,
  Cypher, real Falkor); fixed -> 7/7 pass
hub / recall / cortex-exec store-touching tests: same as main
scripts/check_metric_lineage.py --gate, check_definition_drift.py --gate,
  check_inner_state_registry.py, check_scripts_dir_no_stdlib_shadow.py: ok
```

## Evals run

```text
PYTHONPATH=. python services/orion-substrate-runtime/evals/run_pe_writer_tick_lost_update_eval.py
  fixed: pe_fields_reverted 0, tick_fields_reverted 0, tick_cas_misses 0 (360 writes)
  whole_node_write_baseline: tick_fields_reverted 131/131, tick_cas_misses 103   passed
python -m orion.substrate.evals.run_dynamics_lost_update_eval / run_decay_since_last_eval /
  run_complete_hydration_eval: ok
```

## Docker/build/smoke checks

```text
Not deployed (per instructions). FalkorDB syntax for MERGE ... ON CREATE SET ... ON MATCH SET ...
REMOVE verified against the live FalkorDB on a throwaway graph.
```

## Review findings fixed

Code review ran in a subagent against `origin/main...HEAD`. No blocking defects found. It confirmed the new tests fail on origin/main's code (7 of 7 interleaving/Cypher tests), so none are vacuous.

- Finding (moderate): on a cache miss (node written by another process since the last hydrate), the write-through cache stored the caller's model defaults while Falkor kept the tick's values. The tick's next compare-and-set would then expect an activation the graph does not hold.
  - Fix: the upsert `MERGE` now `RETURN`s the node's native properties and the cache stores the decoded durable row. The merge-based mirror stays as a fallback for clients that return no row (test doubles).
  - Evidence: `test_real_falkor_cache_miss_caches_what_the_graph_kept` (real FalkorDB) passes.
- Finding (nit): a tick run over a Falkor-backed store that lacks `update_node_dynamics` would silently drop pressure/dormancy.
  - Fix: documented at the dynamics fallback and in Risks. No production path does this.
  - Evidence: the worker's store comes from `build_substrate_store_from_env` (routed Falkor, targeted write).
- Finding (nit): the eval's "old" comparison run is not exactly the old writer.
  - Fix: renamed it to `whole_node_write_baseline` and documented what it does and does not reproduce.
  - Evidence: the eval docstring.
- Finding (nit): ON CREATE now writes skipped keys, e.g. `contributing_turn_ids_json="[]"`.
  - Fix: harmless. The cache now holds the durable row, so the cache and the graph agree.
  - Evidence: same as the first finding.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```

Other services importing `orion/substrate` (hub, recall, concept induction) pick up the new `upsert_node` on their next rebuild; not required for this fix.

## Risks / concerns

- Severity: low
- Concern: a Falkor-backed store wrapper without `update_node_dynamics` running the tick would silently stop persisting pressure/dormancy on existing nodes.
- Mitigation: the only tick in production runs on the routed Falkor store (targeted write); covered by `test_writer_lookup_falls_back_only_for_stores_without_targeted_path`.

- Severity: low
- Concern: in-memory and GraphDB stores do not apply the store-level pressure/dormancy rule to blind upserts (Falkor only); only callers that name the keys in `skip_metadata_keys` get it there.
- Mitigation: Falkor is the production backend; the PE writer names the keys explicitly, so its behavior is the same on every store.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2621

🤖 Generated with [Claude Code](https://claude.com/claude-code)
