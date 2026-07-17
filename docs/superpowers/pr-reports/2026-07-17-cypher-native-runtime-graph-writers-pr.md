# PR report: substrate-runtime graph writers proven against Cypher-native Falkor adapter

## Summary

- Promoted 5 dynamics-engine metadata keys (`dynamic_pressure`, `dynamic_pressure_reason`, `dormant`, `dormancy_updated_at`, `prediction_error`) from `BaseSubstrateNodeV1.metadata` to native Cypher scalar properties on concept nodes in `orion/substrate/falkor_codec.py` / `falkor_store.py`.
- Closed a restart-durability gap: `FalkorSubstrateStore`'s cold-start hydration always produced `metadata={}`, so a process restart against a Falkor-backed store would silently reset all `SubstrateDynamicsEngine`/`attention_broadcast` state — masked today only because the in-process cache serves reads while the process stays up.
- Found and fixed a second, related bug during review: `services/orion-substrate-runtime/app/worker.py::_write_prediction_error_node` builds a fresh `ConceptNodeV1` on every call and writes it under a fixed, reused `node_id` (its own documented "re-writes collapse" behavior) — once the codec always includes the 5 dynamics keys in every upsert's `SET` clause, a repeat prediction-error event durably clobbered the dynamics engine's already-computed `dynamic_pressure`/`dormant` state back to defaults. Fixed by carrying forward dynamics-engine-owned metadata from any existing node before overwriting.
- Added integration tests proving `SubstrateDynamicsEngine.tick()` and the runtime worker's `_write_prediction_error_node` / `_dynamics_tick` work correctly against a Falkor-primary `RoutedSubstrateGraphStore`, at both the library level and the service level — closing the design spec's "runtime SPARQL cutover uses Cypher-native adapter, not blob port" acceptance check.
- Updated `docs/superpowers/specs/2026-07-16-cypher-native-substrate-postgres-bus-split-design.md`'s acceptance checklist and status line with this evidence.
- Left `services/orion-substrate-runtime/.env_example`/`.env`/`docker-compose.yml` untouched — no defaults flipped, no live cutover performed in this patch.

## Outcome moved

The Cypher-native Falkor adapter (PR #1120) is now proven, with tests, to correctly carry the specific graph-shaped signals (`prediction_error`, `dynamic_pressure`, dormancy) that `orion-substrate-runtime`'s dynamics tick and attention broadcast already depend on — including across a simulated process restart. Before this patch, cutting runtime over to Falkor would have silently reset all dynamics state on every restart, and separately, on every repeat prediction-error event even without a restart. Both gaps are closed at the code level; the live env cutover itself remains a deliberately separate, later decision.

## Current architecture

`orion/substrate/falkor_codec.py`'s `encode_node_properties()`/`decode_concept_node()` handled only the common `BaseSubstrateNodeV1` scalar fields (confidence, salience, activation, provenance, etc.) plus concept-specific `label`/`definition`/`taxonomy_path`/`evidence_refs`. Anything in the free-form `metadata` dict — including the fields `SubstrateDynamicsEngine` and `attention_broadcast` actually read and write — was dropped on every durable write and every cold-start hydration. `services/orion-substrate-runtime/app/worker.py`'s `_dynamics_tick()` and `_write_prediction_error_node()` already exercise this path today against SPARQL (where the same issue does not apply, since that store's persistence model differs), gated off by default behind `SUBSTRATE_DYNAMICS_TICK_ENABLED` / `SUBSTRATE_WRITE_PREDICTION_ERROR_NODES`.

## Architecture touched

- `orion/substrate/falkor_codec.py`: closed 5-key metadata-promotion allowlist, `DYNAMICS_METADATA_KEYS`/`DYNAMICS_ENGINE_OWNED_METADATA_KEYS` constants, tolerant float coercion.
- `orion/substrate/falkor_store.py`: `NATIVE_NODE_RETURN_FIELDS` carries the 5 new columns through cold-start hydration's Cypher `RETURN` clause.
- `services/orion-substrate-runtime/app/worker.py`: `_write_prediction_error_node` carries forward dynamics-engine-owned metadata before overwriting a node under its fixed `node_id`.

## Files changed

- `orion/substrate/falkor_codec.py`: 5-key native property promotion (encode/decode), `_safe_float()` tolerant coercion, `DYNAMICS_METADATA_KEYS`/`DYNAMICS_ENGINE_OWNED_METADATA_KEYS`.
- `orion/substrate/falkor_store.py`: `NATIVE_NODE_RETURN_FIELDS` extended for cold-start hydration.
- `orion/substrate/tests/test_falkor_codec.py`: encode/decode/round-trip/no-dynamics-defaults tests for the 5 promoted keys.
- `orion/substrate/tests/test_falkor_store.py`: upsert-sets-native-properties + hydrate-into-metadata tests.
- `orion/substrate/tests/test_dynamics_falkor_routed.py` (new): library-level integration proof, including a simulated-process-restart round trip through native Cypher properties.
- `services/orion-substrate-runtime/app/worker.py`: `_write_prediction_error_node` preserves dynamics-engine-owned metadata across rewrites.
- `services/orion-substrate-runtime/tests/test_worker_falkor_routed_store.py` (new): worker-level proof for `_write_prediction_error_node`, `_dynamics_tick`, and env-wiring (`build_substrate_store_from_env` → Falkor-primary routed store), plus the rewrite-preserves-state regression.
- `docs/superpowers/specs/2026-07-16-cypher-native-substrate-postgres-bus-split-design.md`: acceptance evidence + status update.

## Schema / bus / API changes

- Added: none (no new bus channels, no new schema registry entries).
- Removed: none.
- Renamed: none.
- Behavior changed: Falkor durable writes for concept nodes now include 5 additional native Cypher scalar properties (`dynamic_pressure`, `dynamic_pressure_reason`, `dormant`, `dormancy_updated_at`, `prediction_error`); previously these were silently dropped. `_write_prediction_error_node`'s repeat-write behavior now preserves prior dynamics state instead of resetting it.
- Compatibility notes: Purely additive to the Falkor Cypher schema — no existing property renamed or removed. SPARQL store, non-concept node kinds, edges, and drive/goal/tension/contradiction handling are untouched.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no (verified existing Falkor/routed documentation in `services/orion-substrate-runtime/.env_example` already matches the code; no staleness found).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no env template change).
- skipped keys requiring operator action: none.

## Tests run

```text
PYTHONPATH=.:services/orion-substrate-runtime /tmp/orion-test-venv/bin/python3 -m pytest \
  services/orion-substrate-runtime/tests/test_worker_falkor_routed_store.py \
  services/orion-substrate-runtime/tests/test_worker_prediction_error_node.py \
  services/orion-substrate-runtime/tests/test_worker_dynamics_tick.py \
  orion/substrate/tests/ -q
→ 346 passed

git diff --check → clean
```

`scripts/check_schema_registry.py` / `scripts/check_bus_channels.py` referenced in the standard gate list do not exist in this repo (closest matches are `check_activation_saturation.py`, `check_single_consumer_channels.py`, neither applicable — no bus/schema change in this patch).

Note: the broader `services/orion-substrate-runtime/tests/` suite has 12 pre-existing failures / 9 pre-existing errors unrelated to this patch (confirmed identical against the branch's pre-patch state) — an `app.models` package-ambiguity issue between `orion-substrate-runtime`'s own `app` package and `orion-sql-writer`'s `app.models` when both are on `PYTHONPATH` together, and other pre-existing environment/fixture gaps. Not touched or worsened by this patch.

## Evals run

```text
No eval harness exists for the Falkor adapter / substrate-runtime dynamics seam; focused deterministic tests cover the codec, adapter, and worker-level integration.
```

## Docker/build/smoke checks

```text
No Docker smoke run. No runtime/config/dependency/port changes in this patch. Live Falkor restart smoke against a real Hub deployment remains required before any live cutover decision (tracked as an open acceptance check in the design spec, unchanged by this patch).

scripts/safe_graphify_update.sh REFUSED (~92% node-loss guard, the known 2026-07-14
destructive-update failure mode) and auto-restored graph.json/manifest.json; left
graphify-out unchanged rather than re-running or trusting the raw `graphify update`
output.
```

## Review findings fixed

- Finding (High): `_write_prediction_error_node`'s documented "re-writes collapse" behavior (fixed `node_id`, fresh `ConceptNodeV1` per call) durably clobbered `SubstrateDynamicsEngine.tick()`'s already-computed `dynamic_pressure`/`dormant` state back to defaults on every repeat prediction-error event, once the codec started always including those keys in the upsert's `SET` clause.
  - Fix: `_write_prediction_error_node` now best-effort looks up the existing node by `node_id` and carries forward `DYNAMICS_ENGINE_OWNED_METADATA_KEYS` before constructing the new node; lookup failure stays fail-open (some store test doubles only implement `upsert_node`).
  - Evidence: `test_write_prediction_error_node_preserves_dynamics_state_on_rewrite` in `services/orion-substrate-runtime/tests/test_worker_falkor_routed_store.py`; reviewer reproduced the bug live against the real code path before landing the fix.

## Restart required

```text
No restart required. This patch changes library/adapter code and adds tests only;
no service is currently configured to use SUBSTRATE_STORE_BACKEND=falkor/routed,
SUBSTRATE_WRITE_PREDICTION_ERROR_NODES, or SUBSTRATE_DYNAMICS_TICK_ENABLED by
default, so no running orion-substrate-runtime instance's behavior changes until
an operator deliberately flips those flags.
```

## Risks / concerns

- Severity: Medium
- Concern: Live cutover of `orion-substrate-runtime` to `SUBSTRATE_STORE_BACKEND=routed`/`falkor` still requires a real-Redis restart smoke (Concept Atlas survives restart check from the Slice 1 acceptance list is Hub-specific; an equivalent runtime-specific smoke has not been run against a live FalkorDB instance) before that flag flip should happen. This patch proves correctness against a scripted test double, not a live server.
- Severity: Low
- Concern: `bool(metadata.get("dormant", False))` would coerce a non-bool truthy string (e.g. `"false"`) to `True`; not reachable today since the only real producer (`dynamics.py`) always writes real Python bools, but worth remembering if a future producer writes this key from a different serialization path.

## PR link

<link — to be filled in after `gh pr create`>
