# PR report: kill dead drive-seed pressure, give harness-closure prediction-error a shared accumulating node

## Summary

- Removed `drive_seed_pressure()` from `orion/substrate/pressure.py` and its BFS propagation loop from `SubstrateDynamicsEngine._compute_pressures()` (`orion/substrate/dynamics.py`) -- dead code for the retired drive-taxonomy system. Live query against `orion_substrate` confirmed zero `node_kind=="drive"` nodes exist; this matches the Sentience Striving Program's formal, deliberate halt of DriveEngine.
- Removed the now-unreferenced `PressureConfig.drive_base` / `drive_propagation_attenuation` fields after confirming (repo-wide grep) nothing else reads them.
- `services/orion-substrate-runtime/app/worker.py`'s `_write_prediction_error_node()` gained an optional `contributing_id: str | None = None` param. Repeat writes to a shared node_id now carry forward and accumulate bounded per-turn attribution in `metadata['contributing_turn_ids']` (capped at 20, oldest dropped, deduped on same-id re-fire), via the same `get_node_by_id` lookup already used to carry forward `DYNAMICS_ENGINE_OWNED_METADATA_KEYS`.
- `handle_post_turn_closure()` now writes the harness-closure prediction-error lane to a fixed node_id `node:substrate.harness_closure` (was `harness_closure:<correlation_id>`, a brand-new node per turn that could never accumulate), matching the existing `node:substrate.execution` / `node:substrate.transport` pattern, and passes `contributing_id=closure.correlation_id`.
- `_execution_tick` / `_transport_tick` now also pass their last-processed grammar-event id as `contributing_id`, for consistency (it was already in scope, no extra plumbing).
- Added/updated tests in `orion/substrate/tests/` (removal side -- no dedicated drive-seed tests existed to remove; confirmed via grep) and `services/orion-substrate-runtime/tests/` (new node_id + accumulation + cap + dedup + malformed-value + non-string-item test coverage).

## Outcome moved

Intended outcome: the one live prediction-error signal lane (harness-turn surprise) should be able to sustain `dynamic_pressure` past the 30-minute decay horizon when surprise keeps recurring, instead of each surprising turn producing an isolated node that always decays out alone.

**This outcome is NOT fully achieved against the repo's own documented default runtime config -- see "Risks / concerns" below.** The in-process logic (dedup/cap/carry-forward, fixed shared node_id, fail-open contract) is correct and fully tested. But `contributing_turn_ids` specifically does not survive durably against the live `SUBSTRATE_STORE_BACKEND=falkor` path, because of a pre-existing closed-allowlist design in `orion/substrate/falkor_codec.py` that this task's brief explicitly listed as out of scope to touch. This is disclosed prominently, not silently shipped -- see below.

## Current architecture

`SubstrateDynamicsEngine._compute_pressures()` (`orion/substrate/dynamics.py`) previously seeded `dynamic_pressure` from three sources: drive-seed (dead -- zero drive nodes exist), prediction-error (the one live lane), and contradiction-amplification. The harness-closure branch of the prediction-error lane wrote a brand-new `node_id=f"harness_closure:{correlation_id}"` per turn in `_write_prediction_error_node()`, unlike the two sibling producers (`_execution_tick`/`_transport_tick`) which already used fixed node_ids (`node:substrate.execution` / `node:substrate.transport`) so repeated surprise could re-observe and sustain the same node.

## Architecture touched

- `orion/substrate/pressure.py`, `orion/substrate/dynamics.py`: dead-code removal only, no behavior change to the two live pressure lanes (prediction-error, contradiction).
- `services/orion-substrate-runtime/app/worker.py`: `_write_prediction_error_node()` signature and body, all three of its call sites (`handle_post_turn_closure`, `_execution_tick`, `_transport_tick`).
- `services/orion-substrate-runtime/tests/`: `test_worker_prediction_error_node.py`, `test_post_turn_closure_listener.py`.

## Files changed

- `orion/substrate/pressure.py`: deleted `drive_seed_pressure()` and `PressureConfig.drive_base` / `drive_propagation_attenuation`.
- `orion/substrate/dynamics.py`: deleted the drive-seed import and its BFS propagation loop in `_compute_pressures()`.
- `services/orion-substrate-runtime/app/worker.py`: `_write_prediction_error_node()` gains `contributing_id` param + `contributing_turn_ids` carry-forward/append/cap logic; `handle_post_turn_closure()` uses fixed `node:substrate.harness_closure` node_id + passes `contributing_id`; `_execution_tick`/`_transport_tick` pass `contributing_id=last_id`.
- `services/orion-substrate-runtime/tests/test_post_turn_closure_listener.py`: updated the existing node_id assertion for the new fixed node_id + `contributing_id`; added two tests proving two different-correlation-id closures land on the shared node and accumulate, and that a repeat same-correlation-id closure does not duplicate.
- `services/orion-substrate-runtime/tests/test_worker_prediction_error_node.py`: added a `_RecordingStore` fake (persists nodes by node_id, supports `get_node_by_id`, so carry-forward is genuinely exercised) and 7 new tests covering accumulation, dedup, cap-and-drop-oldest, malformed stored value, non-string stored items, an item whose `__str__` raises, and the no-`contributing_id`-provided case.

## Schema / bus / API changes

- Added: `metadata['contributing_turn_ids']` (list[str], capped at 20) on prediction-error concept nodes written via `_write_prediction_error_node()`, when a `contributing_id` is supplied.
- Removed: `drive_seed_pressure()` function; `PressureConfig.drive_base` / `drive_propagation_attenuation` fields; the `"drive_seed"` / `"drive_propagation:<predicate>"` values that used to appear in `metadata['dynamic_pressure_reason']` can no longer be produced (no production code path ever writes them again).
- Renamed: harness-closure prediction-error node_id changed from `harness_closure:<correlation_id>` (per-turn) to `node:substrate.harness_closure` (fixed, shared).
- Behavior changed: repeated harness-turn surprise now writes to the same node instead of spawning a new one each time; `_execution_tick`/`_transport_tick` prediction-error writes now also record `contributing_id` (their last-processed grammar-event id) in the same accumulating field.
- Compatibility notes: old `harness_closure:<correlation_id>` nodes already durably written before this deploy are orphaned (no longer written to or read from) but not deleted -- they will simply decay out on their existing 30-minute horizon and stop being touched. No migration needed. `services/orion-substrate-runtime/tests/test_turn_referent_store.py`'s `coalition_ref == "harness_closure:corr-1"` assertion is a *different*, unrelated field (`app/turn_referent_store.py`'s own Postgres `coalition_ref` column, not the substrate graph node_id) and was intentionally left untouched.

## Env/config changes

None. No `.env_example` keys added, removed, or renamed.

## Tests run

```text
/mnt/scripts/Orion-Sapienform/venv/bin/python -m pytest orion/substrate/tests -q
→ 370 passed

/mnt/scripts/Orion-Sapienform/venv/bin/python -m pytest services/orion-substrate-runtime/tests -q \
  --ignore=services/orion-substrate-runtime/tests/test_grammar_consumer_integration.py
→ 143 passed, 16 failed

/mnt/scripts/Orion-Sapienform/venv/bin/python -m pytest services/orion-substrate-runtime/tests/test_worker_prediction_error_node.py services/orion-substrate-runtime/tests/test_post_turn_closure_listener.py -q
→ 18 passed
```

The 16 failures and the 1 collection error (`test_grammar_consumer_integration.py`, `ModuleNotFoundError: No module named 'app.models'`, a cross-service `sys.path` collision when `orion-sql-writer`'s and `orion-substrate-runtime`'s `app` packages both resolve from repo root) are **pre-existing**, verified by `git stash`-ing this branch's changes and re-running the identical suite against unmodified `main` -- same 16 test names fail, same collection error, byte-for-byte. Not introduced by this patch. One of the 16 (`test_worker_falkor_routed_store.py::test_write_prediction_error_node_preserves_dynamics_state_on_rewrite`) touches the same function this patch modifies but fails identically on `main` before any of these changes -- confirmed via the same stash-and-rerun.

## Evals run

No eval harness exists for `orion/substrate` or `orion-substrate-runtime` beyond the pytest suites above (checked `orion/substrate/evals/` -- has evals for repair-pressure, none for the drive-seed/prediction-error pressure lanes touched here; checked `services/orion-substrate-runtime/` -- no `evals/` directory). Follow-up: none proposed specifically for this patch; the existing gap predates it.

## Docker/build/smoke checks

Not run. This patch touches a live-running service's code (`orion-substrate-runtime`), so a container rebuild/restart is required before the fix takes effect (see "Restart required" below), but no compose/env/Dockerfile changes were made, so `docker compose config` validation is not applicable here.

## Review findings fixed

- Finding (must-fix, review subagent + independently confirmed via code trace): `contributing_turn_ids` is silently dropped by the real Falkor-backed production store. `FalkorSubstrateStore.upsert_node()` → `encode_node_properties()` (`orion/substrate/falkor_codec.py`) only promotes a closed 5-key allowlist (`DYNAMICS_METADATA_KEYS`) to native Cypher properties; `contributing_turn_ids` is not in it. `FalkorSubstrateStore.get_node_by_id()` reads an in-process cache that briefly holds the full object after a write, but that cache is rebuilt from durable Falkor rows via `_hydrate_from_durable()` on essentially every `SubstrateDynamicsEngine.tick()` (any upsert bumps `_write_generation`, forcing the next `snapshot()` to re-hydrate; dynamics ticks run on a ~30s interval), and `decode_concept_node()` reconstructs metadata via the same closed allowlist -- so `contributing_turn_ids` is wiped from the effective read path within seconds to tens of seconds of any substrate write, not durable across turns spread more than that apart.
  - Fix: **not applied** -- the concrete fix (adding `contributing_turn_ids` to `DYNAMICS_METADATA_KEYS`/`_dynamics_properties_from_metadata`/`_dynamics_metadata_from_row` in `orion/substrate/falkor_codec.py`, JSON-encoded since Falkor properties are scalar) requires editing `orion/substrate/falkor_codec.py`, which this task's brief explicitly listed as a non-goal ("Do not touch `DYNAMICS_ENGINE_OWNED_METADATA_KEYS` or `falkor_codec.py`"). Rather than silently override an explicit scope boundary on a file this repo's own comments flag as deliberately closed ("Do NOT add a generic metadata-dump path here"), this is disclosed here as the primary concern for this PR. See "Risks / concerns" below.
  - Evidence: traced end-to-end via `orion/substrate/falkor_store.py:249-572` (`FalkorSubstrateStore`), `orion/substrate/falkor_codec.py:103-286` (`encode_node_properties`, `DYNAMICS_METADATA_KEYS`, `decode_concept_node`), `services/orion-substrate-runtime/app/settings.py` (`dynamics_tick_interval_sec` default ~30s), and confirmed by contrast that `orion/substrate/graphdb_store.py`'s `GraphDBSubstrateStore` (Fuseki backend) persists the full `node.model_dump()` as `payload_json` with no allowlist restriction -- so this gap is specific to the Falkor Cypher-native backend, which is the one confirmed live via `SUBSTRATE_STORE_BACKEND=falkor`.
- Finding (should-fix, review subagent): the `str(item)` coercion of stored `contributing_turn_ids` list items, and its surrounding fail-open `try/except`, were implemented but had no test exercising non-string stored items or an item whose `__str__` raises.
  - Fix: added `test_contributing_turn_ids_coerces_non_string_stored_items` and `test_contributing_turn_ids_tolerates_item_that_raises_on_str` in `services/orion-substrate-runtime/tests/test_worker_prediction_error_node.py`.
  - Evidence: `pytest services/orion-substrate-runtime/tests/test_worker_prediction_error_node.py -q` → all pass, including the two new cases.
- Finding (nit, review subagent, out of scope): `orion/substrate/tests/test_attention_broadcast.py`'s `test_drive_seed_reason_types_as_pressure_despite_stale_raw_prediction_error` and `orion/substrate/tests/test_falkor_store.py` construct fixtures with `dynamic_pressure_reason="drive_seed"`, and the former's docstring calls that "what a real tick would persist" -- no longer true after this diff, since the only code path that ever wrote `"drive_seed"` is deleted. Neither test calls the deleted function and both still validate real, useful behavior (generic reason-string typing / codec round-tripping), so they still pass. Left untouched per the reviewer's own assessment that this is outside this diff's stated scope (removal of test files/cases was scoped to tests that directly exercise `drive_seed_pressure()` or drive-seed propagation, of which none exist in `orion/substrate/tests/`). Flagging here as a documentation-staleness item for a future small follow-up.
- Finding (nit, review subagent, pre-existing/adjacent, not fixed): the pre-existing `DYNAMICS_ENGINE_OWNED_METADATA_KEYS` carry-forward loop directly above the new `contributing_turn_ids` logic (`for key in DYNAMICS_ENGINE_OWNED_METADATA_KEYS: if key in existing.metadata: ...`) is not itself wrapped in its own inner try, unlike the new logic. Not introduced by this diff; still covered by the outer method-level `try/except` so the fail-open contract holds. Not touched, per "keep the patch thin."

## Restart required

`orion-substrate-runtime` is a live-running service and must be rebuilt/restarted to pick up this code change:

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-substrate-runtime/.env \
  -f services/orion-substrate-runtime/docker-compose.yml \
  up -d --build
```

## Risks / concerns

- Severity: **High (material, blocks the stated live-verification acceptance criterion)**
  Concern: `contributing_turn_ids` does not durably survive against the live Falkor-backed production store (see "Review findings fixed" above for the full trace). Checking `orion_substrate` FalkorDB directly for `node:substrate.harness_closure.contributing_turn_ids` after multiple harness turns will very likely show either the key absent or only the most recent single id, not an accumulated list across turns spread more than ~30 seconds apart -- because the periodic `_hydrate_from_durable()` cache rebuild silently drops it on every refresh.
  Mitigation: the fixed shared node_id (`node:substrate.harness_closure` replacing per-turn node_ids) and the `dynamic_pressure`/`prediction_error` accumulation themselves *are* durable and correct -- those two fields *are* in `DYNAMICS_METADATA_KEYS`. Only the `contributing_turn_ids` attribution list is affected; the underlying pressure-sustaining behavior this task set out to fix (repeated surprise sustaining `dynamic_pressure` past the 30-minute decay horizon) is intact and durable. Concrete follow-up: a small, separate patch adding `contributing_turn_ids` to `orion/substrate/falkor_codec.py`'s `DYNAMICS_METADATA_KEYS` (JSON-encoded, since Falkor properties are scalar) plus the corresponding encode/decode functions, scoped and reviewed on its own given this file's deliberately closed allowlist design.

- Severity: Low
  Concern: pre-existing test-suite noise (16 failing tests, 1 collection error) in `services/orion-substrate-runtime/tests` unrelated to this patch, confirmed via stash-and-rerun against unmodified `main`.
  Mitigation: none needed for this PR; tracked here for visibility, not introduced or worsened by this change.

## PR link

Branch pushed: `feat/harness-closure-pressure-accumulation`. Run `gh pr create` from this worktree to open the PR (not run automatically here -- see final response for the exact command/link).
