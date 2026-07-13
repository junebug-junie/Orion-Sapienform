# PR: orch route-arbitration visibility + shadow grammar lane

Branch: `feat/orch-route-grammar-lane` → `main`
Spec: `docs/superpowers/specs/2026-07-12-orch-route-grammar-lane-design.md`

## Summary

- Surfaces orch's per-turn arbitration facts (execution lane pick + reason, mind-gate decision + skip reason, output mode) — previously only `logger.info`/`logger.debug` lines — onto `VerbResultV1.output["_route_metadata"]` and the client-facing `final_meta.route_metadata`. Always on, no schema/bus cost.
- Adds a shadow-only (`PUBLISH_CORTEX_ORCH_GRAMMAR=false` by default) `GrammarEventV1` producer in orion-cortex-orch publishing those same facts under `trace_id` prefix `orch.route:`.
- Adds a fifth substrate-runtime grammar-reduction lane (`route_grammar`, `ENABLE_ROUTE_GRAMMAR_REDUCER=false` by default) that materializes those events into a capped `active_route_arbitration` projection.
- Corrects a design-spec error found mid-implementation: chat-lane requests were believed to produce no step-level grammar trace in cortex-exec. They already do, via `LegacyPlanVerb.execute()` in `verb_adapters.py` — a file the original research pass missed. The two patches that would have "fixed" that (nonexistent) gap were struck before any code was written.

## Outcome moved

Orch's routing/arbitration decisions — which lane, whether "mind" fired, output mode — go from invisible (log lines only) to inspectable on two independent surfaces: the actual client response, and (once flipped on) a durable substrate-runtime projection. This is a distinct signal from cortex-exec's existing execution-trajectory lane: it answers "what got selected and why," not "how did the selected thing execute."

## Current architecture

`orion-cortex-orch::call_verb_runtime()` already computed lane/mind/output-mode facts per turn but they terminated at `logger.info`/`logger.debug`. Substrate-runtime already had four grammar lanes (`biometrics`, `execution_trajectory`, `transport_bus`, `chat_grammar`), each a `(source_service, trace_id_prefix)`-filtered reducer package under `orion/substrate/<name>_loop/`, registered in a shared cursor/health framework (`store.py::GRAMMAR_CURSOR_REGISTRY`, `worker.py::REDUCER_SPECS`, `grammar_truth.py`'s two friendly-key maps).

## Architecture touched

- `orion-cortex-orch`: `orchestrator.py` (route_metadata computation + attach + shadow publish call), `main.py` (merge into `final_meta`), new `grammar_emit.py`/`grammar_publish.py`, `settings.py`, `.env_example`, `docker-compose.yml`.
- `orion-substrate-runtime`: new `route_grammar` lane wired into the existing generic reducer framework (`store.py`, `settings.py`, `worker.py`, `grammar_truth.py`, `.env_example`, `docker-compose.yml`).
- Shared: new `orion/substrate/route_loop/` package + `orion/schemas/route_projection.py`, `orion/bus/channels.yaml` (producer registration), `config/substrate-lattice/grammar_producer_registry.v1.yaml`, new manual migration.

## Files changed

- `services/orion-cortex-orch/app/orchestrator.py` — computes `route_metadata`, attaches to both `VerbResultV1` return paths, fires the shadow grammar publish once per call.
- `services/orion-cortex-orch/app/main.py` — merges `_route_metadata` into `final_meta` before the client response is built (was being silently dropped by the `output["result"]` unwrap otherwise).
- `services/orion-cortex-orch/app/grammar_emit.py` (new) — pure builder for the two-event `orch.route:` trace.
- `services/orion-cortex-orch/app/grammar_publish.py` (new) — fail-open bus-publish wrapper using the existing shared `orion.grammar.publish.publish_grammar_event` helper.
- `services/orion-cortex-orch/app/settings.py`, `.env_example`, `docker-compose.yml` — `PUBLISH_CORTEX_ORCH_GRAMMAR`, `GRAMMAR_EVENT_CHANNEL`.
- `orion/schemas/route_projection.py` (new) — `RouteArbitrationRunStateV1`/`RouteArbitrationProjectionV1`.
- `orion/substrate/route_loop/{__init__,constants,ids,grammar_extract,merge,reducer,projection,pipeline}.py` (new) — mirrors `execution_loop`'s structure, specifically its capped/evicted reducer (not `chat_loop`'s, which has no cap on its `turns` dict).
- `services/orion-substrate-runtime/app/{store,settings,worker,grammar_truth}.py`, `.env_example`, `docker-compose.yml` — fifth reducer lane wired into the existing generic framework.
- `services/orion-substrate-runtime/tests/test_brain_frame_worker.py` — extended with the cursor-key regression pattern from the `54997e89` phantom-lane fix.
- `services/orion-sql-db/manual_migration_route_substrate_loop.sql` (new) — not applied to any database by this PR.
- `config/substrate-lattice/grammar_producer_registry.v1.yaml` — new `orion-cortex-orch` entry; also fixed `orion-cortex-exec`'s stale `status: planned` (code default has been `true` since a prior PR).
- `orion/bus/channels.yaml` — added `orion-cortex-orch` to `orion:grammar:event`'s `producer_services`.
- `docs/superpowers/specs/2026-07-12-orch-route-grammar-lane-design.md` — design spec, with the mid-implementation correction documented inline.

## Schema / bus / API changes

- Added: `RouteArbitrationRunStateV1`, `RouteArbitrationProjectionV1` (not registered in `orion/schemas/registry.py` — matching `ChatSessionProjectionV1`'s existing precedent of not being registered there either; `ExecutionTrajectoryProjectionV1` is the one that is, so the codebase is inconsistent on this already).
- Added: `route_grammar_consumer` cursor on the existing `orion:grammar:event` channel (no new channel).
- Behavior changed: `orion:grammar:event`'s registered `producer_services` list now includes `orion-cortex-orch` (it was already implicitly a valid producer per schema, just undocumented until this PR made it a real one).
- Compatibility notes: everything is additive and default-off (`PUBLISH_CORTEX_ORCH_GRAMMAR=false`, `ENABLE_ROUTE_GRAMMAR_REDUCER=false`) except `VerbResultV1.output["_route_metadata"]`, which is always-on but additive (new dict key, nothing removed).

## Env/config changes

- Added keys: `PUBLISH_CORTEX_ORCH_GRAMMAR=false`, `GRAMMAR_EVENT_CHANNEL=orion:grammar:event` (orch); `ENABLE_ROUTE_GRAMMAR_REDUCER=false`, `ROUTE_GRAMMAR_BATCH_LIMIT=100` (substrate-runtime).
- `.env_example` updated: yes, both services.
- local `.env` synced: not applicable — this worktree has no local `.env` for either service (fresh worktree off `origin/main`); `python scripts/sync_local_env_from_example.py` was run by two of the three implementing subagents and reported nothing to sync for that reason. Operator must add these keys to their real machine's `.env` files before enabling either flag.
- skipped keys requiring operator action: none skipped by the sync script's allowlist; the DB migration (`manual_migration_route_substrate_loop.sql`) is the actual operator action required before `ENABLE_ROUTE_GRAMMAR_REDUCER=true` can be safely flipped.

## Tests run

```
/tmp/orion-test-venv/bin/python -m pytest services/orion-cortex-orch/tests/test_route_grammar_emit.py services/orion-cortex-orch/tests/test_lane_routing.py services/orion-cortex-orch/tests/test_verb_runtime_rpc.py services/orion-cortex-orch/tests/test_execution_lanes.py -q
→ 29 passed

/tmp/orion-test-venv/bin/python -m pytest orion/substrate/route_loop/ tests/test_route_substrate_reducer.py tests/test_execution_substrate_reducer.py services/orion-substrate-runtime/tests/test_brain_frame_worker.py -q
→ 45 passed
```
Includes a load-test regression (>2000 synthetic trace_ids) proving the `active_route_arbitration` projection stays capped and the just-written run in a batch is never evicted (mirroring the `8daeecf7` regression test).

Full `services/orion-substrate-runtime/tests/` run showed 14 pre-existing failures unrelated to this change (confirmed via `git stash` comparison against the same branch without this diff — order/module-global-state issues in `reducer_health`/`grammar_truth` test files). `services/orion-cortex-orch/tests/` has 5 files that fail at collection in this sandbox on a pre-existing `numpy.dtype size changed` ABI issue, unrelated to any file this PR touches.

## Evals run

No eval harness exists for either service's grammar-lane behavior beyond the unit/regression tests above.

## Docker/build/smoke checks

```
docker compose -f services/orion-cortex-orch/docker-compose.yml config -q      → exit 0
docker compose -f services/orion-substrate-runtime/docker-compose.yml config -q → exit 0
```
No live bus/Postgres smoke run — both new producers/reducers are default-off; enabling either is an explicit follow-up operator action gated by the DB migration above.

## Review findings fixed

- Finding: `main.py`'s `verb_result.output["result"]` unwrap dropped the `_route_metadata` sibling key before the client-facing response was built, so Patch E's data never actually reached a human despite being correctly attached upstream.
  - Fix: capture `_route_metadata` before the unwrap, merge into `final_meta["route_metadata"]` (mirroring the existing `final_meta["auto_route"]` pattern).
  - Evidence: `test_lane_routing.py`/`test_verb_runtime_rpc.py` still 21/21 green after the fix; confirmed by reading the unwrap logic directly.
- Finding: `orion:grammar:event`'s `producer_services` list in `orion/bus/channels.yaml` did not include `orion-cortex-orch`, though the new producer patch makes it a real one.
  - Fix: added the entry.
  - Evidence: direct diff review.
- Finding: `grammar_producer_registry.v1.yaml` had `orion-cortex-exec` marked `status: planned` despite `PUBLISH_CORTEX_EXEC_GRAMMAR` defaulting `true` in code since an earlier PR — stale, found while adding the new `orion-cortex-orch` entry.
  - Fix: corrected to `live`.
  - Evidence: direct diff review against `services/orion-cortex-exec/app/settings.py`.
- Finding: `docker-compose.yml` for both services didn't pass through the new env keys despite being added to `.env_example`/`settings.py`.
  - Fix: added passthrough lines to both.
  - Evidence: `docker compose config -q` validates clean on both files.

## Restart required

```bash
# Apply migration before enabling ENABLE_ROUTE_GRAMMAR_REDUCER=true:
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_route_substrate_loop.sql

# Then, after adding the new keys to each service's real .env:
docker compose --env-file .env --env-file services/orion-cortex-orch/.env -f services/orion-cortex-orch/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-substrate-runtime/.env -f services/orion-substrate-runtime/docker-compose.yml up -d --build
```
No restart is required to merge this PR as-is — both new producers/consumers are default-off, and `_route_metadata` on the result is a pure additive field.

## Risks / concerns

- Severity: low. Concern: one of the three parallel implementing subagents (`route_loop` reducer) used `git stash`/`git stash pop` mid-task in this shared worktree while a second subagent was concurrently writing files to `services/orion-cortex-orch/`. Verified after the fact that nothing was lost (all of the second agent's files were intact, diffs reviewed, tests green), but this was closer to luck than to a designed-safe operation — a stash/pop in a shared, concurrently-written worktree can silently interleave or drop another agent's uncommitted work. Mitigation: none needed for this PR (verified clean), but future multi-agent sessions in a shared worktree should avoid `git stash` entirely, or use isolated worktrees per agent.
- Severity: low. Concern: `route_grammar`'s eviction cap (`ROUTE_ARBITRATION_MAX_RUNS`/`MAX_AGE_SEC`) is hardcoded rather than settings-configurable, unlike `execution_trajectory`'s (which gained settings fields in the `8daeecf7` fix after hitting production volume). Mitigation: acceptable for a shadow-only, default-off first cut; add settings fields if/when this lane is enabled at real volume.
- Severity: none (informational). `RouteArbitrationProjectionV1` is not registered in `orion/schemas/registry.py`, matching `ChatSessionProjectionV1`'s existing precedent (only `ExecutionTrajectoryProjectionV1` is registered there) — the codebase is already inconsistent on this, not a regression introduced here.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/orch-route-grammar-lane

(`gh pr create` could not run in this sandbox — `gh auth login` was not configured. Branch `feat/orch-route-grammar-lane` is pushed; open the PR via the link above with this file's contents as the description.)
