# feat(world-pulse): Orion went looking — curiosity followups in the digest

## Summary

- World-pulse now does an inline **gap-fill web fetch** for under-covered digest sections (reusing the existing `web.fetch.readonly` capability policy + `execute_readonly_fetch`) and attaches results to the digest as `curiosity_followups`, rendered as an **"Orion went looking"** block.
- Findings ride the **existing** `world.pulse.run.result.v1` event (no new bus event) via `WorldPulseRunResultV1.digest`.
- The concept-induction worker **reuses** a matching followup instead of doing a second Firecrawl call: `select_reusable_followup` + `outcome_from_followup` → `prefetched_outcome` on `maybe_execute_substrate_act_after_metabolism`. Live fetch remains the fallback.
- Disabled by default; effective on/off = `WORLD_PULSE_CURIOSITY_FETCH_ENABLED` **AND** `ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED`. Skipped on dry runs. Builder never raises.

## Outcome moved

Single shared fetch: an under-covered section is fetched **once** (by world-pulse) and reused downstream by the reactive episode-journal loop, eliminating a duplicate billable Firecrawl call and surfacing the gap-fill findings to the human in the digest.

## Current architecture (before)

World-pulse computed section coverage but never acted on gaps; the reactive concept-induction loop independently ran its own readonly fetch when reacting to `world.pulse.run.result.v1`.

## Architecture touched

- Producer: `orion-world-pulse` pipeline gains a gated, dry-run-skipped inline fetch writing `DailyWorldPulseV1.curiosity_followups`.
- Contract: additive field on `DailyWorldPulseV1` (already registry-registered by class reference; no registry edit).
- Consumer: `orion-spark-concept-induction` bus worker reads matching followups and passes `prefetched_outcome` into the substrate-act path.
- Shared: `orion/autonomy` gains `tokenize_terms`, `curiosity_reuse.py`, and a `prefetched_outcome` param on `policy_act`.

## Files changed

- `orion/schemas/world_pulse.py`: `CuriosityFindingV1`, `CuriosityFollowupV1`, `DailyWorldPulseV1.curiosity_followups`.
- `orion/autonomy/salience.py`: public `tokenize_terms` wrapper.
- `orion/autonomy/curiosity_reuse.py` (new): `select_reusable_followup`, `outcome_from_followup`.
- `orion/autonomy/policy_act.py`: `prefetched_outcome` short-circuit (None-path byte-identical).
- `orion/spark/concept_induction/bus_worker.py`: reuse wiring in the `world.pulse.run.result.v1` act branch.
- `services/orion-world-pulse/app/services/curiosity.py` (new): `build_curiosity_followups` (capability gate + per-section fetch + map; gate guarded so it never fails the run).
- `services/orion-world-pulse/app/services/{digest,pipeline,renderers}.py`: passthrough, wiring, and the "Orion went looking" block.
- `services/orion-world-pulse/app/settings.py` + `.env_example`: 3 `WORLD_PULSE_CURIOSITY_*` flags (off/5/9).
- `services/orion-world-pulse/Dockerfile`: copies `config/autonomy` so the runtime capability gate can load its policy.
- Tests added across all touched packages, incl. a worker-level reuse-wiring integration test.

## Schema / bus / API changes

- Added: `CuriosityFindingV1`, `CuriosityFollowupV1`, `DailyWorldPulseV1.curiosity_followups` (default `[]`).
- Removed / Renamed: none.
- Behavior changed: `build_digest` gained an optional `curiosity_followups` param; `maybe_execute_substrate_act_after_metabolism` gained an optional `prefetched_outcome` kwarg. Both backward-compatible (defaulted).
- Compatibility: additive; `world.pulse.run.result.v1` payload unchanged except the new digest field. No registry edit needed (class-reference registration).

## Env/config changes

- Added keys: `WORLD_PULSE_CURIOSITY_FETCH_ENABLED` (false), `WORLD_PULSE_CURIOSITY_MAX_ARTICLES_PER_SECTION` (5), `WORLD_PULSE_CURIOSITY_MAX_SECTIONS` (9).
- `.env_example` updated: yes.
- local `.env` sync: `scripts/sync_local_env_from_example.py` skips the worktree (no local `.env` there). The keys were added to the operator `.env` at `services/orion-world-pulse/.env` (main checkout) with safe defaults. **After merge, re-run `python scripts/sync_local_env_from_example.py` from the main checkout to formalize.**
- Skipped keys requiring operator action: none new. (Repo's `check_env_template_parity.py` / `check_schema_registry.py` / `check_bus_channels.py` do not exist in this repo; env sync + import checks were used instead.)

## Tests run

```text
# from repo root (worktree), main-checkout venv
pytest services/orion-world-pulse/tests   -> 68 passed
pytest orion/autonomy/tests               -> 125 passed
pytest services/orion-spark-concept-induction/tests -> 5 passed (incl. new reuse-wiring test)
# per-task TDD: schema(2), tokenize(3), settings(2), curiosity(7), pipeline(2), renderers(3),
#               curiosity_reuse(4), policy_act_prefetched(1)
```

Pre-existing (not regressions), verified identical on baseline:
- `test_world_pulse_pipeline.py::{test_run_world_pulse_with_fixture_fetch,test_pipeline_fixture_mode_without_network}` fail only from the service dir (cwd-relative `config/world_pulse/sources.yaml`); pass from repo root.
- `test_graph_profile_repository.py::TestGraphReadModel::{test_shadow_parity_logs_graph_unavailable_reason,test_shadow_parity_logs_expected_mismatch_fields}` fail only under the large combined run (cross-test ordering); pass in isolation.

## Docker/build/smoke checks

```text
docker compose config (world-pulse)          -> world-pulse-config-ok
docker compose config (concept-induction)    -> concept-induction-config-ok
docker build world-pulse image (from branch) -> success
docker run <img> import app.services.curiosity + load_capability_policy() -> IMPORT_OK, CAPABILITY_POLICY_LOADED rules=4
```

## Review findings fixed

- **world-pulse image didn't ship `config/autonomy`** → enabling the feature in-container would miss the capability policy; the unguarded gate call could crash the run. Fix: `COPY config/autonomy` in the Dockerfile + wrap `_gate_open` so a policy load/eval failure degrades to `[]`. Evidence: in-image `load_capability_policy()` returns 4 rules; new `test_gate_evaluation_error_degrades_to_empty`.
- **Highest-risk seam (worker reuse wiring) had no integration test.** Fix: `test_curiosity_reuse_wiring.py` drives a real `world.pulse.run.result.v1` envelope through `ConceptWorker.handle_envelope` with the real select/outcome/policy_act path; asserts fetch backend call count == 0 and emitted `action_id` == the followup's. settrace confirmed the new `bus_worker.py` lines execute.
- **`build_digest` optional param redundant with attribute-attach** (both spec-requested) — kept as documented; harmless (Low).

## Restart required

Run from the **main checkout** after merge (do not skip the env sync):

```bash
python scripts/sync_local_env_from_example.py
docker compose --env-file .env --env-file services/orion-world-pulse/.env -f services/orion-world-pulse/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-spark-concept-induction/.env -f services/orion-spark-concept-induction/docker-compose.yml up -d --build
```

To exercise the feature (billable): set `WORLD_PULSE_CURIOSITY_FETCH_ENABLED=true`, `ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED=true`, ensure `FIRECRAWL_API_KEY` present, and `ORION_AUTONOMY_EPISODE_JOURNAL_ENABLED=true` on the consumer.

## Live smoke (Task 11 — operator-run, NOT executed here)

Not run: requires enabling billable Firecrawl fetches on the live deployment.
1. `curl -sS -X POST http://localhost:8628/api/world-pulse/run -H 'content-type: application/json' -d '{"dry_run": false, "requested_by":"manual"}'` → expect >=1 followup with >=1 article and an `action_id`.
2. Concept-induction logs: expect `wp_curiosity_followup_reused ... action_id=<X>` matching a followup, then journal-compose/episode-journal/action-outcome lines, and **no** `web.fetch.readonly` execution for that reused section.

## Risks / concerns

- Low — Reuse matches only the **first** gap-section label; a mismatch yields a second live fetch (graceful fallback, billable miss).
- Low — `append_action_outcome` runs producer-side but not on the reused consumer path. Shared SQL store → one row; per-container file store → audit row only in world-pulse. `ActionOutcomeEmitV1` bus emit fires on both paths (source of truth). Ensure `ORION_ACTION_OUTCOME_STORE_PATH` writable in world-pulse.
- Low — `build_curiosity_followups` uses `asyncio.run` per section; safe because `run_world_pulse` is synchronous.

## Status

DONE_WITH_CONCERNS (concerns are all Low, documented above). PR must be opened by an authenticated operator — see instructions below.
