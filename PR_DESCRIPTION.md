# fix(mind-enrichment): raise wall/timeout budget so LLM synthesis can complete

Branch: `fix/mind-enrichment-wall-budget` → `main`
Open PR: https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/mind-enrichment-wall-budget

## Summary

- The `orion-thought` Mind stance enrichment (#873) enabled orion-mind's 3-phase LLM pipeline (semantic `quick` → appraisal `metacog` → stance `chat`, each capped by `MIND_LLM_TIMEOUT_SEC=25s`) but budgeted the whole run at `ORION_THOUGHT_MIND_WALL_MS=12000` (12s). `MindRunBudget.phase_timeout_sec` clamps each phase to the remaining wall, so semantic synthesis RPC-timed-out at ~12s, tripped `wall_time_exceeded_after_semantic`, and fail-opened to `contract_only` on **every** run — the coloring (requires `meaningful_synthesis`) never fired. Empty-shell cognition per `AGENTS.md`.
- Raise `ORION_THOUGHT_MIND_WALL_MS` default `12000 → 90000` (≥ 3× the 25s per-phase timeout) and `ORION_THOUGHT_MIND_TIMEOUT_SEC` default `15 → 100` (HTTP read must exceed the wall so Mind's own fail-open result returns instead of the client aborting).
- Add `MIND_ENRICHMENT_MIN_VIABLE_WALL_MS` (=75000) + `mind_enrichment_config_warnings()` — a boot-time, log-only coherence check that warns when the wall is sub-viable or the HTTP timeout ≤ wall while enrichment is enabled.
- Guard `_projection_starvation_summary` on `present`: the light enrichment path intentionally sends no cognitive projection, so stop emitting the misleading "projection starved at Orch preflight" summary that misdirected root-causing (the machine key `mind.projection_starved` was already `present`-gated; this aligns the human summary with it).
- Regression tests: settings viability/warnings (invariant, not literal), budget phase-timeout clamp (12s non-viable vs 90s viable), light-path no-false-starvation.

## Outcome moved

Mind enrichment can now actually complete LLM synthesis and emit `meaningful_synthesis` coloring instead of degrading to `contract_only` on every turn. Diagnostic summary no longer falsely blames projection starvation on the light path.

## Root cause (from the reported trace)

`stance_react` run `13771964-…` (heavy emotional user turn) → `semantic_synthesis` RPC timeout at **12143 ms** on `orion:mind:llm:reply` → `wall_time_exceeded_after_semantic` → `llm_fail_open_to_deterministic` → `fallback_contract_only`. `12143ms ≈ wall(12000) − safety`, i.e. the wall — not the gateway — was the ceiling.

## Current architecture (before this patch)

`orion-thought._maybe_build_mind_coloring` → `build_light_mind_request(wall_time_ms=settings.mind_wall_ms)` → HTTP POST `orion-mind /v1/mind/run` → `run_mind` → `run_mind_llm_synthesis` (budget = `MindRunBudget(wall)`, per-phase timeout = `min(MIND_LLM_TIMEOUT_SEC, remaining − safety)`). With a 12s wall, the first phase alone exceeded budget → fail-open.

## Architecture touched

- `orion-thought` config surface (wall/timeout defaults + boot coherence advisory).
- `orion-mind` engine diagnostic-summary gating (no behavior change to the LLM/deterministic decision path).

## Files changed

- `services/orion-thought/app/settings.py`: new defaults; `MIND_LLM_TIMEOUT_SEC_ASSUMED`/`MIND_ENRICHMENT_PHASE_COUNT`/`MIND_ENRICHMENT_MIN_VIABLE_WALL_MS`; `mind_enrichment_config_warnings()` + boot log
- `services/orion-thought/docker-compose.yml`, `.env_example`, `README.md`: defaults + invariant/latency docs
- `services/orion-mind/app/engine.py`: `_projection_starvation_summary(..., present=...)` guard + caller
- `services/orion-thought/tests/test_settings_mind_enrichment.py`: hermetic reload, viability + warning regressions
- `services/orion-mind/tests/test_engine_budget.py`: phase-timeout clamp viability
- `services/orion-mind/tests/test_projection_starvation.py`: light-path no-false-starvation
# fix(world-pulse): derive Firecrawl key from FCC mount, not .env

## Summary

- World-pulse's "Orion went looking" curiosity readonly fetch needs a Firecrawl key, but the service compose had **no FCC mount** and no key wiring.
- On a real/CI deploy (or after any `up_all_services_batched.sh` restart) world-pulse would silently fall back to the stub backend and fetch nothing — or force an operator to paste a raw secret into the gitignored `.env`, which is not portable and drifts.
- This patch mirrors `orion-spark-concept-induction`: bind-mount host `~/.fcc` read-only and derive the key via `ORION_FCC_ENV_PATH`. `FIRECRAWL_API_KEY` stays **blank** in `.env` / `.env_example` — no secret in the repo or in env files.
- Result: `up_all_services_batched.sh` brings world-pulse up with a working curiosity fetch and zero secret duplication, independent of any other container.

## Outcome moved

Failure mode fixed: world-pulse curiosity fetch no longer requires a raw secret in `.env` and no longer regresses to the stub backend on a clean/batched bring-up. The Firecrawl key now resolves from the same single source of truth (`~/.fcc/.env`) the rest of the mesh uses.

## Current architecture

- `services/orion-world-pulse/app/services/curiosity.py` calls `resolve_fetch_backend()` → `resolve_firecrawl_api_key()` (`orion/autonomy/fetch_backend_resolve.py`), which checks `FIRECRAWL_API_KEY` env first, then falls back to `ORION_FCC_ENV_PATH` (default `~/.fcc/.env`).
- `orion-spark-concept-induction` already mounts `${HOME}/.fcc:/root/.fcc:ro` and sets `ORION_FCC_ENV_PATH=/root/.fcc/.env`.
- `orion-world-pulse/docker-compose.yml` had **no** `volumes:` block, so the FCC file was never present in the container → key never resolved → stub backend.

## Architecture touched

- world-pulse runtime packaging only. No bus/schema/API contract changes.

## Files changed

- `services/orion-world-pulse/docker-compose.yml`: add read-only `${HOME}/.fcc:/root/.fcc:ro` volume mount (mirrors concept-induction).
- `services/orion-world-pulse/.env_example`: document `ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED` (default `false`), `ORION_EPISODE_FETCH_BACKEND=auto`, `ORION_FCC_ENV_PATH=/root/.fcc/.env`, and a blank `FIRECRAWL_API_KEY=` (secret derived from mount, never stored).

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: `_projection_starvation_summary` returns "" when no projection was present (light path); no change to the machine contract keys or decision routing.
- Compatibility notes: no env keys added/renamed — only default values changed.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- Changed defaults: `ORION_THOUGHT_MIND_WALL_MS` `12000→90000`, `ORION_THOUGHT_MIND_TIMEOUT_SEC` `15→100`
- `.env_example` updated: yes
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (also restored operator toggles `ENABLED`/`ARTIFACT_PUBLISH` the sync had reset)
- skipped keys requiring operator action: none

## Tests run

```text
# orion-thought (hermetic)
PYTHONPATH=services/orion-thought:. .venv/bin/python -m pytest \
  services/orion-thought/tests/test_settings_mind_enrichment.py \
  services/orion-thought/tests/test_mind_light_snapshot.py \
  services/orion-thought/tests/test_mind_enrichment_fail_open.py \
  services/orion-thought/tests/test_mind_artifact_mode_tag.py \
  services/orion-thought/tests/test_mind_http_client.py \
  services/orion-thought/tests/test_mind_coloring_selector.py -q
→ 33 passed

# orion-mind touched (hermetic: LLM/bus off, matching CI)
MIND_LLM_SYNTHESIS_ENABLED=false MIND_LLM_USE_BUS=false ORION_BUS_ENABLED=false \
PYTHONPATH=services/orion-mind:. .venv/bin/python -m pytest \
  services/orion-mind/tests/test_engine_budget.py \
  services/orion-mind/tests/test_projection_starvation.py -q
→ 6 passed
```

Pre-existing, unrelated local-`.env` leak failures (salience/reverie default-off, rich-projection needs LLM off, metacog needs bus on) were confirmed to fail identically on `main` without this diff and pass hermetically.

## Evals run

```text
No new eval harness added. The "synthesis reachable" end-to-end claim (real gateway producing meaningful_synthesis under the 90s wall) is UNVERIFIED without the live LLM bus; deterministic gates prove the budget is now dimensioned to allow all three phases.
```

## Docker/build/smoke checks

```text
Not run in this environment. No Dockerfile/dependency/port/health changes — only env default values.
- Behavior changed: none
- Compatibility notes: purely additive compose volume + env template docs.

## Env/config changes

- Added keys (documented in `.env_example`): `ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED`, `ORION_EPISODE_FETCH_BACKEND`, `ORION_FCC_ENV_PATH`, `FIRECRAWL_API_KEY` (blank placeholder).
- `.env_example` updated: yes
- local `.env` synced: operator `.env` updated in place (raw key removed; set `ORION_FCC_ENV_PATH=/root/.fcc/.env`, `ORION_EPISODE_FETCH_BACKEND=auto`, blank `FIRECRAWL_API_KEY=`). `.env` remains gitignored.
- skipped keys requiring operator action: none. Host must have `~/.fcc/.env` containing `FIRECRAWL_API_KEY` (already present on the athena node and used by concept-induction).

## Tests run

```text
docker compose --env-file .env --env-file services/orion-world-pulse/.env \
  -f services/orion-world-pulse/docker-compose.yml config
# -> config OK; volumes source=/home/athena/.fcc target=/root/.fcc (ro); ORION_FCC_ENV_PATH resolved
```

## Evals run

```text
No eval harness change. Curiosity behavior covered by the merged feature's
services/orion-world-pulse/tests/test_curiosity.py (unchanged).
```

## Docker/build/smoke checks (live, athena node)

```text
# env key blank, mount present, key derived:
ENV_FIRECRAWL_LEN=0
FCC_PATH=/root/.fcc/.env
FCC_FILE_KEY_LEN=35
resolve_firecrawl_api_key() -> len 35 ; backend = firecrawl_search_backend

# producer run 8332fa96 (non-dry):
world_pulse_curiosity_gate outcome=allowed reason=allowed auto_execute=True
world_pulse_curiosity_followups sections=1 articles=5
digest section=hardware_compute_gpu articles=5 (real GPU titles)

# consumer reuse (single-shared-fetch, no second fetch):
wp_curiosity_followup_reused run_id=8332fa96 section=hardware_compute_gpu action_id=fetch-8332fa96...-85246d3d articles=5
substrate_policy_act capability=journal.compose.episode outcome=allowed
substrate_episode_journal_dispatched ; action_outcome_emitted success=True ; wp_stream_processed
```

## Review findings fixed

- Finding (prior review): 12s wall makes synthesis structurally impossible (Critical).
  - Fix: wall 90000, timeout 100, viability invariant + boot check.
  - Evidence: `test_default_wall_is_viable_for_three_phase_synthesis`, `test_engine_budget.py` clamp tests.
- Finding (prior review): tests only assert fail-open / enshrine 12000.
  - Fix: invariant-based regression tests.
  - Evidence: settings + budget tests.
- Finding (prior review): misleading "Orch preflight starvation" on light path (Minor).
  - Fix: `present` guard.
  - Evidence: `test_light_path_without_projection_does_not_claim_orch_starvation`; genuine-starvation test still passes (present=True).
- Finding (this fix's review, Minor): cross-service `MIND_LLM_TIMEOUT_SEC` coupling + user-turn latency.
  - Fix: README notes.
  - Evidence: `services/orion-thought/README.md` budget section.

## Restart required

```bash
python scripts/sync_local_env_from_example.py
docker compose --env-file .env --env-file services/orion-thought/.env \
  -f services/orion-thought/docker-compose.yml up -d --build
# no orion-mind change required, but restart it too if its .env changed
- Finding: raw Firecrawl secret was copied into gitignored `services/orion-world-pulse/.env` to make the feature run.
  - Fix: reverted; switched to read-only FCC mount + `ORION_FCC_ENV_PATH` derivation, key blank everywhere.
  - Evidence: `ENV_FIRECRAWL_LEN=0` in the container while `resolve_firecrawl_api_key()` returns len 35 from the mount; live run fetched 5 real articles.

## Restart required

After merging to `main` and pulling into the checkout you run the batched script from:

```bash
docker compose --env-file .env --env-file services/orion-world-pulse/.env \
  -f services/orion-world-pulse/docker-compose.yml up -d --force-recreate
# or simply:
./mesh-utilities/common/up_all_services_batched.sh
```

Until this branch is merged + pulled, the live container is running with an ad-hoc override providing the same mount; a batched restart from an unpatched checkout would drop the mount and regress curiosity to the stub backend.

## Risks / concerns

- Severity Medium — **latency**: enrichment runs synchronously before `stance_react`; a hanging Mind LLM now fails open after ~75–90s (was ~12s). Default-off, so low blast radius. Pair rollout with a turn-level budget/circuit-breaker.
- Severity Low — `MIND_ENRICHMENT_MIN_VIABLE_WALL_MS` assumes orion-mind `MIND_LLM_TIMEOUT_SEC=25`; if that changes, re-derive the wall (documented in README).
- Severity Low — **UNVERIFIED at runtime**: real `meaningful_synthesis` under the new budget not observed end-to-end (needs live LLM bus).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/mind-enrichment-wall-budget
- Severity: low
- Concern: host `~/.fcc/.env` must exist on any node running world-pulse with curiosity enabled (same precondition concept-induction already has).
- Mitigation: with no key resolvable, `resolve_fetch_backend()` degrades to the stub and `build_curiosity_followups()` degrades to empty — the run never fails.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/world-pulse-fcc-mount
