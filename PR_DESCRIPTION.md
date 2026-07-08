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

- Severity: low
- Concern: host `~/.fcc/.env` must exist on any node running world-pulse with curiosity enabled (same precondition concept-induction already has).
- Mitigation: with no key resolvable, `resolve_fetch_backend()` degrades to the stub and `build_curiosity_followups()` degrades to empty — the run never fails.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/world-pulse-fcc-mount
