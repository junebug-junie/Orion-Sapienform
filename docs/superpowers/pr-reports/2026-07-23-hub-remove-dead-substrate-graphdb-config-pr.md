## Summary

- `orion-hub` has run on `SUBSTRATE_STORE_BACKEND=falkor` (FalkorDB/Cypher) live since PR #1153's substrate-runtime cutover; the SPARQL/GraphDB fallback path it used to support is dead.
- Removed the 11 `SUBSTRATE_GRAPH_*`/`SUBSTRATE_GRAPHDB_*` pydantic Settings fields that existed only to configure that dead fallback, plus the matching `.env_example` and `docker-compose.yml` blocks.
- Deleted `services/orion-hub/scripts/seed_substrate_graphdb.py`, a one-shot demo script whose own docstring requires `SUBSTRATE_STORE_BACKEND=graphdb`, a mode Hub never runs in.
- Fixed `docker-compose.yml`'s `SUBSTRATE_STORE_BACKEND` fallback default from `sparql` to `falkor` (review finding — see below).
- Deliberately left `GRAPH_BACKEND`, `RDF_STORE_*`, and `AUTONOMY_GRAPH_*` untouched: these are genuinely consumed via an indirect call chain (`api_routes.py::api_debug_autonomy_goal_archive` → `orion.autonomy.goal_archive.archive_subjects` → `orion/graph/backend_config.py`), not a direct grep hit inside Hub's own files.
- Synced the local `services/orion-hub/.env` by hand (same 11 keys removed).

## Outcome moved

Part of the broader Fuseki decommission: one more service's config surface no longer references SPARQL/GraphDB env vars that have no live code path. Reduces the chance of an operator being misled into thinking Hub still supports a GraphDB backend, and closes a latent config-drift footgun in the compose fallback default.

## Current architecture

Before this patch, `orion-hub/app/settings.py` declared 11 fields for a `SUBSTRATE_STORE_BACKEND=graphdb`/`sparql` SPARQL fallback path that Hub does not run in production. `docker-compose.yml` defaulted `SUBSTRATE_STORE_BACKEND` to `sparql` if unset, and also defaulted the SPARQL query/update URLs to hardcoded Fuseki endpoints — so even with the key unset, the container would start (misdirected to Fuseki, but non-fatal).

## Architecture touched

- `services/orion-hub/app/settings.py`
- `services/orion-hub/.env_example`
- `services/orion-hub/docker-compose.yml`
- `services/orion-hub/scripts/seed_substrate_graphdb.py` (deleted)
- `services/orion-hub/.env` (local, gitignored, synced by hand — not part of this diff)

## Files changed

- `services/orion-hub/app/settings.py`: removed 11 `SUBSTRATE_GRAPH_*`/`SUBSTRATE_GRAPHDB_*` Field declarations; left `SUBSTRATE_STORE_BACKEND` untouched (real, live selector).
- `services/orion-hub/.env_example`: removed matching key blocks; comment notes this reverses a 2026-07-17 decision to keep them as a Fuseki fallback.
- `services/orion-hub/docker-compose.yml`: removed matching `environment:` entries; fixed `SUBSTRATE_STORE_BACKEND:-sparql}` → `SUBSTRATE_STORE_BACKEND:-falkor}` (review finding).
- `services/orion-hub/scripts/seed_substrate_graphdb.py`: deleted — one-shot demo script for a backend mode Hub never runs in, zero remaining references anywhere in the repo (Makefile, README, docs, CI all checked).

## Schema / bus / API changes

- Added: none
- Removed: none (env-only change, no schema/bus/API surface)
- Renamed: none
- Behavior changed: none in the live/intended config (both old default paths and new default point at falkor when `.env` is set correctly, which it is)
- Compatibility notes: if `SUBSTRATE_STORE_BACKEND` is ever unset from `.env` (fresh clone before sync, resync bug, manual edit), the container previously fell back to a working-but-misdirected Fuseki SPARQL config; now it falls back to `falkor` (the correct, intended behavior) instead of crashing at import — this compose default fix directly closes that gap.

## Env/config changes

- Removed keys: `SUBSTRATE_GRAPH_QUERY_URL`, `SUBSTRATE_GRAPH_UPDATE_URL`, `SUBSTRATE_GRAPH_URI`, `SUBSTRATE_GRAPH_TIMEOUT_SEC`, `SUBSTRATE_GRAPH_USER`, `SUBSTRATE_GRAPH_PASS`, `SUBSTRATE_GRAPHDB_ENDPOINT`, `SUBSTRATE_GRAPHDB_GRAPH_URI`, `SUBSTRATE_GRAPHDB_TIMEOUT_SEC`, `SUBSTRATE_GRAPHDB_USER`, `SUBSTRATE_GRAPHDB_PASS`
- Added keys: none
- Renamed keys: none
- `.env_example` updated: yes
- local `.env` synced: yes, by hand (removed the same 11 keys directly; `sync_local_env_from_example.py` was not required since this is a pure removal, not an addition)
- skipped keys requiring operator action: none

## Tests run

```text
pytest services/orion-hub/tests/test_autonomy_goal_archive_api.py services/orion-hub/tests/test_autonomy_runtime_ui_panel.py services/orion-hub/tests/test_situation_settings_env.py services/orion-hub/tests/test_fcc_env_catalog.py -q
36 passed

pytest services/orion-hub/tests -q
957 passed, 32 failed, 2 skipped, 2 errors
(all 32 failures reproduced identically on unmodified main -- pre-existing/environment-related,
 not caused by this change; confirmed via test_fcc_model_labels_api.py's CHANNEL_VOICE_* ValidationError
 reproducing on a clean main checkout with zero diff)

python3 -c "import ast; ast.parse(open('services/orion-hub/app/settings.py').read())"
OK

python3 -c "import yaml; yaml.safe_load(open('services/orion-hub/docker-compose.yml'))"
YAML_OK

scripts/check_service_env_compose_parity.py orion-hub
pass (env_file covers container env; environment: block edits don't affect parity)
```

## Evals run

No eval harness exists for `orion-hub`'s config/settings layer; this is a pure config-removal change with no behavioral eval surface.

## Docker/build/smoke checks

Not run in this environment — no runtime behavior changes for the live/synced config (`SUBSTRATE_STORE_BACKEND=falkor` in `.env` both before and after). The only behavior change is in the *unset* fallback path, verified by code trace (see review below), not live Docker smoke.

## Review findings fixed

- Finding: `docker-compose.yml`'s `SUBSTRATE_STORE_BACKEND:-sparql}` fallback default contradicted this diff's own "provably unreachable" claim — if `.env` ever drops the key, the container would fall back to `sparql`, and with the SPARQL URL defaults now removed, `build_substrate_store_from_env()` (called at hard module-import time in `api_routes.py`) would raise `SubstrateSparqlBackendUnconfiguredError` and crash Hub at startup instead of the old (silently-misdirected-to-Fuseki-but-non-fatal) behavior.
  - Fix: changed the compose default to `SUBSTRATE_STORE_BACKEND:-falkor}` to match the actual intended live default.
  - Evidence: `git diff` on `services/orion-hub/docker-compose.yml` line 245; re-validated with `python3 -c "import yaml; yaml.safe_load(...)"` after the fix.
- Finding (informational, not fixed — pre-existing, out of scope): `Settings.SUBSTRATE_STORE_BACKEND` itself has zero runtime reads via `settings.SUBSTRATE_STORE_BACKEND` anywhere in Hub's code; the real backend selection happens via direct `os.getenv` in the shared `orion/substrate/graphdb_store.py`, bypassing pydantic Settings entirely. Noted for future cleanup, not blocking.

## Restart required

```text
No restart required (env-only removal; live .env already has SUBSTRATE_STORE_BACKEND=falkor set explicitly and unaffected by this change).
```

## Risks / concerns

- Severity: Low
- Concern: `Settings.SUBSTRATE_STORE_BACKEND` field is itself dead code (declared but never read via `settings.X`), discovered during review but out of scope for this patch.
- Mitigation: flagged for a future cleanup pass; does not affect correctness of this change.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1311
