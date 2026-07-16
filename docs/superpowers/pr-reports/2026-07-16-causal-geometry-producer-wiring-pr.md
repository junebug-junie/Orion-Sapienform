# Causal Geometry v1 follow-up: scheduled producer + shared postgres-disk store path

## Summary

- Extracted Phase A's core measurement logic (`scripts/causal_geometry_report.py`) into an importable module, `orion/substrate/causal_geometry_engine.py`, so it can be called from more than one place without duplication.
- Added `orion/substrate/causal_geometry_producer.py`: `run_causal_geometry_production_cycle()`, the missing link between Phase A measurement and Phase B's HITL proposal queue -- measures a snapshot, proposes `field_topology_weight_patch` candidates, and enqueues only non-duplicate ones.
- Wired a new scheduled loop into `services/orion-field-digester/app/worker.py` (`_causal_geometry_producer_loop`, same shape as the pre-existing prune/health loops), off by default via `FIELD_PLASTICITY_PRODUCER_ENABLED`, default cadence 24h.
- Fixed `FIELD_PLASTICITY_SQL_DB_PATH` to a real default (previously empty/in-memory-only) on a shared, durable disk alongside Postgres (`/mnt/postgres/field_topology`, not `/mnt/telemetry`), per operator direction, so the HITL store genuinely persists cross-process between `orion-hub` and `orion-field-digester`.
- Code review (8 parallel angles) found and fixed 5 real issues, most notably an intra-cycle proposal-dedup bug (reproduced live) and a `STORAGE_ROOT`-templating bug that would have silently broken cross-process store sharing the moment an operator relocated Postgres's disk.

## Outcome moved

Prior to this branch (PR #1087, merged), the Phase B plasticity HITL queue was real and correctly gated but permanently empty in production -- nothing ever called `propose_field_topology_patches()`/`store.propose()` on a schedule, and the learned-weights store had no real persisted-path default. This PR closes both remaining gaps from that PR's own documented follow-up list: a scheduled producer now exists, and the store's persistence path is real and correctly shared between both containers.

## Current architecture

Before this PR: `scripts/causal_geometry_report.py` was a standalone CLI script, runnable manually but never scheduled. `orion/substrate/field_topology_plasticity.py`'s `propose_field_topology_patches()` had zero production callers. `FIELD_PLASTICITY_SQL_DB_PATH` defaulted to empty (in-memory only, scoped to whichever single process constructed the store).

## Architecture touched

- `orion/substrate/causal_geometry_engine.py` (new) -- Phase A's measurement engine, extracted for reuse.
- `orion/substrate/causal_geometry_producer.py` (new) -- the scheduled producer function.
- `scripts/causal_geometry_report.py` -- trimmed to CLI-only, re-exporting only the engine names its own code or `tests/test_causal_geometry_report.py` actually use.
- `services/orion-field-digester/app/worker.py`, `app/settings.py`, `app/digestion/diffusion.py` (`get_learned_store()`, now public and thread-safe).
- Both services' `.env_example` and `docker-compose.yml` (new producer env keys, corrected `FIELD_PLASTICITY_SQL_DB_PATH` default, `/mnt/postgres/field_topology` volume mount replacing the earlier `/mnt/telemetry` mount on `orion-hub`).
- `services/orion-field-digester/requirements.txt` (added `numpy==1.26.4`).

## Files changed

- `orion/substrate/causal_geometry_engine.py`: new, Phase A core logic extracted from the CLI script.
- `orion/substrate/causal_geometry_producer.py`: new, the scheduled measure-propose-enqueue cycle.
- `orion/substrate/tests/test_causal_geometry_producer.py`: new, 7 tests (success, cross-cycle dedup, intra-cycle dedup regression, measurement failure, proposal-stage failure, insufficient-data, AST guard against `PatchApplier`).
- `scripts/causal_geometry_report.py`: trimmed to CLI-only + a minimal, accurate re-export list.
- `services/orion-field-digester/app/worker.py`: new `_causal_geometry_producer_tick`/`_causal_geometry_producer_loop`.
- `services/orion-field-digester/app/settings.py`: 3 new fields (`field_plasticity_producer_enabled`/`_interval_hours`/`_window_hours`).
- `services/orion-field-digester/app/digestion/diffusion.py`: `_get_learned_store` renamed to public `get_learned_store`, now thread-safe (double-checked locking).
- `services/orion-field-digester/tests/test_worker_causal_geometry_producer.py`: new, 3 tests (disabled no-op, enabled call-with-expected-args, failure-never-raises).
- `services/orion-field-digester/requirements.txt`: added `numpy==1.26.4`.
- `services/orion-field-digester/.env_example`, `docker-compose.yml`: new producer keys; `FIELD_PLASTICITY_SQL_DB_PATH` corrected to a fixed container-internal path; new `/mnt/postgres/field_topology` volume mount.
- `services/orion-hub/.env_example`, `docker-compose.yml`: `FIELD_PLASTICITY_SQL_DB_PATH` corrected the same way; `/mnt/telemetry` mount replaced with `/mnt/postgres/field_topology`.

## Schema / bus / API changes

None. No schema, channel, or API surface changed in this PR.

## Env/config changes

- Added keys (`orion-field-digester`): `FIELD_PLASTICITY_PRODUCER_ENABLED` (`false`), `FIELD_PLASTICITY_PRODUCER_INTERVAL_HOURS` (`24`), `FIELD_PLASTICITY_PRODUCER_WINDOW_HOURS` (`168`).
- Changed default: `FIELD_PLASTICITY_SQL_DB_PATH` (both services) -- was empty, now `/mnt/postgres/field_topology/learned_weights.sqlite3` (a fixed container-internal path; the real host-side location is controlled by `STORAGE_ROOT` via the new volume mount, not by this variable).
- `.env_example` updated: both services.
- Local `.env` synced: `python scripts/sync_local_env_from_example.py --all-keys` run in the worktree; both services' local `.env` files carry the new keys and corrected default. Also copied to the live deployment's `.env` files (see Restart required).
- Skipped keys requiring operator action: none -- everything defaults safely off/inert (`FIELD_PLASTICITY_PRODUCER_ENABLED=false` means the producer never runs unless explicitly enabled).

## Tests run

```text
orion/substrate/tests + orion/schemas/tests + tests/test_causal_geometry_report.py: 333 passed
services/orion-field-digester/tests (run from its own dir): 63 passed
services/orion-hub/tests/test_causal_geometry_{api,page}.py: 22 passed
```

## Evals run

No eval harness exists for this service tree (same gap noted in PR #1087). `docker compose config` was used as a live-config verification step (see Docker/build/smoke checks) -- not a substitute for a real eval, but the closest available check for the config-correctness bugs this review round found.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-field-digester config --quiet   # exit 0
scripts/safe_docker_build.sh orion-hub config --quiet              # exit 0
# Empirically verified FIELD_PLASTICITY_SQL_DB_PATH resolves to the fixed
# container path (/mnt/postgres/field_topology/learned_weights.sqlite3) in
# both services' rendered config, including under an overridden STORAGE_ROOT
# (confirms the review-fix for the STORAGE_ROOT bug).
scripts/safe_docker_build.sh orion-field-digester build   # built cleanly (pre-review-fix commit; not rebuilt after review fixes since none touch runtime image contents beyond source already COPYed)
scripts/safe_docker_build.sh orion-hub build              # built cleanly (same)
```

## Review findings fixed

- Finding: intra-cycle dedup only checked the store's pre-loop state, so two capability channels aliasing one physical edge could both enqueue a proposal in the same cycle.
  - Fix: add each newly-enqueued `target_ref` to the dedup set as it's proposed.
  - Evidence: new regression test `test_intra_cycle_duplicate_candidates_for_the_same_edge_enqueue_only_one_proposal`, passes.
- Finding: `FIELD_PLASTICITY_SQL_DB_PATH`'s default templated the container-visible env var with `${STORAGE_ROOT}`, but the container-side mount destination is fixed -- an operator override would silently break cross-process store sharing.
  - Fix: corrected to a literal container path in both `.env_example` files.
  - Evidence: `docker compose config` re-verified under an overridden `STORAGE_ROOT` -- env var now stays correctly pinned to `/mnt/postgres/field_topology/learned_weights.sqlite3` regardless.
- Finding: `get_learned_store()`'s singleton construction had an unguarded race once a second asyncio loop could call it concurrently.
  - Fix: `threading.Lock` with double-checked locking.
  - Evidence: full `orion-field-digester` test suite (63) still passes.
- Finding: `scripts/causal_geometry_report.py` re-exported 12 names with zero real callers anywhere in the repo.
  - Fix: trimmed to the 16 names actually used.
  - Evidence: `tests/test_causal_geometry_report.py` (10 tests) still passes unchanged.
- Finding: stale comment (old default, pre-rename function name) and a docstring PR-misattribution.
  - Fix: both corrected.
  - Evidence: read-through confirmation.

Not fixed (documented as known, low-severity gaps):
- Proposal-stage exception handler undercounts already-persisted work in its failure summary -- currently unreachable (`store.propose()` can't raise today), so left as-is rather than adding a guard for an impossible case.
- No health/staleness monitoring for the new 24h producer loop, unlike the existing poll/prune loops -- proportionate-scope follow-up, not every existing loop has this either.

## Restart required

```bash
# New volume mounts require recreate (up -d), not a plain restart.
docker compose --env-file .env --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-field-digester/.env -f services/orion-field-digester/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: Low
  - Concern: the producer's real proposal surface is currently only 2 designed `capability_capability` edges in `config/field/orion_field_topology.v1.yaml` -- even with everything correctly wired, a cycle may legitimately produce zero proposals indefinitely if those two channels don't diverge past the 0.02 meaningful-delta threshold in a given week. "Technically wired" isn't the same as "practically active."
  - Mitigation: none needed -- this is expected, honest behavior (`ok=True, proposals_created=0`), not a bug. Worth knowing when judging whether the feature "works" from the hub UI alone.
- Severity: Low
  - Concern: no per-loop health/staleness alerting for the new producer (unlike poll/prune's coverage in `health_monitor.py`).
  - Mitigation: deferred as a proportionate-scope follow-up.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/causal-geometry-producer-wiring
