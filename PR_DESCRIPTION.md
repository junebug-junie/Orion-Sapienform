## Summary

- Add `services/orion-signals/` — thin mesh launcher for the organ-signal causal spine (not a runtime service).
- Machine-readable `roster.v1.yaml` with tiers: `core`, `tier1`, `tier2`, `routing`, `full`.
- Deterministic `up.sh` / `down.sh` / `smoke.sh` scripts with Redis dedupe when `bus-core` is already running.
- 13 gate tests validating roster compose paths, service names, and script contracts.
- Register `orion-signals` in `scripts/sync_local_env_from_example.py` for env parity.

## Outcome moved

Operators can bring up the organ-signal mesh (bus + gateway + tier1 producers + homeostatic consumer) with one command instead of ad-hoc per-service compose invocations.

## Current architecture

Each organ producer had its own `docker-compose.yml` and `.env`. There was no tiered roster or cumulative launcher for the `orion:signals:*` causal spine around `orion-signal-gateway`.

## Architecture touched

- `services/orion-signals/` — new orchestration seam (roster + scripts + tests + README)
- `scripts/sync_local_env_from_example.py` — `orion-signals` in `DEFAULT_SERVICES`
- Root `README.md` — cross-link to orion-signals operator guide

## Files changed

- `services/orion-signals/roster.v1.yaml`: tier → compose service mapping with organ_ids
- `services/orion-signals/scripts/up.sh`: cumulative tier launcher; optional `--env-file`; gateway `--no-deps` redis dedupe
- `services/orion-signals/scripts/down.sh`: reverse-order stop with optional env files
- `services/orion-signals/scripts/smoke.sh`: bus-core ping + gateway HTTP checks
- `services/orion-signals/.env_example`: operator contract (`ORION_BUS_URL`, `SIGNALS_TIER`, `SIGNALS_USE_BUNDLED_REDIS`)
- `services/orion-signals/README.md`: operator guide
- `services/orion-signals/tests/`: roster + script gate tests
- `scripts/sync_local_env_from_example.py`: include orion-signals in default sync
- `README.md`: pointer to orion-signals README

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: none (launcher only)
- Compatibility notes: Hub remains external (host network); start separately for Organ Signals UI

## Env/config changes

- Added keys: `SIGNALS_TIER`, `SIGNALS_USE_BUNDLED_REDIS` in `services/orion-signals/.env_example`
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: yes (`services/orion-signals/.env_example`)
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes
- skipped keys requiring operator action: `ORION_BUS_URL` (host-specific Tailscale IP — set manually)

## Tests run

```text
bash -n services/orion-signals/scripts/*.sh  → OK
PYTHONPATH=. orion_dev/bin/python -m pytest services/orion-signals/tests -q  → 13 passed
```

## Evals run

```text
None (orchestration/operator tooling; no eval harness for this seam)
```

## Docker/build/smoke checks

```text
Not run in CI agent session (requires live bus + per-service .env on operator host).
Operator smoke: ./services/orion-signals/scripts/smoke.sh
```

## Review findings fixed

- Finding: `docker compose --env-file` fails when per-service `.env` missing
  - Fix: up.sh/down.sh only pass `--env-file` when file exists; warn otherwise
  - Evidence: `test_compose_helpers_skip_missing_env_file`, manual review
- Finding: duplicate log line after gateway `--no-deps` bring-up
  - Fix: echo before command only (consistent with `compose_up`)
  - Evidence: up.sh review
- Finding: `orion-signals` absent from env sync default list
  - Fix: added to `DEFAULT_SERVICES` + `SIGNALS_`/`SIGNAL_GATEWAY_` sync prefixes
  - Evidence: sync script diff

## Restart required

```text
No restart required for merge itself.
After deploy, bring up mesh:
  cp services/orion-signals/.env_example services/orion-signals/.env
  # Set ORION_BUS_URL=redis://<tailscale-node-ip>:6379/0
  python scripts/sync_local_env_from_example.py
  ./services/orion-signals/scripts/up.sh tier1
  ./services/orion-signals/scripts/smoke.sh
```

## Risks / concerns

- Severity: low
- Concern: Per-service `.env` files must exist (sync script or manual copy) before `up.sh` can substitute variables
- Mitigation: WARN logs + README prerequisites; optional env-file handling prevents hard fail on missing launcher `.env`

## PR link

(filled after push)
