# PR: Collapse mirror live-path truth gate and bus readiness

## Summary

Manual collapse intake smoke (`smoke_juniper_collapse_fanout.py`) only proves the downstream organ: `orion-collapse-mirror` can consume `orion:collapse:intake` and fan out to triage + sql-write. It bypasses equilibrium metacog trigger intake, cortex-orch, cortex-exec, metacog draft/enrich, and substrate production truth.

This PR adds operator tooling and service hardening so live-generation failures are diagnosable without mistaking intake fanout health for end-to-end health.

## Changes

### 1. `scripts/collapse_mirror_live_path_truth.sh`

Live-path gate that checks:

- `PUBSUB NUMSUB` on:
  - `orion:equilibrium:metacog:trigger`
  - `orion:cortex:exec:request` (honors `CHANNEL_EXEC_REQUEST` / `CHANNEL_CORTEX_EXEC_REQUEST`)
  - `orion:exec:request:CollapseMirrorService`
  - `orion:collapse:intake`
  - `orion:collapse:sql-write`
- Substrate `/grammar/truth` summary (degraded reasons, backlog, quarantine)
- Aggregate `./scripts/grammar_production_truth.sh`

Exits nonzero when upstream subscribers are missing or substrate truth is degraded — matching PR #706 (dead pub/sub) and PR #707/#708 (reducer backlog / quarantine) failure modes.

### 2. `orion-collapse-mirror` `/ready` endpoint

- Reports NUMSUB-backed readiness for both async consumers:
  - `Hunter` on `CHANNEL_COLLAPSE_INTAKE`
  - `Rabbit` on `EXEC_REQUEST_PREFIX:CollapseMirrorService`
- Stores `rabbit`/`hunter` on `app.state` during lifespan

### 3. Remove legacy `exec_worker.py`

Threaded raw-subscribe worker was never started from `main.py`; `bus_runtime.start_services()` is the canonical path. Removing it prevents accidental dual-subscriber reintroduction.

### 4. `CHANNEL_COLLAPSE_SQL_WRITE` setting

Replaces hardcoded `orion:collapse:sql-write` in `bus_runtime.py`. Wired through `.env_example`, `docker-compose.yml`, and `settings.py`.

### 5. Tests

- `services/orion-collapse-mirror/tests/test_ready.py` — 3 readiness cases
- `services/orion-collapse-mirror/tests/conftest.py` — chdir isolation from repo-root `.env`

## Live stack evidence (pre-PR)

```bash
redis-cli -u "$ORION_BUS_URL" PUBSUB NUMSUB \
  orion:equilibrium:metacog:trigger \
  orion:cortex:exec:request \
  orion:exec:request:CollapseMirrorService \
  orion:collapse:intake \
  orion:collapse:sql-write
```

Observed (2026-06-17): 3 / 1 / 1 / 3 / 2 subscribers — not in PR #706 dead-subscriber mode.

```bash
./scripts/collapse_mirror_live_path_truth.sh
# collapse_mirror_live_path_truth: PASS
```

Grammar production truth also PASS; transport reducer backlog cleared; no unacknowledged quarantine.

## Test plan

- [x] `PYTHONPATH=. pytest services/orion-collapse-mirror/tests/test_ready.py -q` (from repo root and worktree)
- [x] `python3 -m compileall services/orion-collapse-mirror/app`
- [x] `bash -n scripts/collapse_mirror_live_path_truth.sh`
- [x] `./scripts/collapse_mirror_live_path_truth.sh` against live stack — PASS

## Operator notes

- Local `.env` synced: `CHANNEL_COLLAPSE_SQL_WRITE=orion:collapse:sql-write` added to `services/orion-collapse-mirror/.env`
- Intake-only smoke remains valid for downstream organ checks; use live-path gate for production metacog generation readiness

## Non-goals

- No changes to equilibrium/cortex-exec metacog pipeline logic
- No schema/bus channel registry changes (channels already registered)
- Does not replace `trace_metacog.py` end-to-end publish probe
