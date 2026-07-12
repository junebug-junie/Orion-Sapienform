## Summary

- Fixed `GET /state/latest` (orion-state-service's HTTP debug route) throwing `TypeError` on every call.
- `http_get_latest` called `STORE.get_latest(req)` without the required keyword-only `biometrics_stale_after_sec` argument; the bus-RPC path (`_handle_get_latest`) already passed it correctly.
- Found during Phase 4 (`CLUSTER_ROLE_WEIGHTS` research, `docs/notes/2026-07-12-phase4-cluster-weighting-research.md`), flagged there as "not fixed here," fixed now as its own small patch.

## Outcome moved

The HTTP debug endpoint is now callable instead of 500ing on every request. Low-stakes (only the bus-RPC path is exercised by real services today), but it's the only externally-curlable way to inspect this service's state without a bus client.

## Current architecture

`services/orion-state-service/app/main.py` exposes both a bus-RPC handler (`_handle_get_latest`, correct) and an HTTP debug route (`http_get_latest`, was broken) over the same `StateStore.get_latest()` method, which requires `biometrics_stale_after_sec` as a keyword-only argument.

## Architecture touched

`orion-state-service` only. No contract changes.

## Files changed

- `services/orion-state-service/app/main.py`: pass `biometrics_stale_after_sec=float(settings.biometrics_stale_after_sec)` in `http_get_latest`, matching `_handle_get_latest`.
- `tests/test_state_service_http_latest.py` (new): regression test — confirmed it fails against the pre-fix code (`AssertionError: assert 'biometrics_stale_after_sec' in {}`) and passes against the fix.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
PYTHONPATH=. pytest tests/test_state_service_http_latest.py -q
1 passed

(confirmed regression: same test against pre-fix main.py via git stash -> 1 failed)
```

## Evals run

None applicable — deterministic one-line kwarg fix.

## Docker/build/smoke checks

Not run — not deployed.

## Review findings fixed

None — single Explore-agent review pass returned PASS with no findings (settings field confirmed to exist, mock realistic, no adjacent bugs).

## Restart required

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-state-service/.env \
  -f services/orion-state-service/docker-compose.yml \
  up -d --build orion-athena-state-service
```

Not run — left for Juniper to trigger.

## Risks / concerns

None — isolated, deterministic, test-covered fix with no behavioral change to the bus-RPC path.

## PR link

Branch pushed: `fix/state-service-http-latest-missing-kwarg`
