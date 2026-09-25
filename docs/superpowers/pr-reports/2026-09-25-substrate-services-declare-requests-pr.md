## Summary

- orion-feedback-runtime and orion-policy-runtime crash-loop in production since the 2026-09-25 08:19 UTC deploy: `ModuleNotFoundError: No module named 'requests'` at import. Policy (L8) and feedback (L10) write nothing.
- Cause: PR #2343 (`732bb6913`) made their stores import `orion.substrate.pending_marker_reconcile`. Any `orion.substrate.*` import runs the package `__init__`, which imports `graphdb_store`, which does `import requests` at module top. Dispatch-runtime already declared `requests`; feedback and policy never did.
- Fix: declare `requests` in every service that imports `orion.substrate`: feedback, policy, plus heartbeat and substrate-telemetry (no `requests` in their live images today, latent) and cortex-orch (has it only transitively; floor pin, no downgrade).
- New CI gate `tests/test_substrate_services_declare_requests.py` in Static repo gates: fails when a service imports `orion.substrate` without declaring `requests`. Red on main's requirements, green here.

## Outcome moved

Layer 8 (policy) and Layer 10 (feedback) start again; this class of import crash is caught in CI instead of production.

## Current architecture

`orion/substrate/__init__.py` eagerly imports materializer/graphdb_store/policy_profiles/etc. Services pull in the whole package on any submodule import.

## Architecture touched

Service dependency files only; one static test; CI workflow step.

## Files changed

- `services/orion-feedback-runtime/requirements.txt`, `services/orion-policy-runtime/requirements.txt`, `services/orion-heartbeat/requirements.txt`, `services/orion-substrate-telemetry/requirements.txt`: `requests==2.32.3` (same pin as execution-dispatch-runtime).
- `services/orion-cortex-orch/requirements.txt`: `requests>=2.32.3` (live image already has 2.34.2 transitively).
- `tests/test_substrate_services_declare_requests.py`: the gate.
- `.github/workflows/orion-static-gates.yml`: runs it.
- this report.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none. Behavior changed: none. Compatibility notes: none.

## Env/config changes

- None. `.env_example` unchanged; no sync needed.

## Tests run

```text
tests/test_substrate_services_declare_requests.py            3 passed
same test with services/ reverted to main (git stash)        1 failed, 2 passed  (names feedback, policy, heartbeat, substrate-telemetry, cortex-orch)
```

## Evals run

```text
None: dependency declaration only; no behavior to evaluate.
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-feedback-runtime build  -> Built; docker run ... python3 -c "import app.main; import requests" -> import ok, requests 2.32.3
scripts/safe_docker_build.sh orion-policy-runtime build    -> Built; same check -> import ok, requests 2.32.3
Live (pre-fix): both containers status=restarting, RestartCount 14, traceback ends in graphdb_store.py:14 import requests.
```

## Review findings fixed

- Reviewed by the implementing session; no separate review subagent was run because the change is dependency pins plus a static test with a demonstrated red/green. Flagged here per the repo's review rule.

## Restart required

From the primary checkout on main, after merge:

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-policy-runtime up -d --build
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-feedback-runtime up -d --build
```

heartbeat, substrate-telemetry and cortex-orch need no restart now (running and healthy); they pick it up on their next rebuild.

## Risks / concerns

- Severity: low. Concern: the root cause is the eager package `__init__`; any future third-party import added there repeats this. Mitigation: the gate pins the current case; a follow-up could make `orion/substrate/__init__.py` lazy.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2346
