## Summary

- `orion/gpu_pool/tests/test_stage5_4_no_capacity_callers.py` now exempts `scripts/report_dead_env_keys.py`.
- That script (GPU pool stage 6.6) lists `DURABLE_RUNS_CAPACITY_ENABLED` in `KNOWN_DEAD` so it can delete it from live `.env` files; the gate's `DURABLE_RUNS_CAPACITY_` pattern read that removal list as a broker caller.

## Outcome moved

The `gpu-pool — config gate…` CI job has failed on main since stage 6.6 landed, blocking every open PR (e.g. #2456). It passes again.

## Current architecture

The stage 5.4/5.6 gate greps non-test files under orion/services/scripts for any client, route, package or env key of the deleted /capacity permit broker, with an empty allow-list.

## Architecture touched

Test-only: one path added to `ALLOWED`, with a comment saying why.

## Files changed

- `orion/gpu_pool/tests/test_stage5_4_no_capacity_callers.py`: allow-list the dead-key reporter.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
pytest orion/gpu_pool/tests/test_stage5_4_no_capacity_callers.py -> 4 passed (main: 1 failed, 3 passed)
mutation: a real caller (POST :8121/capacity/acquire) placed in another scripts/ file -> 1 failed (gate still bites)
```

## Evals run

N/A (test-only change).

## Docker/build/smoke checks

N/A.

## Review findings fixed

Self-reviewed: exemption is a single exact path, not a prefix of a directory; pattern unchanged.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low. Concern: a future real caller added inside `report_dead_env_keys.py` would not be caught. Mitigation: that file is an operator report/cleanup script with no network client.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
