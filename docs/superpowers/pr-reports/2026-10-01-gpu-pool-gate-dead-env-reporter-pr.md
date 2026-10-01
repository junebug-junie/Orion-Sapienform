## Summary

- The stage 5.4 "no /capacity callers" gate (`orion/gpu_pool/tests/test_stage5_4_no_capacity_callers.py`) flagged `scripts/report_dead_env_keys.py`, added in #2453.
- The script is not a caller. It names retired keys (`DURABLE_RUNS_CAPACITY_ENABLED`) so it can report them as dead.
- It is now allow-listed, with the reason written next to the entry.

## Outcome moved

Main was latently broken. The `orion-gpu-pool tests` workflow is path-filtered, so it did not run when #2453 merged. Any later PR touching those paths failed, for example #2457.

## Tests run

```text
pytest orion/gpu_pool/tests/test_stage5_4_no_capacity_callers.py -q   -> 4 passed (failed on main before)
```

## Evals run

None (a test-gate scope fix).

## Env/config changes

None.

## Review findings fixed

None. Review skipped: a one-line allow-list entry in a test, with the reason inline.

## Restart required

No restart required.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
