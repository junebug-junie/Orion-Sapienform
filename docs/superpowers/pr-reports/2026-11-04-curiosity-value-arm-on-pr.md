# feat(curiosity): turn on the value-order arm (patch 0 of the curiosity replica)

## Summary
- Curiosity's offer experiment gets its second arm. On a fair coin per run, the run is offered priors ordered by how much they are likely to change (expected belief change), instead of by raw uncertainty.
- The switch was off by design until the replay said the measurement was usable. Re-run today against live data, it says usable.
- This is patch 0 of the curiosity replica design (PR #3093). The replica has to beat both rule arms, and until now only one had ever run.

## Evidence (live, 2026-11-04)
Command: `scripts/analysis/replay_curiosity_realized_nats.py --pg`
- 202 completed turns, 0 unknown. 31 failed turns are left out.
- 53% of completed turns moved nothing. Mean 0.064 nats, sd 0.210, p90 0.119.
- Every offer since 09-26 is on the uncertainty arm. The value arm has 0 runs.
- Result: **USABLE**, about 171 runs needed per arm.

## Env/config changes
- `HUB_CURIOSITY_VALUE_ORDER_ENABLED`: false → true, in `.env_example`, the settings default, and the primary checkout's local `.env`. The sync script would have reported a value-only change like this as "diverged" and left it, so I set it by hand.

## Tests run
```text
services/orion-hub/tests/test_curiosity*.py   556 passed, 3 skipped
orion/curiosity                               191 passed, 23 skipped
check_settings_defaults orion-hub --example-drift   OK
```

## Restart required
```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns
- **Severity: low.** Half of runs now see a different order of the same priors.
  - What can't change: the stale bucket, the coverage slots and the sample size are the same on both arms.
  - Rollback: set `HUB_CURIOSITY_VALUE_ORDER_ENABLED=false` in Hub's `.env` and rebuild Hub.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
