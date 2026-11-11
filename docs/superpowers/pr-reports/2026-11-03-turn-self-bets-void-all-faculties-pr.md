## Summary

- Reverts one part of #3077. When a turn crashes or is lost, the settler again voids **every** faculty of that turn (`failed_to_execute` or `expired_unsettled`), whether or not stance stated it.
- **Why:** #3078 (self-monitor, merged before #3077) already fixed the false "my last 26 step guesses failed to start" sentence, in the right layer. The self-monitor skips `failed_to_execute` on guess faculties. It counts a turn as crashed under `turn.run` only when **all** of the turn's rows are `failed_to_execute`. #3077 marked the unstated rows of a crashed turn `ungradeable_no_stated`, so every crashed v2 turn would read as a turn that ran.
- The data fix proposed with #3077 is dropped. No rows need changing.

## Outcome moved

- `turn.run` counts crashed v2 turns as crashed again.
- The track-record sentence stays fixed through #3078.

## Current architecture

- After #3077 (deployed 19:58 UTC 11-03), `settle()` checked "stated" before the void reasons.

## Architecture touched

- `services/orion-prediction/app/turn_self_bets.py`: `settle()` ordering only.

## Files changed

- `services/orion-prediction/app/turn_self_bets.py`: void first, for every faculty, with a comment naming the `turn.run` contract.
- `services/orion-prediction/tests/test_turn_self_bets.py`: the regression test now asserts the contract, namely that a crashed or lost v2 turn voids all six faculties with the same reason. The earlier crash tests are back to checking all rows.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: back to the pre-#3077 void behaviour.
- Compatibility notes: rows settled between 19:58 UTC and this deploy may carry `ungradeable_no_stated` on unstated faculties of crashed turns. Each one makes one crashed turn read as a turn that ran under `turn.run`, until it ages out of the 7-day window. I will count them after deploy.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no change needed
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: no env change
- skipped keys requiring operator action: none

## Tests run

```text
services/orion-prediction: pytest tests/test_turn_self_bets.py tests/test_grading.py -q -> 44 passed
orion/self_monitor -q (with #3078 merged)                                               -> 124 passed
```

## Evals run

```text
None. A behaviour revert covered by the contract test.
```

## Docker/build/smoke checks

```text
UNVERIFIED until deploy.
```

## Review findings fixed

- This PR is itself the review finding: #3077's settler change conflicted with #3078's `turn.run` contract. I found it by reading #3078's merged SQL (`bool_and(... void_reason = 'failed_to_execute')`).

## Restart required

```text
Post-pull rebuild of orion-prediction.
```

## Risks / concerns

- Severity: low
  - Concern: rows settled while #3077's version was live (from 19:58 UTC).
  - Mitigation: they are counted after deploy and listed in the session report.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3083

🤖 Generated with [Claude Code](https://claude.com/claude-code)
