## Summary

- The first weather refresh of a process now runs **before** the per-series loop. Later refreshes still run after it.

## Outcome moved

Live, 2026-10-21 at 04:47 UTC, right after PR #2829 deployed:

- The first selection ran before any weather was stored.
- That left the `athena.central_ac_proxy` column empty, so the covariates were 21% gaps (the limit is 10%).
- `direct_arx` therefore refused 40 of 40 backtest windows, and could not compete until the next selection, 6 h later.

With this fix, a fresh host or an emptied weather table no longer loses the first selection.

## Current architecture

The weather refresh ran after the series loop on every tick. That placement came from an earlier review fix: a slow weather call should never delay issuing.

## Architecture touched

`services/orion-prediction/app/engine.py` (`Engine.tick`).

## Files changed

- `services/orion-prediction/app/engine.py`: a refresh before the loop when `weather_state` is still empty.
- `services/orion-prediction/tests/test_inputs.py`: an ordering regression test.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: the first tick after each process start waits once for the weather call (up to about 25 s) before issuing.
- Compatibility notes: none.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: n/a.
- Skipped keys requiring operator action: none.

## Tests run

```text
services/orion-prediction: pytest tests -q -> 122 passed
reviewer mutation check: deleting the fix makes test_first_weather_refresh_runs_before_the_selection_later_ones_after fail
```

## Evals run

```text
n/a (ordering fix); live check after deploy: next selection's direct_arx n_windows > 0
```

## Docker/build/smoke checks

```text
not rebuilt locally (no dependency change)
```

## Review findings fixed

- Finding: none blocking.
- Noted:
  - A first refresh that fails moves later retries after the loop. The worst case is one more refused cycle, and it heals at the next selection. Accepted, because the alternative is up to 25 s of extra delay on every hourly retry while the feed is dead.
  - The per-restart wait is now documented in the code.

## Restart required

From the clean primary `main` checkout after merge:

```bash
scripts/safe_docker_build.sh orion-prediction up -d --build
```

## Risks / concerns

- Severity: low.
  - Concern: about 25 s of delay before the first issue after a restart.
  - Mitigation: it happens once per boot.

## PR link

<link>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
