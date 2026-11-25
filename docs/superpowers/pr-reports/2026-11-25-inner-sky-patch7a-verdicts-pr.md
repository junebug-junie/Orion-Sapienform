## Summary

- Integration patch 7a (#3960) makes three more organs write into Orion's shared experience record: intention, day letter and personality.
- Their verdict words were not in the Inner Sky colour table. The page would draw them neutral, and the feed eval would list them as unsorted.
- `not_produced` (a promise Prime took on was not carried) is now amber, the same as `refused` and `abandoned`.
- `chose` and `declined_all` (Prime picked an option, or chose to wait) are now normal.
- No colour or organ-mapping change was needed. `inner-sky.js` already has colours for all three organs, and the route reads organs from the registry.

## Outcome moved

When 7a deploys, Inner Sky colours the new experiences on purpose, not by default.

## Current architecture

`services/orion-hub/scripts/inner_sky_verdicts.py` is the one table that both the page and `evals/run_inner_sky_feed_eval.py` use for colour. An unknown word renders neutral and is reported as unsorted.

## Architecture touched

Hub only. No contract, bus, schema or env change.

## Files changed

- `services/orion-hub/scripts/inner_sky_verdicts.py`: three verdict words sorted.
- `services/orion-hub/tests/test_inner_sky_routes.py`: the words added to the severity test.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: these three verdict words get a colour in Inner Sky.
- Compatibility notes: safe to merge before or after #3960.

## Env/config changes

- None.

## Tests run

```text
pytest services/orion-hub/tests/test_inner_sky_routes.py tests/test_inner_sky_page.py -q -> 40 passed
```

## Evals run

```text
run_inner_sky_feed_eval.py --hub http://127.0.0.1:8080 -> PASS (live, before 7a deploys).
organs_with_no_experience: reverie, intention, day_letter, personality. 7a fills the last three.
```

## Docker/build/smoke checks

```text
Not run. A pure table change. Hub rebuilds with #3960.
```

## Review findings fixed

- See the PR conversation (the review ran in a subagent).

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

Fold this into #3960's deploy order (Hub is step 2 there).

## Risks / concerns

- Severity: low
- Concern: amber for `not_produced` assumes an uncarried promise is a real problem. 2 of 5 live closes so far were Hub refusing the order.
- Mitigation: this is a one-line change to revert if it reads too loud.
- Issues opened/closed: none.
