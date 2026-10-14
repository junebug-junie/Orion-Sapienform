## Summary
- Curiosity investigate kickoffs get one fact line from Orion's day record: curiosity runs started today and finished, total minutes, and how many of the runs listed in the thread section are among them, with `[arc:<id>]` (≤5). Stated as fact, no steering.
- Pure helper `orion/curiosity/temporal_fact.py`; `build_kickoff_prompt(today_fact=...)`.
- Hub fetches frame + uncapped `GET /temporal-self/arcs?kind=curiosity` in parallel (frame's `arcs_today` is capped at 64 and live held 0 of 7 curiosity runs), 1.5 s deadline, any failure → no line.

## Match key
A curiosity arc's `subject_ref` is the run's `run_id`; there is no honest per-topic key, so the line gives day totals plus overlap with the listed runs.

## Env/config changes
- Added `HUB_CURIOSITY_TEMPORAL_SELF_FACT_ENABLED=true` (orion-hub); reuses `HUB_CURIOSITY_DURABLE_RUNS_URL`. Local `.env` synced.

## Tests run
```text
orion/curiosity/tests + tests/test_curiosity_*.py + tests/test_dream_hypotheses.py: 627 passed, 1 skipped
services/orion-hub tests/test_curiosity_*.py tests/test_orion_day_letter.py: 490 passed, 3 skipped
```
Live sample:
```text
TODAY SO FAR (day record as of 21:31 local): 7 curiosity runs started today and finished, 136 min in total. 1 of the earlier runs listed above is among them, 19 min of that. [arc:189f7204953ab693] [arc:262805d8c62e202f] [arc:aba4652c47edeca8] [arc:2b12ff149baee429] [arc:1730aa523f55b9cb]
```

## Review findings fixed
- Chronicle lags wall clock → line names its record time and never claims a run wasn't today.
- Arcs filed by start day → "started today and finished".
- Rows without matching `day_id` rejected; future `as_of` > 60 s → None; AST test that `today_fact` reaches the prompt.

## Restart required
```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns
- UNVERIFIED: a curiosity run longer than ~35 min may never become an arc (chronicle late-probe window 1800 s, no `curiosity_run` override). Check `completed_at - turn_started_at` in `curiosity_run_outcomes`.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2659
