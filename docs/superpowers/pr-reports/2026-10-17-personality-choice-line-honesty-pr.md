## Summary

- The curiosity choice line in Orion's personality read-back now appears only from **5 choices**. Before that it shows nothing.
- It now names the window it **really** covers ("over the last day" / "over the last N days", up to 14). Before, it always said 14 days.
- The window starts at the earlier of graph birth (the one-shot `seed` cursor) and the oldest choice held, so a choice never drops out of the count and comes back.
- The Hub panel uses the same day/days wording.

## Outcome moved

Live 2026-10-17, the first chat turn that carried the cue showed Orion: "In curiosity you went your own way in 1 of 1 choices over the last 14 days". The graph was born the night before. One data point was presented as a two-week tendency. The cue no longer overstates its evidence.

## Current architecture

`orion/personality/cue.py` rendered the choice line whenever there was at least 1 choice, with `window_days` fixed at 14 by durable-runs (`CHOICE_STYLE_WINDOW_DAYS`).

## Architecture touched

- `orion/personality/cue.py`: the shared renderer used by chat stance, reverie, curiosity, outreach and the Hub cue route.
- durable-runs `PersonalityStore.frame()`: new `covered_days` and `choice_window_days` helpers.
- Hub panel JS: the choice-style text.

## Files changed

- `orion/personality/cue.py`, `orion/personality/tests/test_cue.py`: the minimum and the wording
- `services/orion-durable-runs/app/personality_store.py`: the real window
- `services/orion-durable-runs/tests/test_personality_loop.py`: window tests, including the flicker regression
- `services/orion-hub/static/js/personality.js`, `personality.test.js`: day/days wording
- `docs/superpowers/pr-reports/2026-10-17-personality-choice-line-honesty-pr.md`: this report

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: `ChoiceStyleV1.window_days` is now the real span (1..14), not a constant 14. The schema already allowed it.
- Compatibility notes: none. The only other consumer of `window_days` is the Hub panel, fixed here.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
orion/personality/tests -> 112 passed
services/orion-durable-runs: pytest tests (throwaway Postgres 16) -> 535 passed, 1 skipped
services/orion-hub: personality route/cue/curiosity/outreach tests -> 57 passed, 4 skipped
services/orion-thought: tests/test_personality_cue.py -> 19 passed
node --test personality.test.js -> 18 passed
```

## Evals run

```text
Live read-only: the seed cursor is fixed at 2026-10-16 22:37:42 while the source cursors keep advancing, so it is a stable birth marker. The only live choice started 1h51m before it, and the regression test pins that case.
```

## Docker/build/smoke checks

```text
Not deployed. UNVERIFIED live until rebuilt.
```

## Review findings fixed

- Finding: I first took graph age from `min(updated_at)` of the source cursors. That value drifts forward as the sources read.
  - Fix: use the one-shot `seed` cursor, which is never moved.
  - Evidence: live query.
- Finding: a choice dated before the seed dropped out of the count and came back at each day boundary.
  - Fix: the window starts at the earlier of birth and the oldest choice.
  - Evidence: `test_a_choice_older_than_the_seed_never_drops_out_of_the_window`.
- Finding: the Hub panel would print "over the last 1 days".
  - Fix: day/days wording.
  - Evidence: a node test.

## Restart required

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build && scripts/safe_docker_build.sh orion-thought up -d --build && scripts/safe_docker_build.sh orion-hub up -d --build
```

There is no schema change, so the order does not matter.

## Risks / concerns

- Severity: low. Concern: if first-boot seeding failed and only succeeded later, the `seed` timestamp is later than true first boot. The window could then undercount by that gap. Mitigation: choices older than the seed still widen the window, and an embedder outage also holds choice folding.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2746

🤖 Generated with [Claude Code](https://claude.com/claude-code)
