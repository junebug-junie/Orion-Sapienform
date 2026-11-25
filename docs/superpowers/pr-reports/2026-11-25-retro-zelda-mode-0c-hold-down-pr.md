## Summary

- Zelda game mode `$0C` (seen live in room 73, leaving a cave) ignores every button but DOWN. Retro's motor went hands-off during every transition, and offers need a controllable Link, so Orion's play froze there.
- The motor now holds DOWN while mode is `$0C`, with or without a goal, capped at 600 frames, and journals `mode_wait_released` / `mode_wait_unreleased`.
- A decision offered just before the stall is cleared; held frames don't count as goal effort; Scripted's idle shortcut knows about the hold.

## Outcome moved

Orion's solo Zelda play (#3774 follow-on). 11-25 11:36Z run: 114 decisions in 13 min, then stuck in room 73 mode `$0C` for 17 min until session cap; end save skipped (`not_controllable`). Same spot froze the 11-16 save 6f1b49c6.

## Current architecture

`services/orion-retro/app/zelda/navigate.py` `ZeldaGoals`: offers decisions only when Link is controllable; `_drive` returns mask 0 on any non-controllable frame; `mask()` only emits while a goal is accepted or death recovery runs.

## Architecture touched

Retro Zelda motor and Scripted idle shortcut. No bus, schema, env change. New journal kinds `mode_wait_released`, `mode_wait_unreleased` (Retro's own session journal).

## Files changed

- `services/orion-retro/app/zelda/navigate.py`: `_track_release`, `RELEASE_MAX_FRAMES`, `HOLD_DOWN_MODE`; hold DOWN in idle and goal paths.
- `services/orion-retro/app/scripted.py`: idle shortcut also skips while releasing.
- `services/orion-retro/tests/test_zelda.py`: two tests (hold with/without goal; cap, stale offer, mid-goal journaling, effort count).
- `services/orion-retro/README.md`: the `$0C` rule.

## Schema / bus / API changes

- Added: Retro journal kinds `mode_wait_released`, `mode_wait_unreleased`.
- Behavior changed: script-grant motor presses DOWN during mode `$0C`.

## Env/config changes

None.

## Tests run

```text
services/orion-retro: pytest tests -q -> 192 passed
new test against origin/main navigate.py -> fails (mask 0 != DOWN)
```

## Evals run

No eval harness change. Live-save replay instead (scratch CoreProbe in orion-athena-retro, new navigate.py, save 6f1b49c6 = the 11-16 stuck save): released in 42 frames (`mode_wait_released` frames=42), then `decision_offered` with Link controllable in room 73.

## Docker/build/smoke checks

Not run; normal retro rebuild picks it up.

## Review findings fixed

- Finding: unbounded DOWN hold if `$0C` means something else elsewhere.
  - Fix: 600-frame cap, then stop and journal `mode_wait_unreleased` with room/level/submode/x/y.
  - Evidence: cap test.
- Finding: an offer made just before the stall stayed acceptable as a `wait` goal.
  - Fix: clear `attempt` when the hold starts (no goal active).
  - Evidence: test asserts accept raises.
- Finding: mid-goal stalls not journaled; held DOWN counted as goal effort.
  - Fix: tracking runs on every frame; `note_applied` skips while releasing.
  - Evidence: test checks `mode_wait_released` frames=9 and applied unchanged.
- Finding: Scripted idle shortcut ignored the hold.
  - Fix: also checks `releasing`.
- Also fixed while revising: a capped hold no longer blocks death recovery (idle short-circuit only on `$0C` or unknown frames).

## Restart required

```bash
scripts/safe_docker_build.sh orion-retro up -d --build
```

After deploy: remove 6f1b49c6 from Redis `orion:retro:leisure:unplayable_saves` so Orion resumes from the 11-16 save (newer than 11-11).

## Risks / concerns

- Severity: low. Concern: meaning of `$0C` outside room 73 is unknown. Mitigation: 600-frame cap plus `mode_wait_unreleased` row names the place.
- Issues: related #3774 (closed), #3887 open.

## PR link

(filled after push)
