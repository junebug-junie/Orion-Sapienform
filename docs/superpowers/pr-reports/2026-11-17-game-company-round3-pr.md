## Summary

- Game company spec, revision 4: answers Orion's third round of review (8 points, thread `7a835850b6ef`).
- The screen-change floor is per game. 0.01 fails on The Legend of Zelda (NES); a game with no validated floor gets no looks.
- A look's temporal event is written only once the look is settled; a contradicted note never becomes an event; the label says "Saw on the game screen".
- New counter `input_stale_skips`; note error rate is contradicted / checkable; patch 3 checks lease priority.
- Step 0 numbers kept as a fixture: `services/orion-hub/evals/fixtures/retro_look_step0.json`.
- Lane-door runbook gains the rule for spike work on a pool card.
- Fourth round (19 more points from Orion): live floor calibration with input as label, miss counter, lane wait, no defer for looks, only checked-and-held notes reach the day, frame expiry follows the last post, stance-cancel gap and hard-race checks, read grant file for Orion's role.

## Outcome moved

The spec no longer promises a floor that skips most Zelda NES play, and no longer lets a contradicted note stand in Orion's day.

## Current architecture

Spec revision 3 (PR #3517, merged). Patch 1 merged (PR #3627). Nothing of patch 2 or 3 is built.

## Architecture touched

Docs only, plus one eval fixture file (data, no code).

## Files changed

- `docs/superpowers/specs/2026-11-16-game-company-watch-and-commentary-design.md`: revision 4.
- `docs/runbooks/2026-11-14-lane-door.md`: spike-work rule (pin by hardware id, print card memory and leases).
- `scripts/sql/2026-11-17_grant_orion_readonly_retro.sql`: SELECT on `retro_sitting`, `retro_play_sessions` for `orion_readonly`. Hand-applied (#3657).
- `services/orion-hub/evals/fixtures/retro_look_step0.json`: step 0 notes, verdicts and screen-change values. No frames, saves or ROMs.

## Schema / bus / API changes

- None built. The spec now names `input_stale_skips`, a per-game floor table, a settled-look rule for the `retro_look` source, and a per-note `checkable` flag, all for patches 2 and 3.

## Env/config changes

- None.

## Tests run

```text
rtk proxy git diff --check: clean
```

## Evals run

```text
None. The fixture is input for patch 2's regression eval.
```

## Docker/build/smoke checks

```text
None: docs only.
```

## Review findings fixed

- Docs-only change; no code review run.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: medium
  - Concern: the reader list for event labels in prompts is partial (dream cycle, `orion/orion_day/gather.py`, world-pulse resonance, dig not checked).
  - Mitigation: patch 2 acceptance check; the label prefix protects readers either way.
- Issues opened: #3631 (GPU memory no lease explains), #3657 (apply retro read grant, needs-juniper), #3658 (orion-thought cancel listener). Closed: none.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3632

🤖 Generated with [Claude Code](https://claude.com/claude-code)
