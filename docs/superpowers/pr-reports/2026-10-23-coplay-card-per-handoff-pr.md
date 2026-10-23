## Summary

- Each co-play memory card's move lines now count only the bets placed during that handoff. Before, they counted the whole Retro session.

## Outcome moved

**Live 2026-10-23, after #2882 deployed.** A 5-minute harness run (`evals/retro_coplay.mjs`) gave:

- 6 handoffs, 6 `retro_play_sessions` co_play rows and 6 memory cards, all carrying "From my own moves" lines. Before #2882 there were 0.
- 25 bets, 17 of which succeeded.
- The first `engage` bet (fight nearest, enemies_cleared) was met.

The bug fixed here: four cards in a row repeated "worked 11 of 15 times", because one Retro session spans every handoff.

## Current architecture

- `watch_coplay` and leisure call `ZeldaExperience.learned_lines(session)`.
- `fetch_learned_lines` read every bet of the Retro session.

## Architecture touched

- `retro_memory_export.fetch_learned_lines(…, since=)` and the new `bets_since()`.
- `ZeldaExperience.learned_lines(…, since=)`.
- Callers pass the handoff start (`watch_coplay` `started_at`) or the leisure start (`run.started_at`).

## Files changed

- `services/orion-hub/scripts/retro_memory_export.py`: `bets_since`, and the `since` parameter.
- `services/orion-hub/scripts/retro_experience.py`: passes `since` through.
- `services/orion-hub/scripts/retro_routes.py`, `retro_leisure.py`: pass the start time.
- `services/orion-hub/tests/retro/test_experience.py`: `test_each_handoff_card_counts_only_its_own_bets`.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
PYTHONPATH=. pytest services/orion-hub/tests/retro -q   174 passed
```

## Evals run

```text
retro_coplay.mjs --minutes 5 (live, before this fix): 6 co_play rows + 6 cards with move lines; repeated "11 of 15" across cards
```

## Docker/build/smoke checks

Not run. The change is pure Hub logic.

## Review findings fixed

- No review subagent was run. This is a 4-file filter change with a regression test, and it follows two full reviews of the same code today.

## Restart required

```bash
git pull   # post-pull rebuild of orion-hub (Hub + retro gateway)
```

## Risks / concerns

- **Severity Low:**
  - Concern: the cutoff compares the Hub's clock with Postgres `now()`. Both run on athena, so they agree.
  - Mitigation: a bet placed in the same second as the handoff start still counts (`>=`).
- **Note:** the service's live check (served model against the skill-rate baseline) is still `pending`. It counts one unit per Retro session and needs 20; 2 exist. On the first sessions the model's Brier (0.20) is no better than always guessing the base rate (0.18). The cards report this plainly.

## PR link

TBD

🤖 Generated with [Claude Code](https://claude.com/claude-code)
