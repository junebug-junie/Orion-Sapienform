## Summary

- Orion's solo Zelda play froze on one save for 8 runs (11-16..23): every run played 30 minutes with zero decisions.
- Cause: the 11-16 09:54Z end save caught Link mid room-exit (game mode `0x0C`), where the game waits for someone to hold DOWN. Retro offers a decision only when Link can move, and leisure presses buttons only for an accepted goal, so nothing ever moved. Each run skipped its end save (`no_goal_played`), so the next run reloaded the same save.
- End and orphan saves now require normal play with Link controllable (the runner waits up to 3 s for a scroll to finish).
- A run with no decision offered in 60 s ends `no_decision_offered`. When Link is also not controllable, the loaded save is marked unplayable and later runs skip it. This frees the save that is stuck today without a manual step.

Fixes #3774

## Outcome moved

Leisure play: a frozen save costs one 60-second run, not every run forever. A frozen state can no longer become the newest save.

## Current architecture

`services/orion-hub/scripts/retro_leisure.py` starts Retro on the newest compatible save, polls `GET sessions/{id}`, plans a goal for each decision Retro offers (`cognition.attempt`), and saves at the end when at least one goal was placed. Retro (`services/orion-retro/app/zelda/navigate.py observe_frame`) offers a decision only when its observation is `observed` with `link.controllable`.

## Architecture touched

Hub only: `scripts/retro_leisure.py`, `scripts/retro_presence.py` (one new Redis key). No Retro, schema, bus or env change.

## Files changed

- `services/orion-hub/scripts/retro_leisure.py`: `_controllable`, `_settled_for_save`, the save gates, the no-offer watchdog, the unplayable-save filter at start.
- `services/orion-hub/scripts/retro_presence.py`: `unplayable_saves()` / `mark_unplayable_save()` on `orion:retro:leisure:unplayable_saves` (JSON list, max 64).
- `services/orion-hub/tests/retro/test_leisure.py`: the fake Retro reports `zelda.observation` and can withhold offers (and, like real Retro, keeps the policy when nothing was offered); 4 new tests; one expectation changed (status unreadable at the end: no save).
- `services/orion-hub/README.md`: "Frozen saves" note.

## Schema / bus / API changes

- Added: Redis key `orion:retro:leisure:unplayable_saves`; journal event `no_decision_offered`; end reason `no_decision_offered`; `end_save_error=skipped:not_controllable`.
- Behavior changed: when Retro status is unreadable at session end, the end save is skipped (was: saved blind).
- Compatibility: no schema or bus change.

## Env/config changes

None.

## Evidence (live, read-only)

- `retro_play_sessions`, kind `leisure`: 11-16 09:54Z 236 decisions; 11-16 12:54Z through 11-23 zero, 108k Orion frames each.
- Leisure journal (`orion:retro:leisure:journal`): every run since loaded save `6f1b49c6`, the end save of the 11-16 09:54Z run; no planner call, rejection or shed in between.
- That save's RAM: mode `0x0C`, room 73, Link y=213, 4/16 hearts. In a scratch emulator inside `orion-athena-retro` (separate process, live session untouched): 7,200 frames with no input stay in `transition`; holding DOWN returns to `play` with Link controllable. The previous save `2fb75d76` plays normally.

## Tests run

```text
cd services/orion-hub && PYTHONPATH=<wt>:<wt>/services/orion-hub .venv/bin/python -m pytest -q tests/retro tests/test_retro_leisure_self_start.py
305 passed
```
The 3 new behavior tests fail on origin/main's `retro_leisure.py`/`retro_presence.py` (the frozen-save test ends `session_cap`, reproducing the live 30-minute stall).

## Evals run

No eval: this is a control-flow fix. The live proof is the first leisure run after deploy (see Risks).

## Docker/build/smoke checks

Not run; Hub rebuild happens on merge (post-pull rebuild).

## Review findings fixed

Review ran in a subagent (orion-repo-agent) on b676bc7ad.

- Finding (medium): an unreadable game (Retro `status: unknown`, e.g. RAM not decodable) read as "frozen", so every run would mark its save and walk the lineage back one save per run.
  - Fix: `_link_controllable` returns None for unknown; a save is marked only when the game is observed, Link is not controllable, and the session is `running`.
  - Evidence: `test_an_unreadable_game_ends_the_run_but_never_marks_the_save`.
- Finding (medium): `_controllable` required `mode == 'play'`, but Retro offers and Link is controllable in caves too, so a cap reached in a cave skipped the end save.
  - Fix: rely on `link.controllable` (observe.py sets it only in play or cave).
  - Evidence: `test_a_save_in_a_cave_counts_as_controllable`.
- Finding (low): the scroll test fake sent `controllable=True` mid-scroll, which real Retro never does.
  - Fix: the fake sends `controllable=False` while scrolling; the test still proves the wait.
- Finding (low): a run could end the instant Link became controllable, before the offer showed in the status.
  - Fix: 10 s grace (`NO_OFFER_GRACE_SEC`) when Link is controllable at the deadline.
- Finding (low): nothing reported saves skipped as unplayable.
  - Fix: `started` journal row carries `unplayable_skipped`.
  - Evidence: `test_the_start_journal_counts_skipped_unplayable_saves`.
- Finding (low, kept deliberately): an unreadable status at session end now skips the end save. Stated under Schema/behavior changes.

## Restart required

```text
Hub rebuild on merge (post-pull rebuild covers it).
```

## Risks / concerns

- Severity: low. The first run after deploy loads the frozen save, ends after 60 s, and marks it. The next run loads the newest older compatible save, so some progress since 11-11 is lost. Recovering it would need a game-specific button press, which this patch deliberately does not add.
- Severity: low. A good save is marked only when no offer arrives in 60 s AND Link reads not controllable at that moment.
- Issues: fixes #3774. Live proof: the first `no_decision_offered` journal row, then a leisure run with decisions > 0.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3799
