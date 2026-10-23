## Summary

- Retro's Zelda decision offer now lists `open_sides`: the sides whose screen edge Link can walk to, by the motor's own search. This applies to the overworld only and counts only the atlas's learned blocks for that room.
- The Hub planner prompt names those sides, each labelled visited or never visited. It marks a finished goal that changed nothing.
- The planner refuses two answers:
  - an `exit` through a closed side;
  - the previous goal again, when it changed nothing or was a completed `explore` toward a closed side.
- A refused answer gets one retry with the reason appended. The retry runs only when the first answer took under 12 s. A second refusal is final.
- New live replay eval (`evals/zelda_planner_replay.py`) runs on the recorded stuck decision.

## Outcome moved

Live, 2026-10-23 16:16 UTC, session c6041e2d: Orion chose "explore S" 8 times in room 119. That room's south side is solid. The last 6 goals moved Link 0 frames, with `start_digest == end_digest`. The rationales misread the text map, for example "open space to the south" and "room 120 is south" (room 120 is east).

Replay of that decision on the real local model, 5 runs each:
- `main`: 5/5 chose `explore S` (dead move).
- this branch: 5/5 chose `explore W` toward room 118, which Orion has never visited (0 dead moves).

## Current architecture

Retro's motor (`navigate.py`) offers a decision with the observation and a text terrain map. The Hub (`retro_planner.plan_zelda`) prompts `metacog_background` once, validates the reply's shape, and posts the goal. Nothing told the model which sides were exits. Nothing refused a goal that had just done nothing.

## Architecture touched

- Retro `ZeldaGoals.offer`: an additive `attempt.open_sides` field (list or `null`).
- Hub `retro_planner`: prompt line, history label, two refusals, and a bounded single retry.
- No bus, schema-registry or env changes.

## Files changed

- `services/orion-retro/app/zelda/navigate.py`: adds `open_sides()` and puts the field on the offer.
- `services/orion-retro/tests/test_zelda.py`: tests open sides, stale-bump isolation, and that dungeons get `null`.
- `services/orion-retro/README.md`: documents the field.
- `services/orion-hub/scripts/retro_planner.py`: adds `sides_line`, `did_nothing`, the refusals, `refusal_note` and the retry.
- `services/orion-hub/tests/retro/test_planner_v2.py`: tests the prompt, the refusals, the retry, wait and stopped repeats, and the slow-answer rule.
- `services/orion-hub/evals/zelda_planner_replay.py` and `evals/fixtures/zelda_room119_south_wall.json`: the live replay eval. The fixture holds RAM-derived observations only, no ROM bytes.
- `docs/runbooks/retro-browser.md`: the planner behaviour.

## Schema / bus / API changes

- Added: `attempt.open_sides` in Retro's Zelda decision. The Hub ignores unknown fields, so this is backward compatible.
- Behavior changed:
  - the planner can now return the 422 rejections `planner_side_closed` and `planner_repeats_no_effect`;
  - a plan can make two GPU calls.

## Env/config changes

None.

## Tests run

```text
PYTHONPATH=. pytest services/orion-hub/tests/retro -q   99 passed
PYTHONPATH=. pytest services/orion-retro/tests -q       127 passed
PYTHONPATH=. pytest tests/retro -q                       27 passed
```

## Evals run

```text
zelda_planner_replay.py --runs 5 (live GPU, metacog_background)
  this branch: PASS 5/5 goals, 0 dead moves (explore W -> 118)
  main:        FAIL 5/5 dead moves (explore S)
```

## Docker/build/smoke checks

Not run. This is pure Python logic in both services. CI runs the Retro preflight.

## Review findings fixed

- Finding (medium-high): the repeat rule used `applied_frames == 0` as its signal. That refused a repeated `wait`, and an exit that a takeover or pause stopped before it moved.
  - Fix: `did_nothing()` compares the start and end digests (room, Link position and facing, hearts, items, enemies). It never counts `wait` or `stopped:*`.
  - Evidence: `test_waits_and_goals_that_never_ran_may_be_repeated`.
- Finding (medium): bumps from the previous goal (`self.blocked`) leaked into the next room's open sides, which could falsely close a passage two tiles wide.
  - Fix: the offer uses only the atlas's learned blocks for that room.
  - Evidence: `test_open_sides_ignore_the_last_goals_bumps_and_skip_dungeons`.
- Finding (low-medium): dungeon doorways and locked doors are not verified against the wall grid, so the refusals could block real exits.
  - Fix: `open_sides` is `null` outside the overworld, so no side refusal happens there.
- Finding (low): two calls can outlast Retro's 30 s offer.
  - Fix: retry only when the first answer took under 12 s.
  - Evidence: `test_slow_first_answer_is_not_retried`.
- Finding (low): retry notes for some refusals showed raw codes.
  - Fix: added plain-English notes for the fight and unsupported-goal refusals.
- Not changed: shed-backoff handling across the two calls. The review judged it correct.

## Restart required

```bash
git pull   # the post-pull rebuild rebuilds orion-hub and orion-retro (a running game ends)
```

## Risks / concerns

- Low: a false learned block (a bump that was really an enemy, which never expires) can still close an overworld side. Exits through that side would then be refused. Before this change the motor could sometimes push through.
- Low: live co-play with this change is UNVERIFIED until it is deployed. Check that the next session's `goal_finished` rows show varied goals.

## PR link

TBD

🤖 Generated with [Claude Code](https://claude.com/claude-code)
