## Summary

- After the door fix (#2924), Link left the dungeon and then crossed rooms 115 <-> 99 thirty times.
- Cause: every room the atlas knew was worth the same 0.3. The round trip had worked 28 of 28 times (chance 0.91), so it always ranked first.
- Fix 1, count-based novelty: a known room's worth is now 0.3 / visits, using the atlas visit counts (`room_value`). Each menu line shows "visited N times".
- Fix 2, shortlist: the planner model sees 2 options, the best move between rooms and the best other move. With the full menu it took the highest chance 10 of 10 times, even when that move ranked 3rd.
- The menu header and the instruction now say that the order already weighs chance, novelty and learning value.

## Outcome moved

Replay of the live room-99 and room-115 decisions with the real atlas, live quotes and the real model: before, 0/10 left the loop (round trip 10/10); after, 10/10 left by the least-visited side (99 -> W to 98, 4 visits; 115 -> W to 114, 2 visits).

## Current architecture

Score = value x (p + 2 EIG), where EIG is the learning value. Value is a hand-set table, and every known room was 0.3. The model saw the full menu, ranked best first.

## Architecture touched

Hub only (`retro_experience.py`, `retro_planner.py`). Leisure play uses the same shortlist. No contract, schema or env changes.

## Files changed

- `services/orion-hub/scripts/retro_experience.py`:
  - `room_value` and `room_visits`;
  - the visit count in the option text;
  - `PLANNER_SHORTLIST = 2`, and `for_planner(limit)` keeps the best traverse plus the best other move;
  - `placed` falls back to the full-menu rank.
- `services/orion-hub/scripts/retro_planner.py`: the menu header and `MENU_INSTRUCTION`.
- `services/orion-hub/tests/retro/test_experience.py`: tests for novelty, the shortlist, a way out next to a fight, and the rank of an off-shortlist pick.

## Schema / bus / API changes

- Behavior changed: the planner prompt lists at most 2 options. Bets still record the full-menu rank.

## Env/config changes

None.

## Tests run

```text
cd services/orion-hub && pytest tests/retro -q   -> 191 passed
pytest services/orion-hub/evals -q                -> 66 passed, 3 skipped
Mutations: rank fallback off -> its test fails; plain top-2 -> way-out test fails
```

## Evals run

```text
zelda_experience_replay.py -> PASS 5/5
zelda_planner_replay.py --with-menu (room 119) -> PASS 5/5
Live replay rooms 99/115 (real atlas, quotes, model): 10/10 leave by least-visited side (was 0/10)
```

## Docker/build/smoke checks

```text
No build. The post-pull rebuild covers orion-hub and the retro gateway.
```

## Review findings fixed

- Finding (must): a pick outside the shortlist lost its rank (`chosen_rank` None).
  - Fix: `placed` falls back to `menu.rank_of`.
  - Evidence: `test_a_pick_outside_the_shortlist_is_placed_with_its_full_menu_rank`; it fails without the fix.
- Finding (should): only exits are discounted by visits, so fight and retreat could fill a plain top 2 and hide every way out.
  - Fix: the shortlist keeps the best traverse plus the best other move.
  - Evidence: `test_the_shortlist_always_keeps_a_way_out_next_to_a_fight`; it fails with a plain `[:2]`.
- Not fixed (low): a cave that was already entered keeps its full worth, so an enter/leave loop is possible. Visit counts for dungeons with level "unknown" mix all dungeons. Both are follow-ups.

## Restart required

```bash
git pull   # rebuilds orion-hub and the retro gateway
```

## Risks / concerns

- Severity: medium
  - Concern: the model's stated reasons are often wrong (for example, "visited more" when the room was visited less). The choice is right because of the order, not because of the reasoning.
  - Mitigation: treat the rationale as narration. The learned world model (next) replaces the hand-set worth.
- Severity: low
  - Concern: visit counts accumulate per save lineage, so rooms near spawn lose worth for good.
  - Mitigation: intended (it pushes outward). New rooms keep a worth of 1.0.

## PR link

(added after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
