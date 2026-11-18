## Summary

Follow-up to spec #2763 ("Spec questions answered" #5: Juniper, YES, every Orion-facing line prints people as "at least <most at once>, at most <clusters>", never a bare cluster count) and the motion-stories plan (PR #2736: "the street line gains what moved").

- The camera clause in every chat turn said "111 individuals counted today (fixtures excluded)" (live turn 4bbaaf2a, 2026-11-03). That number was every appearance cluster of every kind, vehicles included. It now says "today so far (through 13:00), people: at least 2, at most 15 (the camera splits one person into several)". The range comes from today's walkway day rollup, the same row and helper (`walkway_rollup.person_count_range`) the day letter prints. No range means no number.
- The street line now ends with up to two of today's spot-checked motion stories: "Walkway motion stories today (spot-checked): at 17:22, A person walked ...". They appear only while Juniper's spot-check gate passes, using the same rule the rollups use. `STREET_MOTION_LINE_ENABLED=false` removes them.
- The rollup line said "68 unnamed regulars" (live day line 2026-11-02). That was also a bare cluster count of every kind. Rules version 5 says "unnamed regulars (...): some people, some dogs, 60 vehicles". Older stored lines get the clause swapped in place at read time, so every reader is fixed without waiting for a rebuild.
- The walkway journal facts and the Hub "Walkway history" panel no longer carry the regulars total.
- Chat stance's street clip went from 520 to 900 characters so the stories survive. The story line is last, so a cut drops it first.

## Outcome moved

- Live read-only replay at `now=2026-11-03T00:30Z` (yesterday evening, local), run against production Postgres with the patched code:
  - Eyes clause: "receiving frames ...; today so far (through 00:00), people: at least 2, at most 29 (the camera splits one person into several)". It no longer reads "N individuals counted today".
  - Street line gained: "Walkway motion stories today (spot-checked): at 17:22, A person walked left to right and away from the house on the walkway, over 15 s (7 detections), together with 1 other person, together with a dog, a dog; at 12:51, A person walked right to left and toward the house ...". The 17:22 pair (two stories 2 s apart, each `together_story_ids` of the other) is told once.
- Live read at 2026-11-03 08:29Z (01:29 local): the day row is "through 01:00, nobody passed". So the eyes clause has no number, and no stories have happened today yet, so there is no story line. Both are correct: neither says "nobody passed".
- Gate, live tally: rules version 1, 129 right, 2 wrong_other, 1 cant_tell, 0 wrong names. The gate passes.

## Current architecture

- `orion/situational/camera_eyes.py` `fetch_camera_eyes` ran `count(DISTINCT individual_id)` on `vision_individual_sighting` since local midnight for street cameras (all kinds). `eye_phrase` printed it bare. It runs in cortex-exec (situation brief) and in the Hub (curiosity self-inquiry "YOUR EYES").
- `orion/situational/perception_reader.py` `fetch_street_summary` built captions, who, rhythm, unresolved and patio lines. It read no motion stories, even though the rollups already carried gated story sentences.
- `orion/vision/walkway_rollup.py` `render_line` printed "N unnamed regulars" from `regulars.count`. `display_line` returned stored v4 lines unchanged.

## Architecture touched

- `camera_eyes`: one primary-key probe on `vision_rollup` (0.14 ms live) replaces the sightings count. It fails open: a failed read costs only the number.
- `perception_reader`: three new reads. Gate version (5.4 ms live), verdict tally (4.1 ms), today's stories (0.18 ms). A gate or story failure costs only the story line, and the other lines survive (tested). The gate has to pass before any story content is read.
- `walkway_rollup`: `RULES_VERSION` 4 -> 5 and `regulars_phrase`. `display_line` swaps the old clause in v2-v4 lines, and its own failure gives "", never the bare stored count. Because the rules version moved, orion-sql-writer rebuilds the hours inside `catchup_hours` (48 h) once. That is the same bounded cost as the v2/v3/v4 bumps.
- `chat_stance`: `STREET_SUMMARY_CHARS = 900`.

## Files changed

- `orion/situational/camera_eyes.py`: reads the people range from the day rollup; `individuals_today` and the sightings count are retired; `count_individuals_for` is renamed `people_range_for`.
- `orion/schemas/situation.py`: `CameraSenseV1.individuals_today` replaced by `people_today_low/high/through`.
- `orion/situational/context.py`, `services/orion-hub/scripts/curiosity_investigation.py`: the parameter rename.
- `orion/situational/perception_reader.py`: gated motion stories (`_gated_stories`, `street_story_line`, `STREET_MOTION_LINE_ENABLED`).
- `orion/vision/walkway_rollup.py`: rules version 5, `regulars_phrase`, `display_line` swaps the regulars clause.
- `services/orion-actions/app/walkway_forecast.py`: journal facts have no regulars count for living kinds.
- `services/orion-hub/static/js/walkway-history.js`, `services/orion-hub/scripts/walkway_rollup_routes.py`: the regulars chip and its total count are gone.
- `services/orion-cortex-exec/app/chat_stance.py`: street clip is 900.
- `services/orion-cortex-exec/.env_example`, `services/orion-hub/.env_example`: `STREET_MOTION_LINE_ENABLED=true`.
- `services/orion-cortex-exec/README.md`: the eyes clause and the story line are documented.
- Tests: `orion/situational/tests/test_camera_eyes.py`, `orion/situational/tests/test_street_stories.py` (new), `orion/vision/tests/test_walkway_regulars.py` (new), `services/orion-actions/tests/test_walkway_forecast.py`, `services/orion-cortex-exec/tests/test_chat_stance_street_context.py`, `services/orion-hub/tests/test_curiosity_self_inquiry.py`, `services/orion-thought/tests/test_reverie_street_day.py`.

### Every Orion-facing bare people/individual count found (sweep)

| Where | Was | Now |
|---|---|---|
| `camera_eyes.eye_phrase` (every chat turn, Hub self-inquiry) | "111 individuals counted today" | day-rollup range, or no number |
| `walkway_rollup.render_line` regulars (day letter, curiosity study material, reveries, walkway journals, substrate evidence) | "68 unnamed regulars" | "unnamed regulars (...): some people, some dogs, 60 vehicles" |
| `walkway_forecast.compact_rollup_facts` (journal LLM facts) | `unnamed_regulars.count`, `by_kind.person` | `things_by_kind` + `living_kinds_seen_before` |
| Hub walkway-history chip and routes (Juniper-facing) | "3 unnamed regulars", `regulars.count` | removed |

These were checked and are already ranges or not cluster counts: the who line (`_who_line`), the patio one-frame floor, rollup busiest/quietest totals (mixed-kind upper bounds labelled "at most"), and temporal-self story labels.

## Schema / bus / API changes

- Added: `CameraSenseV1.people_today_low`, `people_today_high`, `people_today_through`.
- Removed: `CameraSenseV1.individuals_today`. It is in-process only: built and rendered in the same service, not on the bus, not persisted. The reviewer confirmed this.
- Renamed: `fetch_camera_eyes(count_individuals_for=)` -> `people_range_for=`.
- Behavior changed: the `vision_rollup` line at rules version 5 (regulars). The Hub `/walkway` rollup JSON `regulars` has no `count`.
- Compatibility notes: no bus channel or registry change. `CameraSenseV1` stays registered under the same name.

## Env/config changes

- Added keys: `STREET_MOTION_LINE_ENABLED=true` (orion-cortex-exec, orion-hub). It ships ON.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys orion-cortex-exec orion-hub`: yes. Both live `.env` files hold `STREET_MOTION_LINE_ENABLED=true`.
- Skipped keys requiring operator action: none.

## Tests run

```text
orion/situational/tests orion/vision/tests orion/orion_day/tests/test_gather.py      352 passed
orion-cortex-exec test_chat_stance_street_context + test_situation_perception_context 60 passed
orion-hub test_curiosity_self_inquiry, test_walkway_rollup_routes, browser smoke, street day  56 passed, 1 skipped
orion-actions test_walkway_forecast                                                  25 passed
orion-thought test_reverie_street_day + test_vision_reader                           26 passed
orion-sql-writer test_vision_rollup + evals/test_walkway_reducers_eval               48 passed, 2 skipped
orion-substrate-runtime test_walkway_seen_loop                                       9 passed, 2 skipped
node services/orion-hub/static/js/walkway-history.test.js                            3 pass, 0 fail
scripts/check_env_template_parity.py PASS; check_env_key_single_source OK; check_personal_terms OK
```

The new tests fail on main: 12 test_camera_eyes tests, all 3 test_walkway_regulars tests, and test_street_stories fails at import.

Broader cortex-exec stance/situation set: 20 failures, the same 20 on main (diffed). They are pre-existing.

## Evals run

```text
services/orion-sql-writer/evals/test_walkway_reducers_eval.py  passed (part of the 48 above)
```

There is no eval harness for the street-line wording. The live read-only replay above is the evidence. Its SQL ran on real Postgres (`CAST(:since AS timestamptz)`, `text[]` together ids). CI exercises the story SQL only through a fake connection.

## Docker/build/smoke checks

```text
Not run (not asked to deploy). Live read-only probe of the patched functions against production Postgres: see Outcome moved.
```

## Review findings fixed

- Finding: a raise in the gate/story path (bad verdict row) escaped to `fetch_street_summary`'s outer except and wiped every street line.
  - Fix: `_gated_stories` and the story render each fail alone.
  - Evidence: `test_a_broken_story_gate_costs_only_the_story_line`, `test_a_bad_story_row_costs_only_the_story_line`.
- Finding: `display_line`'s regulars swap was unguarded, so malformed `regulars` would raise into five readers.
  - Fix: guarded; a failure gives "".
  - Evidence: `test_malformed_regulars_never_raise_into_readers`.
- Finding: the stance comment about cut order was wrong, and stories could push the patio head count out of the 900 clip.
  - Fix: the story line moved to last, and the comment was corrected.
  - Evidence: `test_story_line_is_last_so_a_length_cut_drops_it_first`.
- Finding: the cortex-exec README described the old bare count.
  - Fix: updated, and the story line and its flag are documented.
- Finding: `_ids` mis-parsed JSON strings and quoted array literals.
  - Fix: list-only, quotes stripped.
  - Evidence: `test_together_ids_shapes`.
- Finding: a third copy of the local-midnight math.
  - Fix: uses `walkway_rollup.local_midnight`.
- Finding: the Hub route still sent `regulars.count`.
  - Fix: dropped.

## Restart required

Rebuild from the clean primary main checkout after merge (the post-pull rebuild covers these):

```bash
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-thought up -d --build
scripts/safe_docker_build.sh orion-actions up -d --build
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```

## Post-deploy live check

The first chat turn after cortex-exec is rebuilt should contain the range and no bare count:

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "SELECT correlation_id, ts, raw::text LIKE '%individuals counted today%' AS bare, substring(raw::text from position('outdoor camera (walkway' in raw::text) for 400) FROM cockpit_turn_sighting WHERE stage='motor_boot' ORDER BY ts DESC LIMIT 3"
```

Pass: `bare` is `f`, and the clause reads "today so far (through HH:MM), people: at least X, at most Y ..." (or has no number before the first person of the day). On a day with a story, the street bullet ends with "Walkway motion stories today (spot-checked): ...".

## Risks / concerns

- Severity: low. Concern: story sentences carry the writer's own wording: "together with 1 other person" (a co-presence count within one passage, not a cluster total) and repeated companions ("a dog, a dog, a dog"). Mitigation: follow-up in orion-sql-writer `vision_motion.py` to collapse companions ("with 3 dogs" / "with someone else").
- Severity: low. Concern: the eyes range covers closed hours only ("through 01:00"), so it lags up to an hour behind the street line's 15-minute who line. Mitigation: the "through HH:MM" label says so.
- Severity: low. Concern: the rules version bump rebuilds 48 h of hour rows once, and those may shift slightly under today's clusters. Mitigation: the same bounded cost as earlier bumps.
- Severity: low. Concern: the story SQL is tested in CI only through a fake connection. Mitigation: a live read-only run against production Postgres passed (above).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3071

🤖 Generated with [Claude Code](https://claude.com/claude-code)
