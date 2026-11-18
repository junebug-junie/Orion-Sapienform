# Walkway stories: name or correct each person, wired into recognition

## Summary

- Each story card now shows who the story is about: one box per person (the walker, and anyone it walked with). A box shows the name Orion has, or "unnamed person (individual <short id>)".
- In each box Juniper can type a name, say "Someone else" (and optionally type who it really is), or say "Not sure".
- No new identity store. A typed name is a normal "who is this?" answer: an `orion_ask` row that orion-sql-writer applies as a typed label, and the usual `orion:ask:answered` event for the substrate entity. A story name also links the person to an existing individual Juniper already gave that name.
- "Someone else" records one picture verdict per story crop. The existing split pass splits those crops off the individual.
- Correcting the name a story was written with records the story's "Wrong name" verdict, so the gate tally sees it.
- The story sentence shows the name. It is re-derived each time the card loads: Juniper's answer on that story wins; otherwise the same strict naming rule runs on today's names. That rule now lives in one shared module, `orion/vision/story_naming.py`.

## Outcome moved

Before: a story said "A person walked right to left ..." and Juniper had no way to name the person or fix a wrong name. Nothing she saw on the card reached recognition.

After: one click on a story names the individual everywhere Orion recognizes them. The substrate gets the entity, other stories covering that person pick up the name, and a wrong name feeds the gate tally. "Someone else" splits a mis-clustered person.

## Current architecture

- `vision_motion_story` rows (orion-sql-writer `app/vision_motion_loop.py`) store a sentence, `name_label` (the name at write time), and `evidence` crops, each with `individual_id`.
- The Hub card (`scripts/motion_story_routes.py`, `static/js/walkway-stories.js`) listed stories and recorded Right / Wrong name / Wrong motion / Wrong other / Can't tell.
- Names came only from "who is this?" and "same person?" ask cards (`scripts/ask_routes.py`). sql-writer `_apply_answers` turned answered rows into typed labels. Picture verdicts on multi-picture cards fed `app/vision_split.py`.

## Architecture touched

- Hub: new route `POST /api/vision/motion-stories/{story_id}/people/{individual_id}`. `GET /api/vision/motion-stories` adds `people`, `display_name`, `display_sentence`, `name_source`, `together_story_ids` per story.
- sql-writer `_apply_answers`: story names add a same-name `'same'` pair link. New source kind `vision_story_person` records picture verdicts only (the `crop_id` comes straight from the story). `ASKED_TODAY_SQL` no longer counts story names against the opener's daily cap.
- Shared: `orion/vision/story_naming.py` (the naming rule, nouns, subject swap). sql-writer's `name_for` delegates to it.

## Files changed

- `orion/vision/story_naming.py`: the one naming rule, used at write time (sql-writer) and read time (Hub).
- `orion/vision/tests/test_story_naming.py`: rule, evidence reader, subject swap.
- `orion/schemas/ask.py`: documents `vision_story_person` and story-name answers on `source_kind`.
- `services/orion-hub/scripts/motion_story_routes.py`: people, read-time sentence, the person route, the Wrong name rule.
- `services/orion-hub/static/js/walkway-stories.js`: person boxes, re-render from the server's answer, "Orion first wrote: ...", the 60 s refresh never wipes a half-typed name.
- `services/orion-hub/tests/test_motion_story_routes.py`: pure tests and real-Postgres route tests.
- `services/orion-hub/tests/test_walkway_stories_js.py`: JS helpers run in node.
- `services/orion-hub/tests/test_walkway_stories_browser_smoke.py`: 360 px browser smoke of the person boxes.
- `services/orion-sql-writer/app/vision_individuals.py`: same-name link, `vision_story_person`, crop_id passthrough, cap exclusion.
- `services/orion-sql-writer/app/vision_motion.py`: `name_for` and `NOUNS` come from the shared module.
- `services/orion-sql-writer/tests/test_vision_split.py`: real-Postgres apply, link, split, and cap tests.
- `services/orion-sql-writer/README.md`, `services/orion-hub/README.md`: docs.

## Schema / bus / API changes

- Added: Hub route `POST /api/vision/motion-stories/{story_id}/people/{individual_id}`, body `{action: name|someone_else|unsure, name}`. New `orion_ask.source_kind` value `vision_story_person` (free-text column, no migration).
- Removed: none.
- Renamed: none.
- Behavior changed: the list response carries extra fields (additive). `orion:ask:answered` payload is unchanged; story names publish the same `OrionAskAnsweredV1` a "who is this?" answer does.
- Compatibility notes: no migration. `orion_ask.answer_detail` and `vision_ask_picture_verdict` already exist (`manual_migration_vision_ask_names_v1.sql`). Deploy order does not matter for correctness: a Hub without the new sql-writer writes rows the old writer applies as plain typed names (no same-name link) and leaves `vision_story_person` rows unapplied until sql-writer is updated.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change)
- skipped keys requiring operator action: none

## Tests run

```text
orion/vision/tests + orion-sql-writer test_vision_split/confirm_asks/motion/story_strip
  (ORION_TEST_PG_URI = throwaway postgres:16)                        160 passed
orion-hub test_motion_story_routes (real Postgres) + test_walkway_stories_js
  + test_ask_routes + test_ask_names_pg                               87 passed, 1 skipped
orion-hub test_walkway_stories_browser_smoke (Chromium, 360 px)       2 passed
scripts/check_async_routes_not_blocking.py                            no blocking calls
orion-sql-writer full tests: same 14 environment failures as main (biometrics, notify,
  chat history, subscription checks); none touched by this patch
```

Local end-to-end on a throwaway database (Hub route in process, then sql-writer `_apply_answers` on the same rows):

```text
POST 200 Maya walked right to left, over 5 s (3 detections).
applied 1
label ('Maya', False, 'ask-ecf79bd6...')          typed label, provenance = the story ask
pairs [('ind-anon', 'ind-maya', 'same', 'ask-ecf79bd6...')]
list Maya walked right to left ... juniper [('ind-anon', 'Maya')]
published [('vision_individual', 'ind-anon', 'Maya')]
```

## Evals run

```text
No new eval. The naming rule did not change (same checks in the same order, now shared);
the existing motion story replay eval covers it. The new behavior is direct operator input,
so it is covered by real-Postgres tests, not a quality eval.
```

## Docker/build/smoke checks

```text
No image build: no dependency or compose change. Both images already COPY orion/ (orion/vision/story_naming.py ships).
Live path UNVERIFIED until deploy (no deploy from a worktree).
```

## Review findings fixed

- Finding: "Not sure" after "Someone else" brought back the name she had just rejected (the display read only the latest answer).
  - Fix: the display uses her latest name / someone-else on the story; "not sure" never undoes it.
  - Evidence: `test_story_display_rule_uses_todays_names_and_juniper_wins_on_her_story`, plus a display assertion after the unsure POST.
- Finding: story names used up the slow "who is this?" opener's daily cap (2 per day).
  - Fix: `ASKED_TODAY_SQL` skips rows with `answer_detail.story_id`.
  - Evidence: `test_story_names_do_not_use_up_the_who_is_this_daily_cap`.
- Finding: a failed read-back after commit skipped the substrate event.
  - Fix: publish right after the commit; the read-back is fail-open.
  - Evidence: `test_a_name_correction_never_overwrites_a_motion_verdict_and_a_failed_readback_still_publishes`.
- Finding: the automatic Wrong name overwrote Wrong motion / Wrong other.
  - Fix: written only over no verdict, Right, or Can't tell.
  - Evidence: same test.
- Finding: re-saving a name the individual already carries counted as a wrong name.
  - Fix: skipped when the typed name matches the current label.
  - Evidence: `test_wrong_name_only_for_a_correction_of_the_walkers_written_name`.
- Finding: the same-name link gave up when the first candidate had a 'different' verdict.
  - Fix: candidates with any pair verdict are skipped in SQL; the next one is linked.
  - Evidence: `test_story_name_skips_a_different_verdict_and_plain_answers_do_not_link`.
- Finding: a half-typed name blocked filter changes too.
  - Fix: only the 60 s timer waits.
  - Evidence: browser smoke unticks "only checkable" with text in a box.
- Finding (found while fixing): a test edit had moved one test's `_run` call into another test, so one test body never ran.
  - Fix: restored. All 21 route tests run.

## Restart required

Deploy from the clean primary `main` checkout after merge.

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low
  - Concern: a story name links to an existing individual with the same typed name. Two different people with the same name on the same camera would be linked. A plain "who is this?" answer still does not link (unchanged behavior).
  - Mitigation: never over an existing pair verdict; a "Different people" answer removes nothing but blocks new links.
- Severity: low
  - Concern: "Someone else" on an individual made mostly of this one walk is refused by the split (`majority`). The card cannot show that refusal yet.
  - Mitigation: the refusal is recorded in `vision_individual_split`; documented in the sql-writer README.
- Severity: low
  - Concern: saving the same name in a different case ("maya" for "Maya") rewrites the typed label everywhere (last typed name wins), and the label-replaced line is logged.
  - Mitigation: matches every other card; not a wrong-name verdict.
- Severity: low
  - Concern: when a story's walker carries several ids, the latest answer about any of them sets the story's name.
  - Mitigation: those ids all had one name for the rule to fire; edge case.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2753

🤖 Generated with [Claude Code](https://claude.com/claude-code)
