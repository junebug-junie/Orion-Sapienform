## Summary

- Juniper now grades a walkway story sentence in small parts: "Right people?", "Right direction?", "Right place?". Each has Yes / No / Can't tell. Length is left out (her answer 3).
- The card asks only about the parts this sentence actually says. orion-sql-writer records those parts when it writes the sentence. Older stories get their parts worked out once from the facts stored with them. The card never reads them back from the words.
- The story's verdict comes from the part answers, and only once every part is answered. A "no" on people beats a "no" on direction or place, so a wrong name never hides behind a wrong motion.
- The gate's "zero wrong names" rule now also reads Juniper's name corrections when it counts. A story she corrected stays a wrong name even if a later grade says "right people".
- The tally line now says where each verdict came from: the old buttons, the parts, or a name correction. It also says how many stories are only half graded.

## Outcome moved

- One click no longer answers three questions at once.
- The gate stays honest while grading moves from the old buttons to parts:
  - a half-graded story moves no count;
  - an old wrong name is never quietly turned into "right";
  - a name correction cannot hide behind a later "right people? yes".

## Current architecture

Four buttons (Right / Wrong name / Wrong motion / Wrong other) wrote one story verdict. The gate counted those verdicts. Nothing recorded which parts a sentence stated.

## Architecture touched

- **Migration** `manual_migration_vision_motion_story_parts_v1.sql`: three new tables, nothing else.
  - `vision_motion_story_parts`: which parts each sentence states.
  - `vision_motion_story_part_verdict`: Juniper's answer for each part.
  - `vision_motion_story_verdict_source`: who wrote each story verdict.
  - Header: `-- ORION-MIGRATION-REQUIRED-BY: orion-hub`. sql-writer writes the parts table only after checking that it exists, so it does not need the migration.
- **sql-writer:**
  - `vision_motion.sentence_parts` and `Story.parts`.
  - The loop stores the parts when it writes a story, and works them out once for older stories (`backfill_story_parts`).
  - The prune step also clears these tables for removed stories.
  - The rollup gate uses the shared count.
- **Shared** `orion/vision/motion_story_gate.py`:
  - `PARTS` and `derive_story_verdict`;
  - `ACTIVE_CORRECTIONS_SQL` (the list of corrections the gate reads) and `gate_verdicts_sql`.
  - The Hub tally, the walkway rollups and the street line (`perception_reader`) all use this count.
- **Hub:**
  - new route `POST .../parts`;
  - parts on each listed story;
  - the verdict source is recorded;
  - person answers mark name corrections, write the automatic "people = no", and remove it when the correction is undone;
  - the tally shows `by_source`, `half_graded` and `wrong_name_from_corrections`.
- **Card:** the part rows replace the four buttons. "Can't tell" stays for the whole story.

## Files changed

- `services/orion-sql-db/manual_migration_vision_motion_story_parts_v1.sql` (new): the three tables.
- `services/orion-sql-writer/app/vision_motion.py`: `sentence_parts`, `Story.parts`, and the rule for when to bump `RULES_VERSION`.
- `services/orion-sql-writer/app/vision_motion_loop.py`: stores parts at compose time, the one-time backfill, and the prune.
- `services/orion-sql-writer/app/vision_rollup.py` and `orion/situational/perception_reader.py`: use the shared gate count.
- `orion/vision/motion_story_gate.py`: deriving the verdict from parts, and the correction list.
- `services/orion-hub/scripts/motion_story_routes.py`: the parts route, deriving the verdict, the tally, and the correction marks.
- `services/orion-hub/static/js/walkway-stories.js`: the part rows and the tally source line.
- Tests:
  - `services/orion-hub/tests/test_motion_story_routes.py`
  - `test_walkway_stories_browser_smoke.py`
  - `walkway-stories.test.js`
  - `services/orion-sql-writer/tests/test_vision_motion.py`
  - `orion/vision/tests/test_motion_story_parts.py`
- Workflows:
  - `orion-sql-writer-tests.yml`: adds the migration to the test database.
  - `orion-memory-episode-tests.yml`: adds the path filters.
- `services/orion-hub/README.md`.

## Schema / bus / API changes

- Added:
  - three tables;
  - `POST /api/vision/motion-stories/{story_id}/parts {part, verdict}`;
  - `parts` on each story;
  - tally fields `by_source`, `half_graded`, `wrong_name_from_corrections`;
  - `answer_detail.corrects_story_name` on correction asks.
- Removed: nothing. The verdict route stays for "Can't tell" and for old cached pages.
- Renamed: nothing.
- Behavior changed:
  - The gate counts a story under an active name correction as a wrong name, whatever its verdict says. This applies only to corrections made from this patch on. Old corrections still count through their `wrong_name` verdicts, which part grading can no longer overwrite.
  - The gate numbers (50 checked / 80% right / 0 wrong names) are unchanged.
- Compatibility notes: no new bus channel. No new verdict type.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed, because no template changed.
- Skipped keys that need operator action: none.

## Tests run

```text
Hub tests against a throwaway Postgres 16 (with the new migration):
  test_motion_story_routes.py                                        50 passed
sql-writer tests against a throwaway Postgres:
  test_vision_motion.py + test_vision_rollup.py                      108 passed
  test_vision_split.py                                               passed
Pure:
  orion/vision/tests (incl. test_motion_story_parts.py)
    + orion/situational/tests/test_street_stories.py                 passed
  services/orion-hub/tests/test_walkway_rollup_routes.py             7 passed
  node --test walkway-stories.test.js                                17 pass
Browser:
  test_walkway_stories_browser_smoke.py (Chromium, 360 px)           6 passed
Migration gates:
  scripts/check_migration_required_by.py                             OK
  auto_apply_refusals(...)                                           [] (a test pins it)
```

## Evals run

```text
No eval harness covers the story card. Acceptance check 6 (old verdicts read the same:
260 right, 5 wrong_name, 2 wrong_other) still holds: old verdicts have no source row and
count as "old buttons". Re-measure live (#3857).
```

## Docker/build/smoke checks

```text
Not run. The migration is plain CREATE TABLE. The post-pull run applies it before it
rebuilds orion-hub.
```

## Orion's questions settled here

- **Q5 (gate transition):**
  - A half-graded story writes no verdict, stays in the unchecked queue, and moves no count.
  - The tally shows verdicts from the old buttons and from parts separately, plus the half-graded count.
  - The gate numbers are pinned in a test.
  - New grading cannot silently drop the gate. Old wrong names cannot be overwritten by a part grade. A correction counts at gate time.
  - Tests:
    - `test_a_half_graded_story_stays_unchecked_and_the_tally_says_who_wrote_each_verdict`
    - `test_part_grading_never_erases_a_wrong_name_recorded_before_parts_existed`
    - `test_the_gate_numbers_do_not_move`
- **Q6 (automatic "people = no" from a name correction):**
  - The gate reads the list of corrections when it counts (`ACTIVE_CORRECTIONS_SQL`), not only the derived verdict.
  - Undoing the correction removes the automatic part row and the verdict that row made.
  - Tests:
    - `test_a_name_correction_is_a_wrong_name_at_gate_time_and_undoing_it_drops_the_auto_part`
    - `test_undoing_a_name_correction_removes_the_automatic_people_no`
    - `test_a_correction_on_a_linked_person_is_undone_for_every_camera_group`
    - `test_the_shared_gate_count_reads_the_correction_ledger_through_sqlalchemy`
- **Q7 (which parts a story has):**
  - The parts are recorded when the sentence is written (`sentence_parts`, using the same inputs as `sentence_for`) and stored with the story.
  - Older stories get theirs worked out once from stored facts, never from the words.
  - The rule for bumping `RULES_VERSION` is written in `vision_motion.py`: any change to which parts a sentence states, or to how the subject, direction or place reads.
  - A test fingerprints what the sentence writer says over a fixed set of inputs, and fails if that changes without a version bump.
  - Tests:
    - `test_parts_come_from_the_stored_record_never_from_the_prose`
    - `test_the_cycle_stores_parts_when_it_writes_a_story_and_derives_older_ones_once`
    - `test_changing_what_a_sentence_claims_bumps_the_rules_version`

## Review findings fixed

- Finding: undoing a correction left its `wrong_name` verdict in place, so the gate still failed.
  - Fix: the undo also removes that verdict, or works it out again from the parts.
  - Evidence: the undo test now asserts `tally.wrong_name == 0`.
- Finding: corrections made before this patch were not in the correction list, so a part grade could overwrite them with "right".
  - Fix: a part grade never overwrites a wrong name that has no source row.
  - Evidence: `test_part_grading_never_erases_...`.
- Finding: two quick clicks on parts could leave a fully graded story with no verdict.
  - Fix: the story row is locked (`FOR UPDATE`) in the parts route and the person route.
- Finding: the backfill would re-read the same skipped rows forever after a version bump.
  - Fix: it filters on the current version, and inserts each batch in one statement.
- Finding: the fingerprint test hashed source code, so a comment edit would fail it.
  - Fix: it now hashes what the sentence writer outputs over a fixed set of inputs.
- Finding: removed stories left rows behind in two of the new tables.
  - Fix: the prune step clears all three.
- Finding: `half_graded` also counted stories that already had a verdict.
  - Fix: it now counts only stories with no verdict.
- Finding: the comment on the correction list described it wrongly; it works per camera group.
  - Fix: the comment says so, and a linked-person undo test covers it.
- Finding: no test ran the gate's count through SQLAlchemy.
  - Fix: one Postgres-backed test does now.

## Restart required

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

The post-pull run applies the migration first. Its header names `orion-hub`.

## Risks / concerns

- Severity: medium.
  - Concern: a fourth reader of the gate does not use the new count yet. It is in `services/orion-durable-runs/app/temporal_self_sources.py` (temporal-self street events). After a correction, the Hub could say the gate is closed while that reader still treats it as passed.
  - Mitigation: no live effect while the gate is not passed (it is not today). Issue #3872.
- Severity: low.
  - Concern: the gate count looks up the latest answer for each story over all answered story-card asks. That is about 1.4k rows today.
  - Mitigation: fine at today's size. Add a partial index if `orion_ask` grows a lot.
- Issues opened: #3872 (follow-up).
- Issues closed: none.
- #3857 (verify-live) covers this patch.

## PR link

(filled in after the PR is opened)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
