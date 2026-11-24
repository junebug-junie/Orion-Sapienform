## Summary

- The walkway story card now shows camera groups that Juniper already said are the same person as one person. "Sasha, Sasha" becomes "Sasha, seen as 2 camera groups in this story", asked once.
- One answer on that person is saved once for each camera group in it. Each group's answer covers only that group's pictures that the card showed. One answer saves at most four rows.
- A repeat answer saves nothing new. Example: after two groups get linked mid-story, she says "yes" again on the joined person.
- A grouped row has "These are N camera groups - not the same person?". It saves her "same person? no" for one direct link, so a wrong link can be undone from the card.
- The design doc now records Juniper's six answers from 2026-11-24.

## Outcome moved

- The double ask goes away. Today 148 person rows on 97 stories repeat a person who is already linked.
- A wrong link can no longer hide inside a grouped row. Orion asked about this in question 1.

## Current architecture

The card (`walkway-stories.js`) listed one row per camera group (`people`). Each answer wrote one `orion_ask` row about one group. Linked groups showed as separate rows.

## Architecture touched

- **Hub routes** (`motion_story_routes.py`):
  - `GET /api/vision/motion-stories` adds `persons`.
  - The person route takes `group` and `crop_ids`.
  - New route: `POST .../persons/{person_key}/not-same`.
- **Shared pure module** `orion/vision/story_persons.py`. It does the grouping, picks the pictures (at most 4), and builds the answer row. The Hub and the sql-writer tests use the same copy.
- **sql-writer** (`vision_individuals.py`):
  - A "no" written over a "same" link re-works the names that were only copied across that link.
  - A story-card picture answer that repeats an earlier one is not saved twice.
  - Juniper's own "not the same person" rows do not count against Orion's daily "same person?" limit.
- **Self-monitor** (`orion/self_monitor/adapters.py`): Juniper's own cards do not grade any of Orion's guesses.

## Files changed

- `orion/vision/story_persons.py` (new) and `orion/vision/tests/test_story_persons.py` (new): the grouping rules.
- `services/orion-hub/scripts/motion_story_routes.py`:
  - adds `persons`, group answers, and the not-same route;
  - adds a savepoint around each fail-open read.
- `services/orion-hub/static/js/walkway-stories.js`: the grouped row and the not-same choice. Group answers send the crop ids that were shown.
- `services/orion-sql-writer/app/vision_individuals.py`: `rederive_copied_labels`, `same_group_of`, the repeat check in `_record_pictures`, and the daily-limit exclusion.
- `orion/self_monitor/adapters.py`: skips Juniper's own cards.
- Tests:
  - `services/orion-hub/tests/test_motion_story_routes.py`
  - `test_walkway_stories_browser_smoke.py`
  - `walkway-stories.test.js`
  - `services/orion-sql-writer/tests/test_vision_split.py`
  - `orion/self_monitor/tests/test_self_monitor_camera_asks.py`
- `.github/workflows/orion-memory-episode-tests.yml`: adds the new module to the path filter.
- `services/orion-hub/README.md`: describes the change.
- `docs/superpowers/specs/2026-11-24-walkway-who-card-redesign.md`: adds "Juniper's answers (2026-11-24)".

## Schema / bus / API changes

- Added:
  - `persons` on each listed story;
  - `group`, `crop_ids` on `POST .../people/{individual_id}`;
  - `ask_ids`, `skipped_individual_ids`, `skipped_all`, `person_key` in that route's response;
  - `POST /api/vision/motion-stories/{story_id}/persons/{person_key}/not-same {a, b}`.
- Removed: nothing. `people` stays for this release, and Patch B removes it.
- Renamed: nothing.
- Behavior changed:
  - A "no" on a pair re-works the copied names in both groups.
  - A repeated story-card picture answer is skipped.
- Compatibility notes:
  - An old cached page sends no `group` field and keeps the per-group path.
  - No new bus channel. No new answer kind.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed, because no template changed.
- Skipped keys that need operator action: none.

## Tests run

```text
Hub tests against a throwaway Postgres 16:
  services/orion-hub/tests/test_motion_story_routes.py              40 passed
sql-writer tests against a throwaway Postgres:
  services/orion-sql-writer/tests/test_vision_split.py              50 passed
  test_vision_confirm_asks.py + test_vision_individuals.py           pass
Pure tests:
  orion/vision/tests/test_story_persons.py + orion/self_monitor/tests   160 passed
  node --test services/orion-hub/static/js/walkway-stories.test.js      13 pass
Browser:
  test_walkway_stories_browser_smoke.py (Chromium, 360 px)             4 passed
Static gate:
  scripts/check_async_routes_not_blocking.py                           ok
```

## Evals run

```text
No eval harness covers the story card. The design's acceptance queries are the eval:
they are re-measured 14 days after deploy (issue #3857).
```

## Docker/build/smoke checks

```text
Not run: no Docker build. The change touches the Hub's Python and static JS, and the
sql-writer's Python. No dependencies, ports or compose wiring changed.
```

## Orion's questions settled here

- **Q1 (a wrong link hides once linked groups show as one person):** settled.
  - The grouped row offers "not the same person?". It writes the existing pair answer, `different`, for one direct link.
  - sql-writer writes that answer over the `same` row, including a typed-name link.
  - The card shows the two groups apart at once.
  - The route refuses with 409 `still_linked` when the two groups stay joined through other links.
  - Tests: `test_not_the_same_person_writes_her_pair_answer_for_one_direct_link` and `test_not_the_same_person_over_a_typed_name_link_splits_it_and_rederives_copied_names`.
- **Q2 (bound the fan-out):** settled.
  - One answer writes at most 4 rows, one per camera group, each on that group's shown pictures.
  - The writer makes one split per group, and its count matches the number of rows.
  - "Can't see them" writes `unsure` for each group and causes no split.
  - Tests: `test_one_answer_on_a_linked_person_writes_one_row_per_camera_group_on_its_own_pictures`, `test_one_no_on_a_linked_person_splits_each_group_once_and_unsure_splits_nothing`, and `test_a_grouped_person_shows_at_most_four_pictures_...`.
- **Q4 (idempotency after a mid-story merge):** settled.
  - The route skips a group whose latest answer on the story already says the same thing.
  - The writer does not save a story-card picture answer twice.
  - Tests: `test_answering_a_merged_person_again_writes_no_second_answer_per_group` and `test_a_repeated_story_yes_keeps_one_same_picture_per_crop`.
- **Q3, Q5, Q6, Q7, Q8:** Patches C and D.

## Review findings fixed

- Finding: the server rebuilt the person's pictures when she clicked. A link that landed after the card loaded could put picture answers on crops she never saw.
  - Fix: group answers send the crop ids the card showed. The route returns 409 `pictures_changed` when they differ. The card then reloads and saves nothing.
  - Evidence: `test_a_group_answer_on_pictures_the_card_did_not_show_is_refused`.
- Finding: "not the same person" changed nothing on the card until sql-writer ran. A second click saved a second row.
  - Fix: a pending answer leaves the link out of the grouping at once. A second click returns the same answer.
  - Evidence: the first test under Q1 now checks both.
- Finding: re-working copied names rewrote the label's record of which answer set it (`label_ask_id`), even when the name stayed the same. The person model keys its exemplars on that id.
  - Fix: only a different name is rewritten, compared without case.
  - Evidence: `test_a_no_on_an_unlinked_pair_leaves_copied_names_and_their_provenance_alone`.
- Finding: when a fail-open read inside `_enrich` failed, it left the outer savepoint broken, so the request failed instead of failing open.
  - Fix: the read now has its own savepoint.
- Small fixes:
  - the response has `skipped_all`, and the card says "Already recorded";
  - a separate SQL constant replaces a string `.replace`;
  - a guard covers a story row that does not read back;
  - a comment on the repeat check's known limit.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

## Risks / concerns

- Severity: low.
  - Concern: a pending "not the same person" answer hides the link from the card before sql-writer applies it. If sql-writer never applies it, the card and the database disagree.
  - Mitigation: the writer applies answered rows on every cycle. The live check is issue #3857.
- Severity: low.
  - Concern: the repeat check skips a story-card picture answer that matches any earlier one on that crop, including one made before a later correction.
  - Mitigation: the earlier row still stands. The limit is written in a code comment.
- Issues opened: #3857 (verify-live, covers Patches A-D).
- Issues closed: none.

## PR link

(filled in after the PR is opened)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
