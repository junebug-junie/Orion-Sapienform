## Summary

- Two camera groups can carry the same typed name without being linked. Today that is [name 1] with 3 groups, Juniper on the doorbell with 2, and [name 2] with 2. The card now asks: "Orion has another Sasha group. Same person?", and shows both pictures. Nothing gets linked without her answer (her answer 4).
- Her yes is the existing "same person?" answer, and orion-sql-writer links the two groups. Her no records "different", and the card does not ask about that pair again.
- A link made this way shows "Linked as the same person - Undo" for 7 days. Undo asks Orion's next pass to remove the link. The card then shows either "Link undone" or the reason it could not be undone.
- The undo refuses, and says why, in these cases:
  - a split recorded the link;
  - a name was typed on the linked person after the link was made;
  - other "same" links still join the two groups (a split's link, a typed-name link, or another answer), so removing this one would change nothing she could see.
- `scripts/undo_vision_link.py` does the same undo from the command line. It is a dry run unless given `--apply`, the same way the split undo script works.

## Outcome moved

- The last 3 names with groups that are not linked can be joined on the card.
- A wrong "same" answer can be taken back without a database edit.
- The undo never looks like it worked when it did not.

## Current architecture

Same-name groups were linked only when she typed a name she had used before (`_link_same_name`). Nothing asked about groups that were already named but not linked. Only splits could be undone (`undo_vision_split.py`).

## Architecture touched

- **Hub** (`motion_story_routes.py`):
  - `same_name_other` and `card_links` on each person;
  - new route `POST .../persons/{person_key}/same-name`;
  - new route `POST /api/vision/individual-links/{ask_id}/undo`, which only marks the ask with `answer_detail.undo_requested_at`.
- **sql-writer:**
  - new `app/vision_link_undo.py` with `undo_link` and `apply_link_undos`;
  - the individuals cycle runs `apply_link_undos` after it applies answers, under its own lock, so sql-writer stays the only writer of links.
- **Script:** `scripts/undo_vision_link.py`.
- **Card** (`walkway-stories.js`): the same-name question, and the link line with Undo.

## Files changed

- `services/orion-sql-writer/app/vision_link_undo.py` (new): the undo, its refusals, and the contradiction report.
- `services/orion-sql-writer/app/vision_individuals.py`: the cycle runs the requested undos.
- `scripts/undo_vision_link.py` (new): the operator door.
- `services/orion-hub/scripts/motion_story_routes.py`: the same-name question, the links list, and the undo request.
- `services/orion-hub/static/js/walkway-stories.js`: the question and the Undo line.
- Tests:
  - `services/orion-sql-writer/tests/test_vision_link_undo.py` (new)
  - the hub route tests
  - the browser smoke
  - the node tests
- `.github/workflows/orion-sql-writer-tests.yml`: runs the new test file and adds the script to the path filter.
- `services/orion-hub/README.md`.

## Schema / bus / API changes

- Added:
  - two routes;
  - `same_name_other` and `card_links` on each person;
  - `answer_detail` keys `action: same_name`, `undo_requested_at`, `undone_at`, `undo_refused`, `undo_refused_detail`.
- Removed: nothing.
- Renamed: nothing.
- Behavior changed: the individuals cycle now applies undos the Hub requested.
- Compatibility notes: no new table, no new answer kind, no new bus channel. Her answers use `JUNIPER_STORY_CARD_REF`, so no Orion ask statistic or act counts them (Patch A already excluded these rows from the daily cap and the self-monitor).

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- Skipped keys that need operator action: none.

## Tests run

```text
Hub tests against a throwaway Postgres 16:
  test_motion_story_routes.py                                          53 passed
sql-writer tests against a throwaway Postgres:
  confirm_asks, person_model, split, link_undo, motion, story_strip,
  rollup, individuals                                                  349 passed
Pure:
  orion/vision + orion/situational + orion/self_monitor tests          525 passed, 1 skipped
  node --test walkway-stories.test.js                                  18 pass
Browser:
  test_walkway_stories_browser_smoke.py (Chromium, 360 px)             7 passed
```

## Evals run

```text
No eval harness covers links. Acceptance check 7 ("a test links two groups, undoes it,
and the pair row and the copied label are gone; an undo after a dependent name refuses
and names it") is test_undo_removes_the_link_and_rederives_the_copied_name and
test_undo_refuses_out_of_scope_too_old_and_dependents_and_names_them.
```

## Docker/build/smoke checks

```text
Not run. No dependency, port or compose change.
```

## Orion's questions settled here

- **Q3: links versus splits.**
  - The undo first checks whether other "same" links still join the two groups (it walks the links, leaving this one out). If they do, it refuses plainly and names each link still joining them, with its ask.
  - A group that has "same" links and also a "different" row between two of its members is reported in `contradictions` and logged as `[VISION_LINK_CONTRADICTION]`.
  - A split whose `linked_pair` holds the link also refuses the undo.
  - Test: `test_undo_refuses_plainly_when_other_same_links_still_join_the_two`.
- **Q4: idempotency after this card's own merge.**
  - After "Same person? yes" links two groups mid-story, answering the joined person again writes no second who-is-this answer for the group she already answered.
  - Patch A's checks do the work: the route skips a repeated answer, and the writer drops a repeated story-card picture answer.
  - Test: `test_after_a_same_person_yes_the_merged_person_is_not_answered_twice`.
- **Q8: the undo chain test.**
  - Setup: groups a-b-c-d in a chain. Both ends have names she typed: a "Ann" and d "Annie" (typed later). b and c carry copies of the newer name.
  - Undoing the middle link b-c gives b the name "Ann" (from a) and c the name "Annie" (from d). Neither typed name is touched.
  - A three-group variant (p-q-r) is covered as well.
  - Live today: 0 individuals carry a copied label.
  - Test: `test_deleting_the_middle_link_of_a_chain_gives_each_end_its_own_typed_name`.

## Review findings fixed

- Finding: the "name typed since" refusal only looked at who-is-this asks. It missed names typed on the story card, on "Is this <name>?" cards and in "same person?" name boxes.
  - Fix: the check now reads the labels themselves: any typed label in the group with `labeled_at` after the link.
  - Evidence: the refusal test now types the name through the story-card path.
- Finding: an unexpected error on one undo rolled back the whole batch, and the same error came back on every cycle.
  - Fix: every error is caught per undo and written as `undo_refused = "error: ..."`. Other undos go on, and she can try again.
  - Evidence: `test_an_unexpected_error_on_one_undo_is_recorded_not_retried_forever`.
- Finding: `next()` without a default could turn into a 500 error, and could pick a group whose name was only a copy.
  - Fix: the route prefers a group whose name she typed. It returns 409 `same_name_question_gone` when no group matches.
- Finding: a double click could write two answers for one pair.
  - Fix: a lock per pair, then a re-check: one answer per pair.
- Finding: a refused undo said "still joined" without naming the links. Contradictions found on a successful undo were only logged.
  - Fix: the card names the links that still join the pair. Contradictions are also saved on the ask.
- Finding: the Hub and the writer measured the 7-day window differently.
  - Fix: both now use the same interval test.
- Finding: undo was accepted for links the card never shows.
  - Fix: the route now accepts only links made by the card's same-name question (`action = same_name`).
- Finding: the Python side and the SQL side folded names differently.
  - Fix: Python now uses `lower()`, the same as the SQL.
- Decided, not changed: a pair she answered once is never asked about again, even after an undo. The README says so.

## Restart required

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low.
  - Concern: an undo takes effect only on the next individuals cycle, not on the click.
  - Mitigation: the card says "Undo asked for". Each pass applies the undo or writes the reason it refused. A refused undo is never retried on its own; she can press "Try undo again".
- Severity: low.
  - Concern: the same-name question shows the other group's latest sighting thumbnail. A group can lack one, and the card then says "(no picture of the other group)". She can still answer.
  - Mitigation: none needed. The question is shown once per other group.
- Issues opened: none new. #3857 (verify-live) covers this patch.
- Issues closed: none.

## PR link

(filled in after the PR is opened)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
