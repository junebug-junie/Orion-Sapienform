## Summary

- The walkway story card now asks about one person at a time. It shows their pictures, asks "Is this Sasha?" (or "Who is this?"), and offers three answers: "Yes, that's Sasha", "No, it's ..." and "Can't see them". Then "Next person (2 of 4)".
- "No, it's ..." opens a short list of the names Orion knows, most seen first. It also has "Someone not listed" (type a name) and "I don't know who".
- Juniper's guard (her answer 1): picking a name from the list is one tap with no confirm step. The card then shows "Changed to X - Undo" and sends the answer only when she moves on: Next, Done, another answer, or leaving the page.
- Stories where she could only say "not sure" go back in the unchecked queue once (her answer 5). On 2026-11-24, 130 such stories existed. Her first answer on the new card closes each one again.
- The tally line now counts "can't see them" answers separately. `people` is gone from the API: the card uses `persons` only.

## Outcome moved

- Fewer questions per story: one person at a time, and a person without a picture is never asked. Target: at most 2.55 asked persons per story, down from 3.7.
- She can now say "yes" or "no, it's X" directly, instead of pressing "not sure". "Not sure" was 85% of story-card picture answers.

## Current architecture

Patch A, which this branch is stacked on, showed every person in a story as a list with a name box and four buttons each.

## Architecture touched

- **Hub routes** (`motion_story_routes.py`):
  - New: `GET /api/vision/known-names`.
  - The person route takes `card` and `dont_know`.
  - The list response no longer has `people`, and has a new `reopened` flag per story.
  - The tally has `person_answers` and `cant_see`.
- **Card** (`walkway-stories.js`): the one-at-a-time card, the held-answer state, and the known-names list.
- **Shared module** (`orion/vision/story_persons.py`): `card` and `dont_know` in the answer row. The repeat check now also compares which card sent the answer.

## Files changed

- `services/orion-hub/static/js/walkway-stories.js`: the one-at-a-time card (`renderAsk`, `cursorIndex`, `answerSpec`, `nameChoices`), the held answer with Undo (`flush`, `undo`, `go`, `send`), and the tally line.
- `services/orion-hub/scripts/motion_story_routes.py`: known names, re-asking "not sure" stories, `public_story`, the can't-see count, `card`, `dont_know`.
- `services/orion-hub/templates/index.html`: the status line is read out by screen readers (`aria-live`).
- `orion/vision/story_persons.py`: `card` and `dont_know`.
- Tests:
  - `services/orion-hub/tests/test_motion_story_routes.py`: tests moved from `people` to `persons`, plus known names, re-asking once, and the can't-see tally.
  - `test_walkway_stories_browser_smoke.py`: the browser smokes rewritten for the new card, plus a held-pick smoke.
  - `walkway-stories.test.js` and `test_walkway_stories_js.py`.
- `services/orion-hub/README.md`.

## Schema / bus / API changes

- Added:
  - `GET /api/vision/known-names?kind=person&limit=8`;
  - `card` and `dont_know` on the person route;
  - `reopened` on each story;
  - `person_answers` and `cant_see` in the tally.
- Removed: `people` from the list and read-back responses. The card was the only reader; a repo search found no other.
- Renamed: nothing.
- Behavior changed: `unchecked_only=true` also lists the re-asked "not sure" stories.
- Compatibility notes:
  - An old cached page still works. It sends no `card` field, so its answers do not close a re-asked story.
  - No new answer kind. No new verdict type.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed, because no template changed.
- Skipped keys that need operator action: none.

## Tests run

```text
Hub tests against a throwaway Postgres:
  test_motion_story_routes.py + test_walkway_stories_js.py + test_story_persons.py   57 passed
Browser:
  test_walkway_stories_browser_smoke.py (Chromium, 360 px)                          5 passed
  (one run failed once in about 20 earlier runs. No failure after the fixes.
   It looked like a timing issue in the older first smoke test.)
Pure:
  node --test services/orion-hub/static/js/walkway-stories.test.js                  16 pass
Static gate:
  scripts/check_async_routes_not_blocking.py                                        ok
```

## Evals run

```text
No eval harness covers the story card. The design's acceptance checks 2, 3 and 4 (the
'same' share, the 'unsure' share, asked persons per story) are re-measured live (#3857).
```

## Docker/build/smoke checks

```text
Not run. The change touches only the Hub's Python and static JS.
```

## Orion's questions settled here

- **Q2, "Can't see them" writes `unsure` per group:** confirmed on the new card.
  - Its answer is sent with `group: true`, so the Patch A fan-out applies: one `unsure` row per camera group, on that group's shown pictures.
  - Tests: Patch A's `test_one_answer_on_a_linked_person_...` (the route side) and the browser smoke `test_walkway_story_one_person_at_a_time_browser_smoke` (what the card sends).

## Review findings fixed

- Finding: a held pick was lost if sending it failed.
  - Fix: the pick comes back, and she can try again or Undo.
  - Evidence: `test_walkway_story_held_pick_is_never_lost_nor_doubled_browser_smoke` (the server returns 503 first).
- Finding: Undo while the pick was being sent said "nothing was saved".
  - Fix: a "Saving ..." line with no Undo while it is sent. Undo then refuses plainly.
- Finding: a held pick followed by another answer on the same person sent both.
  - Fix: the new answer replaces the held pick.
  - Evidence: the same smoke checks that only one row is sent.
- Finding: the last person had no Next, so a held pick could stay unsent.
  - Fix: a "Done" button appears when a pick is held. A hidden page sends the pick (`visibilitychange`). A page restored from the back-forward cache re-renders.
  - Evidence: the same smoke.
- Finding: the card's place in a story was a number, so it jumped to another person when the list changed.
  - Fix: the card now remembers the person, not the number.
  - Evidence: the node test `the cursor is a person ...`.
- Finding: `load()` could render twice when a 409 came back mid-send.
  - Fix: `load()` sends the held pick first and stops if that fails.
- Finding: a re-asked story with no one left to ask stayed in the queue for ever.
  - Fix: such a story is no longer listed.
  - Evidence: `test_not_sure_stories_go_back_in_the_unchecked_queue_once`.
- Finding: accessibility.
  - Fix: `aria-expanded` and `aria-controls` on "No, it's ..."; an `aria-live` status line; focus moves to the question after Next or Undo.
- Not changed (low, written down here):
  - The re-ask query scans the answered asks each time the unchecked list loads. That is about 1.4k rows today.
  - The known-names list can offer a name she rejected for one camera group, if another group still carries it.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low.
  - Concern: a held pick is only kept in the open page. If the browser is killed with no `pagehide` or `visibilitychange` event, the pick is lost. Nothing is ever saved wrongly in that case.
  - Mitigation: three triggers send it (Next or Done, another answer, page hidden or closed). The card says "Saved when you go to the next person".
- Severity: low.
  - Concern: "I don't know who" on an unnamed person is stored as `unsure` with `dont_know: true`. Nothing reads `dont_know` yet.
  - Mitigation: it stays separate from "can't see them", so a later reader can tell the two apart.
- Issues opened: none new. #3857 (verify-live) covers this patch.
- Issues closed: none.

## PR link

(filled in after the PR is opened)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
