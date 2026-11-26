## Summary

- The Walkway stories card shows one story at a time instead of a long scrolling list.
- Back / Next buttons and a "Story 3 of 20" line sit above the story.
- When a story gets its verdict (a verdict button, or the last sentence part answered), the card moves on to the next story by itself.
- The first load opens on the first story not graded yet. The 60 s refresh keeps the story she is on.
- Pictures of hidden stories are lazy, so they now load only when that story is shown.

## Outcome moved

Juniper asked for "a fixed view where I go one at a time... when I'm done on that card, it moves onto the next one. I can go back too."

## Current architecture

`static/js/walkway-stories.js` `mount()` rendered up to 20 stories into `#walkwayStoriesList`, a scroll box 40rem high.

## Architecture touched

Hub frontend only. No API, schema, bus, or env changes.

## Files changed

- `services/orion-hub/templates/index.html`: pager (Back, position, Next); the list is no longer a scroll box.
- `services/orion-hub/static/js/walkway-stories.js`: `applyView`, `show`, `finished`, plus the pure helpers `startIndex` and `positionText`.
- `services/orion-hub/static/js/walkway-stories.test.js`: helper tests, plus jsdom interaction tests (skipped where jsdom is not installed).
- `services/orion-hub/tests/test_walkway_stories_browser_smoke.py`: the smokes navigate with Next, and check the auto-advance, the last-story message, and Back.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
node --test walkway-stories.test.js (with jsdom)   25 pass
node --test walkway-stories.test.js (no jsdom)     pass, 2 jsdom tests skipped
pytest test_walkway_stories_browser_smoke.py        7 passed (Chromium, 360 px)
pytest test_motion_story_routes.py                  17 passed, 36 skipped
```

## Evals run

None. This is a UI-only change with no eval harness. The browser smoke covers the interaction.

## Docker/build/smoke checks

There was no Docker build. Static files only.

## Review findings fixed

- Finding: the auto-advance moved from wherever she was when the save returned. If she tapped Next during the save, a story was skipped.
  - Fix: `finished(storyId)` advances only if that story is still the one shown, and goes to the story after it.
  - Evidence: jsdom test "a verdict saved after she moved away does not move her again".
- Finding: Back and Next taps could overlap while a held name answer was being sent. The card could then end on the wrong story, and a failed send could land on a hidden story.
  - Fix: `show()` ignores a tap while a move or a send is in flight.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low
- Concern: changing an answer on a story she went back to (a verdict, or the last part) moves her forward again.
- Mitigation: the Back button is right there. This matches "done → next".
