## Summary

- Juniper: "when i click a button, it randomly moves me around the screen."
- Cause, measured in Chromium at 390 px and 1280 px: the card only moved the page when it changed story. After a story finished, the old `pager.scrollIntoView({block: "nearest"})` either did nothing, which left her partway down a different story, or put the Back/Next bar at the very top, under Hub's fixed `h-14` header. Clicks inside a story moved things by 5–20 px at most.
- Fix:
  - The Back/Next bar is pinned (`position: sticky; top: 3.5rem`), so it stays on screen while a tall story scrolls.
  - A new story always starts right under the bar. The page scrolls only when the new story's top would be hidden.
  - "Saved" shows for 600 ms before the card moves on.
  - Refocusing a control never scrolls the page.
  - The status line keeps a fixed minimum height.

## Outcome moved

Moving between stories no longer shifts the page. Measured: Next and Back leave `scrollY` and the positions of the bar and the story unchanged, on both widths, with and without scroll anchoring. Turning scroll anchoring off approximates iPhone Safari, which does not have it.

## Current architecture

PR #4014 shows one story at a time. Its `step()` called `pager.scrollIntoView({block: "nearest"})` after every move.

## Architecture touched

Hub frontend only: `walkway-stories.js` and `index.html`. No API, schema, bus or env changes.

## Files changed

- `services/orion-hub/static/js/walkway-stories.js`: `startUnderPager()`, `ADVANCE_PAUSE_MS`, and `preventScroll` on the rerender focus.
- `services/orion-hub/templates/index.html`: the sticky pager and the status line's minimum height.
- `services/orion-hub/tests/test_walkway_stories_browser_smoke.py`: new test `test_walkway_story_moves_never_jump_the_page_browser_smoke`. It fails on the old code: the bar sat at -374 px instead of 56.
- `services/orion-hub/static/js/walkway-stories.test.js`: the auto-advance test now waits through the pause and checks for "Next story".

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
node --test walkway-stories.test.js (with jsdom)   28 pass
pytest test_walkway_stories_browser_smoke.py        8 passed (new no-jump test included)
no-jump test against the origin/main JS + template  FAILED as expected (pager -374, want 56)
```

## Evals run

None. This is a UI-only change, and the browser smoke covers the interaction.

## Docker/build/smoke checks

No build. Static files only.

## Review findings fixed

See the PR comment once the review finishes.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low
- Concern: real iPhone Safari is UNVERIFIED. WebKit needs system packages this host does not have, so Chromium with scroll anchoring off stood in for it.
- Mitigation: Juniper checks on her phone after the deploy (#4016).
