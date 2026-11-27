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
- `services/orion-hub/templates/index.html`: the sticky pager, the status line's minimum height, and `id="hubTopBar"` on the header.
- `services/orion-hub/evals/walkway_pager_live_geometry.py`: a live geometry eval on the real styled page.
- `services/orion-hub/tests/test_motion_story_routes.py`: a contract test that the header keeps `id="hubTopBar"` and stays fixed.
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

```text
python services/orion-hub/evals/walkway_pager_live_geometry.py
PASS: 18 of 18 moves left the pager at the top bar and the story under it, unmoved as pictures loaded
```

This is a manual eval, not run in CI, because it needs the live Hub. It loads the real Hub page with real Tailwind and 20 real stories at 390 px mobile, using this checkout's JS and pager markup. Every non-GET request is faked, so nothing is written. It walks Next and Back from the bottom of each story, with scroll anchoring on and off. It also checks that no story grows or moves after its pictures load. If Hub cannot be reached, it fails with exit 2; it never skips silently. It is rerun after the deploy as part of #4016.

## Docker/build/smoke checks

No build. Static files only.

## Review findings fixed

- Finding: during the 600 ms pause the finished story was still live, so a correction or a held name pick could be pulled away by the move.
  - Fix: any tap or typing in the story list during the pause, or a held answer, cancels the advance. She sees "Staying on this story; press Next when you are done."
  - Evidence: jsdom test "a tap on the story during the Saved pause keeps her on it".
- Finding: the tally, gate and status lines above the story change length after every save. iPhone Safari has no scroll anchoring, so the story moved by a line or two.
  - Fix: `steady()` wraps those writes and scrolls by exactly how far the shown story moved.
  - Evidence: the smoke test with `overflow-anchor: none` keeps a part row at the same pixel. Without `steady()` it moved 688 -> 669.
- Finding: a reload during the pause that dropped the finished story left her partway down the next one.
  - Fix: `load()` calls `startUnderPager()` when the shown story changed.
- Finding: the "off the bottom" branch scrolled by about 0 px.
  - Fix: it now brings the pager up under the top bar.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low
- Concern: real iPhone Safari is UNVERIFIED. WebKit needs system packages this host does not have, so Chromium with scroll anchoring off stood in for it.
- Mitigation: Juniper checks on her phone after the deploy (#4016).
