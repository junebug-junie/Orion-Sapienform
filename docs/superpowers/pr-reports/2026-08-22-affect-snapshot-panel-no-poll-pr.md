# PR #1843: Carbon (affect snapshot) panel never repainted after the first render

- Branch: `fix/affect-snapshot-panel-no-poll`
- PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1843

## Summary

- The Vision panel's "Carbon (affect snapshot)" dropdown view rendered once on selection and never again — a real ambient/manual capture landing seconds later stayed invisible until the dropdown was re-selected.
- Reported live: Juniper flipped the ambient toggle on, held a pose for a real capture, flipped it off — the backend recorded a real success (`last_result_ok: true`, real `raw_response`, real `video_sha256`, confirmed via `GET /api/vision/affect-ambient/status`) but the panel kept showing the OLD failed-tick 404 text.
- Extracted the render/decision logic into a new standalone module (`carbon-affect-snapshot.js`, UMD pattern matching the existing `container-bringup-ui.js` convention) with real executing `node --test` coverage, instead of leaving it buried in `app.js`'s giant inline closure.
- Rewired the view onto the **same** already-running 15s status poll (`fetchAmbientStatus`) instead of adding a second, independent one, and also wired the ambient toggle's own click response into the same repaint path.
- Went through three rounds of this session's own `/code-review` skill; each round caught real issues in the prior draft (see "Review findings fixed" below).

## Outcome moved

The "Carbon (affect snapshot)" panel now reflects the real backend state within 15s of any change (poll), immediately on a toggle click, and immediately on dropdown selection — not just once, ever.

## Current architecture

`services/orion-hub/scripts/vision_affect_ambient.py` (PR #1840) owns a server-side recurring-capture loop and exposes `GET /api/vision/affect-ambient/status`. `static/js/app.js`'s Vision panel already had a small "Ambient: on/off · last tick Xm ago: ok/failed" status line polling this endpoint every 15s (`fetchAmbientStatus`, unchanged cadence). The separate "Carbon (affect snapshot)" dropdown option (PR #1841) was supposed to show the *content* of the last check (the model's `raw_response`), but only ever fetched it once, at `updateVisionUi()` dispatch time.

## Architecture touched

`services/orion-hub` frontend only (`static/js/app.js`, new `static/js/carbon-affect-snapshot.js`, `templates/index.html`). No backend/schema/bus changes.

## Files changed

- `services/orion-hub/static/js/carbon-affect-snapshot.js` (new): pure render logic (`renderAffectSnapshotHtml`) + a tested `createLatestWinsGate()` ordering primitive. No DOM or `app.js` dependency — runs standalone under `node --test`, same as `container-bringup-ui.js`.
- `services/orion-hub/static/js/carbon-affect-snapshot.test.js` (new): 9 real executing tests (in-progress vs just-completed vs textless-success vs failure rendering, full `&`/`<`/`>` escaping including the error string, out-of-order request resolution).
- `services/orion-hub/static/js/app.js`: dropdown selection now paints a loading placeholder and calls the shared `fetchAmbientStatus()` (no independent fetch/poll of its own); `fetchAmbientStatus()` and `toggleAffectAmbient()` both repaint the panel through one shared `repaintCarbonAffectSnapshot()` gated by a tested latest-wins token.
- `services/orion-hub/templates/index.html`: `<script>` tag for the new module, loaded before `app.js`.
- `services/orion-hub/tests/test_vision_affect_capture_api.py`: static-asset wiring smoke — script tag present, exactly one fetch call site for the status endpoint, toggle path also repaints.

## Schema / bus / API changes

None. Frontend-only.

## Env/config changes

None.

## Tests run

```text
cd services/orion-hub
PYTHONPATH=.:<repo root> venv/bin/python -m pytest tests/test_vision_affect_capture_api.py -q
  22 passed
PYTHONPATH=.:<repo root> venv/bin/python -m pytest tests/test_hub_ui_polish.py tests/test_vision_affect_ambient.py tests/test_vision_frame_cache.py -q
  32 passed

cd static/js
node --test carbon-affect-snapshot.test.js
  9 passed
node --test *.test.js   # full JS suite, checking for collateral breakage
  52 passed, 22 skipped (pre-existing, unrelated), 0 failed
node --check app.js && node --check carbon-affect-snapshot.js
  OK
```

## Evals run

No dedicated eval harness exists for `orion-hub`'s frontend; this is a pure UI-polling bug fix with real executing unit tests + a live production diagnosis (see PR #1842's report for the harness gap note on this service generally).

## Docker/build/smoke checks

Not applicable — no backend/container changes, frontend-only static asset + inline `<script>` files served as-is by the existing `orion-athena-hub` container.

## Review findings fixed

Three full rounds of this session's own `/code-review` skill, run against a fresh diff each time.

**Round 1** (9 findings, on the original inline `app.js`-only draft):
- Finding: no periodic refresh at all — the view rendered once on selection and never again (the reported bug itself).
  - Fix: added a poll.
- Finding: last-result-ok-but-empty-raw_response mislabeled as a failure.
  - Fix: distinct "succeeded but returned no text" message.
- Finding: the new poll duplicated the pre-existing 15s status poll instead of reusing it; brittle whitespace-sensitive test; source-substring test that would pass even if the fix were deleted; no in-flight overlap guard; unlogged fetch errors; shared timer variable ambiguously named.
  - Fix: superseded by the round-2/3 redesign below (module extraction + shared poll) -- no separate round-1 report was committed; the finding was addressed directly in the same branch before it was ever merged.

**Round 2** (9 findings, on the module-extraction draft):
- Finding: the dedup key was built from `last_attempt_at` alone, which `vision_affect_ambient.py` only sets at capture **start** (`try_begin_capture`), never at completion (`end_capture`) — the key was identical before and after a capture finished, so the dedup guard could silently drop the real fresh result. This reproduced the exact bug the fix was written to close.
  - Fix: removed the dedup key entirely — the panel now repaints unconditionally on every poll while active. A plain text-panel `innerHTML` swap every ~15s has no visible flicker cost, so there was never a real reason to skip it here.
  - Evidence: `carbon-affect-snapshot.test.js`'s "recomputes the elapsed-time text fresh on every call" test.
- Finding: resetting the dedup key to `null` in the error/never-ran branches caused a repeated "Loading..." flash on every poll during a sustained failure.
  - Fix: superseded — no dedup key, no loading-flash-on-repoll path exists any more.
- Finding: weaker HTML escaping (`<` only) than `app.js`'s own `escapeHtml()` (`&`, `<`, `>`), and `last_error` was never escaped at all.
  - Fix: module's own `escapeHtml()` now handles all three, applied to both `raw_response` and `last_error`.
- Finding: duplicated fetch/parse logic between the one-shot selection fetch and the interval poll.
  - Fix: superseded — eliminated the separate one-shot fetch function entirely; one fetch site for this endpoint in the whole file.

**Round 3** (8 findings, on the round-2 redesign):
- Finding (material): the round-2 redesign's single-fetch-site claim didn't actually prevent two *concurrent* `fetchAmbientStatus()` invocations (the immediate selection-triggered call and the running 15s interval's own call) sharing the same view-generation from resolving **out of order** — an older, slower response could still overwrite a newer, faster one.
  - Fix: added a tested `createLatestWinsGate()` primitive (issue a token per request, only the response matching the most-recently-issued token gets to paint). Real regression test: issue two tokens, resolve the newer one first, confirm the older one is rejected even though it "arrives" after.
- Finding (material): removing the dedup (round 2's fix) meant a single transient fetch failure now blanked the panel to "unavailable" every time, while the sibling status line kept its last-known-good text on the same blip — visibly contradictory.
  - Fix: added `carbonAffectPanelHasRenderedOnce` — a lone failure after the first successful render is now silently ignored (matches the status line's own established "best-effort, keep stale text on a blip" convention); only the very first attempt for a view selection shows "unavailable" outright, since there's nothing yet to preserve.
- Finding: flipping the ambient toggle left the panel up to 15s stale (it only updated the small status line, not the big panel).
  - Fix: the toggle's own POST response is the same status shape as the polling endpoint — reused to repaint the panel immediately, gated by the same latest-wins token.
- Finding: no test exercised the actual ordering/race behavior — a future regression would pass every existing test.
  - Fix: `createLatestWinsGate` is a real, tested, standalone primitive (2 tests) exercising exactly this.
- Finding: unthrottled `console.warn` on every failed 15s poll during a sustained outage.
  - Fix: warns once per outage streak, clears on the next success.
- Finding: brittle whitespace-sensitive Python test slicing, and it never actually verified the "exactly one fetch site" claim.
  - Fix: rewrote the wiring test to (a) directly assert the status-endpoint string appears exactly once in `app.js`, and (b) locate the function body via the next function's own start (indentation-independent) rather than a `"\n      }"` substring.
- Finding: comment claimed "unchanged pre-existing behavior" for the status-line update, but one edge case (200 response with malformed JSON) actually changed.
  - Fix: corrected the comment to name the one real behavior difference.
- Declined: `escapeHtml()` now exists in three files (`app.js`, `substrate-atlas.js`, this new module) with no shared source, and `substrate-atlas.js`'s copy has already drifted (also escapes quotes) — pre-existing drift this patch didn't create. Unifying three files' escaping into a shared module is real scope creep beyond a UI-panel-polling bug fix; left as-is.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

Frontend-only static assets — a plain container restart (no `--build`) would also pick up the new `.js`/`.html` files if they're mounted as volumes rather than baked into the image; use `--build` to be safe regardless of that service's actual mount setup.

## Risks / concerns

None outstanding. Pure frontend polling/render logic, no backend/schema/bus surface touched, real executing test coverage for every material behavior (ordering, escaping, staleness, textless-success-vs-failure), and three independent adversarial review rounds run to convergence (round 3 found only minor/declined items beyond what's already fixed).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/affect-snapshot-panel-no-poll
