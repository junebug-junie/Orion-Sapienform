# feat(hub): Orion's tiredness gauge in the Biometrics preview card

## Summary

- A small meter in the Hub's Biometrics preview card shows how tired Orion is: a fill against a "sleep line" in the middle of the track, with an "off the scale" marker past twice the line.
- One line under it says why Orion isn't asleep yet (slept recently, waiting for quiet, not tired enough, or overdue) and names the new material in plain words.
- A "?" tooltip explains what tiredness means. It opens on hover or keyboard focus, click pins it, and Escape closes it.
- orion-dream's pressure reply now says whether the 48-hour backstop is due (`overdue`, `lookback_hours`), so the gauge never says "not tired enough" right before a backstop sleep.

## Outcome moved

Tiredness (dream sleep pressure, made real in #2557) was only visible in the Dream tab's raw numbers. It is now a glanceable gauge where Juniper already watches Orion's body signals, with the reason Orion isn't asleep yet.

## Current architecture

- `/api/dream/pressure` (Hub, `dream_routes.py`) proxies orion-dream's `/dreams/cycle/pressure`.
- The Biometrics preview card (`#biometricsPreviewContainer`) refreshes host tiles every 10 s through `loadCardPreview` in `biometrics-view.js`.

## Architecture touched

- Hub frontend: the template, `biometrics-view.js`, a node test, and a browser eval.
- orion-dream: two additive fields on the HTTP pressure reply. No bus, schema, or env change.

## Files changed

- `services/orion-hub/templates/index.html`: `#orionTiredness` slot in the preview card.
- `services/orion-hub/static/js/biometrics-view.js`:
  - `tirednessModel` (pure; exported), a build-once gauge with stateful tooltip, `renderTiredness`, `loadTiredness`.
  - `loadTiredness` refreshes every 60 s, with one fetch in flight at a time.
- `services/orion-hub/static/js/biometrics-view.test.js`: 7 model tests (16 total).
- `services/orion-hub/evals/tiredness_gauge_browser.cjs`: 17 browser checks on the real template, with fixture APIs.
- `services/orion-dream/app/main.py`: `/dreams/cycle/pressure` adds `overdue` and `lookback_hours`.
- `services/orion-dream/tests/test_dream_cycle_v2.py`: endpoint test for `overdue`.

## Schema / bus / API changes

- Added: `overdue` (bool) and `lookback_hours` (number) on orion-dream `GET /dreams/cycle/pressure`. Hub passes them through (`{**data}`).
- Removed / Renamed: none
- Behavior changed: none for existing fields.
- Compatibility notes: additive JSON keys on an HTTP reply. The Hub validates only `pressure` (`SleepPressureV1`) and four named booleans, so an old Hub ignores them. A new Hub against an old dream service treats a missing `overdue` as false.

## Env/config changes

- None.

## Tests run

```text
node --test services/orion-hub/static/js/biometrics-view.test.js        -> 16 pass, 0 fail
all services/orion-hub/static/js/*.test.js                               -> 0 failures
pytest services/orion-dream/tests services/orion-dream/evals -q          -> 129 passed
pytest services/orion-hub/tests/test_dream_routes.py tests/test_dream_hypotheses.py -q -> 27 passed
```

## Evals run

```text
node services/orion-hub/evals/tiredness_gauge_browser.cjs (dream_server.py, Chrome 131, fixture APIs)
{"passed":true,"checks":["renders in preview card","reading and level","off-scale overflow","waiting reason",
 "plain-word new material","meter aria + geometry","tooltip hover","tooltip keyboard focus","Escape closes",
 "pinned tooltip survives 10 s refresh","focus survives refresh","tooltip text click does not open modal",
 "second click closes","help click does not open modal","rested","unavailable is not rested"]}
Screenshots reviewed: the gauge (live-shaped 11.1 / 3.0, off the scale) and the open tooltip.
```

## Docker/build/smoke checks

```text
Live endpoint shape before this change (deployed #2557): pressure 11.06 / 3.0, new_counts {metacog: 14-15, crystallization: 1}.
Post-deploy proof: UNVERIFIED until deployed. Expect the gauge in the Biometrics preview card and
`overdue` in curl localhost:8620/dreams/cycle/pressure.
```

## Review findings fixed

- Finding (should): the 10 s card refresh rebuilt the gauge, closing an open tooltip and dropping keyboard focus.
  - Fix: build the skeleton once; refreshes update text and geometry only.
  - Evidence: browser checks "pinned tooltip survives 10 s refresh" and "focus survives refresh" (waits 11 s).
- Finding (should): below the line but overdue, the gauge said "Not tired enough" right before a backstop sleep.
  - Fix: orion-dream reports `overdue`; the model names it ("Overdue (48 h since last sleep), waiting for N more min with no chat").
  - Evidence: model test (incl. no-candidates and 6 h-first cases); dream endpoint test.
- Finding (should): the tooltip's 48 h claim was broader than the code.
  - Fix: copy now says it counts from the last sleep's start and still waits for a quiet stretch.
- Finding (minor): a second click didn't close the tooltip (focus kept it open); there was no Escape.
  - Fix: explicit open state (hover/focus/pinned); click toggles the pin; Escape closes. A click inside the tooltip no longer blurs the button. The browser eval caught that this re-pinned it.
  - Evidence: checks "second click closes" and "Escape closes".
- Finding (minor): clicking the tooltip text opened the Biometrics modal.
  - Fix: stopPropagation on the gauge header and tooltip.
  - Evidence: check "tooltip text click does not open modal".
- Finding (minor): the card's native "Click to open" tip conflicted with the help tip.
  - Fix: empty `title` on the button and tooltip.
- Finding (minor, unverified clipping): the tooltip could clip in `overflow-hidden` on a narrow column.
  - Fix: anchored to the gauge's left edge, `max-w-full`.
  - Evidence: screenshot.
- Finding (minor): the idle wait ignored `idle_minutes`, and unknown idle read as "waiting for 45 min".
  - Fix: shows minutes left, or "can't tell how long since the last chat".
- Finding (minor): a sleep line of 0 read as unavailable.
  - Fix: reads always ready.
  - Evidence: model test.
- Finding (minor): duplicate first fetch.
  - Fix: one in-flight promise.

## Restart required

From the primary checkout on main after merge:

```bash
git pull --ff-only && scripts/safe_docker_build.sh orion-dream up -d --build && scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low
  - Concern: the gauge refreshes every 60 s, so a sleep can take up to a minute to show.
  - Mitigation: tiredness moves on a scale of hours. Each read runs the dream service's source queries twice (~0.2 s warm), so 10 s polling isn't worth it.
- Severity: low
  - Concern: the "6 h minimum" text is hardcoded while `DREAM_MIN_INTERVAL_HOURS` is configurable.
  - Mitigation: it matches live config (6). Follow-up if it ever changes: add it to the pressure reply like `lookback_hours`.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
