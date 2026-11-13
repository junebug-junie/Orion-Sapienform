## Summary

**Merge order: this PR depends on the durable-runs `GET /physiology/report` PR (branch `feat/physiology-report-api-and-skill`). Merge that one first.** Until it is deployed, the tab shows "durable-runs ... Not Found" in every section. It never shows blank charts.

- The Hub has a new tab, **Body Economy** (`#body-economy`). It is the Phase 3 "Hub body surface" from the body economy spec. It follows the mockup Juniper approved on 2026-10-29.
- The top of the tab shows how Orion's body is doing:
  - "right now" tiles, and a tile for services that should run but are stopped;
  - every limit, hour by hour: amber means binding, and hatching means "over the line, not Orion's doing";
  - wants by family, with deferred wants shown as a dashed ghost bar;
  - the heavy-work rhythm, money this month, wasted GPU time, queue wait per lane, GPU memory headroom, Claude spend, unit costs, value per dollar, and Orion's own `why` lines, word for word.
- Below that is a list of single decisions. You can filter it by outcome. You can search it by family ("why did Orion wait?"). Click a row to see its life, the prices it saw and its reason, word for word.
- Drag across the limits chart or the wants chart to pick a time window. The decision list then shows only that window (spec H2).
- The Hub reads no physiology table. The new route only forwards the request to durable-runs `/physiology/report`.

## Outcome moved

Before this PR, the body economy (shadow prices, wants, deferrals) could only be read with SQL. Now Juniper can open one tab and see three things: which limit is binding, what Orion wanted and did not get, and why each single decision went the way it did.

## Current architecture

- The physiology organs (prices, wants, decisions) write to Postgres. The Hub had no view of them.
- The Hub already reads from durable-runs through thin proxy routes (`temporal_self_routes.py`), with base URL `HUB_CURIOSITY_DURABLE_RUNS_URL`.
- Static iframe tabs (Orion Energy) are wired the same way in `templates/index.html` and `static/js/app.js`.

## Architecture touched

- orion-hub only: one new read route, one new static page with its JS, and tab wiring.
- New upstream dependency: durable-runs `GET /physiology/report` (`physiology.report.v1`). It is built in the sibling PR.

## Files changed

- `services/orion-hub/scripts/body_economy_routes.py`: `GET /api/body-economy/report`, a proxy to `/physiology/report`. It checks the params. If durable-runs cannot be reached, the reply is a 502 that says since when.
- `services/orion-hub/scripts/main.py`: mounts the router.
- `services/orion-hub/static/body-economy.html`: the tab page, in the Hub dark theme.
- `services/orion-hub/static/js/body-economy.js`: draws every chart, the tooltips, the brushing and the drill-down.
- `services/orion-hub/static/js/body-economy.test.js`: node tests. They cover brushing (H2), folding, hatching, wants buckets, empty states and formatting.
- `services/orion-hub/templates/index.html`, `services/orion-hub/static/js/app.js`: the tab button, the section and the hash routing. These follow the Orion Energy tab.
- `services/orion-hub/tests/fixtures/body_economy_report.json`: a fixture in the exact report shape, including `claude`, `vram`, `value_per_cost` and `services`. The node tests, the route tests and the browser smoke all use it.
- `services/orion-hub/tests/test_body_economy_routes.py`: tests for the proxy and the static wiring.
- `services/orion-hub/tests/test_body_economy_browser_smoke.py`: a Playwright smoke (H3) with 3 cases.
- `.github/workflows/schedule-browser-smoke.yml`: runs that smoke in CI. If Playwright is missing, the step fails instead of skipping.
- `services/orion-hub/README.md`: a section for the new tab.

## Schema / bus / API changes

- Added: Hub HTTP `GET /api/body-economy/report?window=24h|7d|30d&sections=&family=&outcome=&limit=`.
- Removed / Renamed: none.
- Behavior changed: none.
- Compatibility notes:
  - This PR adds no bus or schema change. It reads `physiology.report.v1` from the sibling PR.
  - The page accepts a body only when `schema_version` is `physiology.report.v1`. If that field is missing, the body must have `window` and `generated_at`.

## Env/config changes

- Added keys: none. The route reuses `HUB_CURIOSITY_DURABLE_RUNS_URL`.
- Removed keys / renamed keys: none.
- `.env_example` updated: no.
- Local `.env` synced: not needed, because no template changed.
- Skipped keys: none.

## Tests run

```text
node --test --test-reporter=tap services/orion-hub/static/js/body-economy.test.js   -> pass 13, fail 0
node --test $(find services/orion-hub/static/js -name '*.test.js')                  -> pass 392, fail 0
pytest services/orion-hub/tests/test_body_economy_routes.py                          -> 17 passed
pytest services/orion-hub/tests/test_body_economy_browser_smoke.py (Chromium)        -> 3 passed
pytest services/orion-hub/tests (full hub suite, local)                              -> 4433 passed, 146 skipped, 3 xfailed
CI: all 17 checks green, incl. "Hub — full tests folder" and hub-schedule-browser-smoke
```

Mutation check: I removed the family-type guard. The malformed-row smoke then failed with "could not draw ... reading 'length'". I put the guard back.

## Evals run

```text
None. This is a read-only display tab, so there is no quality behavior to score. The browser smoke is its behavior check.
```

## Docker/build/smoke checks

```text
Not built. The change is static assets plus one read route, and it uses no new dependency.
Live path: UNVERIFIED until the /physiology/report PR is deployed.
```

## Review findings fixed

The code review ran in a subagent. It found no XSS, and the app.js wiring matched Orion Energy.

- Finding: a null or non-string `family`, a numeric `day`/`d`, or a string `displaced` threw inside render. The page then stayed on "loading" with the wrong message.
  - Fix: type guards on those values. A render error now shows a named message in the status line.
  - Evidence: a new malformed-row browser smoke, plus the mutation check above.
- Finding: a 200 reply that was not a report (an HTML page, `{}`) was drawn as believable empty data.
  - Fix: `isReport()`. The page now says "the Hub answered, but not with a body report".
  - Evidence: a node test, and a browser smoke with an HTML reply that has status 200.
- Finding: a network error said durable-runs was down. In that case, it is the Hub that did not answer.
  - Fix: the network error now has its own message.
  - Evidence: a node test.
- Finding: during a window switch in flight, a resize drew the old report with the new bucket size, and the old brush stayed on screen.
  - Fix: the window changes only when its own report arrives. The brush is cleared at once.
  - Evidence: the smoke checks the 7 d switch.
- Finding: when the window was not a whole number of buckets, the wants bars and the brush covered different time.
  - Fix: each bar's width now follows the time it really covers.
  - Evidence: not tested directly, because the fixture is aligned. Code review only.
- Finding: a fractional heavy-act count left a heatmap cell with no colour. "What Orion sees" showed the limits' empty message.
  - Fix: the colour step is rounded and clamped. The section has its own empty text.
  - Evidence: the malformed-row smoke uses `heavy: 2.6`.
- Finding (nit): the decision count did not say when the 500-row cap was hit.
  - Fix: the count now adds "(the newest 500 only)".
  - Evidence: not tested; the fixture has 10 rows.

## Restart required

Deploy only after the `/physiology/report` PR is merged and deployed. Run this from the clean primary `main` checkout:

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: medium
  - Concern: the fixture is my reading of the agreed contract. The sibling PR may name a field differently, for example the fields inside `waste.top[]`, the `unit_costs.status` spelling, or the timezone of `rhythm` (I read it as local, as the contract says).
  - Mitigation: unknown values show as "unknown" or "not measured", never as 0. After both PRs land, compare one real `/physiology/report` body with the fixture.
- Severity: low
  - Concern: spec H4 (the stance-cue clause, word for word) is not met. That clause is not wired into Orion's turns yet (Patch F). The page says so. H5 (load time on 30 d) is the upstream's job, and it is not measured here.
  - Mitigation: follow-up after Patch F.
- Severity: low
  - Concern: spec H3 asks for "the displaced want". While bids run in shadow, `displaced` is always empty. The page shows "Displaced: none. Nothing is displaced while bids run in shadow."

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2970

🤖 Generated with [Claude Code](https://claude.com/claude-code)
