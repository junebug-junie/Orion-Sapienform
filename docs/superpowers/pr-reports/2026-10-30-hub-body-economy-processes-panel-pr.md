## Summary

- The Body Economy tab now shows what each process costs. It draws one bar for each container.
- The bar length is the CPU watts above idle. The longest bar is at the top.
- Each bar shows the process name, the host, the watts, and the cores. Cores read "?" because no producer publishes busy cores for each container yet.
- When there are no processes, the panel shows the reason from the report's `missing` list. This is the same rule that the other panels use.
- When the report shows only some of the processes, the panel shows the note under the bars (for example "top 12 of 55 shown").
- The status line now counts each section with a note one time. It says "missing or partial", because a note does not mean the section is empty.

## Outcome moved

- Before: the API at `GET :8124/physiology/report` returned `processes` (live 2026-10-30: 12 rows, `orion-athena-vector-host` on athena at 50.6 W first). The tab did not draw them.
- After: the tab draws panel 9, "What each process costs". The browser smoke shows the four fixture bars in the correct order, and the producer's "partial: ..." note.

## Current architecture

The Hub serves `static/body-economy.html` in an iframe on the `#body-economy` tab. `static/js/body-economy.js` fetches `/api/body-economy/report`. That route is a proxy of durable-runs `/physiology/report` (schema `physiology.report.v1`, `orion/schemas/physiology_report.py`). Each section has one render function. An empty section shows a reason: the fetch failure, then the report's `missing` reason, then a plain default. The producer of `processes` is `shape_processes` in `services/orion-durable-runs/app/physiology_report.py`.

## Architecture touched

Hub static UI only. No service code, contract, or config changed.

## Files changed

- `services/orion-hub/static/body-economy.html`: adds the panel 9 card (`#beProcesses`). Renumbers the later panels 10 to 12.
- `services/orion-hub/static/js/body-economy.js`: adds `processesHtml`, `missingReasons`, and `renderProcesses`. Adds a default empty text for `processes`.
- `services/orion-hub/static/js/body-economy.test.js`: two new node tests for the bars, the order, the labels, the notes, and the empty states.
- `services/orion-hub/tests/fixtures/body_economy_report.json`: adds four `processes` rows (unsorted, on purpose) and one real-shaped `missing` note.
- `services/orion-hub/tests/test_body_economy_browser_smoke.py`: asserts the bars and the note in Chromium. Asserts the empty state reason.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: the UI now reads `processes`, a field that the API already returned.
- Compatibility notes: none.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
node --test services/orion-hub/static/js/body-economy.test.js -> 15 passed (after review fixes)
pytest services/orion-hub/tests/test_body_economy_routes.py tests/test_body_economy_browser_smoke.py -> 17 passed, 3 skipped (no playwright in the main venv)
pytest services/orion-hub/tests/test_body_economy_browser_smoke.py (with a playwright + Chromium interpreter) -> 3 passed
pytest (same interpreter) test_body_economy_browser_smoke.py test_body_economy_routes.py -> 20 passed (after review fixes)
```

## Evals run

No eval. This is a display change. The browser smoke is the check of the changed interaction.

## Docker/build/smoke checks

- The template loads the page: `templates/index.html` has `src="/static/body-economy.html?v={{HUB_UI_ASSET_VERSION}}"`. The page loads `/static/js/body-economy.js` with the same `?v=`.
- The cache-bust token comes from `build_hub_ui_asset_version()`. It includes the git sha and the file mtimes, so there is nothing to bump by hand.
- No Docker build was run. Live tab after deploy: UNVERIFIED.

## Review findings fixed

A review subagent read the branch diff against origin/main. It found no must-fix items. It confirmed that all producer strings are escaped and that the fixture is valid against `ProcessV1`.

- Finding: the status line counted every `missing` entry as a missing section. Notes like "top 12 of N shown" are on sections that are drawn.
  - Fix: count distinct sections. Say "missing or partial".
  - Evidence: the smoke asserts "1 section(s) missing or partial" on the full fixture.
- Finding: the empty state showed only the first `processes` reason.
  - Fix: join all the reasons with "; ".
  - Evidence: a new node test asserts "a; b".
- Finding (nit): the sort was not stable for the same name and watts on two hosts.
  - Fix: sort by watts, then name, then host.
  - Evidence: a new node test asserts the order "a:x", "b:x".
- Finding (nit): the fixture note "top 4 of 9 shown" cannot come from the producer, because it always shows the top 12.
  - Fix: use a real "partial: ..." reason.
  - Evidence: node and smoke tests assert it.
- Finding (nit): `cores: 3.5` is never sent today.
  - Fix: a test comment says it tests the future path.
- Not changed: negative watts show as a negative number with an empty bar. The producer does not send them today.

## Restart required

Rebuild the Hub from the clean primary `main` checkout after merge:

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low. Concern: cores always read "?" today. Mitigation: the tooltip says that busy cores are not published for each container. The label changes by itself when the producer fills `cores`.
- Severity: low. Concern: container names can be short ids (for example `d590a9d48af2`). Mitigation: none in this patch. The name comes from the producer as it is.

## PR link

PR_LINK_PLACEHOLDER
