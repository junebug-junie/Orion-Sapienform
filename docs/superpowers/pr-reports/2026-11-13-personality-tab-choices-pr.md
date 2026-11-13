## Summary

- The Hub Personality tab now shows Orion's **choices**, not Orion's prose. Juniper called the old tab "slop": it listed sentences repeated by automated jobs.
- **Recent choices**: the last 30 choices across lanes. Each shows the lane in words, the time, what Orion picked (with its market rank), the top options Orion passed over (with rank), and "Said nothing" / "Made no move" declines.
- **Tendencies**: choice-built tendencies only, each with "n of m choices went this way", strength, status in words, an "In Orion's Prime block now" badge, and 3 example choices. Each has **Does this look like Orion? Yes / No**.
- A **No** retires the tendency in durable-runs (the single writer). A retired tendency leaves the frame, so it never reaches the Prime tending line or the market again.
- **Changed my mind**: a tendency Juniper said No to sits in a collapsed "You said no to" list. Its button sends a Yes, which revives it as a candidate (same revive as the reducer in #3336) with a weight-history row. A tendency that faded out on its own still gets 409.
- **Is it changing anything?**: market decisions with the care arm vs the holdback arm (from `want_ledger` `personality:arm=` lines). It reads "Not live yet" until #3332 deploys.
- The old excerpt / recurrence panels and their Hub proxies (`/dispositions`, `/evidence`, `/naming`, `/cue`) are removed.

## Outcome moved

Juniper can now see what Orion actually chose and turned down, and can say yes or no to each tendency. Her no has a runtime effect: the tendency stops reaching Prime and the market. This closes spec rev 4 move 4 acceptance M1b ("Juniper reads their labels and records yes or no").

## Current architecture

- The tab was a rev 2 panel: cue text, recurrence nodes with prose excerpts, evidence excerpts, a naming log. Rev 4 (#3321) already made durable-runs read only `origin='choice'` nodes, but the tab still showed the rev 2 view.
- There was no way for Juniper to record an answer on a tendency. The Prime tending line docstring (`orion/prime/tending.py`) already promised "her no in the Hub tab retires a tendency".

## Architecture touched

- **orion-durable-runs**: 3 read routes, 1 write route, store methods, a driver method that holds the reducer tick lock while it writes.
- **orion-sql-db**: one additive table, `personality_juniper_verdict`.
- **orion-hub**: proxy routes, page template, page JS, tab blurb in `index.html`.
- No bus, schema-registry or env changes.

## Files changed

- `services/orion-sql-db/manual_migration_personality_juniper_verdict_v1.sql` (+ `_rollback.sql`): the answer table (insert-only, one row per answer). REQUIRED-BY orion-durable-runs. Passes the auto-apply allowlist (`auto_apply_refusals` is empty with live tables).
- `services/orion-durable-runs/app/personality_store.py`: `recent_choices`, `tendencies`, `record_juniper_verdict` (answer row + retire + weight-history row in one transaction, `FOR UPDATE`, origin `choice` only), `arm_counts`, `choice_view`.
- `services/orion-durable-runs/app/personality_driver.py`: `juniper_verdict` runs under the tick lock. Without it, a tick that loaded the node before the answer would write the old status back.
- `services/orion-durable-runs/app/main.py`: `GET /personality/choices`, `GET /personality/tendencies`, `POST /personality/tendencies/{id}/verdict` (404 unknown, 409 already retired, 422 bad body), `GET /personality/arms`.
- `services/orion-hub/scripts/personality_routes.py`: proxies for the four routes; old prose proxies removed; 409 passes through.
- `services/orion-hub/templates/personality.html`, `static/js/personality.js`: the new tab. Text only via `textContent`; no `innerHTML`.
- `services/orion-hub/templates/index.html`: tab blurb.
- Tests: `services/orion-durable-runs/tests/test_personality_tab_postgres.py`, `test_personality_tab_routes.py`, `services/orion-hub/static/js/personality.test.js`, `services/orion-hub/tests/test_personality_routes.py`, `services/orion-hub/tests/test_personality_panel_browser_smoke.py`.
- CI: `.github/workflows/schedule-browser-smoke.yml` runs the new browser smoke (fails if skipped); `orion-durable-runs-tests.yml` triggers on the new migration.
- Docs: `services/orion-hub/README.md`, `services/orion-durable-runs/README.md`.

## Schema / bus / API changes

- Added: table `personality_juniper_verdict`; durable-runs `GET /personality/choices`, `GET /personality/tendencies`, `POST /personality/tendencies/{id}/verdict`, `GET /personality/arms`; Hub `/api/personality/choices|tendencies|arms`, `POST /api/personality/tendencies/{id}/verdict`.
- Removed: Hub `/api/personality/dispositions`, `/evidence`, `/naming`, `/cue` (only the old panel used them). durable-runs keeps its own `/personality/dispositions|evidence|naming` routes.
- Behavior changed: a "no" sets `personality_disposition.status = 'retired'` for that choice node.
- Compatibility: additive. Without the migration, the verdict route answers 503 naming the file, and the tendencies list shows a warning and no answers.

## Env/config changes

- Added keys: none. Removed: none. Renamed: none.
- `.env_example` updated: no. Local `.env` sync: not needed.
- Skipped keys: none.

## Tests run

```text
services/orion-durable-runs: test_personality_tab_postgres.py, test_personality_tab_routes.py,
  test_personality_loop.py, test_personality_naming.py, test_personality_postgres.py,
  test_personality_choice_replay.py  -> 65 passed (throwaway postgres:16 container as ORION_ADMISSION_TEST_DSN)
services/orion-durable-runs full tests folder (same DSN) -> 1041 passed, 3 skipped
services/orion-hub: tests/test_personality_routes.py, test_hub_panel_reachability.py,
  test_personality_cue_reader.py -> 42 passed
node --test services/orion-hub/static/js/personality.test.js -> 14 passed
tests/test_sql_migration_drift_gate.py, tests/test_deploy_migration_gate.py -> 142 passed
tests/test_no_personal_terms.py, tests/test_no_private_eval_artifacts.py -> 10 passed
scripts/check_metric_lineage.py --gate -> PASS; scripts/check_definition_drift.py --gate -> PASS
```

## Evals run

```text
services/orion-hub/tests/test_personality_panel_browser_smoke.py (Playwright, Chromium) -> 4 passed
  Real template + JS. Covers: choices render (pick, rank, passed over, decline, hostile text stays text),
  tendencies with examples, Yes (saved, card stays), No (confirm dialog, POST, card retired, buttons gone),
  a refused save (named error, buttons back), "Not live yet", and an upstream outage.
Live read-only check (conjourney): the 30 newest choices all join to a choice-built tendency, and
  the example join returns 3 choices per tendency for the top 2 tendencies.
```

No new eval harness: the tab is observability plus one write; the browser smoke is the interaction check.

## Docker/build/smoke checks

```text
No Docker build run. Not deployed (per task).
```

## Review findings fixed

Follow-up from the coordinator: Juniper could never undo her own No (any retired tendency got 409). Fixed: a Yes on a tendency she retired revives it; the tab keeps those under "You said no to" with "Changed my mind". Tests: `test_no_retires_the_tendency_and_the_next_tick_keeps_it_retired` (now also revives), `test_a_faded_tendency_cannot_be_revived_by_yes`, browser smoke `changed_mind` scenario.

Review subagent (hand review of `origin/main...HEAD`): no blockers; tick race, retired-id return, prose-node isolation, SQL, migration allowlist and XSS checked sound.

- Finding: the No dialog said "it cannot come back", but the reducer can found a NEW tendency from the same kind of choice.
  - Fix: honest wording ("If Orion keeps choosing this way, a new tendency can form and you will be asked again."), in the dialog and README. A "remember my no" match would edit `orion/personality/graph.py`, which another agent owns right now.
  - Evidence: node test `the No confirm is honest`; browser smoke checks the dialog text.
- Finding: the "In Orion's Prime block now" badge ignored Prime's 2-line cap and Orion's own rejections.
  - Fix: `orion_verdict` returned by `/personality/tendencies`; badge only on the first 2 eligible (active, weight >= 10, not rejected); node test pins `MIN_WEIGHT` and `MAX_LINES` to `orion/prime/tending.py`.
  - Evidence: node test `tendencyViews marks only the first PRIME_MAX_LINES`.
- Finding: a choice still said "Counts toward: X" after X was retired.
  - Fix: "Counted toward: X (retired)".
  - Evidence: node test `a choice whose tendency was retired says so`.
- Finding: the arms readout depends on #3332, and "funded" in shadow rows.
  - Fix: merge order stated below; wording is "given the go-ahead". `want_ledger.shadow` is true on every row by its own contract (`manual_migration_want_ledger_v1.sql`), so splitting by it would show nothing.
  - Evidence: README + this report.
- Finding: no test that the verdict waits for the tick lock.
  - Fix: `test_driver_verdict_waits_for_the_tick_lock`.
  - Evidence: passes; fails if `juniper_verdict` skips the lock.
- Finding: raw migration file name on screen; jargon ("market's #3").
  - Fix: one plain sentence ("Your answers can't be saved until a database update is applied."), file name only in a tooltip; "ranked #3"; the market glossed once on the page.
  - Evidence: node test `answersBlockedText`; browser smoke `no_table` scenario.
- Finding: Yes/No clickable while the answer table is missing.
  - Fix: buttons disabled when that warning is set.
  - Evidence: browser smoke `no_table` scenario.

## Restart required

Merge order: independent of #3332. Before #3332 deploys, "Is it changing anything" reads "Not live yet".

Deploy order: apply the migration, then durable-runs, then Hub. The post-pull rebuild applies the migration on its own (additive, allowlisted). By hand:

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < services/orion-sql-db/manual_migration_personality_juniper_verdict_v1.sql
```

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low. Concern: on main alone, a "no" retires the tendency but a similar run of choices can found a NEW one. #3336 (feat/personality-loop-wiring) makes Juniper's No absorb those choices instead. Mitigation: merge #3336; until then she can say no again.
- Severity: low. Concern: "Is it changing anything" reads "Not live yet" until #3332 deploys; its `personality:arm=care|holdback` format was checked against #3332's tests. Mitigation: it is labelled, not hidden.
- Severity: low. Concern: UNVERIFIED live: the tab, the Yes/No write and the retire were not exercised on the live stack (not deployed).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3335

🤖 Generated with [Claude Code](https://claude.com/claude-code)
