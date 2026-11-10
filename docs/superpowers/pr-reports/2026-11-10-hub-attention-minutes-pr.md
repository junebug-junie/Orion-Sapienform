## Summary

- The Hub now records, minute by minute, when Juniper is actually looking at it. Juniper asked for this on 2026-11-10: "record somehow that my hub is active and im looking at it".
- A minute counts only when three things are true in the browser: the page is visible, the window has focus, and there was real input (pointer, key, wheel, scroll, touch) in the last 120 s.
- A new small script, `static/js/hub-attention-tracker.js`, runs in the top-level Hub page. While all three are true, it sends one heartbeat per minute. It sends nothing otherwise.
- A new route, `POST /api/hub-attention/heartbeat`, adds each heartbeat to one row per (server minute, client) in a new table, `hub_attention_minute`.
- Flag `HUB_ATTENTION_TRACKER_ENABLED`, on by default. No reader and no bus event in this PR.

## Outcome moved

Before: nothing recorded whether Juniper was present at the Hub. `substrate_hub_presence` records chat turns only, so reading a tab, scrolling a chart, or watching a panel left no trace.
After: each attended minute becomes one row with the client type (`desktop` or `mobile`), the tab (hash route), the heartbeat count, and the first and last times. The prediction service's reachable forecast (separate branch `docs/juniper-reachable-forecast`) will read this table later, after about 2 weeks of live rows pass the metric gate.

## Current architecture

- The Hub shell (`templates/index.html`) loads about 70 deferred scripts, with `app.js` last. Several tabs are same-origin iframes (`/reading`, `/gpu-pool`, `/temporal-self`, `/personality`, ...).
- `scripts/hub_presence.py` mirrors chat-turn liveness to `substrate_hub_presence`. It does not see non-chat activity.
- Route modules use `APIRouter(prefix=...)` and are mounted from `scripts/api_routes.py`.

## Architecture touched

- orion-hub: one new browser script, one new route module, a settings flag, a compose env entry, a README note.
- orion-sql-db: one additive migration (new table + one index).
- `scripts/sync_local_env_from_example.py`: added the `HUB_ATTENTION_TRACKER_` sync prefix. Without it the sync script did not add the new key to the live `.env` (no generic `HUB_` prefix is synced).

## Files changed

- `services/orion-hub/static/js/hub-attention-tracker.js`: the browser tracker. It attaches capture-phase input listeners to the top document and to every same-origin iframe document (on iframe load, plus a 15 s rescan, up to 3 levels deep). It sends a heartbeat at second :30 of each local minute when attended. It stops for the page load when the server says the flag is off. Every path is wrapped, so it never throws into the page.
- `services/orion-hub/static/js/hub-attention-tracker.test.js`: 9 `node:test` cases (CI runs every `*.test.js` in `orion-static-gates.yml`).
- `services/orion-hub/templates/index.html`: loads the tracker once, before `app.js`.
- `services/orion-hub/scripts/hub_attention_routes.py`: the heartbeat route and the upsert.
- `services/orion-hub/scripts/api_routes.py`: mounts the router.
- `services/orion-hub/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: the flag and its documentation.
- `services/orion-hub/tests/test_hub_attention_routes.py`: route, wiring, and node-runner tests.
- `services/orion-sql-db/manual_migration_hub_attention_minute_v1.sql`: the table.
- `scripts/sync_local_env_from_example.py`: the sync prefix.

## Schema / bus / API changes

- Added: HTTP `POST /api/hub-attention/heartbeat`. Body `{client: "desktop"|"mobile", tab?: string}`. Unknown fields are ignored. A bad `client` or a `tab` longer than 512 characters returns 422. The response is `{ok, enabled, minute?, client?, tab?, reason?}`, always HTTP 200 for a valid body.
- Added: table `hub_attention_minute (minute timestamptz, client text, tab text, beats int, first_at timestamptz, last_at timestamptz, PRIMARY KEY (minute, client))`, plus an index on `last_at DESC`. `minute` is `date_trunc('minute', now())` on the Postgres server. The client clock is never used. The tab is the latest non-null tab in that minute.
- Removed / renamed: none.
- Bus: none. No channel or schema-registry change.
- Compatibility: if the migration is not applied, the route logs one warning per 10 min and returns `{ok: false, reason: "db_error"}`. The Hub creates nothing at boot, so a missing table cannot crash the boot.

## Env/config changes

- Added keys: `HUB_ATTENTION_TRACKER_ENABLED=true` (orion-hub).
- `.env_example` updated: yes.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes. The script output was `orion-hub: +HUB_ATTENTION_TRACKER_ENABLED='true'`, and `/mnt/scripts/Orion-Sapienform/services/orion-hub/.env` now has this key. This needed the new `HUB_ATTENTION_TRACKER_` prefix. Without it the first run added nothing.
- Skipped keys requiring operator action: none.
- Migration: `-- ORION-MIGRATION-REQUIRED-BY: orion-hub`. It is additive only. `apply_post_pull_migrations.py --dry-run` reports "would apply (single transaction, lock_timeout=10s)", so the post-pull rebuild applies it before the Hub rebuild.

## Metric quality gate

1. **Provenance.** The values come from real browser events on Juniper's own screen, in `hub-attention-tracker.js`: `visibilityState`, `hasFocus()`, and the capture listeners for `pointermove/pointerdown/keydown/wheel/scroll/touchstart`. The row is written by `hub_attention_routes.write_heartbeat`. The minute comes from the database clock.
2. **Independence.** The Claude typing log (dev-economics / Claude Code session activity) records Juniper typing to Claude in a terminal or IDE. That is a different window and a different input surface. This sensor fires only while the Hub window has focus, so the two are complementary, not overlapping. The overlap is that both need Juniper at a keyboard. `substrate_hub_presence` counts chat turns. That is a subset of Hub activity, and this sensor also covers it. The chat-turn signal is therefore partly redundant with this one, not the reverse. Do not combine the two as independent evidence of presence in one model.
3. **Theory.** Attention to a screen requires three things: the content is on screen (visible), it is the active input target (focus), and the person is interacting with it (recent input). The 120 s window is the usual idle threshold that analytics "engaged time" measures use. A known miss is passive watching without input for more than 120 s, such as reading a long panel without scrolling. That reads as absent.
4. **Live-data sanity.** UNVERIFIED until deployed. No live rows exist yet. Check after deploy: the rows must also return to rest. Expect zero rows overnight and while Juniper is away. Expect `beats` to be about 1 per minute. If a minute shows `beats >= 3` steadily, more than one Hub tab or device is open on the same client class.
5. **Existing mechanism.** `substrate_hub_presence` / `hub_presence.py` (chat turns only), the cockpit and agent-board presence (agent sessions, not Juniper), and the walkway/office camera presence (physical room, not the screen). None of them records screen attention.
6. **Reversibility.** It is cheap to remove. Turn the flag off, or delete the route and the script, then `DROP TABLE hub_attention_minute`. Nothing reads it yet. There is no bus channel, schema registry entry, or manifest.

## Tests run

```text
node --test services/orion-hub/static/js/hub-attention-tracker.test.js
  # pass 9  # fail 0
cd services/orion-hub && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<worktree> pytest tests/test_hub_attention_routes.py tests/test_hub_presence.py tests/test_orion_message_tray_wiring.py -q
  25 passed
pytest tests/scripts/test_sync_local_env_from_example.py tests/test_sql_migration_drift_gate.py scripts/tests/test_check_env_template_parity.py -q
  130 passed
Full Hub suite (services/orion-hub/tests, this branch): 4693 passed, 147 skipped, 4 failed
  All 4 fail identically on main (pre-existing, unrelated):
    test_attention_loops_ui_smoke.py::test_app_js_loads_cognitive_loops
    test_attention_loops_ui_smoke.py::test_template_has_cognitive_loop_container
    test_fcc_model_labels_api.py::test_fcc_model_labels_disabled_by_default
    test_check_async_routes_not_blocking.py::test_the_real_repo_is_clean  (cwd artifact; 12 passed when run from repo root)
git diff --check                                         clean
scripts/check_metric_lineage.py --gate                   PASS
scripts/check_settings_defaults.py orion-hub --example-drift   OK
scripts/check_env_key_single_source.py                   OK
scripts/check_migration_required_by.py                   OK (1 new migration)
scripts/apply_post_pull_migrations.py --dry-run <file>   would apply (single transaction, lock_timeout=10s)
scripts/check_service_env_compose_parity.py orion-hub    cannot run on this service: compose has 2 services (hub-app,
                                                         hub-retro-gateway), checker refuses. Key added to hub-app
                                                         environment: and hub-app also has env_file.
```

The upsert SQL was checked against the live Postgres on a TEMP copy of the table, inside a transaction that was rolled back. Four upserts in the same minute gave one row: `desktop|reading|beats=4`. A null tab kept the earlier tab. Nothing persisted.

## Evals run

```text
None. There is nothing to evaluate until live rows exist. The live sanity check is the planned eval (metric gate step 4), about 2 weeks after deploy, before the reachable forecast reads the table.
```

## Docker/build/smoke checks

```text
docker compose --env-file .env --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml config
  -> hub-app environment: HUB_ATTENTION_TRACKER_ENABLED: "true"
No image build or deploy was run (not asked).
```

## Review findings fixed

The review was done by the dispatching agent, not by a review subagent in this branch. Findings will be recorded on the PR.

## Restart required

```bash
scripts/rebuild_services_from_git_diff.sh   # after merge, from the clean primary main checkout: applies the migration, then rebuilds orion-hub
```

## Risks / concerns

- Severity: low. Concern: input inside cross-origin iframes (for example an external Grafana embed) is not seen. A sub-page opened directly, not inside the Hub shell, does not load the tracker. Mitigation: documented here, in the README, and in the script header. Same-origin tab iframes are covered.
- Severity: low. Concern: a long read with no input for more than 120 s reads as absent. Mitigation: this is a deliberate conservative bias. Measure it in the live sanity check.
- Severity: low. Concern: two Hub tabs open on one machine count twice in `beats` for that minute. The minute row stays single. Mitigation: consumers use row existence, not `beats`, as the attended signal.
- Severity: low. Concern: the heartbeat route has no auth, like the other Hub `/api` routes on the tailnet. A forged beat can only add a row. Mitigation: `client` is enum-checked and `tab` is sanitized and capped at 64 characters.

## PR link

(filled in after `gh pr create`)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
