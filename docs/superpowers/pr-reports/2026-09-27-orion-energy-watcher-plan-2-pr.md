# Orion Energy Watcher — Plan 2: portal, bill reconcile, stakes, Hub strip

Plan 1 (PR #2373, merged) gave Orion a price for every hour of house electricity and for each GPU run. Plan 2 closes the loop with Rocky Mountain Power (RMP). Orion can now take in RMP's own bills and forecast, check its own estimate against them, keep a running "how worried should we be about this month's bill" snapshot, and let that snapshot pause optional curiosity work. The whole thing shows up on a Hub strip. Nothing here has been deployed yet; every live claim below is marked **UNVERIFIED**.

## Summary

- **Bills in, two ways, one format.** Orion reads RMP bills and forecasts from a drop folder, where you can hand-enter a JSON file. It can also read them from an optional headless browser that reuses a saved RMP login session. Both paths produce the same events; only the `source` label differs (`file_drop` vs `rockymountain_power`). (`services/orion-energy/app/bills.py`, `services/orion-energy/portal/`)
- **Orion checks its own math against RMP.** For every bill or forecast, Orion computes its own number for the same period and publishes the dollar difference, the kWh difference, and a per-line breakdown. When Orion is missing usage data, the difference is left blank with a named gap, never a fake $0. (`orion/energy/reconcile.py`, `energy.reconcile.v1`)
- **Importer health is visible.** A status tick says whether the importer is `healthy`, `stale`, `reauth_required` (the saved RMP login died), or `degraded`. A stale importer never invents usage. (`orion/energy/importer_status.py`, `energy.importer.status.v1`)
- **A stakes snapshot, and an optional curiosity hold.** Orion compares its month-end projection to RMP's forecast (`normal` / `near_forecast` / `over_forecast` / `unknown`). With `ORION_ENERGY_STAKES_ENABLED=true`, Hub curiosity holds a scheduled run with `held_off:energy_stakes` when the projection is near or over forecast, one attention row per hold episode (cycle + pressure), not per tick. The flag defaults to off. Unknown or stale data never causes a hold, and a run you start yourself is never held. (`orion/energy/stakes.py`, `services/orion-hub/scripts/energy_stakes_gate.py`)
- **Hub Energy strip.** A strip under the cabinet cooling strip shows importer state, cycle-to-date cost, Orion's projection vs RMP's forecast, the price of the next kWh, stakes pressure with its reason, the newest reconcile lines, and 14 daily kWh bars. Anything missing reads `unknown`, never `$0.00`. A snapshot older than `ORION_ENERGY_STAKES_MAX_AGE_SEC`, or a failed fetch, blanks the tiles instead of leaving old numbers up; a stale snapshot says "stale since <as_of>", and cycle-to-date always says "through <time>". (`services/orion-hub/static/js/energy-strip.js`, `scripts/energy_routes.py`)
- **Persistence.** sql-writer stores all five new event kinds in five new tables, which are created at boot. (`services/orion-sql-writer/app/models/energy.py`)

## Outcome moved

- Before: Orion could price its own GPU work, but nothing checked that price against the real bill, and nothing autonomous could see what a month of house electricity was going to cost.
- After: every bill and forecast produces a reconcile row that says how far off Orion's tariff model is and in which line. That is the evidence needed to patch the tariff config when it drifts. Optional curiosity spend can also see a real, freshness-checked dollar stake. The new replay eval shows reconcile landing on a hand-computed Schedule 1 bill to within $0.000001 for a matching bill, and within one cent for a hand-typed bill rounded to cents.

## Current architecture

Before this patch, `orion-energy` read Green Button XML files from a drop folder, priced them with the versioned Schedule 1 tariff (`config/energy/tariff.rmp_ut_sch1.2026-08-10.yaml`), and published usage, cycle-accrued cost, and per-run cost. sql-writer stored those three kinds. RMP's own bills and forecast never reached Orion, there was no importer health signal, and the Hub had no energy view. Curiosity had no cost input beyond its existing daily caps.

## Architecture touched

- **Contracts:** 5 new schemas in `orion/schemas/energy.py`, registered in `orion/schemas/registry.py`, and 5 new channels in `orion/bus/channels.yaml`.
- **`orion/energy` package:** reconcile math, the importer status state machine, the stakes snapshot, and ledger window helpers.
- **`services/orion-energy`:** bill drop folder, reconcile on bill arrival and on late usage, a status/stakes tick, and a new optional `orion-energy-portal` container (compose profile `portal`, Playwright, `Dockerfile.portal`).
- **`services/orion-sql-writer`:** 5 new tables, routes, and channel subscriptions. The channels are also force-appended in code, so a stale operator `.env` can't leave them unsubscribed.
- **`services/orion-hub`:** the curiosity stakes gate behind a flag, two read-only energy routes, and the Energy strip UI.
- **Ops:** `scripts/sync_local_env_from_example.py` now reaches the new keys (`ENERGY_`, `ORION_ENERGY_STAKES_`, `HUB_ENERGY_` prefixes; `orion-energy` in the default services). The CI workflows list the new hub tests.

## Files changed

- `orion/schemas/energy.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml`: the five new kinds and channels.
- `orion/energy/reconcile.py` (new): Orion-vs-RMP reconcile for actual bills and forecasts.
- `orion/energy/importer_status.py` (new): `healthy` / `stale` / `reauth_required` / `degraded`.
- `orion/energy/stakes.py` (new): stakes snapshot and pressure.
- `orion/energy/ledger.py`, `orion/energy/testing.py`: contiguous-window helpers; UTC-true test intervals.
- `orion/energy/tests/*`: reconcile, importer status, stakes, ledger, and schema tests.
- `services/orion-energy/app/{bills,inbox,main,pipeline,portal_status,settings}.py`: bill drop, reconcile wiring, status tick.
- `services/orion-energy/portal/*`, `Dockerfile.portal`, `Dockerfile.portal.dockerignore`, `requirements-portal.txt`: the headless RMP fetcher and the human reauth tool.
- `services/orion-energy/{.env_example,docker-compose.yml,README.md}`: new keys, the portal service, and docs.
- `services/orion-energy/tests/*`: bills, inbox, pipeline, main wiring, settings, and portal tests.
- `services/orion-energy/evals/test_energy_reconcile_replay_eval.py` (new): reconcile against the real tariff and a hand oracle, plus the file-drop path end to end.
- `services/orion-sql-writer/app/{models/energy.py,models/__init__.py,energy_persist.py,settings.py,worker.py}`, `.env_example`, `tests/test_energy_sql_shape.py`: persistence.
- `services/orion-hub/scripts/{energy_stakes_gate.py,curiosity_investigation.py,main.py,energy_routes.py,api_routes.py}`, `app/settings.py`, `.env_example`: the gate and the routes.
- `services/orion-hub/{templates/index.html,static/js/energy-strip.js,static/js/energy-strip.test.js,static/js/biometrics-view.js}`: the strip.
- `services/orion-hub/tests/{test_energy_stakes_gate.py,test_energy_routes.py,test_energy_strip_panel.py,test_curiosity_investigation.py}`: hub tests.
- `scripts/sync_local_env_from_example.py`, `tests/scripts/test_sync_local_env_from_example.py`: sync reaches the new keys; `ENERGY_USAGE_POINT_ID` is in `NEVER_SYNC_KEYS`.
- `services/orion-hub/README.md`: Energy strip and energy stakes hold (`ORION_ENERGY_STAKES_ENABLED`, `ORION_ENERGY_STAKES_MAX_AGE_SEC`, `HUB_ENERGY_TIMEZONE`).
- `tests/test_energy_bus_catalog.py`: catalog and registry coverage for the new channels and kinds.
- `.github/workflows/orion-reading-tests.yml`: new hub tests added to the explicit list. `orion-energy-tests.yml` already covers `services/orion-energy/**`, including evals, so it is unchanged.
- `config/metrics/metric_definitions.lock.json`: lock refresh for the new schema fields.
- `docs/superpowers/specs/2026-09-26-orion-energy-watcher-design.md`: Status line; Branch/Worktree header lists both plans.
- `docs/superpowers/plans/2026-09-27-orion-energy-watcher-plan-2.md`: the plan.

## Schema / bus / API changes

- **Added kinds and channels:**
  - `energy.bill.actual.v1` on `orion:energy:bill:actual`
  - `energy.bill.forecast.v1` on `orion:energy:bill:forecast`
  - `energy.reconcile.v1` on `orion:energy:reconcile`
  - `energy.stakes.snapshot.v1` on `orion:energy:stakes:snapshot`
  - `energy.importer.status.v1` on `orion:energy:importer:status`

  All five are registered, all five are catalogued, and sql-writer persists them to `energy_bill_actual`, `energy_bill_forecast`, `energy_reconcile`, `energy_stakes_snapshot`, and `energy_importer_status`.
- **Added HTTP routes (Hub, read-only):** `GET /api/energy/latest` (stakes, importer, reconcile, plus top-level `stale`, `as_of`, `covered_through`; `stale` is null when there is no snapshot) and `GET /api/energy/usage/daily?days=14`.
- **Removed:** none.
- **Renamed:** none.
- **Behavior changed:** with `ORION_ENERGY_STAKES_ENABLED=true`, Hub curiosity can record a `held_off:energy_stakes` decision instead of spending. With the flag off (the default), curiosity behavior is unchanged, and the snapshot is never read.
- **Compatibility notes:** Plan 1 kinds and channels are unchanged. The tables are created by sql-writer's `create_all` at boot; no manual migration is needed. Until sql-writer is redeployed from this branch, the Hub strip reads `energy data unavailable` (checked read-only against live Postgres in Task 8: the Plan 2 tables don't exist yet).

## Env/config changes

- **Added keys, `orion-energy`:**
  - `ENERGY_BILL_INBOX_DIR`, `ENERGY_BILL_PROCESSED_DIR`
  - `ENERGY_STATUS_INTERVAL_SEC`, `ENERGY_STALE_AFTER_HOURS`
  - `ENERGY_PORTAL_ENABLED`, `ENERGY_PORTAL_STATUS_PATH`, `ENERGY_PORTAL_INTERVAL_HOURS`, `ENERGY_PORTAL_BASE_URL`, `ENERGY_PORTAL_PROFILE_DIR`, `ENERGY_PORTAL_RAW_DIR`, `ENERGY_PORTAL_BACKFILL_DAYS`, `ENERGY_PORTAL_TIMEOUT_SEC`
  - `ENERGY_STAKES_NEAR_RATIO`, `ENERGY_STAKES_OVER_RATIO`
  - `ENERGY_BILL_ACTUAL_CHANNEL`, `ENERGY_BILL_FORECAST_CHANNEL`, `ENERGY_RECONCILE_CHANNEL`, `ENERGY_STAKES_CHANNEL`, `ENERGY_IMPORTER_STATUS_CHANNEL`
- **Added keys, `orion-hub`:** `ORION_ENERGY_STAKES_ENABLED` (default `false`), `ORION_ENERGY_STAKES_MAX_AGE_SEC` (default `1800`), `HUB_ENERGY_TIMEZONE` (default `America/Denver`).
- **Changed values, `orion-sql-writer`:** `SQL_WRITER_SUBSCRIBE_CHANNELS` gains the 5 new channels and `SQL_WRITER_ROUTE_MAP_JSON` gains the 5 new routes. No new keys.
- **Removed keys:** none.
- **Renamed keys:** none.
- **`.env_example` updated:** yes, for orion-energy, orion-hub, and orion-sql-writer.
- **Local `.env` synced with `python scripts/sync_local_env_from_example.py`:** yes. orion-energy got its keys, and hub got `ORION_ENERGY_STAKES_*` and `HUB_ENERGY_TIMEZONE`.
  - The sync script can't add members *inside* a JSON value, and the sql-writer list keys are outside its default prefixes. So the live sql-writer `.env` was still missing the 5 new channels and 5 new routes.
  - Task 9 added them by hand, additively, and only those two lines changed. Backup: `/tmp/t9/sql-writer.env.bak`.
  - After the edit, `check_service()` from `scripts/check_env_template_parity.py`, run against this branch's templates, reports no blocking drift and no missing keys for orion-energy, orion-hub, or orion-sql-writer. Without this edit, `safe_docker_build.sh orion-sql-writer up` would have refused the deploy after merge.
- **Skipped keys requiring operator action:**
  - `ENERGY_USAGE_POINT_ID` (Plan 1) is now in `NEVER_SYNC_KEYS`: the sync script never adds or overwrites it, even with `--force`. **The operator pastes the meter's usage point id into `services/orion-energy/.env` by hand** (empty = use the only usage point in the feed).
  - Unrelated keys reported as diverged and left alone (Task 9 run, `/tmp/t9b/gates.log`): `DURABLE_RUNS_GRAPH_HOST`, `GPU_POOL_ACTUATE_ROLES`, `COCREATION_SIGNALS_GH_TOKEN`, `COCREATION_SIGNALS_AFFECTIVE_STATE_ENABLED`, `ORION_CURIOSITY_GRAPH_HOST`, `ORION_CURIOSITY_GRAPH_PORT`, `ORION_CURIOSITY_GRAPH_USER`, `CURIOSITY_PEER_CURSOR_BUDGET_STATE`. The final-review sync run additionally lists `HUB_CURIOSITY_ELASTIC_ACTIVATION_ENABLED` (local `true`, example `false`), also unrelated.

### §0A metric gate note

The stakes snapshot is a projection of inputs that are already gated, not a new continuous signal:

- metered usage (RMP Green Button intervals)
- the versioned tariff (`config/energy/tariff.rmp_ut_sch1.2026-08-10.yaml`)
- RMP's own bill forecast
- importer freshness

It adds no new sensor and no new biometric. `pressure` is a ratio of Orion's month-end projection to RMP's forecast, bucketed at two operator knobs (`ENERGY_STAKES_NEAR_RATIO=1.0`, `ENERGY_STAKES_OVER_RATIO=1.10`). Those thresholds are operator settings, not learned values.

There is a known bias. Orion's projection is pre-tax, while RMP's forecast may include taxes. That bias lowers the ratio, so it leans toward *fewer* holds, not more. Pressure is `unknown` whenever the importer isn't healthy, coverage lags more than `ENERGY_STALE_AFTER_HOURS`, the forecast isn't for the current period, or the forecast has no dollar figure. `unknown` never holds.

- **Live-data sanity check (gate step 4): UNVERIFIED.** Live `energy_usage_interval` has 0 rows and no RMP forecast has been ingested yet, so there is no real series to inspect for a flat or saturated signal. Re-run this check once the first real bill and forecast land. The curiosity flag stays off until then.

### Acceptance checks (spec §Acceptance checks)

1. **Green Button XML becomes hourly rows and `energy.usage.observed`.** Covered by:
   - Plan 1: `orion/energy/tests/test_energy_espi.py` and `services/orion-energy/tests/test_energy_pipeline.py::test_ingest_publishes_usage_then_accrual`.
   - Portal path: `test_portal_fetch.py::test_good_fetch_delivers_xml_and_bills` and `test_energy_inbox.py::test_portal_named_file_is_labeled_rockymountain_power`.
   - Postgres: `test_energy_sql_shape.py::test_usage_upsert_newer_retrieval_wins`.
2. **Accrued dollars update, with `tariff_version` recorded.** Covered by `test_energy_pipeline.py::test_ingest_publishes_usage_then_accrual`, `evals/test_energy_bill_replay_eval.py::test_cycle_total_matches_hand_bill`, and the `tariff_version` asserts in `test_energy_run_cost.py` and `test_energy_reconcile.py`.
3. **Run cost is known when joules are known, and null (not zero) when blind.** Covered by `test_energy_run_cost.py::test_blind_settlement_is_unknown_not_free` and `test_incremental_cost_at_cycle_position` (Plan 1).
4. **House share only with an overlapping interval, and never used by autonomy.** Covered by `test_energy_run_cost.py::test_house_share_when_interval_covers_window`, `test_house_share_needs_full_coverage`, and `test_partial_interval_overlap_is_house_share_gap`. The Hub gate never reads `house_share_cost_usd` (`energy_stakes_gate.py` docstring; the gate reads only the stakes snapshot).
5. **Bill and forecast are ingested, and reconcile shows the delta.** Covered by:
   - Unit: `test_energy_reconcile.py`, 13 tests, including the hand oracles `test_actual_full_period_hand_oracle` and `test_forecast_linear_run_rate_hand_oracle`.
   - Pipeline: `test_energy_pipeline_bills.py::test_bill_after_usage_publishes_bill_and_priced_reconcile` and `test_bill_before_usage_reconciles_again_when_usage_lands`.
   - **New eval:** `evals/test_energy_reconcile_replay_eval.py`, three tests. A matching bill reconciles to $0. Ten flat days project to the full-month oracle. A hand-typed bill file dropped in the inbox reaches reconcile, labeled `tax_unknown`, within one cent.
6. **The importer can visibly enter `reauth_required` / `stale` / `degraded`, and stale emits no fake usage.** Covered by:
   - `test_energy_importer_status.py`, 11 tests.
   - `test_energy_pipeline_bills.py::test_status_tick_with_no_data_is_stale_and_emits_no_usage` and `test_status_tick_portal_reauth`.
   - `test_portal_fetch.py::test_login_redirect_is_reauth_and_writes_nothing`.
7. **Flag off leaves curiosity unchanged; flag on produces a `held_off:energy_stakes` row.** Covered by:
   - `test_curiosity_investigation.py::test_energy_flag_off_never_reads_the_snapshot`, `test_energy_over_forecast_holds_and_leaves_an_attention_row`, `test_energy_hold_never_blocks_a_forced_run`, and `test_energy_no_fresh_pressure_never_holds`.
   - `test_energy_stakes_gate.py`, 8 test functions.
8. **Metric gate.** See the §0A note above. The stakes snapshot is a projection of already-gated inputs; its live-data check is UNVERIFIED.

## Tests run

All commands were run from the worktree root with `PY=/mnt/scripts/Orion-Sapienform/.venv/bin/python`.

```text
git diff --check                                                        -> clean
$PY scripts/sync_local_env_from_example.py                              -> no new updates; unrelated diverged keys only
$PY scripts/check_env_template_parity.py                                -> env template parity: PASS (92 service(s) compared)
$PY scripts/check_compose_no_relative_mounts.py                         -> PASS (91 compose files, 0 relative host mounts)
scripts/check_bus_channels.py                                           -> DOES NOT EXIST in this repo (not invented)
scripts/check_schema_registry.py                                        -> DOES NOT EXIST in this repo (not invented)
make agent-check                                                        -> DOES NOT EXIST (Makefile comment confirms)
  real equivalents run instead:
$PY scripts/check_bus_reply_channels.py                                 -> 13 reply-channel prefixes resolved, 0 uncovered
PYTHONPATH=. $PY -m pytest tests/test_energy_bus_catalog.py tests/test_schema_registry_import_light.py \
  tests/test_channel_prefix_guardrail.py tests/test_bus_reply_channel_catalog_coverage.py -q
                                                                        -> 21 passed, 1 failed
  the 1 failure: test_channel_prefix_guardrail::test_literal_publish_subscribe_prefixes flags
  services/orion-context-exec/app/events.py. That file is not touched by this branch, and the test
  fails identically on main at the merge base d16e9adf9. Pre-existing.
PYTHONPATH=. $PY -m pytest tests/test_energy_bus_catalog.py orion/energy/tests -q     -> 142 passed
PYTHONPATH=.:services/orion-energy $PY -m pytest services/orion-energy/tests services/orion-energy/evals -q
                                                                        -> 94 passed
(cd services/orion-sql-writer && PYTHONPATH=.:../.. $PY -m pytest tests/test_energy_sql_shape.py -q)
                                                                        -> 32 passed
PYTHONPATH=.:services/orion-sql-writer $PY -m pytest -q services/orion-sql-writer/tests/test_route_map_completeness.py \
  services/orion-sql-writer/tests/test_route_coverage.py              -> 14 passed
PYTHONPATH=$PWD $PY -m pytest -q services/orion-hub/tests/test_energy_stakes_gate.py services/orion-hub/tests/test_energy_routes.py \
  services/orion-hub/tests/test_energy_strip_panel.py services/orion-hub/tests/test_curiosity_investigation.py \
  services/orion-hub/tests/test_situation_settings_env.py services/orion-hub/tests/test_cabinet_sensors_panel.py \
  services/orion-hub/tests/test_cabinet_cooling_routes.py              -> 233 passed
(cd services/orion-hub && node --test static/js/energy-strip.test.js static/js/cabinet-sensors.test.js static/js/biometrics-view.test.js)
                                                                        -> tests 19, pass 19, fail 0
(cd services/orion-hub && node --test $(find static/js -name '*.test.js' | sort))
                                                                        -> tests 214, pass 192, fail 0, skipped 22
PYTHONPATH=. $PY -m pytest -q tests/scripts/test_sync_local_env_from_example.py scripts/tests/test_check_env_template_parity.py
                                                                        -> 36 passed
git fetch origin main; git log HEAD..origin/main                        -> 0 commits (no drift, no conflicts)
```

## Evals run

```text
PYTHONPATH=.:services/orion-energy $PY -m pytest services/orion-energy/evals -q   -> 5 passed
  test_energy_bill_replay_eval.py (Plan 1)       : 2 passed
  test_energy_reconcile_replay_eval.py (new)     : 3 passed
    test_matching_bill_reconciles_to_zero                 real tariff, 720 kWh; delta $0 within 1e-6, pre_tax basis
    test_ten_flat_days_project_to_the_full_month_oracle   240 h metered, projected to 720 kWh / oracle total
    test_hand_entered_bill_file_reaches_reconcile          JSON dropped into inbox, then scan_bills, then reconcile; tax_unknown, |delta| < $0.01
```

The oracle is written from published Schedule 1 numbers, not by calling the tariff code: 720 kWh = 400 × $0.098332 + 320 × $0.125263, times the rider and tax multipliers, plus the $12.00 customer charge and the $0.16 Schedule 91 lifeline surcharge. So reconcile cannot pass by agreeing with itself.

## Docker/build/smoke checks

The worktree had no root, sql-writer, or hub `.env`. Following the convention in the sibling worktrees, those were symlinked (gitignored) to the primary checkout's live files so the wrapper could run.

```text
scripts/safe_docker_build.sh orion-energy config                         -> exit 0 (env parity PASS, hostname refs OK)
scripts/safe_docker_build.sh orion-energy --profile portal config        -> exit 0 (orion-energy + orion-energy-portal services)
scripts/safe_docker_build.sh orion-sql-writer config                     -> exit 0
scripts/safe_docker_build.sh orion-hub config                            -> exit 0
scripts/safe_docker_build.sh orion-energy build orion-energy             -> exit 0, "Image orion-energy-orion-energy Built"
scripts/safe_docker_build.sh orion-energy --profile portal build orion-energy-portal
                                                                         -> exit 0, "Image orion-energy-orion-energy-portal Built" (~155 s)
docker run --rm --network none --entrypoint python orion-energy-orion-energy -c "import app.main, app.bills, app.pipeline, orion.energy.reconcile, orion.energy.stakes, orion.energy.importer_status"
                                                                         -> core imports ok
docker run --rm --network none --entrypoint python orion-energy-orion-energy-portal -c "import portal.main, portal.fetch, portal.parse, portal.reauth; import playwright"
                                                                         -> portal imports ok
```

**Not run, on purpose:** `up -d` for orion-energy, orion-sql-writer, or orion-hub. An `orion-energy` container is live on the shared deployment (`orion-athena-orion-energy`, still running its previous image after these builds), and redeploying shared services needs Juniper's approval. The live path is therefore **UNVERIFIED**, specifically:

- the `energy_status state=... pressure=...` log line
- `energy_reconcile` and `energy_importer_status` rows in Postgres
- the Hub strip on real data
- a live curiosity hold
- the portal fetch

## Review findings fixed

- **Finding (Task 1):** a stakes snapshot could claim "over forecast" while the importer was stale, and NaN/inf could slip into reconcile and stakes numbers.
  - Fix: a compared pressure now requires `importer_state="healthy"`. Float fields reject non-finite values (`allow_inf_nan=False`, plus a `bucket_deltas` validator).
  - Evidence: `test_stakes_compared_pressure_requires_healthy_importer`, `test_stakes_rejects_non_finite_ratio`, `test_reconcile_rejects_non_finite_bucket_delta` (RED, then GREEN).
- **Finding (Task 3):** reconcile measured durations on the wall clock, so projections were off by an hour across daylight-saving changes. A missing first hour also read as "no usage" instead of "incomplete".
  - Fix: all durations use real UTC seconds, and the run-rate uses real covered time, which also handles 15-minute intervals. A leading hole now reports `usage_incomplete`, and test intervals are UTC-true.
  - Evidence: `test_forecast_dst_partial_coverage_projects_over_real_seconds`, `test_forecast_dst_full_coverage_returns_metered_kwh`, `test_forecast_quarter_hour_intervals_use_real_covered_time` (was 180 vs 720), `test_actual_first_hour_missing_is_usage_incomplete`.
- **Finding (Task 4):** a healthy importer plus a mid-cycle data gap could freeze the projection and still say "over forecast". A closed or future forecast could also be compared.
  - Fix: pressure is `unknown` when coverage lags more than `stale_after_hours` (now a required argument) and when the forecast is not for the current period. There is one shared "is current" check.
  - Evidence: `test_mid_cycle_gap_blocks_pressure_on_stale_coverage`, `test_closed_forecast_is_unknown`, `test_future_forecast_is_unknown`, `test_current_forecast_denver_last_local_day_of_period`.
- **Finding (Task 5):** a malformed bill file (a list or dict `kind`) could crash the inbox pass and drop usage rows that had already moved to `processed/`. Failed files also overwrote each other, and the default env sync never reached the orion-energy keys.
  - Fix: usage and bills now run in separate try blocks, with usage first. A bad `kind` is a `ValueError` and the file moves to `failed/`. Failed files are timestamp-stamped. `orion-energy` and the `ENERGY_` prefix were added to the sync script.
  - Evidence: `test_poison_bill_cannot_block_usage_publish`, `test_bad_bills_go_to_failed_and_publish_nothing`, `test_repeated_failures_with_same_name_both_survive`, `test_energy_keys_are_reached_by_the_default_sync`.
- **Finding (Task 6, round 1):** the portal container could restart-storm against RMP, saved raw HTML could contain tokens, the profile dir could be world-readable, reauth recorded a fake success, and unchanged bill history was resent every run.
  - Fix: each recorded attempt holds the next one for a full interval, including across restarts. Script bodies and hidden-input values are scrubbed, and the raw dir is `0700` with `0600` files. The profile dir is `0700`. Reauth writes `reauth_completed` and keeps the previous success time. Bills are deduplicated by content hash.
  - Evidence: `test_portal_runtime.py` (16 tests), `test_raw_html_is_scrubbed_and_private`, `test_unchanged_bill_history_is_written_once`, `test_reauth_keeps_previous_success`.
- **Finding (Task 6, round 2):** two rows for the same billing period flip-flopped, and an attempt killed mid-run did not count toward the restart wait.
  - Fix: same-period rows collapse to the first row in table order. The attempt start is stamped before the browser launches. Status JSON that isn't an object reads as missing, and csrf/token meta tags are scrubbed.
  - Evidence: `test_same_period_rows_collapse_to_first_and_stay_quiet`, `test_attempt_start_is_stamped_before_browser_launch`, `test_non_object_status_json_reads_as_missing`.
- **Finding (Task 9, found during env verification):** the live sql-writer `.env` was missing the 5 new channels and routes. Nothing reported it: the sync script can't see inside JSON values, and the parity gate compares against the primary checkout's (main's) template, not this branch's.
  - Fix: additive hand-edit of the two lines, with a backup at `/tmp/t9/sql-writer.env.bak`.
  - Evidence: `check_service()` against the branch templates reports blocking=[] and warnings=[] for all three services.
- **Final whole-branch review:**
  - Finding (Important): the Hub Energy strip kept the last good dollar tiles on screen after a failed fetch, and showed an old snapshot's numbers as current.
    - Fix: `/api/energy/latest` returns `stale` (snapshot `as_of` older than `ORION_ENERGY_STAKES_MAX_AGE_SEC`; unreadable `as_of` is stale), `as_of`, and `covered_through`. The JS uses `Promise.allSettled`; a failed/non-2xx/unparseable fetch renders every tile `unknown` and clears the bars, and each endpoint renders independently. Stale renders every tile `unknown` with "stale since <as_of>"; cycle-to-date always shows "through <covered_through>". `importerLabel({state:null})` and a null `billing_period_start` read "unknown".
    - Evidence: node tests "a network failure turns every tile unknown and clears the bars", "bad JSON or an HTTP error counts as unreachable", "one endpoint failing does not block the other", "a stale snapshot never shows its numbers as current"; route tests `test_latest_old_snapshot_is_flagged_stale`, `test_latest_fresh_snapshot_is_not_stale_and_surfaces_coverage`, `test_latest_unreadable_as_of_is_stale_not_fresh` (RED, then GREEN).
  - Finding (Important): `ENERGY_USAGE_POINT_ID` could be flattened by `sync_local_env_from_example.py --force`.
    - Fix: added to `NEVER_SYNC_KEYS`.
    - Evidence: `test_energy_usage_point_id_never_synced_even_with_force`.
  - Finding (Minor): when two versions of a bill share `computed_at`, the reconcile row shown was arbitrary.
    - Fix: `DISTINCT ON` ordering is now `billing_period_start DESC, computed_at DESC, utility_as_of DESC`.
    - Evidence: `test_reconcile_newest_utility_version_wins_a_computed_at_tie`.
  - Finding (Minor): the curiosity hold minted a new attention row every 5-minute tick (`entry_id` keyed on snapshot `as_of`), flooding `recent_attention_cue`; the correlation id was a per-tick uuid4 missing from the log.
    - Fix: `entry_id = curiosity:held_off:energy_stakes:<cycle_start>:<pressure>` (snapshot UTC day if the cycle is unknown), so repeats collapse on the attention PK's ON CONFLICT no-op; `correlation_id = uuid5(NAMESPACE_URL, entry_id)`, logged on `curiosity_investigation_blocked`. Still behind `ORION_ENERGY_STAKES_ENABLED`.
    - Evidence: `test_the_same_hold_episode_is_one_row_across_ticks`, `test_a_new_pressure_or_cycle_is_a_new_episode`, `test_unknown_cycle_falls_back_to_the_snapshot_day_not_one_row_forever`, `test_energy_hold_repeats_one_episode_with_a_deterministic_correlation`.
  - Finding (docs): Hub and orion-energy READMEs didn't document the new keys; the portal section didn't say to stop the running portal before a headed reauth or `--once`.
    - Fix: Hub README sections "House electricity (Energy strip)" and "4.2.4 Energy stakes hold"; orion-energy README key tables for `ENERGY_STALE_AFTER_HOURS`, `ENERGY_STAKES_NEAR_RATIO`, `ENERGY_STAKES_OVER_RATIO`, `ENERGY_PORTAL_TIMEOUT_SEC`, `ENERGY_PORTAL_RAW_DIR`, `ENERGY_PORTAL_BACKFILL_DAYS`, plus `scripts/safe_docker_build.sh orion-energy --profile portal stop orion-energy-portal` before reauth/`--once`, so two Chromium processes never share the profile.
    - Evidence: `.superpowers/sdd/final-fix-report.md`.

## Restart required

Juniper to run these from this worktree (or from main after merge). Order matters: sql-writer first so the tables exist, then energy, then hub.

```bash
cd /mnt/scripts/Orion-Sapienform-orion-energy-watcher-plan-2
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-energy up -d --build orion-energy
scripts/safe_docker_build.sh orion-hub up -d --build

# evidence to collect afterwards
docker logs --tail=50 orion-athena-orion-energy 2>&1 | grep -E "energy_status|energy_ledger_replayed"
#   expect: energy_status state=... pressure=... within ENERGY_STATUS_INTERVAL_SEC (300 s)
# after dropping a real bill JSON into /mnt/storage-warm/orion-energy/bills/inbox/:
#   expect: energy_ingested ... bills=1, then:
#   SELECT reconcile_kind, billing_period_start, orion_total_usd, utility_total_usd, delta_usd, reconcile_gap
#     FROM energy_reconcile ORDER BY computed_at DESC LIMIT 3;
#   SELECT as_of, state, reason FROM energy_importer_status ORDER BY as_of DESC LIMIT 1;

# portal (optional; needs the one-time headed reauth in services/orion-energy/README.md "Portal" first)
scripts/safe_docker_build.sh orion-energy --profile portal run --rm orion-energy-portal python -m portal.main --once --days 730
scripts/safe_docker_build.sh orion-energy --profile portal up -d --build orion-energy-portal
# then set ENERGY_PORTAL_ENABLED=true in services/orion-energy/.env and restart orion-energy
```

Leave `ORION_ENERGY_STAKES_ENABLED=false` until the §0A live-data check has been run on real reconcile and stakes rows.

## Risks / concerns

- **Severity: High.** **Concern:** RMP portal URLs and selectors are **UNVERIFIED**; nobody has run a live login yet. **Mitigation:** the file-drop path works without the portal. An empty download or empty bill table is an error state, not a silent success, and a dead session reads `reauth_required` with no retry storm.
- **Severity: Medium.** **Concern:** the whole live path is **UNVERIFIED**: the status log line, reconcile and importer-status rows, the Hub strip on real data, and a live curiosity hold. The Plan 2 tables don't exist in live Postgres until sql-writer is redeployed. **Mitigation:** the restart commands and evidence queries are listed above. The Hub strip was proven against a scratch Postgres built from the real models in Task 8.
- **Severity: Low (fixed).** **Concern:** `ENERGY_USAGE_POINT_ID` could be wiped by a `--force` sync. **Mitigation:** now in `NEVER_SYNC_KEYS`; the operator pastes it by hand.
- **Severity: Low (follow-up).** **Concern:** no retention on `energy_stakes_snapshot` / `energy_importer_status`. One row each per `ENERGY_STATUS_INTERVAL_SEC` (300 s) is ~105k rows/yr per table. **Mitigation:** small rows and indexed `ORDER BY as_of DESC LIMIT 1` reads; add a bounded-retention pass in sql-writer (same shape as `substrate_attention_schema` retention) before year one.
- **Severity: Low (follow-up).** **Concern:** past closed bills are not re-reconciled after a tariff config patch; reconcile only re-runs on bill arrival or late usage. **Mitigation:** re-drop the affected bill JSON to force a new reconcile row; a follow-up could re-reconcile closed bills when `tariff_version` changes.
- **Severity: Low (deferred).** **Concern:** the spec's optional second hold condition (expensive seasonal block plus weak expected value) is not built; the gate holds only on near/over-forecast pressure. **Mitigation:** deliberate scope cut; needs a live expected-value signal that clears the §0A gate first.
- **Severity: Medium.** **Concern:** the image tag `orion-energy-orion-energy` now points at this branch's build. The running container is untouched, but a plain `up -d` from the primary checkout before merge could recreate it on the Plan 2 image. **Mitigation:** deploy only via the restart commands above, or rebuild from main if a pre-merge restart is needed.
- **Severity: Low.** **Concern:** the live sql-writer `.env` already lists the 5 new channels and routes, but the running sql-writer is still `main`'s code. If sql-writer restarts from `main` before this merges, it would subscribe to the new channels without the matching table models. **Mitigation:** routes are looked up per message, not at boot, and nothing publishes the new kinds until `orion-energy` is redeployed from this branch. Restore `/tmp/t9/sql-writer.env.bak` if this branch is abandoned.
- **Severity: Low.** **Concern:** `check_env_template_parity.py` compares the live `.env` against the *primary checkout's* template, so a branch that adds JSON members passes the gate pre-merge and only blocks at the first post-merge deploy. **Mitigation:** fixed by hand for this branch. Worth a follow-up so the gate compares the template of the tree being deployed.
- **Severity: Low.** **Concern:** `tests/test_channel_prefix_guardrail.py` fails on main (orion-context-exec literals). **Mitigation:** pre-existing and unrelated; follow-up.
- **Severity: Low.** **Concern:** open Minor findings from the per-task reviews, kept for final triage:
  - **T1:**
    - `usage_lag_hours` accepts inf.
    - Some schema tests use `pytest.raises(ValueError)` without `match=`.
    - No tests for a backwards forecast period, UTC normalisation, or `extra=forbid`.
    - The `ImporterSource` value `portal` vs the `EnergySource` value `rockymountain_power` vocabulary split was mandated by the brief.
    - A reconcile can carry `delta_usd` while `utility_total_usd` is None.
  - **T2:**
    - The forecast and stakes insert tests don't assert date coercion.
    - The new channel order in `.env_example` is reversed vs `settings.py`.
    - `_insert_once` returns True on a skipped duplicate (brief-mandated).
  - **T3:**
    - The DST full-coverage test doesn't guard the short-circuit.
    - The forecast's unobserved remainder is priced at the last month's season (an undocumented approximation).
    - Intervals straddling the period end are counted in full.
    - **Plan 1 bug:** `EnergyUsageIntervalV1.interval_seconds` uses wall-clock subtraction across DST (follow-up).
    - RMP proration for non-month periods is UNVERIFIED.
  - **T5:**
    - `_rereconcile` uses a single min/max window, which over-reaches for scattered batches.
    - A stale reconcile may publish before a fresh one in the same pass (harmless if the sql-writer key collapses them).
    - Pre-existing pytest-asyncio and pydantic warnings.
  - **T6:**
    - A killed attempt keeps the previous state in `status.json`.
    - Collapsed same-period rows are dropped without a log line (newest-first order is UNVERIFIED).
    - Dedupe advances on fetcher write, not on ingest.
    - `green_button_range` uses the UTC date.
    - The scrub misses href/action token query strings and `data-*token*` attributes.
  - **T7:**
    - No-hold outcomes are silent when the flag is on.
    - ~~The correlation id is a per-tick uuid4 and is missing from the log line.~~ Fixed in the final review (uuid5 of the episode `entry_id`, logged).
    - Untested: a publish failure still holds, a reader timeout does not hold, and the envelope correlation id equals the payload's.
    - The spec's optional second hold condition (expensive seasonal block plus weak evidence) is not built.
  - **T8:**
    - ~~**Should fix:** the `energy-strip.js` catch leaves the last good dollar tiles on screen after a network failure.~~ Fixed in the final review.
    - ~~`Promise.all` couples the latest and daily fetch failures.~~ Fixed (`Promise.allSettled`).
    - `json.loads(bucket_deltas)` and `float(kwh)` sit outside the try in `energy_routes`.
    - Route warnings drop `exc_info` and repeat every 60 s until the tables are deployed.
    - ~~Cosmetic nulls in the JS labels.~~ Fixed in the final review.
    - Weak tests for the timezone default and the SQL substring.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2376

Status: DONE_WITH_CONCERNS (live path UNVERIFIED pending Juniper-approved redeploy; portal selectors UNVERIFIED; final whole-branch review findings fixed; PR link pending).
