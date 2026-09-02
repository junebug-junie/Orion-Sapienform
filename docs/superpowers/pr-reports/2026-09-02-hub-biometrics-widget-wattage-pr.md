# Hub: retire old biometrics widget, expand voice viz, real wattage cards

## Summary

- Removed the redundant `#biometricsPanel` (strain/homeostasis/stability) widget from the Hub main tab — superseded by the biometrics preview/modal shipped in PR #2027/#2032/#2036 — and its `app.js` wiring; "Oríon's Voice" now takes the full row.
- Added `measurements_by_node` to `BiometricsClusterV1` so the per-node raw measurements `publish_cluster()` already computes (post PDU-proxy fill, pre fleet-sum) survive instead of being discarded — a fleet total can no longer say which machine drew how much watts.
- Forwarded that per-node breakdown through orion-biometrics' `/snapshot` and the Hub's `/api/biometrics/preview/snapshot` route as `cluster_measurements_by_node`.
- Added `chassis_watts` as a real-time wattage default: one tile per node (athena self-reported via iLO, circe via athena's PDU-proxy read) on the compact EKG-card preview, plus a current tile + 24h trended sparkline in the Athena/Circe modal subviews.
- Fixed two review findings: `orion-sql-writer` was silently dropping the new `measurements_by_node` field (no column), and the websocket handler kept computing/shipping a `biometrics` snapshot on every outgoing message even though this same change removed its only client-side reader — see **Review findings fixed** below.

## Outcome moved

- The Hub main tab no longer duplicates biometrics info in two places; the voice visualizer gets real screen space.
- Real-time chassis wattage for both hosts is now visible without opening the modal, and the modal gives current + 24h trended wattage per node — previously only a normalized 0-1 "power" pressure existed, no raw watts anywhere in the UI.
- Circe's proxied wattage reading (previously computed then discarded every cluster-publish tick) now persists to Postgres and is queryable.
- Removed a per-connection websocket loop (`biometrics_heartbeat`) that had become pure network chatter — a `{"biometrics_tick": True}` push every `BIOMETRICS_PUSH_INTERVAL_SEC` (5s) to every connected browser, read by nobody.

## Current architecture

Before this patch: `#biometricsPanel` and the newer biometrics preview/modal (`biometrics-view.js`) both existed side by side in the Hub main tab, showing overlapping strain/homeostasis/stability data through two different pipelines — one pushed over the websocket (`_with_biometrics`/`biometrics_heartbeat`/`d.biometrics`), one polled over HTTP (`/api/biometrics/preview/*`). Circe's PDU-proxied wattage was computed every cluster-publish tick inside `BiometricsHub.publish_cluster()` (`services/orion-biometrics/app/main.py`), summed into a fleet total via `aggregate_fleet_measurements`, and then discarded — the per-node breakdown was never persisted or exposed anywhere.

## Architecture touched

- **Hub frontend**: `templates/index.html`, `static/js/app.js` (widget removal), `static/js/biometrics-view.js` (wattage cards).
- **Hub backend**: `scripts/biometrics_preview_routes.py` (`/snapshot` now forwards `cluster_measurements_by_node`; `chassis_watts` added to `_CHANNEL_COLUMN`), `scripts/websocket_handler.py` (`_with_biometrics` gutted, `biometrics_heartbeat` removed).
- **orion-biometrics**: `app/main.py` (`publish_cluster()` now populates `measurements_by_node`; `_build_snapshot_payload()` forwards it).
- **Schema**: `orion/schemas/telemetry/biometrics.py` (`BiometricsClusterV1.measurements_by_node`, additive/optional).
- **orion-sql-writer**: `app/models/biometrics_cluster.py` (new column), `app/main.py` (boot-time `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`).
- **Env/config**: `services/orion-hub/.env_example`, `app/settings.py`, `docker-compose.yml` (removed dead `BIOMETRICS_PUSH_INTERVAL_SEC`), README updated.

## Files changed

- `services/orion-hub/templates/index.html`: remove `#biometricsPanel`, widen voice-viz container.
- `services/orion-hub/static/js/app.js`: remove `updateBiometricsPanel()` and its wiring.
- `services/orion-hub/static/js/biometrics-view.js`: add `chassisWattsFor()`, wattage tiles in preview + modal, `RAW_CHANNELS`.
- `services/orion-hub/scripts/biometrics_preview_routes.py`: forward `cluster_measurements_by_node`; add `chassis_watts` channel.
- `services/orion-hub/scripts/websocket_handler.py`: gut `_with_biometrics`, remove `biometrics_heartbeat`.
- `services/orion-hub/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: retire `BIOMETRICS_PUSH_INTERVAL_SEC`.
- `services/orion-biometrics/app/main.py`: populate/forward `measurements_by_node`.
- `orion/schemas/telemetry/biometrics.py`: add `measurements_by_node` field.
- `services/orion-sql-writer/app/models/biometrics_cluster.py`, `app/main.py`: persist the new field.
- Tests: `services/orion-hub/tests/test_biometrics_preview_api.py`, `test_biometrics_view_ui.py`; `tests/test_fleet_measurements.py`, `test_pdu_proxy_polling.py`; `services/orion-biometrics/tests/test_measurements_by_node.py` (new), `tests/conftest.py` (new); `services/orion-sql-writer/tests/test_biometrics_cluster_sql_shape.py`.

## Schema / bus / API changes

- **Added**: `BiometricsClusterV1.measurements_by_node: Optional[Dict[str, Dict[str, float]]]` — per-node raw measurements, additive/optional, `model_config = ConfigDict(extra="ignore")` already on this model so old consumers tolerate it silently.
- **Removed**: none.
- **Renamed**: none.
- **Behavior changed**: `/api/biometrics/preview/snapshot` response gains a `cluster_measurements_by_node` key (`null` when the queried node doesn't run the cluster aggregator, i.e. always for circe). Websocket messages no longer carry a `biometrics` key at all.
- **Compatibility notes**: `BiometricsClusterSQL` gains `measurements_by_node` (JSONB, nullable) with its own boot-time `ADD COLUMN IF NOT EXISTS` migration — required before deploy or every biometrics-cluster insert will fail with `UndefinedColumn` the moment the model declares the field (same hazard documented inline for the existing `measurements` migration).

## Env/config changes

- Added keys: none.
- Removed keys: `BIOMETRICS_PUSH_INTERVAL_SEC` (orion-hub) — its only consumer (`biometrics_heartbeat`) was removed in this same patch.
- Renamed keys: none.
- `.env_example` updated: yes (`services/orion-hub/.env_example`).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes — run; `BIOMETRICS_PUSH_INTERVAL_SEC` also removed directly from the primary checkout's live `services/orion-hub/.env` (the sync script reads `.env_example` from the primary checkout, which doesn't have this branch's edit until merge, so a removed key had to be removed by hand). Re-ran the sync afterward: only two pre-existing, unrelated diverged keys remain (`GRAPHITI_ADAPTER_URL`, `HUB_AITOWN_WORLD_ID`).
- skipped keys requiring operator action: none.

## Tests run

```text
# repo-root telemetry (schema + proxy-merge)
PYTHONPATH="services/orion-biometrics" pytest tests/test_fleet_measurements.py tests/test_pdu_proxy_polling.py -q
36 passed

# orion-biometrics service
PYTHONPATH=".:services/orion-biometrics" pytest services/orion-biometrics/tests/ -q
149 passed, 2 pre-existing failures (test_node_catalog.py::test_circe_expected_offline,
test_biometrics_grammar_emit.py::test_circe_node_availability_reflects_expected_offline —
confirmed present on main, unrelated: live node_catalog.yaml expected_online value vs.
fixture expectation)

# orion-hub, biometrics-scoped
PYTHONPATH=".:services/orion-hub" pytest services/orion-hub/tests/test_biometrics_preview_api.py \
  services/orion-hub/tests/test_biometrics_node_client.py \
  services/orion-hub/tests/test_biometrics_view_ui.py \
  services/orion-hub/tests/test_substrate_biometrics_debug_api.py \
  services/orion-hub/tests/test_websocket_agent_claude_routing.py \
  services/orion-hub/tests/test_workflow_schedule_runtime_paths.py -q
88 passed

# orion-hub, full-suite collection (import-breakage check; full run times out at
# repo scale, unrelated to this patch)
pytest services/orion-hub/tests/ --collect-only -q
2044 tests collected, 0 import errors

# orion-sql-writer, biometrics-scoped
PYTHONPATH=".:services/orion-sql-writer" pytest services/orion-sql-writer/tests/test_biometrics_cluster_sql_shape.py \
  services/orion-sql-writer/tests/test_biometrics_summary_node_ts_index.py \
  services/orion-sql-writer/tests/test_biometrics_summary_sql_shape.py -q
21 passed, 1 pre-existing failure (test_biometrics_summary_sql_shape.py::test_every_payload_field_has_a_column —
confirmed present on main, unrelated model: BiometricsSummarySQL missing peak_pressure/
peak_pressure_channel columns — a real latent bug of the same shape as the one this PR
fixes for BiometricsClusterSQL, worth a follow-up, not touched here)

# orion-sql-writer, full-suite collection
pytest services/orion-sql-writer/tests/ --collect-only -q
491 tests collected, 0 import errors

# Verified the review finding directly: stashed the new SQL column, confirmed
# test_every_payload_field_survives_to_a_column_or_is_deliberately_mapped fails
# with "payload field 'measurements_by_node' has no column", restored the fix,
# confirmed it passes.
```

## Evals run

No dedicated eval harness exists for orion-hub, orion-biometrics, or orion-sql-writer's biometrics surface. Gate tests above are the coverage for this change.

## Docker/build/smoke checks

Not run — no live Docker daemon available in this session. Deterministic gate checks below cover what's runnable without live infra; a live smoke (curl `/api/biometrics/preview/snapshot?node=athena`, confirm `cluster_measurements_by_node` populated and non-degenerate) is still recommended before/after deploy.

```text
python scripts/check_metric_lineage.py --gate            -> PASS
python scripts/check_definition_drift.py --gate           -> PASS (0 changed)
python scripts/check_inner_state_registry.py              -> OK (15 entries)
python scripts/check_scripts_dir_no_stdlib_shadow.py       -> clean
python scripts/check_service_hostname_refs.py              -> OK
python scripts/check_compose_no_relative_mounts.py         -> PASS (83 files, 0 relative mounts)
python scripts/check_journal_dispatch_registry.py           -> OK
python scripts/check_sentience_instruments.py --static-only -> all claims hold
python scripts/check_system_health_producers.py             -> OK (11 sites)
python scripts/check_control_surface_store_parity.py        -> OK (3 services)
python scripts/check_daily_schedule_collisions.py          -> pre-existing, unrelated
                                                                (Daily Journal/Daily Pulse
                                                                08:30 collision), report-only
git diff --check                                            -> clean
```

## Review findings fixed

- Finding: `BiometricsClusterV1.measurements_by_node` was silently dropped by orion-sql-writer's only registered consumer of `orion:biometrics:cluster` (`_write_row` filters payload keys against `BiometricsClusterSQL`'s declared columns; the field had no column) — the per-node wattage breakdown never survived into history/audit queries.
  - Fix: added the `measurements_by_node` JSONB column to `BiometricsClusterSQL`, plus a boot-time `ALTER TABLE IF EXISTS orion_biometrics_cluster ADD COLUMN IF NOT EXISTS measurements_by_node JSONB;` migration in `app/main.py`'s lifespan (same pattern and same hazard already documented for the `measurements` column: a live table without the column would `UndefinedColumn` on every insert the moment the model declares the field, silently stopping ALL biometrics-cluster persistence).
  - Evidence: `test_every_payload_field_survives_to_a_column_or_is_deliberately_mapped` reproduced with the column stashed out (`AssertionError: payload field 'measurements_by_node' has no column`), confirmed passing with the fix restored; added `test_measurements_by_node_is_a_column_not_only_the_summed_total` and extended `test_write_row_actually_invokes_the_normalizer` to assert the field round-trips through the real `_write_row` path.
- Finding: the websocket handler kept computing a `BiometricsCache` snapshot (lock + dict copies + per-node construction) and attaching it as `payload["biometrics"]` on 20+ outgoing message sites, even though this same change removed the client's only reader of it (`updateBiometricsPanel`/`d.biometrics`).
  - Fix: `_with_biometrics()` gutted to a passthrough — no cache lookup, no attached key — rather than unwinding all 20+ call sites' `cache`/`biometrics_cache` parameter threading, which is a separate, larger, real-time-path-risk cleanup. `biometrics_heartbeat()`, the one caller whose *entire* job was this enrichment (a periodic `{"biometrics_tick": True}` push nothing reads), was fully removed instead — function, its spawn (`biometrics_task`), and its cancel — along with the now-dead `BIOMETRICS_PUSH_INTERVAL_SEC` env key across `settings.py`/`.env_example`/`docker-compose.yml`/README, synced into the live `.env`.
  - Evidence: `rg -n '\.biometrics\b' services/orion-hub/static/js/` — zero matches, confirming no client reader exists; `rg -n 'biometrics_tick|biometrics_heartbeat|biometrics_task'` post-fix — zero matches in `websocket_handler.py` outside one now-corrected docstring reference.

## Restart required

```bash
# orion-sql-writer first, so the new column exists before orion-biometrics
# publishes a payload carrying it (harmless either order, but this avoids a
# brief gap where a cluster event's measurements_by_node has nowhere to land).
docker compose --env-file .env --env-file services/orion-sql-writer/.env -f services/orion-sql-writer/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-biometrics/.env -f services/orion-biometrics/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: low
  Concern: circe's wattage in the UI depends on athena's `/snapshot` response specifically (the only instance whose PDU-proxy poller has live data); if athena's biometrics service is down, circe's wattage tile reads "—" even if circe itself is reachable.
  Mitigation: this is the existing, correct provenance behavior (circe has no BMC and no LAN path to the PDU) — documented in code comments at every layer touched; not a new failure mode introduced by this patch.
- Severity: low
  Concern: `services/orion-sql-writer` tests surfaced a pre-existing, unrelated defect of the same shape as the one this PR fixes — `BiometricsSummarySQL` is missing `peak_pressure`/`peak_pressure_channel` columns, so `BiometricsSummaryV1` silently drops those two fields on write.
  Mitigation: confirmed present on `main` independent of this patch; flagging as a follow-up rather than fixing here to keep this PR scoped to the reviewed diff.
- Severity: none (pre-existing, unrelated)
  Concern: `scripts/check_daily_schedule_collisions.py` reports a Daily Journal/Daily Pulse collision at 08:30 in `orion-actions`.
  Mitigation: report-only gate, unrelated to any file this PR touches.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2039
