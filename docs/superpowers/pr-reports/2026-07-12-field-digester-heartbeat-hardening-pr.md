# PR: field-digester idle-tick heartbeat with retention hardening

Branch: `chore/field-digester-heartbeat-hardening`

## Summary

- Enables `FIELD_DIGESTER_IDLE_TICK_ENABLED` (the field→attention→self_state heartbeat pacemaker), previously left off after a prior unbounded-Postgres-growth incident on this host.
- Adds a pruner for `substrate_field_applied_deltas`, a dedup ledger with no natural "latest row" reader and (before this patch) zero pruning of any kind — confirmed at 1.69M rows / 373MB with 0% dead tuples before this change. Pruning is gated on the source receipt being confirmed gone from `substrate_reduction_receipts` (structurally correctness-safe), not a guessed time window.
- Adds an edge-triggered health monitor (field-digester-only) watching prune stalls, applied-deltas growth, and database size, alerting via `orion-notify`'s existing `POST /attention/request` (surfaces in Hub's Pending Attention panel) — reusing existing infra rather than building a new alert table/schema.
- Adds a supporting index (`idx_substrate_field_applied_deltas_applied_at`), applied live via `CREATE INDEX CONCURRENTLY`.

## Outcome moved

- `substrate_field_applied_deltas` went from 1,694,533 rows (unbounded growth since service inception) to ~5,400 rows immediately on deploy, confirmed reclaimed by autovacuum within 15s.
- The heartbeat chain (`substrate_field_state` → `substrate_attention_frames` → `substrate_self_state`) now ticks continuously every 2s regardless of chat/biometrics activity, verified live post-deploy.
- A stall/growth/size incident in this specific cascade will now page Juniper via Hub's existing attention panel instead of silently recurring.

## Current architecture

Field-digester polls `substrate_reduction_receipts` every 2s and digests them into `substrate_field_state`. Before this patch, `FIELD_DIGESTER_IDLE_TICK_ENABLED` was hard-disabled because turning it on would tick continuously with no pruning downstream. Retention pruners for `substrate_field_state`/`substrate_attention_frames`/`substrate_self_state` were added in a prior session and confirmed live and healthy (verified via `pg_stat_user_tables` timestamps and log output before starting this work) — but `substrate_field_applied_deltas` had no equivalent, and no health/alerting layer existed for any of it.

## Architecture touched

- `services/orion-field-digester/app/store.py`: new `prune_applied_deltas`, `health_snapshot` (combines 3 read-only checks into one round trip using `pg_stat_user_tables.n_live_tup` instead of `COUNT(*)` on a multi-million-row table).
- `services/orion-field-digester/app/health_monitor.py` (new): edge-triggered check runner + `HealthMonitor`, publishing via `orion.notify.client.NotifyClient`.
- `services/orion-field-digester/app/worker.py`: wires the new pruner into the existing hourly prune loop; adds a health-check loop.
- `services/orion-sql-db/manual_migration_field_digester_v1.sql`: new index.

## Files changed

- `services/orion-field-digester/app/store.py`: applied-deltas pruner (receipt-existence gated) + combined health-snapshot query.
- `services/orion-field-digester/app/health_monitor.py`: new edge-triggered health monitor with retry-until-delivered alerting.
- `services/orion-field-digester/app/worker.py`: wires both into existing loops; adds `_health_loop`.
- `services/orion-field-digester/app/settings.py`: 7 new env-backed settings; `enable_idle_tick` default flipped to `True` to match `.env_example`.
- `services/orion-field-digester/docker-compose.yml`: passes through all new keys (also fixed: `FIELD_STATE_RETENTION_HOURS`/`FIELD_STATE_PRUNE_INTERVAL_SEC` were never actually passed through before this patch, silently relying on code defaults).
- `services/orion-field-digester/.env_example`, `.env`: new keys, idle-tick flipped on, DB-size threshold set from observed baseline.
- `services/orion-field-digester/README.md`: rewritten "Idle tick" section reflecting the now-real guardrails.
- `services/orion-field-digester/requirements.txt`: added `requests` (used by `NotifyClient`).
- `services/orion-field-digester/tests/test_health_monitor.py` (new), `tests/test_worker_prune.py`: 34 tests total, all passing.
- `services/orion-sql-db/manual_migration_field_digester_v1.sql`: new `applied_at` index.

## Schema / bus / API changes

- Added: none (no new bus channels/schemas; reuses `orion-notify`'s existing `ChatAttentionRequest`/`GET /attention` contract).
- Removed: none.
- Renamed: none.
- Behavior changed: `FIELD_DIGESTER_IDLE_TICK_ENABLED` now defaults to `true` (was `false`) at every layer (Settings field default, docker-compose fallback, `.env_example`).
- Compatibility notes: none — additive only.

## Env/config changes

- Added keys: `FIELD_APPLIED_DELTAS_PRUNE_MIN_AGE_HOURS`, `FIELD_DIGESTER_HEALTH_CHECK_INTERVAL_SEC`, `FIELD_STATE_STALL_MULTIPLIER`, `FIELD_APPLIED_DELTAS_ALERT_ROW_COUNT`, `FIELD_DIGESTER_DB_SIZE_ALERT_GB`, `NOTIFY_BASE_URL`, `NOTIFY_API_TOKEN`.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced: yes, via `python3 scripts/sync_local_env_from_example.py --all-keys`. Note: the default (no-args) invocation reported "no changes needed" because these keys aren't in its curated sync-prefix allowlist; `--all-keys` was required. That run also touched 4 unrelated services (including wiping a live `GITHUB_TOKEN` secret in `orion-cortex-exec/.env`) — all reverted by hand before proceeding; only `orion-field-digester/.env` carries this PR's changes.
- Skipped keys requiring operator action: none.

## Tests run

```
$ .venv/bin/python -m pytest services/orion-field-digester/tests/ -q
34 passed in 0.59s
```

Additionally, the new raw-SQL store methods (no DB-integration-test harness exists in this service to extend) were manually verified against live production data before and after deploy — see Docker/build/smoke checks below.

## Evals run

No eval harness exists for this service; this is a wiring/infra-hardening change, not a model-quality question. Flagging per the repo's own convention (matches PR #970's own note on this point).

## Docker/build/smoke checks

```
$ docker compose --env-file .env --env-file services/orion-field-digester/.env \
    -f services/orion-field-digester/docker-compose.yml build
Successfully installed ... requests-2.31.0 ...
Image orion-field-digester-field-digester Built

$ docker compose --env-file .env --env-file services/orion-field-digester/.env \
    -f services/orion-field-digester/docker-compose.yml up -d
Container orion-athena-field-digester Started

$ curl -fsS http://localhost:8116/health
{"status":"ok","service":"orion-field-digester"}
```

Post-deploy live verification:
- `substrate_field_state.tick_id` advancing every ~2s (10 ticks in 20s window), zero errors in logs.
- Downstream cascade confirmed: `substrate_attention_frames`/`substrate_self_state` advancing in lockstep, zero errors in `orion-attention-runtime`/`orion-self-state-runtime` logs.
- `applied_deltas_pruned deleted=1689742 min_age_hours=1.0` logged on first tick; table confirmed at ~5,400 rows immediately after; `n_dead_tup` confirmed cleared to 0 by autovacuum within 15s (`last_autovacuum` timestamp advanced).
- Health-check loop executed on startup with no errors (first live call to `orion-notify`).
- New index confirmed used by the query planner (`EXPLAIN` showed `Index Scan using idx_substrate_field_applied_deltas_applied_at`).

## Review findings fixed

Ran an 8-angle code review (line-by-line, removed-behavior, cross-file, reuse, simplification, efficiency, altitude, conventions) via parallel subagents before deploy.

- Finding: health-monitor's in-memory transition state would go permanently silent on a restart during an ongoing incident (found independently by 3 of 8 angles).
  - Fix: first observation after startup now checks `orion-notify`'s own pending-attention list before deciding whether to suppress.
  - Evidence: `test_health_monitor_suppresses_first_observation_alert_when_notify_already_has_open_item`, `test_health_monitor_fires_first_observation_alert_when_no_open_item_found`.
- Finding: `NotifyClient.attention_request()` can return `ok=False` without raising; a failed delivery was silently treated as handled, permanently losing that transition.
  - Fix: a transition only commits to in-memory state once delivery is confirmed; otherwise it retries every check cycle.
  - Evidence: `test_health_monitor_retries_transition_alert_until_notify_confirms_delivery`.
- Finding: `DB_SIZE_ALERT_GB` default (20.0) was below the actual observed `conjourney` baseline (37.5GB) — would have false-alarmed immediately on deploy.
  - Fix: recalibrated default to 60.0GB with real headroom, documented as baseline-derived not a round guess.
  - Evidence: live `pg_database_size` query before setting the default.
- Finding: no index backed the new pruner's `applied_at` predicate on a 1.69M-row table, forcing full scans every hourly cycle.
  - Fix: added `idx_substrate_field_applied_deltas_applied_at`, applied live via `CREATE INDEX CONCURRENTLY`.
  - Evidence: `EXPLAIN` output confirming index-scan usage.
- Finding: `run_checks()` did 3 separate DB round-trips per 15-minute cycle, one an expensive `COUNT(*)` on a multi-million-row table.
  - Fix: combined into one query using `pg_stat_user_tables.n_live_tup` as a cheap estimate (fine against a 5-million-row alert threshold).
  - Evidence: live query re-verified single-round-trip after the change.
- Finding: inconsistent disable-semantics between the two pruners (`< 0` vs `<= 0`), a stale README default, a `generated_at`/`created_at` column mismatch between what the pruner acts on and what the health check measures, and a hardcoded database name in the alert message.
  - Fix: aligned all four.
  - Evidence: `test_prune_tick_always_prunes_applied_deltas_even_when_field_state_retention_is_zero`, updated store/health_monitor code, README diff.

Deferred (not material, noted for follow-up): a shared batched-delete helper and a shared `AttentionPublisher`-style component already exist in `orion-mesh-guardian` and could be generalized rather than duplicated — left as-is to keep this patch's blast radius contained. Also: **`orion-attention-runtime` and `orion-self-state-runtime` prune on the identical 72h/hourly pattern but have no equivalent health/alert wiring of their own** — this PR closes the gap for field-digester only; the same class of silent-stall risk still exists one and two hops downstream.

## Restart required

Already done as part of this change:

```bash
docker compose --env-file .env --env-file services/orion-field-digester/.env \
  -f services/orion-field-digester/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: medium
  - Concern: `orion-attention-runtime` and `orion-self-state-runtime` have no equivalent health monitoring, despite sharing the identical retention/pruning shape this PR just hardened for field-digester.
  - Mitigation: recommended as the next immediate follow-up; not blocking this PR since field-digester was the specific service being re-enabled here.
- Severity: low
  - Concern: `substrate_field_applied_deltas` still shows ~395MB on disk after the prune (dead tuples cleared, but `VACUUM FULL` would be needed to shrink the file itself).
  - Mitigation: none needed — reclaimed space is reusable by future inserts; growth going forward is bounded by the new hourly pruner.

## PR link

Branch pushed to `origin/chore/field-digester-heartbeat-hardening`. `gh` is not authenticated in this environment, so the GitHub PR itself was not opened automatically — compare/create at:
https://github.com/junebug-junie/Orion-Sapienform/pull/new/chore/field-digester-heartbeat-hardening
