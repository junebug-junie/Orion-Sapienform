# PR: health monitors for orion-attention-runtime and orion-self-state-runtime

Branch: `chore/attention-selfstate-health-monitoring`

## Summary

- Ports the edge-triggered health-monitor pattern from the `orion-field-digester` heartbeat-hardening PR (merged, #975) to the two other services sharing the identical 72h/hourly retention-pruning shape: `orion-attention-runtime` (`substrate_attention_frames`) and `orion-self-state-runtime` (`substrate_self_state` + `self_state_predictions` + `identity_snapshots`, pruned together).
- Each service gets one health check — "has my hourly pruner stalled?" — since neither has field-digester's extra unbounded-ledger problem (`substrate_field_applied_deltas`) or needs to duplicate its database-size check.
- Fixes the same docker-compose gap independently found in both services: retention/prune-interval env vars were documented in `.env_example` but never passed through in the compose `environment:` block, so both running containers were silently using code defaults (72.0h/3600s) regardless of `.env`.

## Outcome moved

- A stalled prune loop in either service (the same failure shape that motivated the original field-digester incident response) now pages Juniper via Hub's existing Pending Attention panel instead of silently recurring.
- `orion-self-state-runtime`'s check is `severity="critical"` — `substrate_self_state` is what `felt_state_reader.py` reads for chat context and what golden phi ticks off, making it the single highest-impact table in this cascade.

## Current architecture

Both services already had working, verified retention pruners (confirmed live and healthy via `pg_stat_user_tables` timestamps during the field-digester work) but zero alerting if the pruner itself silently stalled. Neither had any wiring to `orion-notify`.

## Architecture touched

- `services/orion-attention-runtime/app/{settings.py,store.py,worker.py}`, new `app/health_monitor.py`.
- `services/orion-self-state-runtime/app/{settings.py,store.py,worker.py}`, new `app/health_monitor.py`.
- Both services' `docker-compose.yml`, `.env_example`, `.env`, `requirements.txt`, `README.md`.

## Files changed

- `services/orion-attention-runtime/app/health_monitor.py` (new): one check, `attention_frame_prune_stalled`, severity `error`.
- `services/orion-attention-runtime/app/store.py`: `attention_frame_oldest_age_hours()` — `SELECT min(created_at)`, matching `PRUNE_ATTENTION_FRAMES_SQL`'s cutoff column exactly.
- `services/orion-attention-runtime/app/worker.py`: constructs `HealthMonitor` in `__init__`; adds `_health_loop` (same scaffolding as `_prune_loop`).
- `services/orion-attention-runtime/app/settings.py`: 4 new settings (`attention_frame_stall_multiplier`, `health_check_interval_sec`, `notify_base_url`, `notify_api_token`).
- `services/orion-attention-runtime/docker-compose.yml`: adds the previously-missing `ATTENTION_FRAME_RETENTION_HOURS`/`ATTENTION_FRAME_PRUNE_INTERVAL_SEC` passthrough, plus the 4 new keys.
- `services/orion-attention-runtime/requirements.txt`: adds `requests==2.31.0`.
- `services/orion-attention-runtime/README.md`: new "Health monitor" section.
- `services/orion-attention-runtime/tests/test_health_monitor.py` (new): 13 tests total in the suite (all passing).
- `services/orion-self-state-runtime/app/health_monitor.py` (new): one check, `self_state_prune_stalled`, severity **`critical`** (see reasoning above).
- `services/orion-self-state-runtime/app/store.py`: `self_state_oldest_age_hours()` — `SELECT min(created_at)`, matching `_prune_sql`'s cutoff column. Since `substrate_self_state`, `self_state_predictions`, and `identity_snapshots` are all written together in one `_tick()` and pruned together in one `prune_history()` call, checking the primary table alone is sufficient — a stall in one means a stall in all three.
- `services/orion-self-state-runtime/app/worker.py`: constructs `HealthMonitor` in `__init__` alongside existing store/policy/deviation-gate construction; adds `_health_loop`. Does not touch the existing perception-listener, bus-publish, or deviation-probe logic.
- `services/orion-self-state-runtime/app/settings.py`: 4 new settings (`self_state_stall_multiplier`, `health_check_interval_sec`, `notify_base_url`, `notify_api_token`).
- `services/orion-self-state-runtime/docker-compose.yml`: adds the previously-missing `SELF_STATE_RETENTION_HOURS`/`SELF_STATE_PRUNE_INTERVAL_SEC` passthrough, plus the 4 new keys.
- `services/orion-self-state-runtime/requirements.txt`: adds `requests==2.31.0`.
- `services/orion-self-state-runtime/README.md`: new "Health monitor" section (appended; existing sections — data flow, dependencies, inner-state registry — untouched).
- `services/orion-self-state-runtime/tests/test_health_monitor.py` (new): 37 tests total in the suite (all passing), including an explicit empty-table edge case.

## Schema / bus / API changes

- None. Reuses `orion-notify`'s existing `ChatAttentionRequest`/`GET /attention` contract, same as field-digester.

## Env/config changes

- Added keys (both services): `<SERVICE>_STALL_MULTIPLIER` (or `ATTENTION_FRAME_STALL_MULTIPLIER`/`SELF_STATE_STALL_MULTIPLIER`), `<SERVICE>_HEALTH_CHECK_INTERVAL_SEC`, `NOTIFY_BASE_URL`, `NOTIFY_API_TOKEN`.
- `.env_example` updated: yes, both services.
- local `.env` synced: yes, both services (mirrored `.env_example` values directly rather than the repo's `sync_local_env_from_example.py` script, since these keys aren't in that script's curated allowlist even with `--all-keys` scoped correctly this time — done by hand, confirmed both `.env` files match their `.env_example` key-for-key).
- Skipped keys requiring operator action: none.

## Tests run

```
$ .venv/bin/python -m pytest services/orion-attention-runtime/tests/ -q
13 passed in 0.54s

$ .venv/bin/python -m pytest services/orion-self-state-runtime/tests/ -q
37 passed in 2.39s
```

## Evals run

No eval harness exists for either service; this is wiring/infra-hardening, not a model-quality question (same as the field-digester PR's own note).

## Docker/build/smoke checks

```
$ docker compose --env-file .env --env-file services/orion-attention-runtime/.env \
    -f services/orion-attention-runtime/docker-compose.yml build
Image orion-attention-runtime-attention-runtime Built
$ docker compose --env-file .env --env-file services/orion-attention-runtime/.env \
    -f services/orion-attention-runtime/docker-compose.yml config --quiet
exit 0

$ docker compose --env-file .env --env-file services/orion-self-state-runtime/.env \
    -f services/orion-self-state-runtime/docker-compose.yml build
Image orion-self-state-runtime-self-state-runtime Built
$ docker compose --env-file .env --env-file services/orion-self-state-runtime/.env \
    -f services/orion-self-state-runtime/docker-compose.yml config --quiet
exit 0
```

**Not yet restarted live** — both images are built and validated, but the running containers have not been recreated with this change. Unlike the field-digester PR, this round did not include a live deploy step; see Restart required below.

## Review findings fixed

Built via two parallel subagents (one per service, disjoint files), each instructed to mirror the exact, already-reviewed field-digester pattern verbatim. I then independently verified every diff myself (read every changed file, did not just trust the agent reports) and re-ran both full test suites and both docker builds/config checks myself before committing.

- Finding (caught by the self-state-runtime agent itself during implementation, independently verified by me): the naive port of field-digester's `_check()` factory would format `age_hours:.1f` unconditionally, which crashes on `None` (empty table — a state where the check is trivially healthy, not unhealthy). Fixed by computing the message string inside an explicit `if stalled:` block before calling `_check()`, so `age_hours` is guaranteed non-`None` whenever it's formatted. Verified by reading the actual file content, not just the agent's description.
- Both agents correctly scoped their changes to their assigned service only — confirmed via `git status`/`git diff --stat` before staging; no cross-service or unrelated file bleed.
- Both agents correctly left `.env` unstaged/untouched by git (confirmed via `git check-ignore`).

## Restart required

Images are built and config-validated but **not yet applied**. To deploy:

```bash
docker compose --env-file .env --env-file services/orion-attention-runtime/.env \
  -f services/orion-attention-runtime/docker-compose.yml up -d --build

docker compose --env-file .env --env-file services/orion-self-state-runtime/.env \
  -f services/orion-self-state-runtime/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: low
  - Concern: restarting `orion-self-state-runtime` briefly interrupts the `orion:substrate:self_state` publish stream that phi/chat-context consumers depend on (a normal, few-second restart gap, same as any other redeploy of this service).
  - Mitigation: none needed beyond normal restart discipline; not deploying automatically in this PR pending explicit confirmation, given how central this service is.
- Severity: low
  - Concern: neither new health monitor's `_has_open_alert`/`_publish` HTTP calls have been exercised against a live `orion-notify` yet (unlike field-digester's, which was verified live post-deploy) — they're only unit-tested against mocks.
  - Mitigation: field-digester's identical code path was already verified live (first-boot health check ran cleanly with a real `orion-notify` call); these are byte-for-byte the same client usage, just a different `_SOURCE_SERVICE`/check key, so the residual risk is low, but real verification should happen at deploy time.

## PR link

Branch pushed to `origin/chore/attention-selfstate-health-monitoring`. `gh` was not authenticated in the environment for the prior PR either — open manually at:
https://github.com/junebug-junie/Orion-Sapienform/pull/new/chore/attention-selfstate-health-monitoring
