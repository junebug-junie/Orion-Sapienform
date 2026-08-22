## Summary

- New durable decision log for endogenous outreach: every decision cycle (sent / orion_passed / blocked / generation-failed) now writes one row to a new Postgres table, not just `EndogenousOutreach._last_result` (in-process, wiped on restart) and `logger.warning` lines (wiped on container recreate, and in some branches never even called).
- New `GET /api/debug/endogenous-outreach/decisions` route reads the durable history.
- Code review found two real data-corruption races in the first draft (forced/tension_reason read from unlocked shared instance state); fixed by threading both explicitly through `_record()` as parameters captured locally at decision time.
- Live-verified end-to-end on `orion-athena-hub`, including one incidental, real finding: the outreach pipeline's `_generate()` call has been silently failing on `stance_react_failed: RPC timeout waiting on orion:exec:result:*` — a failure mode that produced **zero** docker log lines even before this patch, so the new table is the only place it was ever visible.

## Outcome moved

Diagnosing "why hasn't Orion reached out" no longer requires reconstructing indirect evidence from `substrate_field_state` after the fact — the real per-tick decision trail is now durable and queryable. This closes the exact gap that made the 2026-08-22 investigation (61 qualifying tension-trigger episodes since 2026-08-19 vs. 1 confirmed send) unable to distinguish "Orion keeps legitimately PASSing" from "generation is silently failing."

## Current architecture

`services/orion-hub/scripts/endogenous_outreach.py`'s `EndogenousOutreach._record()` was the single choke point every decision branch already funneled through, but it only ever wrote to `self._last_result` (in-process dict, `status()`-visible, wiped on restart) and emitted `logger.warning` on some (not all) failure branches.

## Architecture touched

- `services/orion-hub/scripts/endogenous_outreach.py` — `_record()` now persists via a new writer; `forced`/`tension_reason` threaded explicitly as parameters instead of read from shared instance state
- `services/orion-hub/scripts/endogenous_outreach_decisions.py` (new) — best-effort, fire-and-forget writer, mirrors `hub_presence.py`'s shape, reuses `scripts.pg_engine.get_engine()`
- `services/orion-hub/scripts/api_routes.py` — new `GET /api/debug/endogenous-outreach/decisions`
- `services/orion-sql-db/manual_migration_endogenous_outreach_decisions_v1.sql` (new) — applied live

## Files changed

- `services/orion-sql-db/manual_migration_endogenous_outreach_decisions_v1.sql`: new `endogenous_outreach_decisions` table (structured columns + full `result_json`)
- `services/orion-hub/scripts/endogenous_outreach_decisions.py`: new writer module
- `services/orion-hub/scripts/endogenous_outreach.py`: `_record()` hook, explicit `forced`/`tension_reason` threading (race fix)
- `services/orion-hub/scripts/api_routes.py`: new debug read route
- `services/orion-hub/app/settings.py`, `services/orion-hub/.env_example`: new `HUB_ENDOGENOUS_OUTREACH_DECISION_LOG_ENABLED` key (default true; env-first at runtime like `HUB_PRESENCE_WRITER_ENABLED`)
- `services/orion-hub/tests/test_endogenous_outreach_decisions.py`: new writer unit tests
- `services/orion-hub/tests/test_endogenous_outreach.py`: integration tests for the persist hook across every branch + a regression test reproducing the review-found race

## Schema / bus / API changes

- Added: table `endogenous_outreach_decisions`; route `GET /api/debug/endogenous-outreach/decisions`
- Removed: none
- Renamed: none
- Behavior changed: none to existing outreach behavior — purely additive observability
- Compatibility notes: table absent (migration not applied) degrades to a silent no-op write and `{"ok": false, "reason": "table_not_migrated"}` on read; no crash either way

## Env/config changes

- Added keys: `HUB_ENDOGENOUS_OUTREACH_DECISION_LOG_ENABLED` (default `true`)
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: yes
- local `.env` synced with `python3 scripts/sync_local_env_from_example.py`: yes
- skipped keys requiring operator action: none

## Tests run

```
/mnt/scripts/Orion-Sapienform/.venv/bin/python3 -m pytest services/orion-hub/tests/test_endogenous_outreach_decisions.py services/orion-hub/tests/test_endogenous_outreach.py -q
95 passed, 2 warnings

/mnt/scripts/Orion-Sapienform/.venv/bin/python3 -m pytest services/orion-hub/tests -q
1406 passed, 30 failed (pre-existing, unrelated -- confirmed by reproducing one, test_substrate_effect_pipeline.py::test_pipeline_handles_internal_failure_without_raising,
against the pristine main checkout with no worktree cd: same Settings ValidationError on missing CHANNEL_VOICE_* keys, an environment/invocation-context issue, not caused by this diff)
```

## Evals run

No eval harness exists for `orion-hub`'s endogenous-outreach subsystem; not added in this patch (observability-only change, no new cognition behavior to eval).

## Docker/build/smoke checks

```
scripts/safe_docker_build.sh orion-hub build            # clean build
scripts/safe_docker_build.sh orion-hub up -d --build     # deployed live (with explicit go-ahead, since it drops the live websocket connection)
python3 scripts/check_sql_migrations_applied.py --file manual_migration_endogenous_outreach_decisions_v1.sql
  -> [ok] applied (after applying the migration live via psql)

# Live smoke against the real running container:
docker exec orion-athena-hub curl -s http://localhost:8080/api/debug/endogenous-outreach/status
docker exec orion-athena-hub curl -s "http://localhost:8080/api/debug/endogenous-outreach/decisions?limit=5"
  -> real organic decision rows landing correctly (reason=no_tension_trigger)

docker exec orion-athena-hub curl -s -X POST http://localhost:8080/api/debug/endogenous-outreach/trigger
  -> {"outreach":false,"reason":"empty_generation","generation":{"error":"no_final_frame","frame_type":"turn_deferred",
      "detail":"stance_react_failed: RPC timeout waiting on orion:exec:result:...","elapsed_sec":129.875}}
  -- a real, previously-invisible failure, now durably captured
```

## Review findings fixed

- Finding: `self._last_forced` (unlocked shared instance state) could be overwritten by a concurrent forced debug-trigger call before the original tick's own `_record()` read it, persisting the wrong `forced` flag.
  - Fix: removed `self._last_forced` entirely; `forced` is now the `force` parameter already local to each call, threaded explicitly into every `_record()` call site.
  - Evidence: new regression test `test_concurrent_ticks_persist_their_own_forced_flag_not_each_others` reproduces the exact interleaving and passes.
- Finding: `self._last_tension_reason` read inside `_record()` after multiple `await` points could be clobbered by a concurrent tick's `_should_roll()`/force-clear, persisting the wrong (or a stale) tension episode.
  - Fix: captured into a local `tension_reason` variable immediately after `_should_roll()` returns, threaded explicitly through every subsequent `_record()` call in that coroutine instead of re-reading shared state.
  - Evidence: same regression test above; also `test_record_persists_a_forced_trigger_with_no_stale_tension_reason` (added in the base commit) still passes.
- Finding: the new writer built and disposed a private SQLAlchemy engine on every write instead of reusing `scripts.pg_engine.get_engine()`, which the sibling debug-read route in the same commit already used correctly.
  - Fix: switched to `pg_engine.get_engine()`.
  - Evidence: live-verified after redeploy — decision rows still land correctly.
- Finding: module docstring overclaimed writes "cannot write faster than roughly once every 5 seconds"; the `already_sending` bypass path is actually reachable at unbounded rate via the unauthenticated debug trigger.
  - Fix: corrected the docstring to disclose this as an accepted, not a settled, gap.
- Finding: decisions-list route hand-listed all 11 SELECT columns a second time to build the response dict.
  - Fix: build from `dict(row)`, special-casing only the two fields that actually need it.
- Finding: duplicated synchronous-thread-stub lambda across two tests.
  - Fix: extracted to one shared helper.
- Finding (no fix needed, confirmed intentional): `HUB_ENDOGENOUS_OUTREACH_DECISION_LOG_ENABLED` is a typed `Settings` Field but the runtime gate reads `os.getenv(...)` directly — this is the same, already-documented split `HUB_PRESENCE_WRITER_ENABLED` uses (env-first to keep the hot write path free of the full settings import), not a new inconsistency.

## Restart required

```bash
# Already done live as part of this PR, with explicit go-ahead (drops the live websocket connection momentarily):
scripts/safe_docker_build.sh orion-hub up -d --build
```

No further restart required — `orion-athena-hub` is already running the reviewed, fixed code.

## Risks / concerns

- Severity: low
- Concern: the `already_sending` decision-write path has no rate limit and is reachable via the unauthenticated `POST /api/debug/endogenous-outreach/trigger` while a slow generation is in flight (each hit spawns a thread + a pooled connection acquire).
- Mitigation: disclosed explicitly in the module docstring; every write is still best-effort/fire-and-forget with no unbounded in-process accumulation. This endpoint's unauthenticated nature predates this patch. A real rate limit is a fair, separate follow-up if it turns out to matter in practice.
- Severity: informational, not a concern with this PR — a genuine finding surfaced by it
- Concern: `_generate()` is failing with `stance_react_failed: RPC timeout waiting on orion:exec:result:*` on the live container right now, which is very likely the actual root cause of outreach's near-total silence since the 2026-08-19 pipeline rewrite (not the LLM legitimately declining).
- Mitigation/follow-up: root-causing the RPC timeout itself (likely in `orion.hub.turn_orchestrator`/`ThoughtClient.react()`'s wait on `orion:exec:result:*`, or the upstream service it's waiting on) is out of scope for this PR and is the natural next investigation — the new decision log is exactly what makes it now traceable instead of invisible.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1837
