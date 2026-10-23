## Summary

- Retro co-play handoffs now open and close a `play` act (actor `hub.retro_coplay`). Before this, co-play recorded no act at all.
- `watch_coplay` opens the act when it first sees a Zelda cognition policy, stamped at the handoff start. The memory export closes it: 1 with at least one real planner decision, else 0.
- The Retro gateway process, which runs every co-play handoff, now opens its own bus client in the background, with retry. Before, it had none (`bus_provider=lambda: None`).
- The export closes a co-play act only if this process published its open, so a handoff that never opened cannot grade a 0.
- Regression tests that fail on the old code (4 checked by restoring the old files).

## Outcome moved

Body-economy A5, item 6: "the retro emitter (0 acts)". Root cause, verified live and read-only on 2026-10-23:

- Every played session went through **co-play**: 7 `retro_play_sessions` rows of kind `co_play` on 10-23, with 2 to 7 planner decisions and about 3,490 Orion frames each. Co-play opened no act, by design (`retro_gateway.py`: `bus_provider=lambda: None  # co-play has no play act to close`; `retro_memory_export._close_play` returned when `dispatch_id` was empty).
- The only path that had an act was **leisure** (the motor allocator's `retro_leisure_session` dispatch). Its runner gate refuses almost every time. Since the Hub restart at 19:35Z the log shows `browser_connected` and `human_recent`. Exactly one leisure session ran in 14 days (10-23 16:53Z). It ended in 0.7 s (`policy_ended_by_retro`, 0 decisions). Its act did land: `act_980b36e2…`, closed 0 `no_planner_decisions`. So the leisure emitter works, but it almost never runs.

The emitter was never wrongly called and never failed to land. The path where play actually happens just had no emitter.

## Current architecture

- Leisure: dispatch → `retro_leisure_listener` → `act_emit.publish_play_opened` → export closes it.
- Co-play: browser handoff → `retro_routes.watch_coplay` (gateway process) → `RetroSessionEnded` → `retro_session_end_sink` → export. There was no act.

## Architecture touched

- Hub only: `scripts/act_emit.py`, `scripts/retro_routes.py`, `scripts/retro_memory_export.py`, `scripts/retro_gateway.py`, `scripts/main.py`.
- New seam `app.state.retro_session_start_sink`. It mirrors the existing `retro_session_end_sink`.

## Files changed

- `services/orion-hub/scripts/act_emit.py`: `publish_coplay_opened` / `publish_coplay_closed`, horizon 2400 s, and a stable uuid5 envelope correlation id.
- `services/orion-hub/scripts/retro_routes.py`: `emit_session_start`, called once at the first Zelda cognition.
- `services/orion-hub/scripts/retro_memory_export.py`: `coplay_key`, `make_session_start_sink`, the opened-key set, and the co-play close.
- `services/orion-hub/scripts/retro_gateway.py`: background bus with retry and backoff, a closed half-built client on failure, the start sink registered only when the exporter exists.
- `services/orion-hub/scripts/main.py`: registers the start sink.
- `services/orion-hub/README.md`: documents the co-play act.
- Tests: `tests/test_retro_memory_export.py`, `tests/retro/test_planner_v2.py`, `tests/retro/test_experience.py`.

## Schema / bus / API changes

- Added: actor `hub.retro_coplay` (verb `play`, signal `artifact_produced`) on the existing `orion:grammar:event` channel. `orion-hub` is already a listed producer. The act ledger has no actor allowlist, and the new actor gets its own cold posterior.
- Act key and correlation id: the `retro_play_sessions` row key `<retro session id>:<handoff start epoch>`. `test_before=retro_play_sessions:<key>`, `object_ref=retro_session:<retro sid>`.
- Removed / renamed: none.

## Env/config changes

- None. The gateway already loads the Hub `.env` (`ORION_BUS_URL`, `ORION_BUS_ENABLED`). Confirmed live: `ORION_BUS_URL` is in the running gateway container's env.

## Tests run

```text
cd services/orion-hub && PYTHONPATH=<wt> pytest -q tests/test_retro_memory_export.py tests/retro \
  tests/test_retro_leisure_listener.py tests/test_retro_memory_export_pg.py tests/test_act_ledger_producers.py
236 passed, 1 skipped
PYTHONPATH=<wt> pytest -q tests/retro tests/test_grammar_event_producer_catalog.py   30 passed
make check-body-economy-contract   OK (66 state-reading modules, all registered)
Mutation check: old scripts + new tests -> 4 FAILED (the four regression tests)
```

## Evals run

```text
None run. The live proof is the first co-play handoff after deploy (query below). evals/retro_coplay.mjs drives
handoffs end to end; not run here (needs the deployed gateway).
```

## Docker/build/smoke checks

```text
Not deployed (per brief). Post-deploy check, read-only:
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select actor, count(*), count(closed_at) from grammar_acts where verb='play' group by 1"
```

## Review findings fixed

- Finding (material): the export could close a co-play act that was never opened, either because the policy was gone on the first poll or because there was no bus at open time. The ledger would then grade a 0 for a handoff Orion never played.
  - Fix: a process-local set of opened keys. The close happens only for keys in that set.
  - Evidence: `test_a_co_play_handoff_never_opened_closes_no_act`, `test_coplay_handoff_gone_before_first_poll_opens_no_act`.
- Finding (minor): the gateway tried to connect the bus once, with no retry, and leaked a half-built client.
  - Fix: retry with backoff of 2 to 60 s; the client is closed on failure or cancel; the start sink logs `coplay_act_open_skipped reason=no_bus`.
  - Evidence: `test_gateway_act_bus_retries_and_closes_a_failed_client`.
- Finding (minor): the row key is not a UUID, so the open and close envelopes got different random bus correlation ids.
  - Fix: a stable uuid5 per key.
- Finding (nit): lazy imports ran on the relay loop.
  - Fix: pre-imported in `register_session_start_sink`.
- Finding (nit): `if started_at` treated 0 as missing.
  - Fix: `is not None`.
- Accepted (minor): `fill_close_cost`'s GPU lookup by turn correlation id never matches the row key, so it does one empty query per close. Retro play does not use the GPU pool.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build hub-app hub-retro-gateway
```

## Risks / concerns

- Severity: low.
  - Concern: if the gateway restarts mid-handoff, the open act is left. It expires at 0 after 2400 s, the same confound as leisure.
  - Mitigation: none needed. The expiry is honest.
- Severity: low.
  - Concern: the leisure path is still almost always refused by its gate. Co-play is now the only steady source of retro acts.
  - Mitigation: none in this PR. Out of scope (gate policy).
- Severity: low.
  - Concern: grading is UNVERIFIED live until the first post-deploy handoff.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2885
