# PR: recheck before paging on fresh substrate-runtime health degradation

Branch: `fix/substrate-health-recheck-debounce` → `main`

## Summary

- `reducer_cursor_commit_failing:biometrics_grammar_consumer` fired CRITICAL at 2026-07-13 16:55:42 UTC and self-resolved within minutes with no reproducing evidence -- the same "transient blip" signature already investigated and closed once before in the prior health-alerting PR (`3a4fe1df`).
- `HealthMonitor` pages on a single degraded `/grammar/truth` observation with no debounce. Given `grammar_poll_interval_sec=1.0`, a one-tick reducer-health blip self-heals on the very next poll -- long before the 900s health-check interval would ever see it recover -- yet it still fires a CRITICAL page with a 60-minute ack deadline and email escalation.
- Adds a bounded recheck before paging on a *fresh* transition (one not already backed by an open orion-notify alert): wait `SUBSTRATE_RUNTIME_HEALTH_RECHECK_DELAY_SEC` (default `15.0`) and confirm the condition is still true.
- Confirmation is cached per unhealthy streak (`_recheck_confirmed`) so a stuck retry (e.g. orion-notify itself unreachable) doesn't re-sleep or re-query Postgres on every tick -- only once per incident, as the docstring claims.
- A recheck that itself fails to complete (e.g. hits the same DB pressure that caused the degradation) fails toward alerting rather than silently treating an unconfirmable recheck as recovery.

## Outcome moved

A single self-healing reducer-health tick no longer produces a CRITICAL page with email escalation. A genuinely sustained incident still pages within `health_recheck_delay_sec` (default 15s) of the original detection -- functionally immediate relative to the 900s check interval.

## Current architecture

`app/health_monitor.py::HealthMonitor` is edge-triggered: it fires an `orion-notify` attention request only on a healthy->unhealthy transition (not every tick), plus a recovery note on the way back, and retries an undelivered alert every tick until `orion-notify` confirms delivery. It had no way to distinguish "unhealthy right now" from "was unhealthy for the whole 15-minute window since the last check" -- both looked identical to a single point-in-time `/grammar/truth` read.

## Architecture touched

- `services/orion-substrate-runtime/app/health_monitor.py` -- the only runtime logic change.
- `services/orion-substrate-runtime/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md` -- new `SUBSTRATE_RUNTIME_HEALTH_RECHECK_DELAY_SEC` setting, documented and threaded through.

## Files changed

- `services/orion-substrate-runtime/app/health_monitor.py` -- new `_confirm_still_unhealthy`, `_recheck_confirmed_or_confirm`, `_page_if_confirmed` methods; `run_tick` now routes both "fresh unhealthy" branches through `_page_if_confirmed` instead of publishing immediately.
- `services/orion-substrate-runtime/app/settings.py` -- `health_recheck_delay_sec: float` (alias `SUBSTRATE_RUNTIME_HEALTH_RECHECK_DELAY_SEC`, default `15.0`).
- `services/orion-substrate-runtime/.env_example`, `docker-compose.yml`, `README.md` -- new key documented and passed through; README explains the specific incident that motivated this.
- `services/orion-substrate-runtime/tests/test_health_monitor.py` -- autouse `time.sleep` patch fixture (so the suite doesn't actually block); 6 new tests: suppress-blip-on-first-observation, suppress-blip-on-healthy-to-unhealthy-transition, page-on-sustained-degradation, configurable-delay, no-resleep-on-retry, page-when-recheck-itself-raises.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: `HealthMonitor` now delays a fresh unhealthy page by up to `health_recheck_delay_sec` seconds to confirm it isn't a single-tick blip. An already-open alert (detected via `_has_open_alert`) still pages immediately, unchanged.
- Compatibility notes: none -- internal alerting behavior only, no external contract change.

## Env/config changes

- Added keys: `SUBSTRATE_RUNTIME_HEALTH_RECHECK_DELAY_SEC=15.0`.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not applicable -- this worktree has no local `.env` for this service (pre-existing pattern from the prior health-alerting PR). The key has a safe Python-level default (`15.0`), so the running container is not broken by its absence; sync via the script once this merges to main.
- skipped keys requiring operator action: none.

## Tests run

```text
POSTGRES_URI="postgresql://unused/unused" PYTHONPATH=. /tmp/orion-test-venv/bin/python -m pytest services/orion-substrate-runtime/tests/test_health_monitor.py -q
→ 19 passed

POSTGRES_URI="postgresql://unused/unused" PYTHONPATH=. /tmp/orion-test-venv/bin/python -m pytest services/orion-substrate-runtime/tests -q --ignore=services/orion-substrate-runtime/tests/test_grammar_consumer_integration.py
→ 122 passed, 11 failed (same 11 pre-existing failures confirmed identical on unmodified main; test_grammar_consumer_integration.py excluded, pre-existing unrelated app.models import collision)
```

## Evals run

No eval harness exists for this service's health-check behavior beyond the unit/regression tests above (same gap noted in the prior health-alerting PR).

## Docker/build/smoke checks

Not run against a live rebuild -- this is a pure Python logic change to an already-deployed module, verified via live production data (the actual incident, its DB-side correlation, and the current healthy `/grammar/truth` state) plus the unit test suite above. No schema/dependency/port changes requiring a compose smoke.

## Review findings fixed

- Finding: Recheck-before-paging re-ran the 15s sleep and a duplicate live-DB `run_checks()` call on *every* retry tick while `orion-notify` publish kept failing, not once per transition (contradicted the docstring and re-hammered Postgres during exactly the pressure window the feature exists to tolerate). Caught independently by both review agents.
  - Fix: Added `_recheck_confirmed: set[str]`, consulted by `_recheck_confirmed_or_confirm` so a streak is only recheck-confirmed once; cleared when the key is observed healthy again.
  - Evidence: `test_health_monitor_does_not_resleep_on_retry_after_confirmed_transition` -- 3 `run_tick()` calls across a publish-failing retry sequence, asserts `time.sleep` call count stays at 1.
- Finding: `_confirm_still_unhealthy`'s internal `run_checks()` call had no exception handling; if it raised (plausible under the same DB pressure causing the degradation), the exception propagated out of `run_tick()` uncaught, silently skipping the alert for a real, already-observed incident until the next 900s check.
  - Fix: Wrapped in try/except, fails toward alerting (mirrors `_has_open_alert`'s existing fail-open bias) -- an unconfirmable recheck is treated as "still unhealthy," not as recovery.
  - Evidence: `test_health_monitor_pages_when_recheck_itself_raises` -- recheck raises `RuntimeError`, asserts the alert still fires exactly once.
- Finding: The confirm→publish→set-state sequence was duplicated near-verbatim across the "first observation" and "healthy->unhealthy" branches, risking future drift between the two copies.
  - Fix: Extracted the shared `_page_if_confirmed(check)` helper, called from both branches.
  - Evidence: Diff review -- both branches now call the same method; all 19 tests pass.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-substrate-runtime/.env -f services/orion-substrate-runtime/docker-compose.yml up -d --build
```
Not urgent -- the current incident already self-resolved and the live process is healthy. Restart whenever convenient to pick up the debounce.

## Risks / concerns

- Severity: low. Root cause of the original `reducer_cursor_commit_failing` is not directly proven (old container's logs were gone on recreate before this investigation started) -- the Postgres statement-timeout correlation is circumstantial, not a confirmed causal chain. If this alert fires a third time, that would be strong evidence of a real, recurring issue (not just alerting noise) worth a deeper dive with the old container's logs preserved this time (`docker logs <container> > /tmp/<name>.log` before any recreate).
- Severity: none. The alert's `ack_note` in orion-notify itself did not get recorded due to an ack-payload field-name mismatch (`ack_actor`/`ack_note` vs. the schema's `actor`/`note`) on the first (successful, idempotent) ack call -- the resolution reasoning is preserved in this PR instead.

## PR link

`gh` is unauthenticated in this environment. Branch pushed: `fix/substrate-health-recheck-debounce`. Open via: https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/substrate-health-recheck-debounce
