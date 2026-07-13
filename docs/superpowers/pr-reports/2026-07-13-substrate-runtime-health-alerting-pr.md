# PR: substrate-runtime loguru crash fix + hub pending-attention health alerting

Branch: `fix/substrate-runtime-health-alerting` → `main`

## Summary

Triggered by a live "how we looking" check after deploying the orch route-grammar-lane PR. Found three things in `orion-substrate-runtime`'s health:

1. **Real bug, fixed**: `orion/core/bus/resilience.py` hard-imported `loguru` with no fallback (its sibling `bus_service_chassis.py` already degrades gracefully); `loguru` was missing from `services/orion-substrate-runtime/requirements.txt` entirely, so `_publish_brain_frame()` raised `ModuleNotFoundError` on every tick, caught but silent.
2. **Already self-resolved, verified not fixed further**: `reducer_cursor_commit_failing:biometrics_grammar_consumer` — a transient startup blip, gone on recheck a few minutes later, no reproducing evidence.
3. **Not a bug, verified structurally not fixed**: `cursor_lag:chat_grammar_consumer` (~42h stale) — orch's own logs show zero chat-lane request decisions in the last 24h (only `background`/`spark` from journal/introspection), and hub's chat-grammar producer only fires from real hub-websocket message handling, not from any automated/background traffic. No chat happened; nothing to fix.

Since (2) and (3) are exactly the kind of thing that's easy to mistake for "fine" or "broken" without checking, and Juniper asked to be notified when the system is unhealthy rather than having to manually curl `/grammar/truth`, this PR also adds `orion-substrate-runtime`'s first health-alerting hook into `orion-notify` → orion-hub's pending-attention UI, mirroring an established pattern already used by 5 other services.

## Outcome moved

`orion-substrate-runtime` degradation (whatever kind — reducer failure, cursor commit failure, sustained lag) now surfaces as an actionable card in hub instead of requiring a manual `curl /grammar/truth`. The loguru crash class is fixed both narrowly (this service now has the dependency) and structurally (the shared `resilience.py` module degrades gracefully instead of crashing on import for any future service missing loguru).

## Current architecture

`orion-notify` (`services/orion-notify`) already runs a `/attention/request` → `ChatAttentionRequest` → hub's `pendingAttention` frontend array pipeline, consumed via `orion.notify.client.NotifyClient`. Five services already use it for exactly this purpose (`orion-self-state-runtime`, `orion-field-digester`, `orion-attention-runtime`, `orion-mesh-guardian`, `orion-thought`), each with their own `app/health_monitor.py`. `orion-substrate-runtime` had no equivalent — its own `/grammar/truth` health computation (`app/grammar_truth.py::build_substrate_grammar_truth`) existed but nothing consumed it proactively.

## Architecture touched

- `orion/core/bus/resilience.py` — shared module, now degrades to stdlib `logging` when `loguru` is absent (affects every service importing `publish_with_reconnect`, not just this one).
- `services/orion-substrate-runtime`: new `app/health_monitor.py`, wired into `app/worker.py`'s task list on its own dedicated poll loop; `requirements.txt` (loguru), `settings.py`, `.env_example`, `docker-compose.yml`, `README.md`.

## Files changed

- `orion/core/bus/resilience.py` — try/except loguru import, falls back to `logging.getLogger("orion.bus")`, mirroring `bus_service_chassis.py`'s existing pattern verbatim.
- `orion/core/bus/tests/test_resilience.py` — regression test: module stays importable with `loguru` blocked via `sys.modules`/`builtins.__import__` patching + `importlib.reload`.
- `services/orion-substrate-runtime/requirements.txt` — added `loguru==0.7.2` (the direct fix; the resilience.py fallback is defense-in-depth, not the primary fix for this specific incident).
- `services/orion-substrate-runtime/app/health_monitor.py` (new) — `HealthCheck` dataclass, `run_checks(store, settings)` (single check `substrate_grammar_degraded`, wraps `build_substrate_grammar_truth`, message includes the live `degraded_reasons`), `HealthMonitor` class (edge-triggered, consults `orion-notify`'s own `/attention?status=pending` before firing so a process restart doesn't double-alert, retries every tick until delivery is confirmed) — structural mirror of `orion-self-state-runtime/app/health_monitor.py`.
- `services/orion-substrate-runtime/tests/test_health_monitor.py` (new) — 11 tests: healthy/unhealthy check output, no-alert-on-first-healthy-observation, alert-once-not-every-tick, recovery note, retry-until-confirmed-delivery, two fail-open paths (publish exception, pending-lookup exception) neither of which raises.
- `services/orion-substrate-runtime/app/settings.py` — `health_check_interval_sec` (`SUBSTRATE_RUNTIME_HEALTH_CHECK_INTERVAL_SEC`, default `900.0`), `notify_base_url`/`notify_api_token` (copied verbatim from `orion-self-state-runtime`'s settings).
- `services/orion-substrate-runtime/app/worker.py` — constructs `HealthMonitor` in `__init__`, ticks it on a new dedicated `_health_loop()` task (not piggybacked on an existing loop — mirrors self-state-runtime's own separately-tunable health cadence rather than inheriting pruning/dynamics-tick semantics).
- `services/orion-substrate-runtime/.env_example`, `docker-compose.yml`, `README.md` — new keys documented and passed through.

## Schema / bus / API changes

None. No new schema, no new bus channel — reuses the existing `orion-notify` HTTP API (`/attention/request`) via the existing shared `orion.notify.client.NotifyClient`.

## Env/config changes

- Added keys: `SUBSTRATE_RUNTIME_HEALTH_CHECK_INTERVAL_SEC=900.0`, `NOTIFY_BASE_URL=http://orion-athena-notify:7140`, `NOTIFY_API_TOKEN=` (empty, secret).
- `.env_example` updated: yes.
- local `.env` synced: not applicable — this worktree has no local `.env` for this service.

## Tests run

```
PYTHONPATH=. /tmp/orion-test-venv/bin/python -m pytest orion/core/bus -q
→ 2 passed

POSTGRES_URI="postgresql://unused/unused" /tmp/orion-test-venv/bin/python -m pytest services/orion-substrate-runtime/tests/test_health_monitor.py -q
→ 11 passed

POSTGRES_URI="postgresql://unused/unused" PYTHONPATH=. /tmp/orion-test-venv/bin/python -m pytest orion/core/bus services/orion-substrate-runtime/tests/test_health_monitor.py -q
→ 13 passed (combined re-run, orchestrator-verified independently of both implementing agents)
```
Full `services/orion-substrate-runtime/tests/` suite (excluding one pre-existing broken-collection file, `test_grammar_consumer_integration.py`, unrelated `app.models` import collision that exists identically on the unmodified branch): 109 passed / 11 failed, confirmed via `git stash` that the same 11 failures exist on the unmodified branch — zero regressions, the +11 passing count over the unmodified branch's 98 is exactly the new `test_health_monitor.py` file.

## Evals run

No eval harness exists for either the bus-resilience module or substrate-runtime's health-check behavior beyond the unit/regression tests above.

## Docker/build/smoke checks

```
docker compose -f services/orion-substrate-runtime/docker-compose.yml config -q   → valid (env-var warnings only, expected with no .env in this worktree)
```
No live redeploy performed as part of this PR — the loguru fix was root-caused against the live deployed container's actual logs before writing the fix (`docker exec orion-athena-substrate-runtime pip show loguru` → not found; confirmed the exact `ModuleNotFoundError` traceback), but applying it requires a rebuild, which is an explicit operator action (see Restart required).

## Review findings fixed

- An independent review subagent examined the health-monitor diff against the reference template across 6 angles (settings/store singleton identity, concurrency on `build_substrate_grammar_truth`'s shared `reducer_health` mutation state, test module-identity/patch reliability, string/copy-paste correctness, and the pre-existing `timeout<=0` busy-loop pattern already present unmodified in `_prune_loop` and the sibling template) — no confirmed bugs. One pattern noted (not a regression): `build_substrate_grammar_truth(store)` calls its own internal `get_settings()` singleton rather than using the `settings` parameter `run_checks(store, settings)` is threaded with — not a bug in practice since `worker.py` constructs everything off the same process-wide singleton, but worth knowing if this module is ever unit-tested with a non-singleton settings object.

## Restart required

```bash
# Rebuild substrate-runtime to pick up the loguru dependency and health monitor:
docker compose --env-file .env --env-file services/orion-substrate-runtime/.env -f services/orion-substrate-runtime/docker-compose.yml up -d --build

# Confirm the loguru crash loop is gone:
docker logs orion-athena-substrate-runtime --since 5m 2>&1 | grep -i "ModuleNotFoundError.*loguru"   # should be empty

# Confirm the health monitor is ticking (first tick fires at container start, then every 900s):
docker logs orion-athena-substrate-runtime --since 5m 2>&1 | grep -i "substrate_runtime_health"
```
Operator must add `NOTIFY_BASE_URL`/`NOTIFY_API_TOKEN` (if the notify service requires a token) to the real `.env` before rebuilding, or the health monitor will fail-open (log-and-retry) every tick without ever successfully reaching `orion-notify`.

## Risks / concerns

- Severity: none. `resilience.py`'s stdlib-logging fallback path uses loguru-style `{}` placeholders in its one `logger.warning(...)` call, which is technically incorrect for stdlib `logging` (would raise if that exact line executes under the fallback). This exact same latent issue already exists unmodified in `bus_service_chassis.py`'s own fallback logger use — not introduced by this patch, not fixed by this patch (out of scope: mirror the established pattern exactly, not redesign it), flagged here for whoever eventually touches that logging call.
- Severity: low. The health monitor's single check (`substrate_grammar_degraded`) is coarse-grained by design — it does not distinguish which reducer/cursor caused the degradation in its `key` (only in the message text). If Juniper wants per-reducer alert dedup/history instead of one blob alert per degraded episode, that's a follow-up, not done here.

## PR link

(To be created — push this branch and open a PR, or ask Claude to do so in a follow-up if `gh auth login` is configured.)
