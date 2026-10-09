# fix(llm-gateway): a gateway-initiated cancel is `upstream_cancelled`, not the worker's `upstream_failed`

## Summary

- When the gateway hangs up on an upstream call itself (caller's time budget ran out, the GPU pool took the lease back, or the caller went away), the log no longer blames the worker. It used to write `upstream_failed ... error=upstream_error reason=RemoteProtocolError` at ERROR; it now writes `upstream_cancelled reason=<why>` at WARNING with `served_by`, `url`, `corr`, `elapsed_ms`, `after_cancel_ms`.
- The check sits at the top of all six exception handlers across the three places the backend opens an upstream HTTP client (`_execute_ollama_chat`, `_execute_llamacpp_native_completion`, `_execute_openai_chat`).
- What the caller receives, the lease release reason, and the stage 6.2 telemetry class are unchanged. Real worker failures (no gateway cancel) still log `upstream_failed` at ERROR exactly as #2536 made them.
- `UpstreamCancel` now records when the call started and when it was cancelled, plus a `cancelled_by_gateway()` helper.

## Outcome moved

False worker-blame ERROR lines on every gateway-side cancel. Evidence for the bug: `/tmp/gw-disconnect-2026-10-09/` — 16/16 `upstream_failed reason=RemoteProtocolError` lines landed 5-9 ms after `gateway_caller_budget_exhausted`. After this patch the same sequence produces one `upstream_cancelled reason=caller_budget_exhausted` WARNING and no ERROR (pinned by a real-socket test that reproduces the exact production line on the old code).

## Current architecture

`main._run_on_grant` runs `run_llm_chat` in the granted role's thread under `upstream_cancel.run_cancellable`. On budget exhaustion / lease revocation / caller cancel it calls `handle.cancel(reason)`, which `SHUT_RDWR`s the worker's sockets. The blocked httpx read raises, `llm_backend._upstream_exception_result` classified that as the upstream's failure and logged `upstream_failed` at ERROR. `main.py` then discarded that result and returned its own (`timeout`/`caller_budget_exhausted` or `gpu_pool_recalled`), and released the lease itself.

## Architecture touched

`services/orion-llm-gateway` only. No bus, schema, or env change.

Telemetry double-count check: the stage 6.2 recorder is called once per bus call in `main.handle_chat`, on the result `_dispatch_chat` returns — i.e. main.py's substituted reply. The backend's own failure result never reached it, so there was no double count before and none now (pinned by `test_bus_call_cancelled_on_budget_is_counted_once_as_upstream_timeout`, which passes on both old and new code). Race check: `_run_on_grant` keeps the backend result only when it has no error; any error result from a call it cancelled is replaced. The backend's cancel result still carries `raw.error` (mirroring main.py's class: `gpu_pool_recalled` for lease cancels, `timeout` otherwise) so that replacement keeps happening, and so it would classify as the cancel even if some future path forwarded it.

## Files changed

- `services/orion-llm-gateway/app/upstream_cancel.py`: `started_at` / `cancelled_at` on the handle; `cancelled_by_gateway()`.
- `services/orion-llm-gateway/app/llm_backend.py`: `_gateway_cancelled_result()`; called first in all six except blocks.
- `services/orion-llm-gateway/tests/test_gateway_cancel_not_upstream_failure.py`: new real-socket tests.
- `services/orion-llm-gateway/README.md`: the recall/loss step names the new log line.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: log line and level only (`upstream_failed` ERROR -> `upstream_cancelled` WARNING) for gateway-initiated cancels.
- Compatibility notes: anything grepping `upstream_failed` to count worker faults now counts real ones only.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
PYTHONPATH=$PWD .venv/bin/python -m pytest services/orion-llm-gateway/tests -q -p no:cacheprovider
427 passed

Regression proof (old llm_backend.py, new tests):
3 failed -- the two real-socket gateway tests fail on the exact production line
  ERROR [LLM-GW] upstream_failed backend=llamacpp error=upstream_error reason=RemoteProtocolError ...
```

New tests:
- caller budget exhausted on a real silent socket, real `run_llm_chat`: no `upstream_failed`, no ERROR record, one `upstream_cancelled` WARNING; reply still `timeout`/`caller_budget_exhausted`; lease released `timeout`; class `upstream_timeout`.
- lease lost on a real socket: same, reply `gpu_pool_recalled`/`lease_lost`, release `cancelled`.
- bus `handle_chat` with budget cancel: telemetry totals exactly `{"upstream_timeout": 1}`.
- each of the three executors (ollama, native completion, openai chat) on a real socket cancelled through `upstream_cancel`: `upstream_cancelled` only.
- genuine `RemoteProtocolError` (server closes without responding, no cancel): still `upstream_failed error=upstream_error reason=RemoteProtocolError` at ERROR, release `upstream_error`.
- the cancel result classifies as the cancel (`upstream_timeout` / `gpu_pool_recalled`) for every reason, never `upstream_error`.

## Evals run

```text
No eval harness for the gateway's log classification; the real-socket tests above are the behavioural check.
```

## Docker/build/smoke checks

```text
Not run: log-only change, no dependency/config/compose change. Live path UNVERIFIED until deployed.
```

## Review findings fixed

Code-review subagent: no must-fix; verified the thread-local handle is visible at except-time, no inner handler catches a cancel-caused exception first, main.py never forwards the backend result after a cancel, telemetry counted once.

- Finding: any exception after a cancel (incl. a real gateway bug like a KeyError in post-processing) was relabelled `upstream_cancelled` at WARNING, losing its traceback.
  - Fix: only `httpx.TransportError`/`OSError` count as the hang-up; anything else stays `gateway_exception` at ERROR with `exc_info`.
  - Evidence: `test_a_gateway_bug_after_a_cancel_stays_a_gateway_exception`.
- Finding: the three `except httpx.TimeoutException` cancel checks were never exercised (real sockets always end in RemoteProtocolError/ReadError).
  - Fix: parametrized tests over all three executors: timeout after cancel -> `upstream_cancelled`, no ERROR; timeout without cancel -> still `upstream_failed` at ERROR, `upstream_timeout`.
  - Evidence: 6 new test cases pass.
- Finding (nit): `elapsed_ms` included executor queueing.
  - Fix: `run_cancellable` resets `started_at` when the worker thread starts.
- Finding (nit): log line lacked the exception message.
  - Fix: `message=%r` (trimmed, same cap as `upstream_failed`).
- Finding (nit): narrow race (genuine drop caught just after a cancel) undocumented; test name mismatched.
  - Fix: documented in the helper docstring (`after_cancel_ms` near 0 marks it); renamed test and added a real cross-thread visibility test.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-llm-gateway up -d --build
```

## Risks / concerns

- Severity: low
- Concern: if a genuine upstream failure and a gateway cancel land within the same few milliseconds, that one call is logged as `upstream_cancelled`. main.py replaces the reply with the cancel's anyway, so only the log line is affected.
- Mitigation: none needed; the cancel is what the caller saw.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
