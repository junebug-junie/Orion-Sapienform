# Listener resubscribe: services batch B

**Depends on the `feat/substrate-listener-health` PR.** That branch adds
`orion/core/bus/resilience.py::run_subscription`, which every change here calls.
Merge it first. This branch is built on top of it.

## Summary

- Nine long-lived bus listeners in seven services now come back after Redis drops.
  Before this, one Redis connection error or timeout ended the listener for good,
  and the process stayed up and looked healthy.
- Each listener now runs through `run_subscription`. It opens the subscription
  again with a backoff (1 s, doubling to 60 s), logs each failure, and records
  liveness under a unique listener name.
- Each site keeps its own decode, filter, and handler logic. Sites that started a
  task or a thread per message still do that and do not wait for it. Sites that
  handled messages in order still do that.
- One regression test per service. Each test makes the first subscribe raise
  `ConnectionError`, then proves the next session's message reaches the real
  handler. Each new test fails on the old code.

## Outcome moved

Failure mode: "a listener dies on one Redis error and nothing notices." After a
Redis restart or a network blip, these services now pick their channel up again
by themselves. Live incident that prompted this: 2026-10-20, orion-substrate-runtime
(fixed in the dependency PR).

Example: orion-energy ran its settlement listener inside `asyncio.gather`, so one
Redis error used to end the whole process. Now the listener retries.

## Current architecture

Each site used the same pattern: `async with bus.subscribe(ch) as pubsub: async for msg in bus.iter_messages(pubsub): ...`.
Nothing caught an error from the subscription itself. The coroutine ended, or its
thread ended. The process kept running.

## Architecture touched

Service code only. No contract, schema, channel, env, or Docker changes.

| Site | Listener name | Handler shape |
|---|---|---|
| `orion-energy` `settlement_loop` | `energy-power-settled` | Inline (awaits the lock and the publish, as before). No stop event in production, as before. |
| `orion-room-companion` `run_consumer` | `room-companion-claude-request` | Inline (the turn already runs in a thread; it was awaited before). |
| `orion-self-study-enrichment` `run_consumer` | `self-study-enrichment-requested` | Inline, sync handler (as before). |
| `orion-curiosity-peer` `run_help_consumer` | `curiosity-peer-help-request` | Inline, `await asyncio.to_thread(...)` (as before). |
| `orion-curiosity-peer` `run_consumed_consumer` | `curiosity-peer-brief-consumed` | Inline, `await asyncio.to_thread(...)` (as before). |
| `orion-whisper-tts` `stt_listener_worker` | `whisper-stt-intake` | Inline (transcription was awaited before). |
| `orion-whisper-tts` `listener_worker` (TTS) | `whisper-tts-intake` | Fire-and-forget. The handler starts the task and returns None. |
| `orion-voip-endpoint` `bus_listener_loop` | `voip-bus-command` | Inline, sync handler (as before). Runs in its daemon thread. |
| `orion-rag` `_listener_worker_async` | `rag-request` | Fire-and-forget. The handler starts a thread and returns None. |

`iter_messages` and `run_subscription` give the handler the same raw redis dict
(`type`, `channel`, `data`), so no handler needed a new decode step.

Other changes:

- `orion-rag`: the listener no longer calls `bus.connect()`. The subscription opens
  its own connection. A failed startup ping used to end the listener thread before
  any retry could run.
- `orion-voip-endpoint`: the outer `except` that published `listener_crashed` is
  gone. `run_subscription` does not let a subscription error escape, so that branch
  could no longer run. Failures now go to the `orion.bus.resilience` log.
- `settlement_loop`, `stt_listener_worker`, `listener_worker`, `bus_listener_loop`,
  and `_listener_worker_async` take an optional `stop` event. Production callers do
  not pass it. Tests use it to end the loop.

## Files changed

- `services/orion-energy/app/main.py`: settlement listener uses `run_subscription`.
- `services/orion-energy/tests/test_energy_settlement_resubscribe.py`: regression test.
- `services/orion-room-companion/app/main.py`: consumer uses `run_subscription`.
- `services/orion-room-companion/tests/test_consumer_resubscribe.py`: regression test.
- `services/orion-self-study-enrichment/app/main.py`: consumer uses `run_subscription`.
- `services/orion-self-study-enrichment/tests/test_consumer_resubscribe.py`: regression test.
- `services/orion-curiosity-peer/app/worker.py`: both consumers use `run_subscription`.
- `services/orion-curiosity-peer/tests/test_consumer_resubscribe.py`: regression tests (2).
- `services/orion-whisper-tts/app/stt_worker.py`, `app/tts_worker.py`: both intake listeners use `run_subscription`.
- `services/orion-whisper-tts/tests/test_listener_resubscribe.py`: regression tests (2). The TTS test also proves the listener does not wait on synthesis.
- `services/orion-voip-endpoint/app/bus_integration.py`: command listener uses `run_subscription`.
- `services/orion-voip-endpoint/tests/test_bus_listener_resubscribe.py`: regression test.
- `services/orion-rag/app/listener.py`: request listener uses `run_subscription`; no `connect()`.
- `services/orion-rag/tests/test_listener_resubscribe.py`: first test for this service (new `tests/` dir).

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: these listeners reopen their subscription after a Redis failure instead of stopping.
- Compatibility notes: same channels, same payloads.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

Each command uses `/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest -q`. PYTHONPATH
matches the CI workflow where one exists (energy, curiosity-peer). Other services
use `.:services/<svc>`.

```text
service                       baseline (origin/main 6b48ef9)   this branch
orion-energy tests            131 passed                       132 passed
orion-energy tests+evals (CI) -                                140 passed
orion-room-companion          54 passed                        55 passed
orion-self-study-enrichment   30 passed                        31 passed
orion-curiosity-peer          77 passed                        79 passed
orion-whisper-tts             85 passed                        87 passed
orion-voip-endpoint           5 passed                         6 passed
orion-rag                     (no tests dir)                   1 passed
orion/core/bus/tests/test_resilience.py                         8 passed
```

No failures exist at baseline.

Mutation check: each new test was run against the old version of its service file.
Every one failed (energy, voip, whisper, and rag also fail on the new `stop` keyword,
or on the removed `connect()`; room-companion, self-study, and curiosity-peer fail on
the `ConnectionError` itself).

## Evals run

```text
PYTHONPATH=.:services/orion-energy python -m pytest -q services/orion-energy/tests services/orion-energy/evals -> 140 passed
```

The other services have no eval harness that touches the bus listener. This is
a transport-reliability fix, and the regression tests cover the behavior.

## Docker/build/smoke checks

```text
Not run. No Dockerfile, compose, requirements, or env change. Live resubscribe after
a real Redis drop is UNVERIFIED until these services are redeployed.
```

## Review findings fixed

A review subagent read the full diff and found no material bugs. It checked
logic, `continue` -> `return`, stop conditions, handler shape, names, the STT
closure, rag without `connect()`, and voip without `listener_crashed` (nothing
consumes that event). It re-ran every suite.

- Finding (nit, pre-existing): the TTS listener did not keep a reference to the synthesis task it started. asyncio can garbage-collect such a task mid-run.
  - Fix: `tts_worker.py` keeps running tasks in `_inflight` and removes each one when it finishes.
  - Evidence: orion-whisper-tts 87 passed.
- Not fixed, reported:
  - Five of the new tests do not run in CI, because rag, whisper-tts, voip-endpoint, room-companion, and self-study-enrichment have no workflow. They run locally only.
  - Each new test waits about 1 s on the real backoff. The suites stay under 7 s.
  - The rag test sets env defaults and stubs `app.processor` for the whole pytest process. This is safe when each service runs alone, as it does today.
  - The STT test covers only the decode-failed path. It does not cover transcribe or publish.

## Restart required

Redeploy each service from the clean primary `main` checkout after both PRs merge:

```bash
scripts/safe_docker_build.sh orion-energy up -d --build
scripts/safe_docker_build.sh orion-room-companion up -d --build
scripts/safe_docker_build.sh orion-self-study-enrichment up -d --build
scripts/safe_docker_build.sh orion-curiosity-peer up -d --build
scripts/safe_docker_build.sh orion-whisper-tts up -d --build
scripts/safe_docker_build.sh orion-voip-endpoint up -d --build
scripts/safe_docker_build.sh orion-rag up -d --build
```

## Risks / concerns

- Severity: low
  - Concern: no service here exposes `subscription_health()` on its `/health` yet. A listener that keeps failing now retries and logs, but `/health` still does not report it.
  - Mitigation: follow-up to wire `subscription_health()` into each service's health surface.
- Severity: low
  - Concern: short-lived reply waits still use a bare `bus.subscribe` (`orion-curiosity-peer/app/worker.py` near line 376, `orion-rag/app/processor.py` near line 51). They wait for one reply per request, so they are not long-lived listeners and were left alone.
  - Mitigation: none needed; they end per request.
- Severity: low
  - Concern: five of the new tests have no CI workflow (rag, whisper-tts, voip-endpoint, room-companion, self-study-enrichment). A later regression would only show up in a local run.
  - Mitigation: follow-up to add those services to a workflow.
- Severity: low
  - Concern: orion-energy used to exit on a settlement-listener error, and a container restart policy would restart it. Now the listener retries in place. The other two loops in that `gather` are unchanged.
  - Mitigation: retry is the intended behavior.

## PR link

Not opened (branch pushed only, per task).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2790
