## Summary

- A bus listener now notices when its connection has died silently. Before this, it waited forever.
- `OrionBusAsync.iter_messages` reads in 1 s slices. After 30 s of silence it sends PING on the listening connection. If nothing comes back in 15 s, it raises `PubSubSilentError` (a redis `ConnectionError`), and the caller's existing reconnect code resubscribes.
- Hunter and Rabbit (the shared bus listener loops) no longer cancel handlers that are still running when only the listening connection died. They still cancel them on stop.
- An RPC reply wait reports a dead reply line as `TimeoutError`, so each caller's existing timeout handling and metrics apply.

## Outcome moved

On 2026-11-02, circe's gpu-lane controller was deaf for about 13 h (05:45 to 18:28 UTC). Its listening socket still showed ESTABLISHED inside the container. The bus had no subscriber for it, and nothing was logged. The mesh guardian raised `gpu_controller_no_answer` for 92 checks in a row, and a manual restart fixed it. With this patch, the same failure fixes itself in 45 s or less, and it logs `Hunter subscriber loop failed ... pubsub silent`.

## Current architecture

`iter_messages` wrapped `pubsub.listen()`. That call blocks with no read timeout (`socket_timeout=None`). redis-py runs its health check only before a read, so on a socket where no message ever arrives, the health check never runs. Hunter's reconnect branch could not fire because no exception was ever raised.

## Architecture touched

- `orion/core/bus/async_service.py`: liveness check in `iter_messages`, `PubSubSilentError`, and `_iter_reply` (dead line becomes timeout) for `rpc_request` and `rpc_legacy_dict`.
- `orion/core/bus/work_queue.py`: `queue_rpc_request` uses `_iter_reply`.
- `orion/core/bus/bus_service_chassis.py`: Hunter and Rabbit keep in-flight handlers running on a transport failure.

## Files changed

- `orion/core/bus/async_service.py`: the fix.
- `orion/core/bus/work_queue.py`: RPC reply wait.
- `orion/core/bus/bus_service_chassis.py`: in-flight handlers survive a resubscribe.
- `orion/core/bus/tests/test_iter_messages_dead_socket.py`: new tests (dead line raises; quiet live line survives; pong buffered during a stall is not declared dead; RPC wait becomes timeout).
- `tests/test_hunter_keeps_inflight_on_transport_error.py`: new tests (handler survives a dead-line resubscribe; handler cancelled on stop).
- 5 test files: inline pub/sub fakes now imitate `get_message`, `ping` and `subscribed` instead of `listen()`.

## Schema / bus / API changes

- Added: `PubSubSilentError` (Python exception, not a schema).
- Behavior changed: `iter_messages` can raise `PubSubSilentError` on a dead line. It sends one PING per 30 s of idle time on each subscription.
- Compatibility notes: no payload or channel changes.

## Env/config changes

None.

## Tests run

```text
bus + chassis + rpc + work_queue + gpu_pool + hub relay: 682 passed
services: embodiment 123, actions 281, harness-governor 96, gpu-lane-controller 110, affectgpt-worker 43 passed
cortex-orch: 31 failed / 255 passed on branch vs 32 failed / 254 passed on main (pre-existing)
```

## Evals run

No eval harness covers bus transport. The live smoke below is the behavioral proof.

## Docker/build/smoke checks

```text
Live smoke: real redis-py -> TCP relay -> real bus (100.92.216.81:6379); the relay then swallows all traffic without closing.
redis-py 5.0.1 (repo venv) and 8.1.0 (mesh-guardian image), same result:
  1.0s got b'hello'
 10.0s alive after idle pings: True
 10.0s relay now silent
 13.0s ConnectionError after 3.0s: pubsub silent: no pong 3s after idle ping
(smoke used idle=3 s, deadline=3 s)
```

## Review findings fixed

- Finding: a `PubSubSilentError` cancelled in-flight concurrent handlers (for example, a cortex-exec turn that runs for minutes).
  - Fix: Hunter and Rabbit cancel in-flight handlers only on stop or cancel, not on a transport failure.
  - Evidence: `test_inflight_handler_survives_dead_line_and_resubscribe`, `test_inflight_handler_cancelled_on_stop`.
- Finding: RPC waits could raise a redis `ConnectionError`, which `except TimeoutError` callers do not catch.
  - Fix: `_iter_reply` converts it to `asyncio.TimeoutError`.
  - Evidence: `test_rpc_reply_wait_reports_dead_line_as_timeout`.
- Finding: an event-loop stall of 15 s or more right after the PING could expire the read before a pong that had already arrived was consumed, a false "dead" on a live line (the reviewer reproduced it 3/3).
  - Fix: before raising, drain once with `get_message(timeout=0)`.
  - Evidence: `test_buffered_pong_after_stall_is_not_declared_dead`.
- Finding: stale docstring on `_pubsub_redis_kwargs`.
  - Fix: updated.

## Restart required

Each service picks this up on its next rebuild from main. Nothing is urgent: the controller that hit this was restarted by hand on 2026-11-02 18:28 UTC and is subscribed again.

```text
No restart required.
```

## Risks / concerns

- Severity: medium
  - Concern: `orion/core/bus/resilience.py` `run_subscription` (about 30 services) and the shared RPC reply worker `_run_rpc_only` read with `get_message(timeout=1)` and have no pong deadline, so they are still deaf on a half-open socket. About a dozen scripts call `pubsub.listen()` directly.
  - Mitigation: follow-up PR ("other weak spots"). Reuse the same check there.
- Severity: low
  - Concern: one PING per 30 s per idle subscription.
  - Mitigation: negligible load on the bus.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3033

🤖 Generated with [Claude Code](https://claude.com/claude-code)
