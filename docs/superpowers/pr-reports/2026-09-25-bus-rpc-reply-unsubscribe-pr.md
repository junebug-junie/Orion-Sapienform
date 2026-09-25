# fix(bus): release RPC reply-channel subscriptions after each call

## Summary

- Every service that asks another service a question over the bus got its answer on a "reply channel", and the shared RPC listener subscribed to that channel and never let go. Callers that make up a fresh channel name per request (GPU pool leases, exec/cortex results, council, state, embeddings) leaked one Redis subscription per call, forever.
- `OrionBusAsync.rpc_request()` (worker path) now unsubscribes the reply channel when the call ends -- reply, timeout, publish error, or cancellation -- unless another in-flight call is still waiting on the same channel (refcount over `_pending_rpc`, under the existing `_rpc_lock`).
- The RPC worker now holds one permanent "anchor" subscription (`orion:rpc:worker-anchor:<uuid>`, nothing publishes to it), so its connection never drops to zero subscriptions. Without it, redis-py would run a health-check read inside the next SUBSCRIBE and collide with the worker's own blocked read (found in review, see below).
- The hub's harness governor client, which drives the same worker machinery by hand (`_run_via_worker`), releases its per-turn channel the same way.
- The ad-hoc fallback path in `rpc_request()` and `rpc_legacy_dict()` already unsubscribe/close via the `subscribe()` context manager -- verified, unchanged.
- `orion/gpu_pool/client.py` and the llm-gateway are deliberately untouched (another PR owns them); the shared fix covers their leak.

## Outcome moved

Live bus Redis (2026-09-25) had 6,668 subscribed pubsub channels: `orion:gpu_pool:reply:*` 2,719, `orion:exec:result:*` 1,428, `orion:gpu_pool:state:*` 768, `orion:cortex:result:*` 737, `orion:council:reply:*` 380, `orion:state:reply:*` 299, `orion:embedding:result:*` 169, ... Growth was tens of thousands per day per busy caller, and each worker reconnect re-sent `SUBSCRIBE` for the whole accumulated set.

Real-Redis smoke (throwaway `redis:7-alpine`, real redis-py 5.0.1, real `OrionBusAsync` worker, 200 sequential + 100 concurrent unique-channel RPCs + 50 concurrent on one stable channel + 1 timeout):

```text
== fixed
lingering smoke:reply:* channels: 0 bus _rpc_subscribed: 0
post-idle rpc ok; worker alive: True
== old code
lingering smoke:reply:* channels: 302 bus _rpc_subscribed: 302
```

The live bus count itself is UNVERIFIED until the listed services are rebuilt (not deployed in this PR).

## Current architecture

`OrionBusAsync.start_rpc_worker()` runs one pubsub connection (`_run_rpc_only`) that dispatches replies to futures keyed `(reply_channel, correlation_id)` in `_pending_rpc`. `_rpc_subscribe()` subscribed a channel once and added it to `_rpc_subscribed`; nothing ever removed it. Reconnect re-subscribed all of `_rpc_subscribed`.

## Architecture touched

- `orion/core/bus/async_service.py`: new `_rpc_release(reply_channel)` (under `_rpc_lock`: no-op if any `_pending_rpc` key still names the channel or it is not subscribed; otherwise drop from `_rpc_subscribed` and `UNSUBSCRIBE`, never raises) and `rpc_release_reply_channel()` (shielded wrapper for `finally` blocks, so cancellation mid-release still completes it). `rpc_request()` worker-path `finally` calls it after popping its own key. During an outage (pubsub connection down) release skips the network call rather than dialing Redis while holding the lock, and clears redis-py's own channel bookkeeping so redis-py cannot quietly re-subscribe a released channel. Per-worker anchor channel subscribed on the worker's first subscribe, on reconnect, and by the idle worker loop.
- `services/orion-hub/scripts/harness_governor_client.py`: `_run_via_worker` `finally` calls `bus.rpc_release_reply_channel(reply_to)`.

Correctness under concurrency: the release decision and the subscribe decision are both made under `_rpc_lock`, and a caller registers in `_pending_rpc` before taking the lock. So a second caller on the same channel is either already counted (channel kept) or subscribes again after the release (channel re-added before its publish). redis-py 5 `subscribe()` clears the channel from `pending_unsubscribe_channels`, so a late UNSUBSCRIBE confirmation cannot drop a re-subscribed channel. The anchor keeps redis-py's `PubSub.subscribed` True after every reply channel is released (checked on real Redis: `subscribed=True`, channels = just the anchor; without the anchor: `subscribed=False`). That matters because redis-py 5 `PubSub.execute_command()` passes `check_health = not self.subscribed`.

## Files changed

- `orion/core/bus/async_service.py`: release seam + call site.
- `services/orion-hub/scripts/harness_governor_client.py`: release after hand-rolled worker RPC.
- `tests/test_bus_async_rpc_worker.py`: 8 new regression tests; the fake pubsub now models redis-py's zero-subscription health-check race (raises on SUBSCRIBE when it has zero channels and a read is in flight).
- `services/orion-hub/tests/test_harness_governor_client_liveness.py`: fake worker bus gains `rpc_release_reply_channel`; worker-path test asserts the channel it subscribed is released.
- `docs/superpowers/pr-reports/2026-09-25-bus-rpc-reply-unsubscribe-pr.md`: this report.

## Schema / bus / API changes

- Added: `OrionBusAsync.rpc_release_reply_channel()` (in-process method; no bus contract). One subscribe-only channel per RPC worker, `orion:rpc:worker-anchor:<uuid>`: never published to, so not a catalog entry (the catalog governs publishes). It shows up in `PUBSUB CHANNELS` once per running worker.
- Removed / Renamed: none.
- Behavior changed: reply channels are unsubscribed after each worker-path RPC. Callers that reuse a stable reply channel re-subscribe on each call (subscribe happens before publish, so no reply can be missed).
- Compatibility notes: no channel, schema, or payload change. `channels.yaml` / registry untouched.

## Env/config changes

- Added / Removed / Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced: not needed (no template changed).
- skipped keys requiring operator action: none.

## Tests run

```text
pytest tests/test_bus_async_rpc_worker.py -q                                   -> 17 passed
  (first-commit test set against the pre-fix async_service.py                  -> 6 failed, 9 passed)
  (anchor disabled, release kept                                               -> 4 failed, incl. the new
   anchor/idle-race test)
pytest <10 root test files touching OrionBusAsync/rpc_request> -q              -> 163 passed
  (tests/test_execution_dispatch_runtime_worker.py fails collection on base too; excluded)
static gates run locally (metric_lineage, definition_drift, service_hostname_refs,
  async_routes_not_blocking, system_health_producers, chat_route_poachers,
  grammar_event_producer_catalog)                                              -> all pass
cd services/orion-hub && pytest tests/test_harness_governor_client_liveness.py -q -> 10 passed
  (with the pre-fix harness_governor_client.py                                 -> 1 failed)
cd services/orion-thought && pytest tests/test_rpc_health_publish.py -q        -> 8 passed
```

New tests: idle worker keeps its anchor across calls and after a reconnect, and a later SUBSCRIBE never hits the race; release during an outage makes no network call; 20 sequential unique-channel RPCs leave 0 subscriptions; stable channel reused 3x works and is released each time; two concurrent RPCs on one channel keep it until the second finishes; timeout path releases; cancelled RPC releases; 10 concurrent RPCs on one channel all succeed and end at 0.

## Evals run

```text
No eval harness exists for orion/core/bus. The real-Redis before/after smoke above is the behavioral check.
```

## Docker/build/smoke checks

```text
Throwaway container redis:7-alpine on 127.0.0.1:16399 (not the bus), removed after.
Fixed: 0 lingering channels, worker alive after idle. Old: 302 lingering.
Idle-gap smoke (15 calls, health check forced due, worker blocked in read before each):
  ok=15 err=0 lingering=0 reconnects=0; pubsub.subscribed stays True (anchor only).
  This end-to-end smoke did NOT trigger the race even without the anchor (it is timing-
  dependent); the reviewer reproduced it with raw redis-py. The anchor removes its
  precondition, which is what the smoke verifies.
No service image rebuilt; not deployed (per task).
```

## Review findings fixed

Code review ran in a subagent against the first commit.

- Finding (material): once all reply channels are released the worker pubsub has zero subscriptions, so redis-py's next SUBSCRIBE runs a health-check PING + read on the socket the worker is already reading -> `RuntimeError('read() called while another coroutine is already waiting ...')`, the RPC fails and the connection is torn down. The reviewer reproduced this against a real Redis with redis-py 5.0.1. Expected about 1 in 30 RPCs after an idle gap (`health_check_interval=30`).
  - Fix: a permanent per-worker anchor subscription, so `subscribed` never becomes False; subscribed with the first reply channel, on reconnect, and by the idle worker loop before it starts reading.
  - Evidence: `test_worker_keeps_anchor_so_idle_resubscribe_cannot_race_the_read` (the fake raises exactly this error), which fails with the anchor disabled; on real Redis `subscribed` stays True with the anchor and goes False without it.
- Finding (material): during an outage, release called `unsubscribe()` on a dead connection, and redis-py would dial Redis (up to 10s connect timeout) while holding `_rpc_lock`, blocking every new RPC and the worker's reconnect.
  - Fix: skip the network call when the connection is None or not connected. The Orion reconnect re-subscribes only the anchor plus `_rpc_subscribed`.
  - Evidence: `test_release_during_outage_does_not_dial_redis_under_the_lock`.
- Finding (nit): a failed UNSUBSCRIBE left the channel in redis-py's `pubsub.channels`, so redis-py's own reconnect could silently re-subscribe it.
  - Fix: `_rpc_forget_pubsub_channel()` clears `channels` / `pending_unsubscribe_channels` on failure or skip.
- Finding (nit, deferred): stable reply channels now cost SUBSCRIBE + UNSUBSCRIBE per call. Correct. The cost is small, so it is not optimized here.
- Finding (info): if a caller is cancelled while its shielded release waits for the lock, the caller sees CancelledError. That is standard asyncio behavior, and there is no deadlock path. No other code bypasses the release (only the harness governor client touches `_pending_rpc`, and it is fixed here).

## Restart required

The `orion/` package is baked into each image by `COPY orion ...` in the service Dockerfile (80 services; only vision-frame-router and vision-retina also bind-mount it). So a rebuild, not a restart, is required.

Services that actually run the RPC worker (via `fork_rpc_client` / `fork(start_rpc_worker=True)` / `start_rpc_worker()`) and therefore carry the leak -- rebuild these for the fix to take effect:

```text
orion-actions
orion-chat-memory
orion-cocreation-signals
orion-context-exec
orion-cortex-exec
orion-cortex-gateway
orion-cortex-orch
orion-hub            (also gets the harness_governor_client change)
orion-llm-gateway
orion-vision-council
```

```bash
for s in orion-actions orion-chat-memory orion-cocreation-signals orion-context-exec \
         orion-cortex-exec orion-cortex-gateway orion-cortex-orch orion-hub \
         orion-llm-gateway orion-vision-council; do
  scripts/safe_docker_build.sh "$s" up -d --build
done
```

Every other service that copies `orion/` picks the change up harmlessly on its next routine rebuild (it never starts the RPC worker, so the new code path is inert there). After rebuild, verify with `redis-cli -u redis://100.92.216.81:6379/0 PUBSUB NUMSUB`-style counts: `PUBSUB CHANNELS 'orion:gpu_pool:reply:*' | wc -l` should stop growing and fall to roughly the number of in-flight calls. Existing leaked subscriptions disappear when each rebuilt container's old connection closes.

## Risks / concerns

- Severity: low. Concern: one extra `UNSUBSCRIBE` round trip per RPC on the worker connection (and one `SUBSCRIBE` per call for stable-channel callers, which previously subscribed only once). Mitigation: both are sub-millisecond Redis commands on an already-open connection; replaces an unbounded subscription set.
- Severity: low. Concern: a reply arriving after its caller timed out now hits an unsubscribed channel and is dropped by Redis instead of by `_handle_rpc_result` -- same outcome as before (no pending future). Mitigation: none needed.
- Severity: low. Concern: `rpc_release_reply_channel` is shielded; if the caller is cancelled during release, the release finishes as its own task. Mitigation: `_rpc_release` never raises.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2336
