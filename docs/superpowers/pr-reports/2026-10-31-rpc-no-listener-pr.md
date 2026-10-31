## Summary

- A bus RPC that nobody hears now fails fast instead of waiting out its timeout while the caller holds a GPU lane.
- `OrionBusAsync.rpc_request(require_listener=True)` raises `RpcNoListener` in two cases:
  - **At publish:** `PUBSUB NUMSUB` on the request channel, read in the same `MULTI` as the `PUBLISH`, is 0. That means no worker received the request.
  - **While waiting:** every `listener_check_sec` (30 s), the request channel must still have a subscriber. Two zero readings a few seconds apart are required, because a listener that is only re-subscribing is still running the request.
- durable-runs opts in on all seven of its RPCs. A lost listener becomes an ordinary failed attempt: the hold is released and the attempt retried with the graph's backoff.
- A lost listener is recorded like a timeout (RPC health and transport grammar), so it stays visible.

## Outcome moved

**Incident (2026-10-31, verified from container logs):**
- durable-runs restarted at 00:51:48 and published two curiosity turns on `orion:curiosity:turn:request` at 00:52:26.
- The restarted Hub subscribed at 00:53:38. Pub/sub keeps nothing, so both requests were lost.
- Run `bad2deec1be8` held circe's agent lane on an 8940 s (2.5 h) RPC timeout. memory_distill was blocked behind it until the run was cancelled by hand about 35 min later.

**After this PR:** the same publish raises at once, the attempt fails, and the hold is released.

## Current architecture

`rpc_request` published and awaited the first reply up to `timeout_sec` with no notion of whether anyone heard the request.

## Architecture touched

- `orion/core/bus/async_service.py` (shared bus client; opt-in only)
- `services/orion-durable-runs/app/runner.py`

## Files changed

- `orion/core/bus/async_service.py`:
  - `RpcNoListener`
  - `_publish_counting_listeners` (MULTI PUBLISH + NUMSUB)
  - `_listener_count`
  - `_await_reply` (two-zero rule)
  - `_publish_rpc_request`
  - `_record_rpc_unheard`
  - `rpc_request(require_listener=False, listener_check_sec=30.0)`
- `services/orion-durable-runs/app/runner.py`: `require_listener=True` on all seven `rpc_request` calls.
- `orion/core/bus/tests/test_rpc_no_listener.py`: new.
- `services/orion-durable-runs/tests/test_runner_turn_contract.py`: the turn RPC opts in, and `RpcNoListener` gives `ok=False, error=rpc:RpcNoListener`.
- `services/orion-durable-runs/tests/{acceptance_bus,pool_fixture,test_pool_refusal_postgres}.py`: the fakes accept the new kwarg.

## Schema / bus / API changes

- Added: `rpc_request(..., require_listener=False, listener_check_sec=30.0)` and the `RpcNoListener(ConnectionError)` exception.
- Behavior changed: none for callers that do not opt in.
- Compatibility notes: `publish()` is unchanged. No envelope or channel changes.

## Env/config changes

- None.

## Tests run

```text
orion/core/bus/tests                                118 passed
  test_rpc_no_listener.py 6 tests; mutation-checked:
    publish-time check disabled  -> 1 failed
    two-zero rule disabled       -> 2 failed
    health/grammar record off    -> 2 failed
services/orion-durable-runs/tests (throwaway Postgres)  728 passed, 1 skipped
  (test_admission_review_regressions.py not collected locally: missing deps; CI runs it)
Real Redis (redis:7, throwaway), with a PSUBSCRIBE orion:* client like orion-bus-mirror:
  PUBLISH count                          -> 1  (pattern subscriber counted; never 0)
  MULTI [PUBLISH, NUMSUB], no worker     -> [1, [(ch, 0)]]
  MULTI [PUBLISH, NUMSUB], with worker   -> [2, [(ch, 1)]]
  real OrionBusAsync rpc_request(require_listener=True), no worker -> RpcNoListener at once
```

## Evals run

```text
None: transport behavior, covered by the real-Redis check above.
```

## Docker/build/smoke checks

```text
Deploy: durable-runs rebuild on athena (post-pull). The shared bus client change is opt-in; nothing else changes behavior.
```

## Review findings fixed

- Finding: the publish-time check was dead. PUBLISH counts pattern subscribers, and orion-bus-mirror PSUBSCRIBEs `orion:*`.
  - Fix: read NUMSUB in the same MULTI as the PUBLISH.
  - Evidence: real-Redis check above, and test `..._fails_at_once_though_the_mirror_heard_it`.
- Finding: a single zero reading while a listener re-subscribes (resilience backoff 1-60 s) would release a hold whose turn is still running.
  - Fix: require two zero readings a few seconds apart.
  - Evidence: `test_a_listener_only_resubscribing_is_not_lost`.
- Finding: the new failure was invisible to RPC health and the transport grammar.
  - Fix: record it like a timeout.
  - Evidence: assertions in the tests.
- Finding: an outer cancel could be swallowed during cleanup.
  - Fix: `asyncio.wait({task})` after cancel.
  - Evidence: `test_an_outer_cancel_still_cancels_the_caller`.

## Restart required

```bash
# athena: post-pull rebuild of orion-durable-runs (automatic on git pull of primary main)
```

## Risks / concerns

- Severity: low. Concern: a request channel with 2+ subscribers where only one restarts, or a worker restart faster than two polls, is not detected. Mitigation: only a worker-side "accepted" acknowledgement closes this. Follow-up, not needed for today's single-listener channels.
- Severity: low. Concern: a Hub down for more than ~10 min now fails a durable run after its 3 attempts instead of waiting 2.5 h. Mitigation: that is the point; the hold goes back to the pool.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3022

🤖 Generated with [Claude Code](https://claude.com/claude-code)
