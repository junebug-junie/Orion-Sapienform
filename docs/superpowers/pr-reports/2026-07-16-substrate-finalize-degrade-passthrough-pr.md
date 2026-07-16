# PR Report: substrate finalize-appraisal RPC timeout degrades gracefully instead of hard-failing the turn

## Summary

- Root-caused a live `Turn error (harness): RPC timeout waiting on orion:substrate:finalize_appraisal:result:*` — traced end to end to `orion-athena-fuseki`'s JVM (`-Xmx96g`) sitting at 94.88GiB/110GiB with 185% CPU (GC thrashing), 17 consecutive Docker healthcheck failures. Confirmed this is unrelated to the same-day `chore/kill-knowledge-forge` merge (PR #1088) — that PR touched zero bus channels/schemas per its own commit message and the graph, and Fuseki had already been running (and degrading) for 42 hours before it merged.
- Added a scoped fallback: when the harness-governor's substrate finalize-appraisal RPC times out specifically (infra unavailable, not a content/logic failure), the harness draft is passed through as the final response instead of hard-failing the turn.
- New `SubstrateAppraisalUnavailableError` in `orion-harness-governor` distinguishes this one failure mode from all other finalize-chain exceptions, which continue to hard-fail exactly as before.
- New optional `HarnessRunV1.finalize_degraded_reason` field carries the cause through the RPC reply/artifact so Hub can tell "degraded success" apart from "hard failure" without inventing a new sentinel on top of existing fields.
- Hub (`turn_orchestrator.py`) now treats a degraded run as a success: publishes chat history normally, returns a new `turn_degraded` frame (soft, non-red) instead of `turn_error`, then the normal success frames.
- `app.js` renders `turn_degraded` as a yellow system note: "Response delivered without reflection (\<reason\>)".

## Outcome moved

Previously: any finalize failure (including simple infra hiccups) hard-failed the whole turn, showed a red "Turn error" banner, and — critically — **never persisted the turn to chat history** (the early-return in `turn_orchestrator.py` happened before the chat-history publish call), so the user-visible partial-draft response was never actually remembered by Orion. Now: a substrate RPC timeout still delivers the response, still writes it to chat history, and only shows a soft non-blocking notice.

## Current architecture

`orion-harness-governor`'s `handle_harness_run_request` runs the harness motor (produces `draft_text`), then `run_harness_finalize_chain` (5a substrate appraisal → 5b reflection → 5c voice-finalize → 6b turn-outcome). The chain's own docstring: "The motor draft is not the final Hub response; this chain is what may change it." Any exception from that chain fell into one of two existing branches: `HarnessFinalizeFailedError` (partial state after 5a/5b succeeded, 5c failed) or a generic `except Exception` (anything else, including a bare 5a RPC timeout) — both set `final_text=None`, `finalize_ran=False`, and Hub's `execute_unified_turn` turned that into a hard `turn_error` frame whenever `not run.finalize_ran or not run.final_text`.

## Architecture touched

- `orion/schemas/harness_finalize.py` — `HarnessRunV1` (already registered in `orion/schemas/registry.py`; new field is optional/backward-compatible, no registry change needed).
- `services/orion-harness-governor/app/bus_listener.py` — new exception class, new except branch in `handle_harness_run_request`.
- `orion/hub/turn_orchestrator.py` — new gate in `execute_unified_turn`.
- `services/orion-hub/static/js/app.js` — new websocket frame handler (same pattern as the existing `turn_deferred` type; not a registered bus channel, this is Hub↔browser websocket protocol only).

## Files changed

- `orion/schemas/harness_finalize.py`: added `HarnessRunV1.finalize_degraded_reason: str | None = None`.
- `services/orion-harness-governor/app/bus_listener.py`: added `SubstrateAppraisalUnavailableError`; `_substrate_client` now wraps `TimeoutError` from the substrate RPC into it; new `except SubstrateAppraisalUnavailableError` branch builds a degraded-success `HarnessRunV1` (`final_text=motor.draft_text`, `finalize_ran=False`, `finalize_degraded_reason` set) and emits a `system.error` event (`phase="substrate_appraisal_unavailable"`) for durable observability before replying.
- `orion/hub/turn_orchestrator.py`: new branch ahead of the existing hard-fail gate — when `finalize_degraded_reason` and `final_text` are both present, publish chat history and return `[turn_degraded frame, *success_frames]` instead of an error frame.
- `services/orion-hub/static/js/app.js`: new `turn_degraded` websocket message handler, yellow (non-red) system note.
- `services/orion-harness-governor/tests/test_harness_governor_rpc.py`: new regression test — substrate client raising `TimeoutError` degrades to draft-passthrough, `finalize_ran=False`, `finalize_degraded_reason` set, system-error emitted.
- `services/orion-hub/tests/test_turn_orchestrator_ws_frames.py`: new regression test — a degraded `HarnessRunV1` produces `turn_degraded` + `final` frames (never `turn_error`), and chat history still gets published.

## Schema / bus / API changes

- Added: `HarnessRunV1.finalize_degraded_reason: str | None = None` (backward-compatible; old RPC replies without the field validate fine via `HarnessRunV1.model_validate(...)`, which is exactly how `HarnessGovernorClient.run()` decodes it in `services/orion-hub/scripts/harness_governor_client.py:142`).
- Added: `turn_degraded` websocket frame type (Hub↔browser protocol, not a registered `orion/bus/channels.yaml` channel — same category as the existing `turn_deferred`/`connection_ready` types).
- Removed: none.
- Renamed: none.
- Behavior changed: a substrate finalize-appraisal RPC timeout no longer hard-fails the turn; it delivers the harness draft as the final response with a soft notice.
- Compatibility notes: none needed — purely additive/optional field, scoped exception handling.

## Env/config changes

None. No new env keys; `.env_example` unaffected.

## Tests run

```text
PYTHONPATH=services/orion-harness-governor:. ./orion_dev/bin/python -m pytest services/orion-harness-governor/tests/test_harness_governor_rpc.py -q
10 passed

PYTHONPATH=services/orion-harness-governor:. ./orion_dev/bin/python -m pytest services/orion-harness-governor/tests/ -q
14 passed

PYTHONPATH=services/orion-hub:. ./orion_dev/bin/python -m pytest services/orion-hub/tests/test_turn_orchestrator_ws_frames.py -q
19 passed

PYTHONPATH=. ./orion_dev/bin/python -m pytest tests/test_unified_turn_schemas.py tests/test_unified_turn_bus_catalog.py orion/harness/tests/ -q
193 passed, 3 pre-existing failures (test_grounding_capsule_consumers.py x2, test_harness_runner.py x1 —
confirmed failing identically on unmodified main, unrelated to this patch)

PYTHONPATH=services/orion-hub:. ./orion_dev/bin/python -m pytest services/orion-hub/tests/ -q
814 passed, 33 pre-existing failures (memory-consolidation/substrate-effect/recall-strategy suites —
confirmed the same failures reproduce on unmodified main in isolation; missing DB/schema
fixtures in this sandbox, unrelated to turn_orchestrator/finalize code touched here)
```

## Evals run

No dedicated eval harness exists for harness-governor or hub turn orchestration beyond the pytest suites above; none added — this is a scoped error-handling patch, not a quality/behavior dimension an eval would score.

## Docker/build/smoke checks

Not run — no config/dependency/Docker/port changes in this patch. Live diagnosis (not part of this patch's own smoke) confirmed the failure signature directly from running containers: `docker logs orion-athena-harness-governor` showed the exact `[rpc] timeout waiting for reply ... orion:substrate:finalize_appraisal:result:c881be8b-...` line this patch now degrades gracefully instead of hard-failing.

## Review findings fixed

Review was run as direct multi-angle manual review (line-by-line, removed-behavior, cross-file tracer, reuse, simplification, efficiency, altitude, CLAUDE.md conventions) — the 8 code-review subagents dispatched for this all failed immediately on launch with "session limit · resets 8:20pm (UTC)" (an account-level rate limit, not a code issue), so the review was done directly instead of re-attempting the subagent dispatch.

No correctness bugs found. Two non-blocking observations carried forward rather than fixed (out of scope for what was asked):

- **Altitude**: every turn still pays the full `SUBSTRATE_FINALIZE_TIMEOUT_SEC` (default 5.0s) before degrading, for as long as the underlying outage persists — this patch stops the turn from hard-failing but doesn't add a circuit breaker to fail fast after repeated timeouts. Worth a follow-up if sustained substrate outages become common.
- **Efficiency**: the new `emit_harness_finalize_system_error` call publishes one `orion:system:error` event per degraded turn; during a sustained outage this is one event per turn, same cadence as the existing voice-finalize failure path already uses for the same channel — not a new problem this patch introduces, but worth knowing if that channel's consumers (`consumer_services: ["*"]` per `channels.yaml`) are sensitive to volume.

## Restart required

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-harness-governor/.env \
  -f services/orion-harness-governor/docker-compose.yml \
  up -d --build

docker compose \
  --env-file .env \
  --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml \
  up -d --build
```

Separately, and not part of this patch: `orion-athena-fuseki` is currently unhealthy (94.88GiB/110GiB heap, GC-thrashing, 17 consecutive failed healthchecks) — this patch stops that condition from hard-failing chat turns, but the underlying Fuseki memory pressure is still live and unaddressed. Juniper asked specifically for the pass-through fix in this session; Fuseki itself was not restarted or otherwise touched.

## Risks / concerns

- Severity: low
- Concern: `finalize_ran=False` combined with a UI-visible "successful" turn is a dual-meaning state — a future reader of `HarnessRunV1` in isolation could misread "finalize didn't run" as "this must be an error state" without knowing about `finalize_degraded_reason`.
- Mitigation: field has an explicit doc-comment in the schema explaining the exact semantics and pointing at how Hub uses it; the new tests encode the expected combination directly.

- Severity: low
- Concern: reflection/appraisal are genuinely skipped for degraded turns — this is a real (if narrow and infra-triggered) change to Orion's cognition pipeline for that turn, not just error-handling polish.
- Mitigation: scoped tightly to `TimeoutError` on the substrate RPC only (verified all three raise sites in `orion/core/bus/async_service.py` are genuine "no reply within timeout" cases, never a masked connection/subscribe error); every other finalize failure mode (voice-finalize content failures, cortex reflection timeouts, malformed payloads) still hard-fails exactly as before. Scope was confirmed with Juniper before implementation given this touches the finalize/cognition chain.

## PR link

<!-- filled in after push -->
