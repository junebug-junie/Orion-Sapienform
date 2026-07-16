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
- `services/orion-hub/scripts/api_routes.py` — the HTTP `mode=="orion"` chat route now also carries the degraded reason.

## Files changed

- `orion/schemas/harness_finalize.py`: added `HarnessRunV1.finalize_degraded_reason: str | None = None`.
- `services/orion-harness-governor/app/bus_listener.py`: added `SubstrateAppraisalUnavailableError`; `_substrate_client` now wraps `TimeoutError` from the substrate RPC into it; new `except SubstrateAppraisalUnavailableError` branch builds a degraded-success `HarnessRunV1` (`final_text=motor.draft_text`, `finalize_ran=False`, `finalize_degraded_reason` set to a sanitized constant, `compliance_verdict` downgraded from "completed" to "partial"), emits a grammar lifecycle event (no fabricated appraisal/reflection molecules), and does **not** emit a `system.error` event (see review findings below).
- `orion/hub/turn_orchestrator.py`: new branch ahead of the existing hard-fail gate — when `finalize_degraded_reason` and `final_text` are both present, publish chat history and return `[turn_degraded frame, *success_frames]` instead of an error frame.
- `services/orion-hub/static/js/app.js`: new `turn_degraded` websocket message handler, yellow (non-red) system note.
- `services/orion-hub/scripts/api_routes.py`: the `mode=="orion"` HTTP chat route now merges the `turn_degraded` frame's reason onto the returned final-frame dict instead of silently dropping it via `frames[-1]`.
- `services/orion-harness-governor/tests/test_harness_governor_rpc.py`: regression test — substrate client raising `TimeoutError` degrades to draft-passthrough with the sanitized reason, downgraded compliance_verdict, and no system-error emission.
- `services/orion-hub/tests/test_turn_orchestrator_ws_frames.py`: regression test — a degraded `HarnessRunV1` produces `turn_degraded` + `final` frames (never `turn_error`), and chat history still gets published.
- `services/orion-hub/tests/test_handle_chat_request_orion_mode_degraded.py` (new): regression tests for the HTTP route's degraded-reason passthrough and normal-success no-op.

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

PYTHONPATH=services/orion-hub:. ./orion_dev/bin/python -m pytest services/orion-hub/tests/test_handle_chat_request_orion_mode_degraded.py -q
2 passed (new)

PYTHONPATH=services/orion-hub:. ./orion_dev/bin/python -m pytest services/orion-hub/tests/ -q
818 passed, 31 pre-existing failures (memory-consolidation/substrate-effect/recall-strategy/
llm-route-selector/fcc-model-labels suites — confirmed the same failures reproduce on
unmodified main in isolation; missing DB/schema fixtures in this sandbox, unrelated to
turn_orchestrator/finalize code touched here)
```

## Evals run

No dedicated eval harness exists for harness-governor or hub turn orchestration beyond the pytest suites above; none added — this is a scoped error-handling patch, not a quality/behavior dimension an eval would score.

## Docker/build/smoke checks

Not run — no config/dependency/Docker/port changes in this patch. Live diagnosis (not part of this patch's own smoke) confirmed the failure signature directly from running containers: `docker logs orion-athena-harness-governor` showed the exact `[rpc] timeout waiting for reply ... orion:substrate:finalize_appraisal:result:c881be8b-...` line this patch now degrades gracefully instead of hard-failing.

## Review findings fixed

`/code-review high` ran for real this time (8 independent finder-angle subagents + direct verification against the actual code, after an earlier attempt in this same session hit an account-level rate limit and had to be done manually instead). 8 findings reported; 5 fixed in this pass per explicit instruction, 3 carried forward as known/lower-severity.

- Finding: internal RPC error text (bus channel name + correlation UUID) leaked verbatim to the end user via `finalize_degraded_reason` -> `turn_degraded` frame -> rendered directly in the chat UI, on every single degraded turn.
  - Fix: `finalize_degraded_reason` now uses a sanitized module-level constant (`"substrate appraisal unavailable (RPC timeout)"`); the real exception detail stays in the operator-facing log line only.
  - Evidence: `test_harness_run_substrate_timeout_degrades_to_draft_passthrough` now asserts the channel name and correlation id are absent from `finalize_degraded_reason`.

- Finding: the new `emit_harness_finalize_system_error` call shares one global tension-rate-limit bucket (cap=3/60s, verified via `orion/autonomy/tension_ratelimit.py`'s `(kind, drive_names)` signature and `orion/autonomy/signal_tension.py`'s fixed `sdm.match("failure_event","severity")` rule) with every other failure_event across ~20 producer services. A sustained substrate outage — the exact scenario this patch targets — would exhaust that budget within 2-3 turns, silently starving out unrelated failure visibility mesh-wide.
  - Fix: removed the `emit_harness_finalize_system_error` call (and its now-unused import) from the degraded branch entirely; the warning log line is the durable operator-facing trace for this path instead.
  - Evidence: `test_harness_run_substrate_timeout_degrades_to_draft_passthrough` now asserts `channel_system_error` is absent from published channels.

- Finding: degraded turns never emitted `turn_outcome_molecule`/`post_turn_closure`/grammar `result_assembled`/`result_emitted` lifecycle events, unlike every other exception path in this function — verified `run_substrate_finalize_appraisal` (5a) sits with zero exception handling as the first statement of `run_harness_finalize_chain`, so `SubstrateAppraisalUnavailableError` propagates before any of those emissions.
  - Fix: added a `_emit_finalize_lifecycle_grammar` call (`status="degraded_passthrough"`, `reflection_ran=False`) to close the honest, no-fabrication-required grammar trace. Deliberately did **not** attempt `turn_outcome_molecule`/`post_turn_closure`: `emit_turn_outcome_molecule`'s `substrate_appraisal`/`reflection` params are non-optional and its `surprise_resolved` computation reads both — fabricating placeholder ones to force the emission would be exactly the empty-shell-cognition anti-pattern this repo prohibits. Documented in code comments as a deliberate scope boundary, not an oversight.

- Finding: `compliance_verdict=motor.compliance_verdict` passed through unmodified, unlike every other exception branch in this function (all force `compliance_verdict="failed"`) — so `compliance_verdict=="completed"` could now coexist with `finalize_ran=False`, a combination previously impossible here. Verified against `orion/harness/runner.py`'s own state machine that a real motor failure/refusal can *not* be smuggled through this way (both require empty `draft_text`, already hard-gated earlier), but "completed" alongside a skipped finalize chain would still mislead any wildcard consumer of `orion:harness:run:artifact` that treats `compliance_verdict=="completed"` as shorthand for "finalize genuinely ran."
  - Fix: downgrade `compliance_verdict` from `"completed"` to `"partial"` in the degraded branch (leaves `"partial"` as `"partial"` — already honest), restoring the invariant that `compliance_verdict=="completed"` never coexists with `finalize_ran=False` anywhere in this function.

- Finding: the HTTP (`mode=="orion"`) chat route (`services/orion-hub/scripts/api_routes.py`) returned only `frames[-1]`, silently discarding the `turn_degraded` notice for any caller not using the websocket (response text and chat history were unaffected — only the soft notice was lost).
  - Fix: the route now merges the `turn_degraded` frame's reason onto the returned final-frame dict as `finalize_degraded_reason`, matching the existing convention of carrying `finalize_ran` on that same dict.
  - Evidence: new `test_handle_chat_request_orion_mode_degraded.py` covers both the degraded and normal-success cases for this route.

Carried forward (lower severity, not fixed this pass):

- **Altitude**: only `TimeoutError` on the substrate RPC is treated as infra-unavailable; a late-but-garbled reply (decode failure or a `system.error` envelope, both raising plain `RuntimeError` in `orion/harness/substrate_client.py`) still hard-fails under the identical Fuseki-overload scenario. Likely intentional (a `RuntimeError` could also indicate a real substrate-side bug worth surfacing loudly), but worth confirming as deliberate rather than assumed.
- **Conventions**: this PR report's "Evals run" section reports the missing eval harness but doesn't separately file a tracked follow-up issue, which CLAUDE.md section 11 technically also requires ("report ... and create a follow-up issue"). Noting it here in lieu of a separate issue.
- **Simplification**: `_publish_unified_turn_chat_history(...)` is called identically from both the degraded branch and the normal-success branch in `turn_orchestrator.py` — a future kwarg change to that call has two sites to keep in sync. Left as-is since collapsing the two branches would touch more of the existing control flow than this patch's scope warranted.

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
- Mitigation: field has an explicit doc-comment in the schema explaining the exact semantics and pointing at how Hub uses it; the new tests encode the expected combination directly. Review additionally confirmed `compliance_verdict` needed the same protection (see findings above) — it's now downgraded to `"partial"` rather than left as `"completed"`, so the one other field a naive consumer might check is also honest.

- Severity: low
- Concern: reflection/appraisal are genuinely skipped for degraded turns — this is a real (if narrow and infra-triggered) change to Orion's cognition pipeline for that turn, not just error-handling polish.
- Mitigation: scoped tightly to `TimeoutError` on the substrate RPC only (verified all three raise sites in `orion/core/bus/async_service.py` are genuine "no reply within timeout" cases, never a masked connection/subscribe error); every other finalize failure mode (voice-finalize content failures, cortex reflection timeouts, malformed payloads) still hard-fails exactly as before. Scope was confirmed with Juniper before implementation given this touches the finalize/cognition chain.

## PR link

<!-- filled in after push -->
