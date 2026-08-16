# cortex-exec: accept "agent" as a valid Compute llm_route override

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1644
Branch: `fix/cortex-exec-agent-route-override`

## Summary

Real root cause of the Muse Glimmer live-testing thread (#1634/#1636/#1640): Hub's "Compute" selector correctly sends `options.llm_route="agent"`, but `services/orion-cortex-exec/app/executor.py`'s accepted-override allowlist only recognized `{"chat", "quick", "metacog", "quick_background"}` — `"agent"` was silently rejected, falling through to verb-based default mapping, which never resolves to `"agent"` for a normal `chat_general` turn.

Confirmed live 2026-08-14: Hub **Mode: Quick + Compute: Agent** produced a response, but nothing reached Circe's dedicated agent-lane worker (Muse Glimmer).

## Outcome moved

Hub's Compute: Agent selector now actually reaches the agent route regardless of which Mode is selected, instead of silently falling back to verb-based defaults (quick/chat).

## Current architecture

Hub → Cortex (`orion-cortex-exec`) → LLM Gateway (`orion-llm-gateway`) → llama.cpp worker, over the bus. Hub's "Compute" dropdown (`hubComputeSelect`) sets `options.llm_route` on the outgoing `CortexChatRequest`. Cortex's `executor.py` (the ~1000-line-per-function core execution engine) reads that value, validates it against an allowlist, and if accepted, uses it as the `route` field on the `ChatRequestPayload` published to the gateway's bus intake. The gateway's `_resolve_route()` reads that top-level `route` field directly (its `LLM_LANE_ROUTING_ENABLED`-gated alternate resolution path is off by default and irrelevant here) and looks it up in `LLM_GATEWAY_ROUTE_TABLE_JSON`.

## Investigation trace (for the record — several dead ends ruled out first)

1. Two prior PRs (#1636, #1640) fixed real bugs in the gateway's route table (`agent` aliasing `chat`, then pointed at the wrong physical host) — necessary fixes, but not sufficient, because the request never reached the gateway's route-table lookup with `route="agent"` in the first place.
2. Checked `services/orion-llm-gateway`'s `LLM_LANE_ROUTING_ENABLED` — `false` by default in both `.env_example` and the live local `.env`. The lane-based `resolve_llm_lane_route()` mechanism (which reads `options.llm_lane`/`options.execution_lane`) is inactive.
3. Grepped the entire gateway service for `options.get("llm_route")` — zero matches. The gateway's active, non-lane `_resolve_route()` only reads `body.route`, `options.route`, `options.routing_key` — none of which match the key Hub actually sends (`llm_route`).
4. Traced one hop further back to Cortex, which turned out to be the actual bridge: `executor.py` reads `options.llm_route` and correctly sets it as `ChatRequestPayload.route` — but only if it passes an allowlist. That allowlist excluded `"agent"`. This is the real choke point.

## Architecture touched

- `services/orion-cortex-exec/app/executor.py`: allowlist + extracted helper.
- `services/orion-cortex-exec/tests/test_executor_llm_route_override.py`: new.

## Files changed

- `services/orion-cortex-exec/app/executor.py`: extracts the override-resolution logic (previously inline in a giant function with zero isolated test coverage) into `_resolve_llm_route_override()`, matching this file's existing pattern for testable extracted helpers (`_skip_journal_pageindex_for_automated_trigger`, same file). Adds `"agent"` to `_ACCEPTED_LLM_ROUTE_OVERRIDES`.
- `services/orion-cortex-exec/tests/test_executor_llm_route_override.py`: 8 new test cases.

## Schema / bus / API changes

- Added: none.
- Behavior changed: `options.llm_route="agent"` (or top-level `ctx["llm_route"]="agent"`) now resolves to the `agent` route instead of being silently rejected. Behavior for `chat`/`quick`/`metacog`/`quick_background` and their legacy aliases (`chat_quick`/`quick_chat`/`chat_kids_story` → `quick`) is unchanged.
- Compatibility notes: purely additive to the accepted set — nothing that previously worked stops working.

## Tests run

```text
$ pytest services/orion-cortex-exec/tests/test_executor_llm_route_override.py tests/test_executor_journal_pageindex_skip.py -q
12 passed

$ pytest services/orion-cortex-exec/tests/ -q
13 pre-existing collection errors (ValueError: Verb already registered) across
unrelated test files -- a global verb-registry singleton re-registers on every
test-file import when the full suite collects together. Confirmed identical on
unmodified origin/main before this change; not caused by or related to this fix.
```

No larger functional test of the full `executor.py` step-execution flow was written: the enclosing function is ~1000+ lines with heavy runtime dependencies (bus client, LLM gateway RPC client, step/context objects) and no existing test harness exercises it directly — extracting and testing the specific buggy logic in isolation (matching this file's own established pattern) was the right-sized fix, not standing up a new integration-test harness for a one-line allowlist bug.

## Evals run

Not applicable — routing logic, not model behavior.

## Docker/build/smoke checks

Not run from this dev environment (no Docker/GPU access). See "Restart required".

## Review findings fixed

- `/code-review medium` against `fix/cortex-exec-agent-route-override`: the `llm_route_selected` log line's `override=%s` field always logged `None` for a rejected/unrecognized override after the extraction, whereas the original inline code logged the raw normalized value even when it was invalid — collapsing "no override supplied" and "override supplied but rejected" into the same signal. That distinction is exactly what would let a future analogous bug (a typo, or another value missing from the allowlist) be traced from logs the same way this one was.
  - Fix: `_resolve_llm_route_override()` now returns `(accepted, attempted)` instead of a single value — `accepted` is `None` unless the override passed the allowlist (used for routing), `attempted` is the raw normalized value whenever one was supplied at all, regardless of acceptance (used for logging). The `llm_route_selected` log line now emits both `override=` (accepted) and `override_attempted=` (raw) — strictly more diagnostic than the original single field, not just restored.
  - Evidence: `test_unrecognized_value_rejected_but_still_visible_as_attempted` — asserts `_resolve_llm_route_override({"options": {"llm_route": "bogus"}}) == (None, "bogus")`.

## Restart required

```bash
docker compose -f services/orion-cortex-exec/docker-compose.yml up -d --build
# Then in Hub: Compute: Agent (any Mode) should now actually reach the agent route.
# Full chain to verify Muse Glimmer specifically also needs #1636 + #1640's
# gateway route-table fixes already live (pointing "agent" at Circe).
```

## Risks / concerns

- Severity: low
  - Concern: no live end-to-end verification yet that a Hub turn with Compute: Agent actually reaches Muse Glimmer after this fix — this fix addresses the Cortex-level allowlist specifically; the full chain also depends on the gateway route table (#1636/#1640) being live on whichever host runs it.
  - Mitigation: none beyond restarting cortex-exec and retesting from Hub; recommend checking the gateway's `gateway_llm_route_selected` log line (see #1640's investigation) for `route=agent served_by=circe-worker-agent-1` to confirm the full chain.
- Severity: low
  - Concern: `executor.py`'s full test suite has pre-existing collection errors unrelated to this change, so CI-style "run everything" coverage for this service is already degraded independent of this PR.
  - Mitigation: none proposed here — flagged as a known, pre-existing gap, not something to silently work around in this PR's scope.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1644
