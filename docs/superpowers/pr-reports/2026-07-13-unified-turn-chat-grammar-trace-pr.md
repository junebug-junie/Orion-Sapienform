# PR: unified-turn chat grammar trace + stance disposition signal

Branch: `feat/unified-turn-grammar-trace` → `main`

## Summary

Direct follow-up to a correction Juniper caught: I'd concluded the `chat_grammar_consumer` cursor's 42h staleness meant "no chat traffic," based on `--since 24h` docker logs against containers that had actually only been up ~30 minutes (post-deploy restart). That conclusion was wrong on two counts: the log window was invalid, and — verified from source code, not logs — the real chat pipeline (`orion/hub/turn_orchestrator.py::execute_unified_turn`, the "unified turn" path every `orion` mode message actually goes through) never called the chat-grammar-emit code at all. The grammar-emit code only ever existed in the older `websocket_handler.py` classic path, which is now a dead branch for `orion` mode traffic specifically (still live for other client modes, so left untouched).

Rather than just restore parity, this adds something the old path never captured: the **Thought stance decision** — `proceed`/`defer`/`refuse`, with reasons and a boundary-register flag. That's Orion choosing whether and how to engage with a given turn, and until this patch it had zero representation anywhere in the substrate ladder.

## Outcome moved

`chat_grammar_consumer` will advance again on real unified-turn traffic. The substrate ladder gains a genuinely new signal (self-regulatory/boundary decisions) it never had before, captured for every turn including the ones Orion declines to engage with — not just the ones that reach a successful answer.

## Current architecture

Five things confirmed by reading the actual code, not inferring from logs:
1. `websocket_endpoint` in `websocket_handler.py` routes `orion` mode + `ORION_UNIFIED_TURN_ENABLED=true` to `run_unified_turn` → `execute_unified_turn`, with an explicit `continue` — never reaching the classic path's `build_chat_request`/grammar-emit block further down in the same function.
2. The classic path's grammar-emit block (`services/orion-hub/scripts/websocket_handler.py` ~line 1136) is still live code for other client modes — not dead, not deleted here.
3. `execute_unified_turn` has a real Thought-stance gate: `ThoughtClient(bus).react(...)` → `thought.disposition in ("proceed", "defer", "refuse")`, with `defer`/`refuse` returning early before any harness call.
4. `orion-harness-governor` (the service that runs the actual FCC motor/finalize chain for unified turns) is already a registered producer on the `execution_trajectory` reducer (`cortex.exec:` prefix) — so step/plan mechanics for unified turns are already captured elsewhere in the ladder. This patch deliberately does not duplicate that; it only covers the conversational-envelope + stance layer, which is `chat_grammar`'s actual scope.
5. `orion/substrate/chat_loop/grammar_extract.py::extract_chat_turn_state` recognized exactly three semantic roles (`user_utterance`, `repair_signal`, `session_context`) and `ChatTurnStateV1` had no field to hold a stance fact even if one were emitted — a new atom without reducer + schema support would have been an "empty-shell" addition, present in the raw event log but invisible in the materialized projection.

## Architecture touched

- `orion/hub/turn_orchestrator.py` — new `_publish_unified_turn_chat_grammar` helper, called at all three of `execute_unified_turn`'s stance-resolution exit points (stance-RPC-timeout, defer/refuse, proceed).
- `services/orion-hub/scripts/grammar_emit.py` — `build_chat_turn_grammar_events` extended with an optional `stance_disposition` atom + edge, mirroring the existing `repair_signal` atom's conditional-emission pattern exactly.
- `orion/substrate/chat_loop/grammar_extract.py` + `orion/schemas/chat_projection.py` — reducer-side parsing and `ChatTurnStateV1` schema extension so the new atom is actually materialized, not just logged.

## Files changed

- `orion/hub/turn_orchestrator.py` — `_publish_unified_turn_chat_grammar` (async, awaited directly — not scheduled as a background task, unlike the classic path's `_schedule_publish`, because this file's own established convention already awaits its other publish calls (`publish_chat_history`, `publish_chat_turn`) directly, and the added latency of one lightweight bus publish is negligible next to the harness governor call it precedes). Wired at all 3 exit points so `defer`/`refuse` turns are captured, not just successful ones.
- `services/orion-hub/scripts/grammar_emit.py` — `stance_disposition`/`stance_disposition_reasons`/`stance_boundary_register` params; new optional atom (`semantic_role="stance_disposition"`, `layer="organ_signal"`, `atom_type="signal"`, `text_value` holds the disposition string directly — mirroring `session_context`'s pattern of using a structured field rather than regex, since disposition is a small enum with no privacy concern) plus a `derived_from` edge to `user_utterance`, matching `repair_signal`'s existing edge pattern.
- `orion/substrate/chat_loop/grammar_extract.py` — parses the new atom (`text_value` for disposition, a light regex for the parenthetical reasons list, a literal substring check for `[boundary_register]`) into the three new state fields.
- `orion/schemas/chat_projection.py` — `ChatTurnStateV1` gains `stance_disposition: str = "unknown"`, `stance_disposition_reasons: list[str] = []`, `stance_boundary_register: bool = False`. Additive with defaults — backward-compatible with existing stored `ChatSessionProjectionV1` rows (JSONB, no migration needed).
- `services/orion-hub/tests/test_hub_grammar_emit.py` — 3 new tests: stance atom absent when not provided, present with correct summary/text_value when provided, edge links to `user_utterance`.
- `services/orion-hub/tests/test_turn_orchestrator_ws_frames.py` — 3 new tests directly exercising `_publish_unified_turn_chat_grammar`: no-op when `PUBLISH_HUB_CHAT_GRAMMAR` is off, stance fields carried through to the built events when on (plus confirms `user_utterance.text_value is None` — raw text never stored), and fail-open on a publish exception.
- `tests/test_chat_substrate_reducer.py` — 3 new tests for `extract_chat_turn_state`'s new parsing: stance present with reasons + boundary register, defaults to `"unknown"` when the atom is absent, no-reasons case doesn't choke on the word-count atom's own parentheses (`"(N words)"`) since role-gating happens before the reasons regex runs.
- `services/orion-hub/README.md` — new subsection under "Chat & Cognition" documenting the unified-turn grammar trace, right after the existing repair-pressure documentation since they're now emitted together.

## Schema / bus / API changes

- Added: `ChatTurnStateV1.stance_disposition`/`stance_disposition_reasons`/`stance_boundary_register` (additive, defaulted).
- Added: `stance_disposition` semantic role on the existing `hub.chat:` grammar trace (no new atom_type, no new event_kind — `"signal"`/`"atom_emitted"` already existed in the schema's enums).
- No new channel, no new cursor, no new reducer, no new env key, no migration.
- Compatibility: fully additive on both the producer and consumer side. Existing consumers of `ChatTurnStateV1` that don't know about the new fields are unaffected (`model_config = ConfigDict(extra="forbid")` only forbids *unexpected* extra keys on incoming data, not missing-with-default fields).

## Env/config changes

None. Reuses the existing `PUBLISH_HUB_CHAT_GRAMMAR` flag (default `true`) and `GRAMMAR_EVENT_CHANNEL`.

## Tests run

```
cd /mnt/scripts/Orion-Sapienform-unified-turn-grammar
/tmp/orion-test-venv/bin/python -m pytest services/orion-hub/tests/test_hub_grammar_emit.py services/orion-hub/tests/test_turn_orchestrator_ws_frames.py tests/test_chat_substrate_reducer.py -q -W ignore::UserWarning -W ignore::DeprecationWarning
→ 34 passed, 8 failed
```
All 8 failures verified pre-existing via `git stash` comparison against unmodified `origin/main` — identical failure set either way (`AttributeError: 'ThoughtEventV1' object has no attribute 'thought'`, a stale test-mock fixture in `_hub_client_patches` that returns a bare `ThoughtEventV1` instead of the `ThoughtReactResult` wrapper `ThoughtClient.react()` actually returns — a real bug in the test suite, not something this patch touches or introduces). Because those existing fixtures were broken, my new `turn_orchestrator.py`-level tests call `_publish_unified_turn_chat_grammar` directly (bypassing the broken `ThoughtClient`/`HarnessGovernorClient` mock chain) rather than through the full `execute_unified_turn` integration path — still exercises the real code, just at the seam that's actually testable in this environment.

## Evals run

No eval harness exists for hub's chat-grammar production.

## Docker/build/smoke checks

None run — no env/compose changes in this patch, nothing to validate at that layer.

## Review findings fixed

None from an external review pass; self-caught during implementation:
- Initial draft scheduled the grammar publish as a fire-and-forget background task (mirroring `websocket_handler.py`'s `_schedule_publish`) to avoid adding latency ahead of the harness dispatch. Reconsidered: this file's own convention already awaits its other publish calls directly, the latency of one bus publish is negligible next to the harness governor call, and fire-and-forget against `MagicMock()` (non-async) test buses produces orphaned background tasks that fail silently — worse for testability with no real latency benefit. Switched to awaiting directly before writing any tests.

## Restart required

```bash
# Hub picks this up on its next normal restart/redeploy -- no new env keys, no migration.
docker compose --env-file .env --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml up -d --build
```
Confirm the cursor is advancing again post-restart:
```bash
curl -fsS http://localhost:8115/grammar/truth | python3 -c "import json,sys; d=json.load(sys.stdin); print([r for r in d['cursor_positions'] if r['cursor_name']=='chat_grammar_consumer'])"
```

## Risks / concerns

- Severity: low. `repair_pressure_grammar_scalars(pre_turn_bundle=repair_bundle, substrate_summary=None)` is called with `substrate_summary` always `None` in the unified-turn path, since `execute_unified_turn` only ever has `repair_bundle` (`TurnAppraisalBundleV1 | None`) in scope, not a separate `substrate_summary` dict the way the classic path does. Confirmed the helper's implementation checks `pre_turn_bundle` first and only falls back to `substrate_summary`, so this is correct, not a gap — noting it because it's the one place the unified-turn call site's available data differs slightly from the classic path's.
- Severity: none (informational). The `_STANCE_REASONS_RE = re.compile(r"\((.*?)\)")` regex used in `grammar_extract.py` is intentionally non-greedy and scoped by the `role == "stance_disposition"` gate before it ever runs, so it doesn't collide with `user_utterance`'s own `(N words)` parenthetical on the same atom set — verified with a dedicated test (`test_extract_stance_disposition_no_reasons_no_parens_in_summary`).

## PR link

(Push and open via `git push -u origin feat/unified-turn-grammar-trace`, then `gh pr create` if authenticated, or open manually via the GitHub compare link the push prints.)
