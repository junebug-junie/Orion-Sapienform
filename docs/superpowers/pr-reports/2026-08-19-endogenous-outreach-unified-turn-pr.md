# Route endogenous outreach through the real unified-turn pipeline

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1739
Branch: `feat/endogenous-outreach-unified-turn`

## Summary

- Root-caused live why outreach's generation was structurally lightweight: it called `CortexGatewayClient.chat()` directly — a bare bus RPC that never reached `orion-harness-governor`, no fcc motor, no substrate appraisal/reflect/voice finalize beats, no post-turn learning closure, no audit artifact — plus a verb-less `chat_general` default whose hidden stance-brief pre-step overflowed context on every attempt.
- Juniper's call once this was traced: **"if orion is going to reach out to me, it needs to be real and not bullshit."**
- Generation now calls `orion.hub.turn_orchestrator.execute_unified_turn` — the exact same function `websocket_handler.py` calls for a real `client_mode == "orion"` turn.
- This is bigger than swapping the client: the outreach prompt is recorded into Orion's real observation stream (`emit_observation()`) and evaluated by a real `ThoughtClient.react()` stance pass that can `defer`/`refuse` the turn — accepted deliberately, since `HarnessRunRequestV1` requires a `thought_event` and there is no shallower entry point.
- `HUB_ENDOGENOUS_OUTREACH_LLM_ROUTE` removed (killed, not deprecated) — route selection is the harness governor's decision now, identically for outreach and real chat.
- `HUB_ENDOGENOUS_OUTREACH_TIMEOUT_SEC` raised 60s → 300s — confirmed live the old value was sized for the light path and timed out mid-Thought-evaluation on the new one.
- **Live-verified end to end**, including a real infra outage caught correctly: first two attempts failed on a real upstream worker (circe) being down — the existing error-shaped-text backstop correctly refused to ship the failure text. Third attempt, after the worker recovered, succeeded: a real 267-char message, `harness_grounding_status=grounded`, delivered through all 3 rails, correctly tagged in `chat_history_log`.

## Outcome moved

Orion's unprompted outreach now goes through the same real cognition pipeline — observation, Thought evaluation, harness governor, fcc motor, finalize chain — that a genuine Juniper-initiated conversation gets. Verified live: the first successful message was grounded, substantive prose (referencing real shared context), not a telemetry readout dressed up in first-person language.

## Current architecture

`endogenous_outreach.py::_generate()` called `CortexGatewayClient.chat()` (a direct `orion-cortex-gateway` bus RPC) with `mode="brain"` and no `verb`, which the router defaulted to `chat_general` — a lighter pipeline than a real `orion`-mode turn, and (root-caused the same day) broken on the `quick` route for any nontrivial prompt.

## Architecture touched

- `services/orion-hub/scripts/endogenous_outreach.py`: `_generate()` rewritten around `execute_unified_turn`; `start()` now threads a `harness_rpc_bus`; `llm_route` removed everywhere in this module.
- `services/orion-hub/scripts/main.py`: wiring updated to pass `rpc_bus` as `harness_rpc_bus`.
- `services/orion-hub/app/settings.py`: `HUB_ENDOGENOUS_OUTREACH_LLM_ROUTE` removed; `HUB_ENDOGENOUS_OUTREACH_TIMEOUT_SEC` default raised to 300.
- No schema/bus contract changes — reuses the existing `execute_unified_turn`/harness-governor pipeline as-is.

## Files changed

- `services/orion-hub/scripts/endogenous_outreach.py`: core rewrite (see above); module docstring rewritten with the full account.
- `services/orion-hub/scripts/main.py`: `start()` call site updated.
- `services/orion-hub/app/settings.py`: `HUB_ENDOGENOUS_OUTREACH_LLM_ROUTE` removed; `TIMEOUT_SEC` default 60→300.
- `services/orion-hub/.env_example`, live `.env`: same two changes.
- `services/orion-hub/README.md` §4.1: new "Through the real unified turn, not a lookalike" section; safety-posture bullets updated to match (permissions structurally enforced, not option-pinned).
- `services/orion-hub/tests/test_endogenous_outreach.py`: `_cortex_client`-based tests rewritten around an `execute_unified_turn` stub (`_stub_unified_turn`/`_final_frame` helpers); `llm_route` removed from the test fixture; new tests for `start()`'s `harness_rpc_bus` threading and the `no_write` payload.

## Schema / bus / API changes

None — reuses `execute_unified_turn`/`HarnessRunRequestV1`/`HarnessRunV1` exactly as they already exist for real turns.

## Env/config changes

- Removed: `HUB_ENDOGENOUS_OUTREACH_LLM_ROUTE` (killed, not deprecated — no fallback to the old direct-call path).
- Changed: `HUB_ENDOGENOUS_OUTREACH_TIMEOUT_SEC` 60 → 300.
- `.env_example` updated: yes.
- local `.env` synced: yes, hand-edited directly (both changes).
- skipped keys requiring operator action: none.

## Tests run

```text
rtk proxy /mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest \
  services/orion-hub/tests/test_endogenous_outreach.py \
  services/orion-hub/tests/test_tension_outreach_trigger.py -q
101 passed
```

## Evals run

None — no eval harness for this trigger.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-hub build      # success
scripts/safe_docker_build.sh orion-hub up -d --build   # recreated, started, /health 200 ok
```

**Live-fired 4 times against the real deployed container** via `POST /api/debug/endogenous-outreach/trigger`:
1. Timed out at the old 60s ceiling with Thought stance-react still in flight (~33s in) — led to raising `HUB_ENDOGENOUS_OUTREACH_TIMEOUT_SEC`.
2 & 3. Full real pipeline ran end to end (~134s each) but the harness governor's own reflect/voice finalize beats hit a genuine backend outage (circe worker down) — `orion-harness-governor`'s own internal LLM call returned literal `"[Error: llamacpp timed out after waiting]"` text (a separate, pre-existing bug in that service, not this patch — traced to `harness_finalize_reflect_payload_unparseable_using_degraded_reflection` in its logs), and this module's existing `looks_like_error_text()` backstop correctly refused to ship it.
4. After Juniper brought circe back up: succeeded. Real 267-char message, `harness_grounding_status=grounded`, delivered and persisted correctly, tagged `endogenous_outreach` in `chat_history_log`.

## Review findings fixed

- **Finding: `test_disabled_instance_starts_no_task` lost its own two assertions** — an earlier edit inserting two new tests above it left the trailing `assert`s attached to the wrong (last-inserted) test function instead, silently reducing the named test to a no-op.
  - Fix: moved the two assertions back to `test_disabled_instance_starts_no_task`; removed the now-duplicate copy from the unrelated test.
  - Evidence: verified directly against the file before/after; 101/101 tests still pass.
- **Finding: README's "gate re-checked immediately before delivery" bullet still documented the old 60s `HUB_ENDOGENOUS_OUTREACH_TIMEOUT_SEC` default** after this same patch raised it to 300s.
  - Fix: updated to 300s with a pointer to the section explaining why.

## Restart required

Already deployed and live-verified during this session:
```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low
- Concern: generation now takes materially longer (~100-135s observed) than the old bare-LLM-call path (seconds), since it runs the full real-turn pipeline. `_send_lock` is held for the duration, so a stuck/slow attempt delays (not blocks — gates report `already_sending`) the next tick's evaluation.
- Mitigation: `HUB_ENDOGENOUS_OUTREACH_TIMEOUT_SEC=300` bounds worst-case hold time; disclosed in settings.py's own comment as a reasoned estimate to re-derive from real observed latency, not yet a measured distribution.
- Separate, out-of-scope finding: `orion-harness-governor`'s reflect/voice finalize beats can return literal `"[Error: ...]"` text instead of a proper failure signal when their own internal LLM call times out (same "upstream failure reported only in text" anti-pattern this module's own `looks_like_error_text()` backstop was built to catch, but living in a different, shared service). Only observed against 3 test calls during a real circe outage — not yet confirmed whether this affects real chat traffic. Flagged for separate investigation, not fixed here.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1739
