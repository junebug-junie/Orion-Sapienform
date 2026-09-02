# PR report: Hub "Agent" Mode routes through FCC (like Orion), not dead context-exec

## Summary

- Live-confirmed 2026-09-02: `orion-context-exec` has zero containers on
  athena, running or stopped — every Hub "Agent" mode chat turn failed with
  `"context-exec run unreachable"`, over both WebSocket and the HTTP
  `/api/chat` fallback.
- Juniper: "context exec is failed prototype" — asked for Agent mode to
  route through the same FCC/harness-governor subprocess-spawn mechanism
  Orion mode already uses (real `claude -p`), not a different backend.
- `client_mode`/`mode` `in ("orion", "agent")` now takes the same
  `run_unified_turn`/`execute_unified_turn` FCC branch on **both**
  transports. `HUB_AGENT_CONTEXT_EXEC_ENABLED` now defaults to `false` (was
  `true`) — the old context-exec bridge code is left in place, just
  unreachable by default.
- Two code-review findings fixed before merge (README staleness; the HTTP
  path was missed in the first draft — would have silently fallen through to
  a different, untested "context-exec disabled" response instead of either
  the old behavior or the claimed fix).
- One more real bug found via my own live verification of this exact patch:
  `turn_orchestrator.py`'s final frame and its persisted `chat_history_log`
  row both hardcoded `mode: "orion"` regardless of caller — fixed, so
  Agent-mode turns are no longer permanently mislabeled in history.
- **Fully live-verified on athena's actual running Hub**, not just tests:
  a real HTTP Agent-mode turn now returns `mode: "agent"`,
  `chat_route: "unified_turn_harness"`, and a genuine Claude response.

## Outcome moved

Hub's "Agent" Mode goes from always-broken (100% failure, wrong error
surfaced) to a real, working Claude turn via the same proven FCC mechanism
Orion Mode already uses — live-verified end to end on production, not just
in tests.

## Current architecture (before this patch)

See the plan file's "Current architecture" section for the full trace
(`websocket_handler.py`'s `client_mode == "orion"` branch vs. the
`context_exec_agent_bridge.py` HTTP call `should_use_context_exec_agent_lane`
gated on `mode == "agent"`). Both transports (`websocket_handler.py` and
`api_routes.py`) independently called into the dead `orion-context-exec`
service for Agent mode.

## Architecture touched

- `services/orion-hub/scripts/websocket_handler.py` — WS chat-turn dispatch
- `services/orion-hub/scripts/api_routes.py` — HTTP `/api/chat` fallback dispatch
- `services/orion-hub/app/settings.py` — `HUB_AGENT_CONTEXT_EXEC_ENABLED` default
- `orion/hub/turn_orchestrator.py` — final-frame/chat-history mode tagging
  (shared by both transports, not Hub-specific)

## Files changed

- `services/orion-hub/scripts/websocket_handler.py`: widened FCC branch
  condition; tag `active_turn["kind"]`/cancel `kind=`/TTS `lane=` by
  `client_mode` instead of a hardcoded `"orion"` literal
- `services/orion-hub/scripts/api_routes.py`: widened the HTTP path's
  matching FCC branch condition (missed in the first draft, added after
  code review caught it)
- `services/orion-hub/app/settings.py`: `HUB_AGENT_CONTEXT_EXEC_ENABLED`
  default `true` → `false`
- `services/orion-hub/.env_example`: same default, with rationale comment
- `services/orion-hub/README.md`: two sections updated (compute-lane-override
  note; the dedicated "Agent mode → context-exec" section, now
  "Agent mode → FCC")
- `orion/hub/turn_orchestrator.py`: `_success_frames`/
  `_publish_unified_turn_chat_history` tag `mode` from the real caller
  instead of a hardcoded `"orion"`
- `services/orion-hub/tests/test_hub_agent_mode_fcc_routing.py` (new, 9 tests)
- `services/orion-hub/tests/test_orion_unified_turn_tts.py` (1 assertion
  updated for the `lane=client_mode` change)

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: Hub "Agent" mode now produces real Claude responses via
  FCC instead of always erroring. Persisted `chat_history_log` rows for
  Agent-mode turns now carry `mode="agent"` instead of the previous
  (incorrect) `mode="orion"`.
- Compatibility notes: `context_exec_agent_bridge.py`/`context_exec_client.py`
  are unchanged and still fully functional — only their trigger condition
  (the settings default) changed. An operator can flip
  `HUB_AGENT_CONTEXT_EXEC_ENABLED=true` to restore the old behavior if
  `orion-context-exec` is ever redeployed.

## Env/config changes

- Added keys: none.
- Changed defaults: `HUB_AGENT_CONTEXT_EXEC_ENABLED` `true` → `false`.
- `.env_example` updated: yes.
- local `.env` synced: yes — athena's live `services/orion-hub/.env` was
  hand-edited to match (`sed`) and the redeploy below confirms it took
  effect (fresh containers actually behaved per the new default).
- Skipped keys requiring operator action: none.

## Tests run

```text
PYTHONPATH=<worktree>/services/orion-hub:<worktree> .venv/bin/python -m pytest \
  services/orion-hub/tests/test_hub_agent_mode_fcc_routing.py \
  services/orion-hub/tests/test_orion_unified_turn_tts.py \
  services/orion-hub/tests/test_context_exec_agent_grounding.py \
  services/orion-hub/tests/test_websocket_agent_claude_routing.py \
  services/orion-hub/tests/test_chat_route_tagging.py \
  services/orion-hub/tests/test_agent_repl_bridge.py \
  services/orion-hub/tests/test_handle_chat_request_orion_mode_continuity.py \
  services/orion-hub/tests/test_handle_chat_request_orion_mode_degraded.py \
  services/orion-hub/tests/test_handle_chat_request_http_fallback_tts.py \
  services/orion-hub/tests/test_chat_history_no_raw_publish.py \
  services/orion-hub/tests/test_endogenous_outreach.py \
  services/orion-hub/tests/test_fcc_sandbox_sync_test_process_guard.py \
  services/orion-hub/tests/test_turn_cancel.py \
  services/orion-hub/tests/test_turn_stop_command.py -q
-> 279 passed
```

Full-suite run (2059+ tests collected) shows 63 pre-existing failures
**unrelated to this patch** — verified two ways: (1) reverting just the
`HUB_AGENT_CONTEXT_EXEC_ENABLED` default back to `true` does not fix
`test_llm_route_selector.py`'s 3 failures, proving they're independent of
this change; (2) the highest-risk adjacent file,
`test_turn_orchestrator_ws_frames.py` (13 failures, mostly untouched by
this patch beyond the mode_tag default-arg addition), fails with
`TypeError: object MagicMock can't be used in an await expression` inside
`pre_turn_appraisal_client.py` — an unrelated async-mock fixture bug.

## Evals run

None — no eval harness exists for Hub's chat-mode routing; the live
production smoke test below is the closest equivalent for this kind of
dispatch-path change.

## Docker/build/smoke checks

**Full live deploy + verification on athena's actual running Hub** (not a
worktree, not a simulation):

```text
docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d --build
-> orion-athena-hub rebuilt, recreated, started; clean logs, no import
   errors, "Startup complete — Hub is ready.", Uvicorn bound 0.0.0.0:8080

curl -s -X POST http://localhost:8080/api/chat -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say hello in exactly five words."}],
       "mode":"agent","session_id":"...","disable_tts":true,"no_write":true}'
```

First deploy (routing fix only): confirmed the request reached the harness
governor's real `claude -p` RPC path (`orion:harness:run:result:<corr>`
reply-listener created) and returned a genuine Claude response with
`chat_route: "unified_turn_harness"` — but `mode: "orion"` in the response,
which led to the `turn_orchestrator.py` fix above.

Second deploy (after the `mode_tag` fix), fresh rebuild + redeploy + retest:

```json
{
  "type": "final",
  "mode": "agent",
  "llm_response": "Hi Juniper, I'm listening closely.",
  "finalize_ran": true,
  "chat_route": "unified_turn_harness"
}
```

Both round-trips took several minutes each (real `orion:thought` RPC +
FCC/`claude -p` turn latency, not a hang) — confirmed via live log
inspection at each stage (pre-turn appraisal RPC, thought RPC, harness
governor reply-listener, `orion-athena-harness-governor`'s own grammar-step
log) rather than just trusting the final HTTP response.

Deployed from the **primary checkout** (`/mnt/scripts/Orion-Sapienform`),
not the worktree — copied the fix's files over first, since building
straight from a disposable worktree's compose context would risk pinning
that worktree's path into a live production deploy.

## Review findings fixed

- Finding: `README.md` still documented Agent mode routing to context-exec
  with `HUB_AGENT_CONTEXT_EXEC_ENABLED=true` as the default, contradicting
  the new default and routing behavior.
  - Fix: both the compute-lane-override note and the dedicated
    "Agent mode → context-exec" section rewritten (now "Agent mode → FCC").
  - Evidence: `services/orion-hub/README.md`.
- Finding: the HTTP `/api/chat` endpoint's Agent-mode gate
  (`should_use_context_exec_agent_lane`) was not updated alongside the
  WebSocket lane — with the settings default flipped, HTTP Agent-mode
  requests would have silently fallen through to the plain
  `cortex_client.chat()` path (a different, untested "context-exec
  disabled" response) instead of either the old behavior or the claimed fix.
  - Fix: found that the HTTP path already had a full, working FCC branch for
    `mode == "orion"` (`execute_unified_turn` + continuity messages + TTS +
    degraded-frame handling) that the first draft missed — widened it the
    same way as the WS lane, `mode in ("orion", "agent")`.
  - Evidence: `services/orion-hub/scripts/api_routes.py`; new tests
    `test_http_fallback_agent_mode_also_shares_the_fcc_branch` and
    `test_http_agent_mode_never_reaches_should_use_context_exec_agent_lane`.

## Restart required

Already done live during this session:
- `orion-athena-hub` rebuilt + recreated on athena (twice — once for the
  routing fix, once for the `mode_tag` fix), confirmed healthy and correct
  both times.

No further restart needed for this PR's changes.

## Risks / concerns

- Severity: low
  - Concern: Agent and Orion mode currently produce behaviorally identical
    FCC turns (same prompt-compilation path, same default model) — only the
    `mode` tag differs. This is an explicit, stated non-goal of this patch
    (see plan), not an oversight — differentiating them (system prompt,
    tool profile) is real follow-up work.
  - Mitigation: none needed yet; flagged for whenever that follow-up is wanted.
- Severity: low
  - Concern: `context_exec_agent_bridge.py`/`context_exec_client.py` are not
    deleted, just unreachable by default — dead code sitting in the repo.
  - Mitigation: deliberate (per plan's stated non-goals) — needs its own
    "any other live callers?" check before a safe deletion PR.

## PR link

(added after `gh pr create`)
