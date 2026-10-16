## Summary

- Orion's own replies now go into the Hub chat's per-connection conversation history, so on the next turn Orion can see what they just said.
- The history list now has a size limit on the Orion/Agent lane. Before, it grew for as long as the browser tab stayed open.
- One shared pair of helpers does the work for both chat lanes: `append_live_assistant_reply` and `trim_history_window` in `services/orion-hub/scripts/chat_history_rehydrate.py`.
- The shared trim also fixes an old slicing bug in the legacy lane. With `context_turns=0`, it added the whole list back instead of trimming it.
- New tests drive the real `websocket_endpoint` through several turns on one connection.

## Outcome moved

Before this fix, inside one open tab, the RECENT CONVERSATION block that Orion received held only Juniper's messages, one after another, with none of Orion's replies between them. A follow-up such as "what did you just say?" reached Orion without the answer it referred to. After this fix, the second turn's continuity reads Juniper, Orion, Juniper, and each item has its time.

## Current architecture

- `websocket_handler.websocket_endpoint` keeps one in-memory `history` list per socket. That list feeds `build_continuity_messages`, which builds `continuity_messages` for `run_unified_turn`, and then `orion/harness/prefix.py::_format_recent_turns`.
- On connect, `chat_history_rehydrate.rehydrate_history` fills an empty list from `chat_history_log`. It only restores and never overwrites.
- **Root cause:** `services/orion-hub/scripts/websocket_handler.py`. The unified orion/agent lane appends the user message at line 1370 (main). It then runs the turn and does `continue` at line 1621 (main). It never appends Orion's reply and never trims. The only assistant append and trim were in the legacy lane at lines 2456-2471 (main), and the unified lane never reaches them.

## Architecture touched

- Hub websocket chat, both lanes. No bus, schema, or env changes.

## Files changed

- `services/orion-hub/scripts/chat_history_rehydrate.py`: adds `append_live_assistant_reply`, which uses the same shape and the same error filter as a rehydrated message. Also adds `trim_history_window`, which keeps the system message plus the newest `2 * turns` messages and floors `turns` to 1.
- `services/orion-hub/scripts/websocket_handler.py`: the unified lane now appends and trims after its TTS block. The legacy lane now uses the same helpers.
- `services/orion-hub/tests/test_ws_unified_history_orion_replies.py`: new tests (14).

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: in the unified lane, the live per-socket history now holds Orion's final replies and has a size limit. In the legacy lane, an error-shaped reply (`[error`, `error:`, traceback) is no longer added to history. Rehydrate already filtered these out.
- Compatibility notes: none

### Which text gets appended

- The text is the `final` frame's `llm_response`, which is what the browser shows. This is the same `run.final_text` that `turn_orchestrator` persists to `chat_history_log`, so a reconnect restores the same words.
- A cut-short turn already carries `[Cut short - not a finished answer.]`. The finalize chain adds it, and the governor's `ensure_cut_short_marked` is a backstop.
- A `turn_error` `partial_draft` is not appended. It is never persisted either, so adding it would make live history differ from what a reconnect restores.
- If a turn produces no final reply, Juniper's own message stays in live history on purpose. Rehydrate cannot restore it, because the orchestrator persists the user row only together with a final reply. This trade-off is documented in the helper's docstring.
- The `ts` is the UTC time when Hub delivered the reply. This is the shape #2725 introduced, so `prefix.py` renders it.

### Duplicates after reconnect

- A new socket starts with a new list. Rehydrate refills it from `chat_history_log` only.
- A second `session_hello` on the same socket is a no-op once live turns exist, because of the `history_is_empty` guard. A test covers this.

### HTTP orion path (`api_routes.handle_chat_request`)

- This path does not have the bug. It keeps no server-side list.
- Its history comes from the client payload, or from `chat_history_log` through `_augment_thin_history_from_persisted`. That table already holds both halves of each completed turn.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change)
- skipped keys requiring operator action: none

## Tests run

```text
pytest services/orion-hub/tests/test_ws_unified_history_orion_replies.py  -> 14 passed
  (before the fix: second-turn, cut-short and bounded tests fail -- reproduced first)
pytest test_ws_unified_history_orion_replies + test_orion_unified_turn_tts + test_chat_history_rehydrate
       + test_workflow_schedule_runtime_paths + test_websocket_agent_claude_routing -> 82 passed
Mutation check (each mutant run against the new test file):
  drop unified append -> 3 fail; drop unified trim -> 1 fail;
  drop legacy append -> 1 fail; drop legacy trim -> 1 fail
Full hub suite, worktree without .env (pytest services/orion-hub/tests):
  main 4fbdcc45e: 35 failed, 3729 passed, 111 skipped
  branch:         35 failed, 3743 passed, 111 skipped  (+14 = the new tests)
  failure sets identical (diffed by test id) -- all 35 pre-existing on main
orion/harness: main 599 passed, branch 599 passed
```

## Evals run

```text
No eval harness covers per-connection history. Gap: a live two-turn smoke on the deployed Hub
(see Risks). The endpoint-driven tests are the closest deterministic proxy.
```

## Docker/build/smoke checks

```text
Not run. Not deployed (per task). Live path UNVERIFIED.
```

## Review findings fixed

- Finding: moving the extract call broke 4 source-shape tests in `test_orion_unified_turn_tts.py`.
  - Fix: the TTS block is unchanged again. The append and trim now sit after it, before the lane's `continue`.
  - Evidence: `test_orion_unified_turn_tts.py` 24/24 passed.
- Finding: the claim of parity with rehydrate was only half true. A failed turn's user message stays live, but rehydrate cannot restore it.
  - Fix: this is kept on purpose and documented in the `append_live_assistant_reply` docstring and the test docstring.
- Finding: the legacy lane's behavior changed (error filter and new trim) and had no test.
  - Fix: `test_legacy_lane_appends_reply_and_stays_bounded` drives the real endpoint in brain mode. The fake socket now uses the real `WebSocketState`; before, `_safe_ws_send_json` read it as disconnected and skipped the code under test.
- Finding (nit): the bounded test had a thin margin.
  - Fix: raised n from 8 to 12.
- Not fixed (nits):
  - Context-overflow hint text can enter continuity. It matches what is persisted, so parity holds.
  - A client-supplied `context_turns` can be very large. This already existed before this change.

## Restart required

```bash
python3 scripts/safe_docker_build.sh orion-hub up -d --build
```

Run from the clean primary main checkout after merge.

## Risks / concerns

- Severity: low
  - Concern: on a failed turn, two user messages in a row reach the harness. This is accurate to what happened.
  - Mitigation: documented. The pop-on-failure pattern already used at the import-guard exits is available if this turns out to be wrong.
- Severity: low
  - Concern: live path UNVERIFIED.
  - Mitigation: after deploy, send two orion-mode turns in one tab. Check the governor's harness request `recent_turns` for the second turn: it should hold the first reply with a time.

## PR link

<filled in after creation>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
