## Summary

- Every line in the unified chat turn's RECENT CONVERSATION block now shows when it was said, in Juniper's local time (America/Denver). Example: `- User (Fri Oct 16, 1:47 am):`, `- Orion (you, prior turn, Thu Oct 15, 6:22 pm):`.
- A plain marker line, `(— about 9 hours pass —)`, sits between two messages when more than an hour passed. It also sits after the last message when more than an hour passed before now.
- The TASK block now states the current time: `Current time: Fri Oct 16, 10:56 am (Juniper's local time)`.
- The times are real stored times. A message without a time renders exactly as before. No time or gap is guessed.
- No other prompt section changes.

## Outcome moved

Live 2026-10-16: Orion saw three events as one moment. The events were a cut-off turn (00:22 UTC), "night night" (07:47 UTC), and "what's the latest?" (16:56 UTC). Orion apologised for being cut off "just now" and claimed Juniper said night night "before I could finish". With this patch, the prompt shows the order and the two long gaps explicitly.

Before (live prompt):

```text
RECENT CONVERSATION (this session so far, 8 prior message(s) -- you are mid-conversation, not opening a new one)
- User: It's your choice on what you want to focus on. Decide then queue it up in your Curiosity run.
- Orion (you, prior turn): [Cut short - not a finished answer.] ...
- User: night night
- Orion (you, prior turn): Night night, Juniper. Sleep well.
TASK THIS TURN (respond to this message, read with the recent conversation above):
User message: what's the latest?
```

After (rendered by the new code from the real chat_history_log times):

```text
RECENT CONVERSATION (this session so far, 6 prior message(s) -- you are mid-conversation, not opening a new one)
- User (Thu Oct 15, 5:38 pm): so, who do you see outside?
- Orion (you, prior turn, Thu Oct 15, 5:38 pm): I see you, Juniper...
- User (Thu Oct 15, 6:22 pm): It's your choice on what you want to focus on. Decide then queue it up in your Curiosity run.
- Orion (you, prior turn, Thu Oct 15, 6:22 pm): [Cut short - not a finished answer.] ...
(— about 7 hours pass —)
- User (Fri Oct 16, 1:47 am): night night
- Orion (you, prior turn, Fri Oct 16, 1:47 am): Night night, Juniper. Sleep well.
(— about 9 hours pass —)
TASK THIS TURN (respond to this message, read with the recent conversation above):
Current time: Fri Oct 16, 10:56 am (Juniper's local time)
User message: what's the latest?
```

## Current architecture

- One function renders the block: `orion/harness/prefix.py::_format_recent_turns`. It is called by `compile_harness_prefix`, which is called by `orion/harness/runner.py::build_harness_prompt`.
- Two places call the renderer:
  - The harness governor (`runner.py`). This builds the real `claude -p` prompt. The warm pool (`fcc_warm_pool.py`) and the spawn path both receive the same prompt string, so one renderer covers both.
  - The Hub's motor_boot preview (`turn_orchestrator.py`). This is shown in the cockpit only.
- History reaches the governor as `HarnessRunRequestV1.recent_turns` (`list[TurnWindowMessageV1]`). Before this patch, `TurnWindowMessageV1` held only `role` and `content`.
- The Hub builds that list from `continuity_messages`. The websocket path takes it from the per-connection `history` list. The HTTP orion-mode path takes it from client messages, plus persisted rows when the thread is thin. `build_continuity_messages` reduced every item to role and content only.

## Architecture touched

- **Timestamp sources (real code, not invented):**
  - Websocket user message: the Hub's receive time, written as `"ts": datetime.now(timezone.utc)` when it appends to `history` (`websocket_handler.py`). The classic lane's assistant append gets the same stamp.
  - Rehydrated rows: `chat_history_log.created_at`. That column is `server_default=func.now()` (`services/orion-sql-writer/app/models/chat_history_log.py`) and is set when the row is first inserted. The column is naive, and the live DB `timezone` is `Etc/UTC` (verified), so a naive value is treated as UTC. One row holds a prompt and its response, so both halves share the row time. The assistant line therefore shows when Juniper's message landed, not when the reply finished. This is a known approximation.
- **Data path:**
  1. `build_continuity_messages(keep_timestamps=True)`. Only the two unified-turn call sites set this.
  2. `build_turn_window(keep_timestamps=True)`.
  3. `TurnWindowMessageV1.at`.
  4. `_format_recent_turns`.
- **Timezone:** the code reads `ORION_SITUATION_TIMEZONE`, the same key and default as the Situation block, so the two zones never disagree inside one prompt. No new env key. If the zone cannot load, times render in UTC and are labeled `(UTC)`, never "local".

## Files changed

- `orion/harness/prefix.py`: per-line local time, gap markers, and the `Current time:` line.
- `orion/schemas/pre_turn_appraisal.py`: new optional `TurnWindowMessageV1.at`. A serializer omits it when it is None.
- `orion/substrate/appraisal/turn_window.py`: `keep_timestamps` flag, which parses `ts` to `at`.
- `orion/hub/turn_orchestrator.py`: the harness `recent_turns` keeps timestamps. The pre-turn appraisal window does not change.
- `services/orion-hub/scripts/cortex_request_builder.py`: `keep_timestamps` flag. The default output does not change.
- `services/orion-hub/scripts/websocket_handler.py`: stamps `ts` on history appends. The unified lane keeps timestamps.
- `services/orion-hub/scripts/api_routes.py`: the HTTP orion-mode lane keeps timestamps.
- `services/orion-hub/scripts/chat_history_rehydrate.py`: selects `created_at` and emits `ts`.
- `services/orion-harness-governor/requirements.txt`: adds `tzdata==2025.3`, the same pin as the Hub.
- Tests: `orion/harness/tests/test_recent_turn_timestamps.py` (new), plus additions to the hub tests `test_chat_history_rehydrate.py`, `test_cortex_request_builder.py`, and `test_turn_orchestrator_ws_frames.py`.

## Schema / bus / API changes

- Added: `TurnWindowMessageV1.at: datetime | None = None`.
- Removed / Renamed: none.
- Behavior changed:
  - The harness prompt shows times and gap markers.
  - Every harness turn that has a user message now gets a `Current time:` line. This includes curiosity, outreach, reading, and collapse-mirror turns, which have no RECENT block.
- Compatibility notes:
  - Every `TurnWindowMessageV1` consumer validates with `extra="forbid"`. When `at` is None, it is omitted from serialization. That keeps every untimed payload byte-identical, including every pre-turn appraisal request (consumer: orion-cortex-exec).
  - A Hub request that has a timed message is rejected by an un-rebuilt governor. **Rebuild the harness governor before the Hub.** The normal post-pull rebuild already orders them that way, but nothing enforces it.

## Env/config changes

- Added / Removed / Renamed keys: none. The code reads the existing `ORION_SITUATION_TIMEZONE`, default `America/Denver`.
- `.env_example` updated: no. Local `.env` sync: not needed. Skipped keys: none.

## Tests run

```text
pytest orion/harness/tests                                      576 passed
pytest services/orion-harness-governor/tests                    85 passed
hub: test_chat_history_rehydrate, test_cortex_request_builder,
     test_turn_orchestrator_ws_frames,
     test_handle_chat_request_orion_mode_continuity             125 passed
hub full suite (branch): 36 failed, 3726 passed
hub full suite (main):   43 failed, 3717 passed
  Every branch failure also fails on main, except one:
  test_memory_graph_routes.py::test_memory_graph_validate_fixture_roundtrip.
  It fails in the worktree because the worktree has no services/orion-hub/.env
  (postgres_uri_not_configured -> 503). It passes in the main checkout. The
  cause is the environment, not this patch.
orion/substrate + orion/schemas: 4 failures, all of which also fail on main
  (felt_state lane, context_provenance)
```

Mutation checks: I removed each piece of the fix in turn, and each removal made at least one test fail.

- The prefix rendering.
- `turn_window` `ts` parsing.
- `rehydrate` `created_at`.
- The serializer `at` drop.
- The orchestrator `keep_timestamps`.
- The builder `ts` carry.
- The websocket `keep_timestamps` and the `ts` stamp.
- The api_routes `keep_timestamps`.
- The gap `>` versus `>=` threshold.
- The UTC fallback label.
- The tail gap line.

## Evals run

```text
None. No eval harness covers prompt-time rendering. The deterministic tests
cover the behavior. Proposed follow-up: a replay eval that checks whether the
motor stops conflating order across a long gap.
```

## Docker/build/smoke checks

```text
Live containers (read-only): ZoneInfo('America/Denver') resolves in both
orion-harness-governor and orion-hub (governor: 16:56 UTC -> 10:56 -06:00).
No image built. Not deployed.
```

## Review findings fixed

- Finding: if the timezone fails to load, UTC times would be labeled "Juniper's local time".
  - Fix: fall back to UTC with a `(UTC)` label and a warning log. Add `tzdata` to the governor requirements.
  - Evidence: `test_unloadable_timezone_renders_utc_labeled_utc`.
- Finding: the hard-coded zone could disagree with the Situation block's `ORION_SITUATION_TIMEZONE`.
  - Fix: read the same key and default.
  - Evidence: `test_timezone_follows_the_situation_setting`.
- Finding: no test pinned the Hub wiring points or the bus round-trip.
  - Fix: a source-pin test for websocket and api_routes `keep_timestamps=True` and the `ts` stamp, plus a `HarnessRunRequestV1` json round-trip assertion in the orchestrator test. Both are mutation-checked.
- Finding: the threshold was `>=` while the docs said "more than an hour", rounding was banker's rounding, and the switch to days at 36 h was odd.
  - Fix: `>` threshold, half-up rounding, hours until 48 h.
  - Evidence: threshold and rounding assertions in `test_gap_marker_threshold`.
- Finding: an old governor rejects a timed request (rollout order).
  - Fix: governor-first restart order, stated below. This is not enforced in code.
- Not fixed, noted:
  - The Hub preview and the governor prompt each read their own "now", so the two can differ by the RPC latency. This affects the cockpit only.
  - An HTTP client could supply its own `ts`. The client is Juniper's own UI, and a bad value parses to None.

## Restart required

Rebuild the governor first, then confirm it has the new field, then rebuild the Hub:

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

```bash
docker exec orion-athena-harness-governor python -c "from orion.schemas.pre_turn_appraisal import TurnWindowMessageV1 as M; print('at' in M.model_fields)"
```

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

Deploy from the clean primary `main` checkout after merge. The container name in the second command is assumed; check it with `docker ps`.

## Risks / concerns

- Severity: high if the order is wrong, otherwise none.
  - Concern: the Hub is deployed before the governor. Every timed chat turn is then rejected, and chat fails until the governor is rebuilt.
  - Mitigation: governor-first order (above). The post-pull rebuild sorts the governor first.
- Severity: low.
  - Concern: a rehydrated assistant line shows the time of Juniper's message, not the time the reply finished.
  - Mitigation: documented in code. The difference is minutes, and the gap markers are about hours.
- Severity: medium (pre-existing, out of scope).
  - Concern: the unified websocket lane never appends Orion's own replies to `history`. The lane `continue`s before the assistant append. Inside one live socket, RECENT CONVERSATION therefore holds rehydrated pairs plus Juniper's later messages, with none of Orion's live replies, and that `history` list is never trimmed.
  - Mitigation: follow-up issue recommended.

## PR link

<filled after open>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
