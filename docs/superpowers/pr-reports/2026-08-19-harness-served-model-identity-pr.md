# PR Report: real served-model identity for the harness/unified-turn chat path

## Summary

- The dominant Hub chat path (`client_mode=="orion"` -> `execute_unified_turn` ->
  `orion-harness-governor`'s FCC motor) had no identity threaded into
  `chat_history_log.response_identity` at all -- `harness_req.fcc_model_label`
  was computed before dispatch but never passed to the persistence call.
  Confirmed live 2026-08-19 by Juniper sending a real message through the
  deployed fix and finding the field still blank for this path (PR #1742 had
  only fixed the `mode="brain"`/HTTP/agent-claude lanes).
- Fixed the threading gap, then answered a live follow-up challenge: is
  `fcc_model_label` (e.g. `"MODEL_SONNET"`) actually the real served model,
  or just a Claude-Code-harness routing alias? Confirmed live that
  `MODEL_SONNET` and `MODEL_OPUS` in `~/.fcc/.env` both route to the
  identical `llamacpp/chat` target -- the alias alone cannot distinguish
  which real backend served a turn.
- Traced the full proxy chain (`claude CLI -> orion-fcc:8082 ->
  orion-llm-gateway:8210 -> llama.cpp`) and confirmed by direct `curl` that
  llama.cpp's own Anthropic-compat `/v1/messages` endpoint echoes the real
  served weights file (e.g. `/models/gguf/Qwen_Qwen3-8B-Q4_K_M.gguf`) in the
  response's top-level `"model"` key regardless of the alias requested, that
  `orion-llm-gateway`'s `anthropic_passthrough.py` is a raw byte passthrough
  that never rewrites that field, and that
  `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1` is already set on both FCC
  subprocess launch sites -- the documented Claude Code CLI flag for exactly
  this scenario.
- Wired up real served-model discovery: extract it from the CLI's own
  stream-json `"assistant"` events, thread it through
  `HarnessMotorResult` -> `HarnessRunV1` -> `turn_orchestrator.py`, and
  prefer it over the requested alias for `response_identity` whenever
  discovery fires. Falls back to the alias when it doesn't (e.g. the motor
  failed before any assistant turn, or a non-discovery-aware backend).
- Also closed the same gap for `endogenous_outreach.py`'s separate
  chat-history publish path, which previously had no model tracking at all.

## Outcome moved

`chat_history_log.response_identity` for the dominant Hub chat path goes
from **always blank** to **the real served backend model** (e.g.
`"Qwen_Qwen3-8B-Q4_K_M"`) when the CLI's stream-json discovery fires, or the
requested route alias (e.g. `"MODEL_SONNET"`) as a fallback -- never blank
for a real completed turn.

## Current architecture

Two independent code paths both write `chat_history_log`:

1. `mode="brain"` (fallback) -> `CortexGatewayClient.chat()` -> cortex-orch
   /cortex-exec's LLM-gateway `chat_general` verb path. Fixed in PR #1742
   via `_last_model_used`/`llm_backend.py`'s existing `_served_model()`.
2. `client_mode=="orion"` (the dominant, default path for everyday chat) ->
   `run_unified_turn()` -> `execute_unified_turn()` ->
   `orion-harness-governor`'s bus RPC -> `HarnessRunner.run()` ->
   `orion/harness/fcc_motor.py`'s `run_fcc_turn()`, which spawns the real
   `claude -p --output-format stream-json` subprocess against the FCC proxy.
   This is a completely separate mechanism from (1), with no built-in model
   tracking before this PR.

`orion/harness/fcc_motor.py`'s `run_fcc_turn()` already parses every
stream-json line into a step frame (`build_step_frame`), including the
`"assistant"` event's `message` object -- which is the CLI's own
repackaging of the raw upstream Anthropic Messages response, `model` field
included, given `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1`. That value
was simply never extracted.

## Architecture touched

- `orion/harness/fcc_motor.py` -- FCC-Claude subprocess motor
- `orion/harness/runner.py` -- `HarnessRunner`/`HarnessMotorResult`
- `orion/schemas/harness_finalize.py` -- `HarnessRunV1`
- `services/orion-harness-governor/app/bus_listener.py` -- governor saga
- `orion/hub/turn_orchestrator.py` -- unified-turn response persistence
- `services/orion-hub/scripts/endogenous_outreach.py` -- unprompted-message
  publish path

## Files changed

- `orion/harness/fcc_motor.py`: added `_served_model_from_assistant()` --
  extracts `message.model` from a stream-json `"assistant"` event, reduced
  to a basename with any weights-file extension stripped (the raw value is
  a full server-side filesystem path; `response_identity` is a user-facing
  "who answered" field, not an infra debug surface). Tracked across the
  `run_fcc_turn()` loop and included in the final `metadata` dict as
  `fcc_served_model`, alongside the existing `fcc_model_label`. Also carried
  into the three mid-stream error exits (stall/timeout,
  `LimitOverrunError`, context-ceiling-exceeded) that previously dropped an
  already-discovered value on those paths.
- `orion/harness/runner.py`: `HarnessMotorResult` gains `fcc_served_model`.
  Added a shared `_served_model_from_metadata()` helper used by both the
  `"final"` and `"error"` event branches, so the extraction/stripping rule
  can't drift between a clean turn and a degraded one. Threaded into both
  `HarnessMotorResult(...)` return sites.
- `orion/schemas/harness_finalize.py`: `HarnessRunV1` gains
  `fcc_served_model: str | None = None`.
- `services/orion-harness-governor/app/bus_listener.py`: threaded
  `fcc_served_model=motor.fcc_served_model` onto all 5
  `HarnessRunV1(...)` construction sites that have a motor result available
  (the 2 that don't -- request validation refusal, and the outer
  bus-message exception handler -- correctly stay `None`).
- `orion/hub/turn_orchestrator.py`: computes
  `resolved_model_label = run.fcc_served_model or harness_req.fcc_model_label`
  once in `execute_unified_turn`, used at all 4 call sites that previously
  passed `harness_req.fcc_model_label` directly (both
  `_publish_unified_turn_chat_history` calls, both `_success_frames` calls).
  Strengthened the `_success_frames` comment to make clear the frame's
  `fcc_model_label` key is dual-purpose (usually the real served model, not
  strictly one of the fixed route aliases) so a future generic consumer
  doesn't assume otherwise.
- `services/orion-hub/scripts/endogenous_outreach.py`: no functional change
  needed beyond what already reads `final_frame.get("fcc_model_label")` --
  it inherits the resolved value automatically since `_success_frames`
  already carries it.
- Tests: `orion/harness/tests/test_fcc_motor_served_model.py` (new, unit
  tests for the extraction/basename helper),
  `orion/harness/tests/test_harness_runner.py` (2 new tests: metadata
  threading present/absent), `services/orion-harness-governor/tests/test_harness_governor_rpc.py`
  (1 new test: `HarnessRunV1` construction threading),
  `services/orion-hub/tests/test_turn_orchestrator_ws_frames.py` (1 new
  test: served-model preference over requested alias).

## Schema / bus / API changes

- Added: `HarnessRunV1.fcc_served_model: str | None = None` on the existing
  `orion:harness:run:result:*` bus reply -- additive, backward compatible.
- Removed: none.
- Renamed: none.
- Behavior changed: `chat_history_log.response_identity`/`ChatHistoryTurnV1.response_identity`
  for the `client_mode=="orion"` path now resolves to the real served model
  when discovery fires, instead of always being blank.
- Compatibility notes: `HarnessMotorResult.fcc_served_model` and
  `HarnessRunV1.fcc_served_model` both default to `None`; every existing
  caller/test that doesn't set them keeps prior behavior unchanged
  (confirmed: full existing test suites pass unmodified).

## Env/config changes

None. No `.env_example` changes.

## Tests run

```text
ENABLE_PRE_TURN_APPRAISAL=false python3 -m pytest \
  orion/harness/tests/test_fcc_motor_served_model.py \
  orion/harness/tests/test_harness_runner.py \
  orion/harness/tests/test_fcc_motor_mcp.py \
  orion/harness/tests/test_fcc_motor_summarize.py \
  orion/harness/tests/test_fcc_motor_cancel.py \
  services/orion-harness-governor/tests \
  -q
# 82 passed, 1 pre-existing unrelated failure
# (test_harness_runner_surfaces_fcc_error_code -- confirmed via git stash to
#  fail identically on unmodified main, unrelated grounding_status formatting
#  bug in an existing "error" event branch this PR does not touch)

ENABLE_PRE_TURN_APPRAISAL=false python3 -m pytest \
  tests/test_turn_orchestrator_ws_frames.py tests/test_endogenous_outreach.py \
  tests/test_agent_claude_chat_history.py tests/test_chat_history_no_raw_publish.py \
  tests/test_chat_turn_spark_meta_turn_effect.py tests/test_chat_history_rehydrate.py \
  -q   # (from services/orion-hub)
# 164 passed
```

Also confirmed pre-existing (unrelated to this diff, via `git stash`):
`orion/harness/tests/test_grounding_capsule_consumers.py`'s 2 failures --
same pattern, both reproduce identically on unmodified main.

## Evals run

No eval harness exists for `orion-harness-governor` or the `orion/harness`
package. Not added in this PR -- pure plumbing fix, covered by unit tests
above; flagging as a follow-up gap rather than claiming eval coverage that
doesn't exist.

## Docker/build/smoke checks

Not run. This changeset touches Python logic only in `orion-harness-governor`
and `orion-hub` (already-running services); no config/dependency/Dockerfile
changes. `chat`/`agent` llama.cpp workers were down for the entire session
(confirmed live via `GET /routes` and via 504s in `orion-athena-fcc`'s own
logs, unrelated to this PR), so a live end-to-end harness turn could not be
exercised in this session. Verified the mechanism empirically at the layer
below (`orion-llm-gateway`'s `/v1/messages`, direct `curl`) instead. **A live
re-test once the chat/agent workers are back up is the natural follow-up.**

## Review findings fixed

- Finding: three of `run_fcc_turn`'s mid-stream `"error"` yields
  (stall/timeout, `LimitOverrunError`, context-ceiling-exceeded) never
  attached a metadata dict, silently dropping an already-discovered
  `served_model` from an earlier assistant event on exactly those failure
  exits.
  - Fix: added `"metadata": {"fcc_served_model": served_model}` to all
    three yields.
  - Evidence: `orion/harness/fcc_motor.py` diff; no test previously covered
    this shape (partial discovery followed by a later failure), so no
    regression test existed to catch the loss.
- Finding: `_served_model_from_assistant` validated the model string was
  non-blank via `.strip()` but returned the original untrimmed value; the
  identical pattern was duplicated (and had the same bug) in
  `runner.py`'s `"final"`/`"error"` branches.
  - Fix: return the stripped value; deduplicated both `runner.py` branches
    into one shared `_served_model_from_metadata()` helper.
  - Evidence: `test_served_model_strips_whitespace` in
    `test_fcc_motor_served_model.py`.
- Finding: `response_identity` could be filled with a raw internal
  filesystem path (e.g. `/models/gguf/Qwen_Qwen3-8B-Q4_K_M.gguf`) instead
  of a friendly identity -- a user-facing "who answered" field, not an
  infra debug surface.
  - Fix: `_served_model_from_assistant` reduces the value to a basename
    with any weights-file extension (`.gguf`/`.bin`/`.safetensors`)
    stripped.
  - Evidence: `test_served_model_extracted_and_reduced_to_basename` in
    `test_fcc_motor_served_model.py`.
- Finding: the WS frame's `fcc_model_label` key now carries a dual-purpose
  value (usually the real served model, not strictly one of the fixed
  route aliases) under the same key name `HarnessRunRequestV1.fcc_model_label`
  uses strictly for the alias -- a future generic consumer could
  misinterpret it.
  - Fix: strengthened the comment at the assignment site in
    `turn_orchestrator.py`'s `_success_frames` to make the dual-purpose
    intent explicit for future readers. No live consumer currently
    misreads it (checked `app.js` -- it only *writes* this key on outgoing
    requests, never reads it off a response frame), so a rename was judged
    unnecessary scope for this PR.
  - Evidence: `orion/hub/turn_orchestrator.py` diff.
- Finding: the 3-line metadata-extraction block was copy-pasted verbatim
  into both `runner.py`'s `"final"` and `"error"` branches.
  - Fix: extracted into the shared `_served_model_from_metadata()` helper
    (also fixes the untrimmed-value bug above in one place).
  - Evidence: `orion/harness/runner.py` diff.

## Restart required

```bash
# Rebuild/restart orion-harness-governor and orion-hub to pick up the new code:
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

## Risks / concerns

- Severity: low
- Concern: real served-model discovery through the CLI's stream-json
  `message.model` field has not been confirmed live end-to-end (the
  `chat`/`agent` llama.cpp workers were down for this entire session). The
  mechanism is proven one layer down (direct `curl` to
  `orion-llm-gateway`'s `/v1/messages`), and
  `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1` is the documented flag for
  exactly this, but the CLI itself is closed-source, so there's a small
  chance it doesn't preserve the field the way expected.
  - Mitigation: falls back cleanly to the requested alias
    (`harness_req.fcc_model_label`) whenever discovery doesn't fire --
    never worse than before this PR, and the gap is a good live-verification
    target once the chat/agent workers recover. Recommend Juniper send a
    real message through Hub chat once workers are back up and check
    `chat_history_log.response_identity` for a real model name (e.g.
    `Qwen_Qwen3-8B-...`) rather than `MODEL_SONNET`/`MODEL_OPUS`.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/chat-history-unified-turn-model-label
