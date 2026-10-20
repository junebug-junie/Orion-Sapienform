## Summary

- The stance pass (the step that writes the imperative and tone the motor follows) only ever saw the current message, never the conversation before it.
- Hub now builds the timestamped RECENT CONVERSATION block before the stance call and passes it as `stance_inputs.recent_conversation` (each message capped at 400 chars).
- `stance_react.j2` tells the pass to read the message against that history: what it answers, corrects, or points back to.
- `orion/harness/prefix.py::_format_recent_turns` becomes public `format_recent_turns` with an optional per-message cap; the motor's copy is unchanged.

## Outcome moved

Live 2026-10-20 05:00Z (corr `98bde7cf`, cockpit `cockpit_turn_sighting`): Juniper wrote "That was on Sunday. Today is Monday :) It was eventful, thanks for asking!" ("it" = Lagoon, Sunday). Stance had only that message and wrote `imperative: "Meet Juniper's Monday update ... acknowledge the 'eventful' nature"`, `mind_work_shape.user_intent: "Juniper is updating the current day and time."`. The motor prompt (cockpit `motor_boot`) held the full timestamped history, including the Lagoon line and a 45 h gap marker, and still replied "Eventful Monday!". With this patch the stance pass reads the same history.

## Current architecture

`execute_unified_turn` built `stance_inputs = {user_message, utterance_origin, harness_user_message?}` and called stance_react; `recent_turns` was built afterwards, only for the harness motor.

## Architecture touched

Hub turn orchestrator (`orion/hub/turn_orchestrator.py`), stance prompt template, harness prefix renderer. No bus/schema change: `StanceReactRequestV1.stance_inputs` is `dict[str, Any]` and the template already renders every key.

## Files changed

- `orion/hub/turn_orchestrator.py`: build `recent_turns` before stance; add `stance_inputs["recent_conversation"]`.
- `orion/harness/prefix.py`: public `format_recent_turns(..., max_content_chars=None)`.
- `orion/cognition/prompts/stance_react.j2`: one guidance block, only when history is present.
- `services/orion-hub/tests/test_turn_orchestrator_stance_recent_conversation.py`: Lagoon-thread regression (timestamps, gap marker, own message stripped, cap applies to stance only).
- `orion/thought/tests/test_stance_react_prompt_imperative_discipline.py`: template renders history + guidance under StrictUndefined; nothing when absent.

## Schema / bus / API changes

- Added: `stance_inputs.recent_conversation` (list of rendered lines) inside the existing open dict.
- Removed: none
- Renamed: `_format_recent_turns` -> `format_recent_turns` (no outside callers).
- Behavior changed: stance prompt now includes recent dialogue.
- Compatibility notes: cortex-exec renders `stance_react.j2`; a Hub deployed before cortex-exec still shows the history through the generic loop (escaped), so deploy order does not matter.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
orion/thought/tests + orion/harness/tests                                  692 passed
services/orion-hub tests -k "turn_orchestrator or unified or stance or recent"  253 passed, 2 failed (both fail on main too:
  test_execute_unified_turn_uses_mind_appraisal_text_for_stance_not_harness, test_recent_effects_endpoint_returns_newest_first)
services/orion-thought tests                                               707 passed, 1 failed (fails on main too: test_mind_enrichment_defaults_off)
```

## Evals run

```text
None. Live check after deploy: a follow-up that refers back past the latest turn; the cockpit stance_decision imperative should name the earlier referent. UNVERIFIED until then.
```

## Docker/build/smoke checks

```text
None run.
```

## Review findings fixed

- Finding (low): the generic `stance_inputs` loop rendered the history through `tojson`, so the stance pass read an escaped one-line array (`\u2014`, `&#39;`).
  - Fix: the loop skips `recent_conversation`; the guidance block renders it as plain lines, once.
  - Evidence: `test_stance_prompt_reads_message_against_recent_conversation` asserts one plain occurrence and no escapes.
- Finding (low): guidance said "Juniper's local times" while the block labels her "User".
  - Fix: guidance says `"User" is Juniper`.
- Reviewer confirmed: other `execute_unified_turn` callers (outreach, curiosity, collapse mirror, reading) pass no history, so their stance passes are unchanged.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
```

## Risks / concerns

- Severity: low
- Concern: stance prompt grows by up to 8 messages x 400 chars (~3.5k chars).
- Mitigation: per-message cap; block omitted when there is no history.

## PR link

TBD

🤖 Generated with [Claude Code](https://claude.com/claude-code)
