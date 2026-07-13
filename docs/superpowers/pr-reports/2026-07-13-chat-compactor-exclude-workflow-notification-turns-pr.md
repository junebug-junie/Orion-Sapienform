# PR report — exclude workflow trigger/status turns from the chat compactor digest window (fix/chat-compactor-exclude-workflow-notification-turns)

## Summary

- Juniper ran a real chat compaction and got back a summary describing "compaction events" instead of actual conversation.
- Root cause: every Skill Runner workflow invocation (prompt = e.g. `"Compact the last N hours of chat..."`, response = the templated `"Workflow: ...\nStatus: ..."` block) gets logged into `chat_history_log` through the exact same pipeline as organic chat, with no distinguishing source or tag. The compactor's discussion-window fetch had no filter, so these round-trips — including the compactor's own trigger turns and any other workflow (dream cycle, schedule management, github compactor, etc.) invoked via chat — get swept into the digest window and correctly summarized as exactly what they are: talk about running workflows, not real conversation.
- Adds `exclude_workflow_notification_turns()` in `orion/cognition/chat_history_compactor/window.py`, which drops turns whose response matches one of the literal templates cortex-orch itself emits for workflow completion / failure / schedule-accepted / schedule-manage text. Matches on exact code-controlled prefixes only, so organic chat cannot false-positive.
- Wired into `_execute_chat_history_compactor_pass` immediately after the window fetch, so a window left with nothing but workflow noise is now correctly treated as quiet (no card/journal persisted) instead of digesting the noise.
- Filters at read time on response content shape rather than requiring a write-time tag, so it also retroactively cleans historical `chat_history_log` pollution (including test traffic from earlier live-verify sessions the same day) the next time any window covers it — no data migration needed.
- Companion fix landed the same day and merged first: `fix/workflow-notify-full-digest-not-truncated` (PR #1011) — stopped completion-notification emails from truncating the digest to 280 chars, and surfaced `digest.card_summary` into the workflow's `main_result` so it's present at all. That fix made the compactor's response *carry* real content; this fix makes sure that content is real conversation, not workflow chatter.

## Outcome moved

Chat digests now reflect what was actually discussed, not the fact that workflows were run. A window whose only activity was workflow triggers is honestly reported as quiet instead of fabricating a "here's what happened" summary out of its own status messages.

## Current architecture

`orion/discussion_window/sql_fetch.py`'s `fetch_discussion_window` reads `chat_history_log` for a bounded time window, keeps rows with both prompt and response, then trims to the most recent *contiguous* cluster (`_trim_contiguous_suffix`, breaking on any gap > 90 minutes) capped at `max_turns`. This is a shared, generic skill (`skills.chat.discussion_window.v1`) used by multiple consumers, not compactor-specific. `_execute_chat_history_compactor_pass` in `services/orion-cortex-orch/app/workflow_runtime.py` calls this skill, then feeds `window.turns` straight into `trim_chat_history_compactor_input` → the digest LLM, with zero filtering on turn content or origin.

## Architecture touched

- `orion/cognition/chat_history_compactor/window.py` — new `_is_workflow_notification_response`, `_rebuild_transcript_text`, `exclude_workflow_notification_turns` functions, co-located with the existing `resolve_chat_compactor_window`.
- `services/orion-cortex-orch/app/workflow_runtime.py` — one new call site: `window = exclude_workflow_notification_turns(window)` right after `DiscussionWindowResultV1.model_validate(skill_blob)` in `_execute_chat_history_compactor_pass`.

## Files changed

- `orion/cognition/chat_history_compactor/window.py`: adds the filter. Matches five literal response shapes cortex-orch controls: `_workflow_summary_text`'s `"Workflow: {title}\nStatus: {status}\n..."` block (used by all `_workflow_summary_text` call sites across every workflow), the pre-execution `"Workflow failed before completion."` literal, `_schedule_workflow_dispatch`'s `"Workflow request accepted: ..."` summary, `execute_workflow_schedule_management`'s `"Workflow schedule {op}: ok/failed"` summary, and `main.py`'s outer-exception fallback `"Workflow '{id}' failed and was not replaced with chat_general."`. All five verified against a subagent grep sweep of every `final_text=` assignment in `workflow_runtime.py` and `main.py` — no uncovered workflow-status shape remains.
- `orion/cognition/chat_history_compactor/tests/test_window.py`: 6 new tests — drops workflow-trigger round trips, does not false-positive on organic mentions of "workflow"/"schedule", covers the schedule-management and chat_general-fallback shapes, and confirms an all-noise window reduces to a fully empty (quiet) window.
- `services/orion-cortex-orch/app/workflow_runtime.py`: 2-line wiring change plus import.
- `services/orion-cortex-orch/tests/test_workflow_lane.py`: 1 new integration test (`test_chat_history_compactor_pass_window_of_only_workflow_triggers_is_treated_as_quiet`) reproducing the exact reported scenario end-to-end through `execute_chat_workflow` — a window with only workflow-trigger turns must skip card/journal persist entirely, matching quiet-day behavior. Verified **red** against pre-fix code (reverted the wiring, confirmed the test fails, restored the fix, confirmed green) before this was committed.

## Schema / bus / API changes

- Added: none (pure function, no new schema/bus/API surface).
- Removed: none.
- Renamed: none.
- Behavior changed: `_execute_chat_history_compactor_pass` now excludes workflow-notification-shaped turns from the digest window before quiet-check and digest generation.
- Compatibility notes: filter runs on already-fetched `DiscussionWindowResultV1` turns; does not touch `orion/discussion_window/sql_fetch.py` or the shared `skills.chat.discussion_window.v1` skill, so no other consumer of that skill is affected.

## Env/config changes

- None. No `.env_example`, `orion/bus/channels.yaml`, or `orion/schemas/registry.py` changes; no env sync required.

## Tests run

```text
source venv/bin/activate
python -m pytest orion/cognition/chat_history_compactor/tests -q
  → 15 passed

python -m pytest orion/cognition/chat_history_compactor/tests orion/cognition/compactor/tests \
    services/orion-cortex-orch/evals/test_chat_history_compactor_digest_eval.py \
    services/orion-cortex-orch/tests -k "compactor or notify or schedule" -q
  → 44 passed

Red-before-green check (temporarily reverted the exclude_workflow_notification_turns() call site):
python -m pytest services/orion-cortex-orch/tests/test_workflow_lane.py::test_chat_history_compactor_pass_window_of_only_workflow_triggers_is_treated_as_quiet -q
  → FAILED (confirms the new test catches the bug)
  → restored fix, re-ran, PASSED
```

## Evals run

No dedicated eval harness change needed — the existing deterministic eval lane at `services/orion-cortex-orch/evals/test_chat_history_compactor_digest_eval.py` covers digest budget/honesty and passed unchanged (5 passed, included in the 44 above).

## Docker/build/smoke checks

```text
docker compose --env-file .env --env-file services/orion-cortex-orch/.env \
  -f services/orion-cortex-orch/docker-compose.yml up -d --build
  → orion-athena-cortex-orch rebuilt and restarted

curl -fsS http://127.0.0.1:8080/health → {"status":"ok","service":"hub"}

Live verify against real (not synthetic) chat_history_log data:
POST /api/chat "Compact the last 24 hours of chat into a memory digest."
  → Result: "No Hub chat turns for chat_compactor:rolling:24h:...  Indexed chat digest
     card and journal entry skipped." (correctly quiet — confirmed via direct SQL query
     and a standalone fetch_discussion_window() call that the raw pre-filter window for
     this specific 24h span already contained only the 4 workflow-notification turns
     from earlier live-verify testing, walled off from the prior night's real conversation
     by the shared skill's pre-existing 90-minute contiguous-suffix cutoff — not something
     this patch introduced or could regress further; self-resolves on the next organic message)
```

## Review findings fixed

- Finding: `_is_workflow_notification_response` missed `execute_workflow_schedule_management`'s `"Workflow schedule {op}: ok/failed"` summary text (real, reachable when a user manages workflow schedules via chat).
  - Fix: added `first.startswith("Workflow schedule ")` check.
  - Evidence: `orion/cognition/chat_history_compactor/tests/test_window.py::test_exclude_workflow_notification_turns_covers_schedule_management_and_chat_general_fallback` passes; verified against `workflow_runtime.py:497`.
- Finding: `_is_workflow_notification_response` missed `main.py`'s outer-exception fallback `"Workflow '{id}' failed and was not replaced with chat_general."` — the exact text seen in this session's own earlier live-verify testing.
  - Fix: added `first.startswith("Workflow '") and "failed and was not replaced with chat_general" in first` check.
  - Evidence: same test as above; verified against `main.py:298`.
- Second review pass (independent subagent, searched every `final_text=` site in both files): no remaining uncovered workflow-status shape found; no false-positive risk identified for either new check.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-cortex-orch/.env \
  -f services/orion-cortex-orch/docker-compose.yml up -d --build
```

Already run during this session — `orion-athena-cortex-orch` is live on this fix as of this report.

## Risks / concerns

- Severity: low
- Concern: the filter runs *after* the shared skill's contiguous-suffix clustering, not before. A dense burst of workflow-trigger turns (as happened during this session's own testing) can still act as a temporal "wall" that keeps the 90-minute-gap cutoff from reaching back to real conversation on the other side of that burst — the window then correctly reports quiet (honest, not fabricated noise) but doesn't recover the older real content either.
- Mitigation: self-resolves on the next organic chat message, since it becomes the new tail of the contiguous cluster and the noise between it and now gets filtered out cleanly. A deeper fix (filtering workflow-noise turns before contiguity clustering, inside the shared `skills.chat.discussion_window.v1` skill or via a compactor-side bypass of the skill's built-in trim) would close this fully but touches shared, multi-consumer code — flagged to Juniper as a follow-up, not implemented here to keep this patch scoped to the reported bug.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/chat-compactor-exclude-workflow-notification-turns
