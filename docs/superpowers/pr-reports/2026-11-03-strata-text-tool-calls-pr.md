## Summary

Some Orion turns ended with a tool call written as plain text. The call never ran, and finalize then rewrote it into a summary of work that never happened. This patch makes the turn try again, and stops finalize from inventing that summary.

- When a turn's last answer ends with a `<tool_call>` block, the motor resumes the same session once and tells the model the call did not run. For an MCP tool the note says to load it with `ToolSearch` `select:<tool>` first. This works for warm chat turns and spawned turns.
- If the resumed answer is a text call again, the runner builds a "cut short" draft from what the turn really found. Finalize skips response repair for that draft. With no findings, the turn fails instead.
- The Context Mode brief in the motor prefix now says to load the `ctx_*` tools with ToolSearch before first use.
- New replay eval over the live failures: `python -m orion.harness.evals.text_tool_call_replay`.

## Outcome moved

Before: 23 turns from 2026-10-31 08:00 to 2026-11-03 ended on a text call to a context-mode tool. Each one went to finalize, and finalize wrote an answer that was not grounded in any result (example: 4bbaaf2a, 03:24Z, the "capacities" answer).
After, in a replay over the same 23 drafts: 23 of 23 are detected and resumed, and 23 of 23 resume notes name `ToolSearch select:<tool>`. If a resume fails, no turn reaches response repair.
Live effect after deploy: UNVERIFIED until the next such turn.

## Root cause (verified)

- FCC runs the claude CLI with `ENABLE_TOOL_SEARCH=true` (`orion/fcc/context_budget.py:214`). MCP tool schemas go to the engine only after `ToolSearch` loads them.
- Strata 0.2.7 (engine 0.1.52) reads a `<tool_call>` only as a call of a tool the request declared. The Anthropic route uses `declared_only=True` (`/opt/strata/terminal/serve/server.py` in the image). Anything else goes back as text and is counted. The chat engine's `/metrics` reports `tool_calls_unread: 23` since its restart. That is exactly the 23 failed turns.
- In every failed turn the model called a `ctx_*` tool without loading it first. In every successful `ctx_*` call in `cockpit_turn_sighting` the model loaded it with ToolSearch first. The call format was correct, so `tool_call_recovery` (#3021) was not the problem.
- The old code resumed only reading turns (`undeclared_tool_call`, docs/superpowers/pr-reports/2026-10-31-reading-turn-undeclared-tool-resume-pr.md). Ordinary turns ended on the text call.

## Current architecture

`fcc_motor.run_fcc_turn` drives one claude CLI turn (warm-pool process for chat replies, spawn for everything else). It held and resumed a final only on reading turns. `runner.HarnessRunner` took the final's text as the draft. Finalize's response repair rewrote any draft that 5b judged misaligned.

## Architecture touched

Harness motor and runner (harness-governor image). No bus, schema, or env changes.

## Files changed

- `orion/harness/fcc_motor.py`: `text_tool_call`, `text_tool_call_note`, `UNLOADED_TOOL_NOTE`, `OTHER_TEXT_TOOL_NOTE`. `_held_text_tool_call` and `_resume_text_tool_call` replace the inline reading-only resume. They are used by the spawn path and (new) the warm path.
- `orion/harness/runner.py`: a final that still ends on a text call becomes a cut-short draft (`fcc_text_tool_call`), or a motor failure when the turn recorded nothing.
- `orion/harness/cut_short.py`: `TEXT_TOOL_CALL_CODE`, its reason text, and the code in `CUT_SHORT_CODES`.
- `orion/fcc/self_index_brief.py`: the Context Mode brief tells the model to load the ctx tools first.
- `orion/harness/tests/test_fcc_text_tool_call.py` and `strata_text_tool_call_drafts.json`: new tests on four real drafts.
- `orion/harness/tests/test_fcc_reading_undeclared_tool.py`: the "only reading turns are resumed" test now asserts the new behavior.
- `orion/harness/evals/text_tool_call_replay.py` and `test_text_tool_call_replay.py`: replay eval.
- `services/orion-harness-governor/README.md`: documents the path.

## Schema / bus / API changes

- Added: none. The final-frame metadata `fcc_undeclared_tool_resume` (already used by reading turns) now also appears on resumed ordinary turns.
- Removed: none.
- Renamed: log line `fcc_reading_undeclared_tool_resume` is now `fcc_text_tool_call_resume` (no code or script greps for the old name).
- Behavior changed: new `grounding_status` value `fcc_text_tool_call`.
- Compatibility notes: none.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
pytest orion/harness/tests orion/fcc/tests  -> 773 passed, 1 skipped, 1 deselected
  (deselected: test_claude_spawn.py::test_no_undiscovered_root_callers_of_claude_permission_argv,
   fails the same way on clean main -- local untracked files)
pytest services/orion-harness-governor/tests -> 96 passed
pytest orion/harness/tests/test_fcc_text_tool_call.py orion/harness/tests/test_fcc_reading_undeclared_tool.py -> 19 passed
```

## Evals run

```text
pytest orion/harness/evals/test_text_tool_call_replay.py -> 2 passed
python -m orion.harness.evals.text_tool_call_replay   (live harness_turn_trace since 2026-10-31 08:00)
{"turns": 23, "resumed_now": 23, "note_names_toolsearch_select": 23,
 "kept_from_response_repair_if_resume_fails": 23,
 "tools": {"ctx_execute": 7, "ctx_batch_execute": 14, "ctx_search": 1, "ctx_execute_file": 1},
 "not_detected": []}
```

The replay proves detection and routing. It does not prove the model completes the step after the note. That is UNVERIFIED until a live turn hits this path.

## Docker/build/smoke checks

```text
Not run. Code-only change in the harness-governor image; no compose, env or dependency change.
```

## Review findings fixed

REVIEW_PLACEHOLDER

## Restart required

Deploy from the clean primary main checkout after merge (the post-pull rebuild covers it):

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

The circe Strata engines need no restart. Nothing changed on circe.

Live check after deploy:

```bash
docker logs --since 24h orion-athena-harness-governor 2>&1 | grep -E "fcc_text_tool_call_resume|fcc_text_tool_call_not_resumed|harness_text_tool_call_final"
```

## Risks / concerns

- Severity: low. Concern: a resume adds one more CLI run (time and GPU) to a turn that ended on a text call. Mitigation: once only, and skipped when less than min(stall window, 60 s) is left.
- Severity: low. Concern: an answer that ends on a literal `</tool_call>` block (for example a reply that quotes one as the last thing) is resumed. Mitigation: the resumed answer is final; the note says the call did not run, which is true.
- Severity: medium. Concern: the model may ignore the note and write the text call again. Mitigation: the runner then ships a cut-short draft of real findings, not an invented summary.

## PR link

PR_LINK_PLACEHOLDER
