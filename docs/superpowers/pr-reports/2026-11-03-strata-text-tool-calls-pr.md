## Summary

Some Orion turns ended with a tool call written as plain text. The call never ran, and finalize then rewrote it into a summary of work that never happened. This patch makes the turn try again, and stops finalize from inventing that summary.

- When a turn's last answer ends with a `<tool_call>` block, the motor resumes the same session once and tells the model the call did not run. For an MCP tool the note says to load it with `ToolSearch` `select:<tool>` first. This works for warm chat turns and spawned turns.
- If the resumed answer is a text call again, the runner builds a "cut short" draft from what the turn really found. Finalize skips response repair for that draft. With no findings, the turn fails instead.
- The Context Mode brief in the motor prefix now says to load the `ctx_*` tools with ToolSearch before first use.
- New replay eval over the live failures: `python -m orion.harness.evals.text_tool_call_replay`.

## Outcome moved

Before: 23 turns from 2026-10-31 08:00 to 2026-11-03 ended on a text call to a context-mode tool. Each one went to finalize, and finalize wrote an answer that was not grounded in any result (example: 4bbaaf2a, 03:24Z, the "capacities" answer).
After, in a replay over the same 23 drafts: 23 of 23 are detected for the resume, and 23 of 23 resume notes name `ToolSearch select:<tool>`. If the resume fails too, 0 of 23 hand finalize a raw call or a draft it could repair (the replay runs the real runner on the call step plus the final).
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

- `orion/harness/fcc_motor.py`: `text_tool_call`, `text_tool_call_note`, `UNLOADED_TOOL_NOTE`, `OTHER_TEXT_TOOL_NOTE`. `_resume_need` and `_resume_held_final` replace the inline reading resume (main's undeclared-tool and not-JSON reasons are kept as they were). They are used by the spawn path and (new) the warm path.
- `orion/harness/runner.py`: a final that still ends on a text call becomes a cut-short draft (`fcc_text_tool_call`), or a motor failure when the turn recorded nothing.
- `orion/harness/cut_short.py`: `TEXT_TOOL_CALL_CODE`, its reason text, the code in `CUT_SHORT_CODES`, and `TurnFindings.drop_trailing_text_call`.
- `orion/fcc/self_index_brief.py`: the Context Mode brief tells the model to load the ctx tools first.
- `orion/harness/tests/test_fcc_text_tool_call.py` and `strata_text_tool_call_drafts.json`: new tests on four real drafts.
- `orion/harness/tests/test_fcc_reading_undeclared_tool.py`, `test_fcc_reading_json_resume.py`: the "only reading turns are resumed" test now asserts the new behavior; the resume tests put a transcript lookup in place, because a resume now needs the transcript on disk.
- `orion/harness/evals/text_tool_call_replay.py` and `test_text_tool_call_replay.py`: replay eval.
- `services/orion-harness-governor/README.md`: documents the path.

## Schema / bus / API changes

- Added: final-frame metadata `fcc_text_tool_call_resume` (resumed ordinary turns, with `held_metadata` from the first run) and `fcc_text_tool_call_resume_skipped`. Reading turns' `fcc_reading_resume_skipped` gains the reasons `no_transcript` and `mcp_config`.
- Removed: none.
- Renamed: `fcc_reading_resume_mcp_failed` log line is now `fcc_resume_mcp_failed`. Reading log lines are otherwise unchanged.
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
pytest orion/harness/tests orion/harness/evals orion/fcc/tests services/orion-harness-governor/tests
  -> 925 passed, 1 skipped, 1 deselected   (after rebase onto main + review fixes)
  (deselected: test_claude_spawn.py::test_no_undiscovered_root_callers_of_claude_permission_argv,
   fails the same way on clean main -- local untracked files)
pytest test_fcc_text_tool_call.py test_fcc_reading_undeclared_tool.py test_fcc_reading_json_resume.py -> 46 passed
```

## Evals run

```text
pytest orion/harness/evals/test_text_tool_call_replay.py -> 2 passed
python -m orion.harness.evals.text_tool_call_replay   (live harness_turn_trace since 2026-10-31 08:00)
{"turns": 23, "detected_for_resume": 23, "note_names_toolsearch_select": 23,
 "runner_if_resume_fails": {"motor_failed": 22, "cut_short_draft": 1},
 "raw_call_or_repairable_draft_reaches_finalize": 0,
 "tools": {"ctx_execute": 7, "ctx_batch_execute": 14, "ctx_search": 1, "ctx_execute_file": 1},
 "not_detected": []}
```

The replay feeds the runner only the final step, so "motor_failed" is the worst case. A real turn
with earlier tool results (4bbaaf2a had five Bash results) gets a cut-short draft of those results.

The replay proves detection and routing. It does not prove the model completes the step after the note. That is UNVERIFIED until a live turn hits this path.

## Docker/build/smoke checks

```text
Not run. Code-only change in the harness-governor image; no compose, env or dependency change.
```

## Review findings fixed

Code review ran in a subagent on the first commit. All should-fix findings are fixed:

- Finding: the cut-short draft repeated the raw `<tool_call>` text as a finding, and the "nothing recorded" branch could not happen live.
  - Fix: `TurnFindings.drop_trailing_text_call()` cuts the call out of the last note (prose before it stays). Runner tests now stream the call as a step, as the CLI does.
  - Evidence: `test_runner_turns_a_surviving_text_call_into_a_cut_short_draft` asserts `<tool_call>` is absent; `test_runner_fails_a_text_call_with_nothing_recorded` streams the step and still fails.
- Finding: durable runs marked the session completed before the resume, so a killed resume would not be resumed by the retry.
  - Fix: the session end is recorded on whichever frame finally leaves the motor.
  - Evidence: `test_a_killed_resume_of_a_durable_run_stays_resumable` (exit -9 -> `interrupted`, `plan_attempt(...).resume is True`).
- Finding: the resume did not check that the transcript exists; `claude --resume` would fail and replace a usable held final.
  - Fix: skip with `reason=no_transcript`; and a resume that fails on its own before any step keeps the held final (`reason=resume_failed`). Kills (exit < 0) still surface as errors for durable retry.
  - Evidence: `test_no_resume_without_a_transcript_keeps_the_held_final`, `test_a_resume_that_fails_on_its_own_keeps_the_held_final`.
- Finding: a text-call turn still published the clean-final fetch hint and skipped `record_step_failed`.
  - Fix: the runner branch sets `error_path_taken` and records the last step as failed.
  - Evidence: governor and harness suites pass.
- Finding: the eval's fields overstated what it measured.
  - Fix: renamed to `detected_for_resume`; the synthetic-findings metric is replaced by a run of the real runner on the call step plus final.
  - Evidence: eval output above.
- Nits fixed: warm env built in a try (keeps the held final on failure); held final cannot be dropped by a warm fallback; no `--append-system-prompt` on a warm resume; first run's metadata kept as `held_metadata`; warm test asserts MCP kwargs and spawn env.
- Nit checked, no change: chat replies have no reply budget, and held curiosity and reading turns already cap the motor below theirs, so a resume cannot starve finalize.

Rebase: main merged a concurrent change to the same reading-resume code (not-JSON resume). Both are kept in one seam.

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

- Severity: low. Concern: the warm-path resume (a spawn resuming a warm slot's transcript) is tested only with fakes. The transcript file for 4bbaaf2a's session was confirmed on disk in the governor's claude home, so the lookup should work. Live: UNVERIFIED.
- Severity: low. Concern: a resume adds one more CLI run (time and GPU) to a turn that ended on a text call. Mitigation: once only, and skipped when less than min(stall window, 60 s) is left.
- Severity: low. Concern: an answer that ends on a literal `</tool_call>` block (for example a reply that quotes one as the last thing) is resumed. Mitigation: the resumed answer is final; the note says the call did not run, which is true.
- Severity: medium. Concern: the model may ignore the note and write the text call again. Mitigation: the runner then ships a cut-short draft of real findings, not an invented summary.

## PR link

PR_LINK_PLACEHOLDER
